import itertools
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial, reduce
from queue import Queue
from typing import List, Optional

import islpy as isl
import numpy as np
import sympy
from sympy import Matrix

import polycim.utils.mat_utils as inv
from polycim.op.base_vector import Base
from polycim.passes.base import DepthFirstPass, Schedule, SchedulePassResult
from polycim.passes.loop_padding import loop_padding_to_box_all


def add_constraints_exclude_null_space(base, exclude_null_space_of):
    if exclude_null_space_of is None:
        return base
    assert isinstance(
        exclude_null_space_of, sympy.Matrix
    ), f"{type(exclude_null_space_of)=}"
    if exclude_null_space_of.rows == 0:
        return base

    assert isinstance(base, isl.Set), f"{type(base)=}"

    row, col = exclude_null_space_of.shape
    new_base = None
    for row_idx in range(exclude_null_space_of.rows):
        cons_pos = isl.Constraint.inequality_alloc(base.get_space())
        cons_neg = isl.Constraint.inequality_alloc(base.get_space())
        for col_idx in range(exclude_null_space_of.cols):
            entry = int(exclude_null_space_of[row_idx, col_idx])
            cons_pos = cons_pos.set_coefficient_val(isl.dim_type.set, col_idx, entry)
            cons_neg = cons_neg.set_coefficient_val(isl.dim_type.set, col_idx, -entry)

        cons_pos = cons_pos.set_constant_val(isl.Val(-1))
        cons_neg = cons_neg.set_constant_val(isl.Val(-1))

        base_pos = base.add_constraint(cons_pos)
        base_neg = base.add_constraint(cons_neg)

        if new_base is None:
            new_base = base_pos.union(base_neg)
        else:
            new_base = new_base.union(base_pos)
            new_base = new_base.union(base_neg)

    return new_base


def add_constraints_positive_first_nonzero(set_):
    """
    a_0 >= 1 or
    (a_0=0 and a_1>=1) or
    (a_0=0 and a_1=0 and a_2>=1) or
    ......

    This constraint also exclude zero point (0,0,...,0)
    """
    n_dim = set_.dim(isl.dim_type.set)

    cons_eq_zero_list = []
    for j in range(0, n_dim):
        cons = isl.Constraint.equality_alloc(set_.get_space())
        cons = cons.set_coefficient_val(isl.dim_type.set, j, 1)
        cons_eq_zero_list.append(cons)

    new_set = None
    for i in range(n_dim):

        cons = isl.Constraint.inequality_alloc(set_.get_space())
        cons = cons.set_coefficient_val(isl.dim_type.set, i, 1)
        cons = cons.set_constant_val(isl.Val(-1))

        current_set = set_.add_constraints([*cons_eq_zero_list[:i], cons])
        if new_set is None:
            new_set = current_set
        else:
            new_set = new_set.union(current_set)

    return new_set


cache_for_find_base = {}


def _find_base_cache(n_dim, dim_sizes, min_reuse_factor, hyperplanes):
    global cache_for_find_base
    key = (n_dim, dim_sizes, min_reuse_factor, hyperplanes)
    if key in cache_for_find_base:
        return cache_for_find_base[key]

    dims = [f"j{i}" for i in range(n_dim)]
    dims_str = ",".join(dims)
    base = isl.Set(f"{{ [{dims_str}] }}")

    for hyperplane in hyperplanes:
        assert type(hyperplane) == tuple, f"{hyperplane=}"
        assert len(hyperplane) == n_dim, f"{len(hyperplane)=}"
        cons = isl.Constraint.equality_alloc(base.get_space())
        for i, coef in enumerate(hyperplane):
            cons = cons.set_coefficient_val(isl.dim_type.set, i, coef)
        base = base.add_constraint(cons)

    for i in range(n_dim):
        bound = dim_sizes[i] - 1  # // min_reuse_factor

        cons_lb = isl.Constraint.inequality_alloc(base.get_space())
        cons_lb = cons_lb.set_coefficient_val(isl.dim_type.set, i, min_reuse_factor)
        cons_lb = cons_lb.set_constant_val(isl.Val(bound))
        base = base.add_constraint(cons_lb)

        cons_ub = isl.Constraint.inequality_alloc(base.get_space())
        cons_ub = cons_ub.set_coefficient_val(isl.dim_type.set, i, -min_reuse_factor)
        cons_ub = cons_ub.set_constant_val(isl.Val(bound))
        base = base.add_constraint(cons_ub)

    base = add_constraints_positive_first_nonzero(base)
    cache_for_find_base[key] = base
    return base


second_array = False
call_find_base_times = 0


def find_base(
    n_dim, dim_sizes, min_reuse_factor, hyperplanes, exclude_null_space_of, lex_lt_set
):
    global second_array
    global call_find_base_times
    call_find_base_times += 1

    base = _find_base_cache(n_dim, dim_sizes, min_reuse_factor, hyperplanes)
    base = add_constraints_exclude_null_space(base, exclude_null_space_of)

    if lex_lt_set is not None:
        base = base.lex_lt_set(lex_lt_set).domain()
        assert type(base) == isl.Set, f"{type(base)=}"

    return base


def empty_or_zero_point(set_):
    return set_.count_val() == 0


def make_bases_to_matrix(bases):
    base_matrix = []
    for base in foreach_nontrival_point(bases):
        base_matrix.append(base)
    base_matrix = Matrix(base_matrix)
    base_matrix = base_matrix.transpose()
    return base_matrix


def find_base_with_max_reuse(
    n_dim,
    dim_sizes,
    max_reuse_factor,
    hyperplanes,
    exclude_null_space_of=None,
    lex_lt_set=None,
):
    global second_array
    """
    try reuse factor from 1 to max_reuse_factor, 
    """

    # binary search to find the largest reuse factor
    low = 1
    high = max_reuse_factor
    base = None

    while low < high:

        mid = (low + high + 1) // 2
        base = find_base(
            n_dim, dim_sizes, mid, hyperplanes, exclude_null_space_of, lex_lt_set
        )
        if empty_or_zero_point(base):
            high = mid - 1
        else:
            low = mid

    base = find_base(
        n_dim, dim_sizes, low, hyperplanes, exclude_null_space_of, lex_lt_set
    )
    # if second_array:
    #     import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    if base is not None and empty_or_zero_point(base):
        base = None

    # if base is not None and exclude_null_space_of is not None:
    #     np_base_matrix = np.array(make_bases_to_matrix(base))
    #     np_exclude_null_space_of = np.array(exclude_null_space_of)
    #     check = np.matmul(np_exclude_null_space_of , np_base_matrix)
    #     ge_zero_per_row = (check>=0).all(axis=1)
    #     le_zero_per_row = (check<=0).all(axis=1)
    #     ge_or_le_zero_per_row = np.logical_or(ge_zero_per_row, le_zero_per_row)
    #     assert ge_or_le_zero_per_row.all(), f"{check}"
    # print("")
    # print(f"{np_base_matrix=}")
    # print(f"{exclude_null_space_of=}")
    # print(f"{ge_zero_per_row=}")
    # print(f"{le_zero_per_row=}")
    # print(f"{hyperplanes=}")

    return base, low


@dataclass
class SearchStatus:
    bases: Matrix
    max_reuse: int
    final_row_as_set: isl.Set


def list_to_set(ls):
    ls = [str(i) for i in ls]
    set_ = isl.Set(f"{{ [{','.join(ls)}] }}")
    return set_


def find_bases_with_max_reuse(
    n_dim, dim_sizes, max_reuse_factor, hyperplanes, pre_bases=Matrix([[]])
):
    global second_array
    result = []
    queue = Queue()

    if pre_bases.rows > 0:
        orth_subspace = orthogonal_sub_space(pre_bases)
    else:
        orth_subspace = None

    bases, reuse = find_base_with_max_reuse(
        n_dim, dim_sizes, max_reuse_factor, hyperplanes, orth_subspace
    )

    # import pdb; pdb.set_trace()
    if bases is None:
        return result

    for base in foreach_nontrival_point(bases):
        search_status = SearchStatus(
            bases=pre_bases.col_join(Matrix([base])),
            max_reuse=reuse,
            final_row_as_set=list_to_set(base),
        )
        queue.put(search_status)

    while not queue.empty():
        search_status = queue.get()
        if search_status.bases.rows == n_dim:
            result.append(search_status)
            continue

        subspace = orthogonal_sub_space(search_status.bases)
        if subspace.rows == 0:
            result.append(search_status)
            continue

        new_bases, new_reuse = find_base_with_max_reuse(
            n_dim,
            dim_sizes,
            max_reuse_factor,  # // search_status.max_reuse,
            hyperplanes,
            exclude_null_space_of=subspace,
            lex_lt_set=search_status.final_row_as_set,
        )
        if new_bases is None:
            result.append(search_status)
            continue

        for new_base in foreach_nontrival_point(new_bases):
            new_search_status = SearchStatus(
                bases=search_status.bases.col_join(Matrix([new_base])),
                max_reuse=search_status.max_reuse * new_reuse,
                final_row_as_set=list_to_set(new_base),
            )
            queue.put(new_search_status)
    # import pdb; pdb.set_trace()
    return result


@dataclass
class MultiArraySearchStatus:
    search_status_per_array: List[SearchStatus]


def concat_bases(ma_search_status):
    search_status_per_array = ma_search_status.search_status_per_array
    base = search_status_per_array[0].bases
    for i in range(1, len(search_status_per_array)):
        base = base.col_join(search_status_per_array[i].bases)
    return base


def find_bases_for_multi_array_reuse(
    n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays
):
    global second_array
    assert type(max_reuse_factor_for_arrays) == tuple
    assert len(max_reuse_factor_for_arrays) == n_array
    assert type(hyperplanes_for_arrays) == tuple
    assert len(hyperplanes_for_arrays) == n_array
    assert type(dim_sizes) == tuple
    assert len(dim_sizes) == n_dim

    ma_result = []

    array_id = 0
    max_reuse_factor = max_reuse_factor_for_arrays[array_id]
    hyperplanes = hyperplanes_for_arrays[array_id]
    result = find_bases_with_max_reuse(n_dim, dim_sizes, max_reuse_factor, hyperplanes)

    for search_status in result:
        ma_search_status = MultiArraySearchStatus(
            search_status_per_array=[search_status]
        )
        ma_result.append(ma_search_status)

    second_array = True
    for array_id in range(1, n_array):
        new_ma_result = []

        max_reuse_factor = max_reuse_factor_for_arrays[array_id]
        hyperplanes = hyperplanes_for_arrays[array_id]
        for ma_search_status in ma_result:
            pre_bases = concat_bases(ma_search_status)
            result = find_bases_with_max_reuse(
                n_dim, dim_sizes, max_reuse_factor, hyperplanes, pre_bases
            )
            # import pdb; pdb.set_trace()
            for search_status in result:
                new_ma_search_status = MultiArraySearchStatus(
                    search_status_per_array=[
                        *ma_search_status.search_status_per_array,
                        search_status,
                    ]
                )
                new_ma_result.append(new_ma_search_status)

        ma_result = new_ma_result

    return ma_result


def record_points(point, record):
    multi_val = point.get_multi_val()
    if multi_val.is_zero():
        return
    val = [int(str(multi_val.get_val(i))) for i in range(len(multi_val))]
    record.append(val)


def foreach_nontrival_point(set_, return_isl_obj=False):
    points = []
    record_points_fn = partial(record_points, record=points)
    set_.foreach_point(record_points_fn)

    nonzero_count = []
    for point in points:
        nonzero_count.append(sum([1 for i in point if i != 0]))
    min_nonzero_count = min(nonzero_count)

    new_points = []
    for i in range(len(points)):
        if nonzero_count[i] == min_nonzero_count:
            new_points.append(points[i])
    # new_points = points

    return new_points


def orthogonal_sub_space(A):
    n = A.cols
    At = A.transpose()
    B = A * At

    B_inv = B.inv()
    C = At * B_inv * A
    D = Matrix.eye(n) - C

    E = inv.scale_to_integer(D)
    F = inv.find_independent_rows(E)

    return F


def base_to_coor_transform_matrix(bases_matrix):
    assert type(bases_matrix) == Matrix, f"{type(bases_matrix)=}"
    bases_matrix = bases_matrix.transpose()
    coor_transform_matrix = bases_matrix.inv()
    coor_transform_matrix = inv.scale_to_integer_per_row(coor_transform_matrix)
    return coor_transform_matrix


def matrix_to_schedule(coor_transform_matrix):
    """
    Each row of coor_transform_matrix is a coordinate transformation

    Rows top->down corresponds to the inner->outer of the domain iterators

    So we need to reverse the order of rows
    """
    domain_iters_def = ", ".join([f"i{i}" for i in range(coor_transform_matrix.rows)])
    range_iters_def = ", ".join([f"o{i}" for i in range(coor_transform_matrix.rows)])
    schedule = isl.BasicMap(f"{{ [{domain_iters_def}] -> [{range_iters_def}] }}")

    for row_idx in range(coor_transform_matrix.rows):
        cons = isl.Constraint.equality_alloc(schedule.get_space())
        for col_idx in range(coor_transform_matrix.cols):
            coor = coor_transform_matrix[row_idx, col_idx]
            cons = cons.set_coefficient_val(isl.dim_type.in_, col_idx, coor)

        cons = cons.set_coefficient_val(
            isl.dim_type.out, coor_transform_matrix.rows - row_idx - 1, -1
        )
        schedule = schedule.add_constraint(cons)
    return schedule


def base_to_coor_transform_schedule(bases_matrix):
    # assert bases_matrix.rows==len(iter_names), f"{bases_matrix.rows=}, {len(iter_names)=}"
    # print(f"{bases_matrix=}")
    coor_transform_matrix = base_to_coor_transform_matrix(bases_matrix)
    # print(f"{coor_transform_matrix=}")
    schedule = matrix_to_schedule(coor_transform_matrix)
    # print(f"{schedule=}")
    return schedule


def batch_base_to_coor_transform_schedule(bases_matrix_list):
    schedule_list = []
    for bases_matrix in bases_matrix_list:
        schedule = base_to_coor_transform_schedule(bases_matrix)
        schedule_list.append(schedule)
    return schedule_list


def padding_for_base_matrix(base_matrix):
    n_rows = base_matrix.rows
    n_cols = base_matrix.cols
    if n_rows == n_cols:
        return base_matrix
    assert n_rows < n_cols, f"{n_rows=}, {n_cols=}"
    I = Matrix.eye(n_cols)
    base_matrix = base_matrix.col_join(I)
    base_matrix_ = inv.find_independent_rows(base_matrix)
    return base_matrix_


def find_schedules_for_multi_array_reuse(
    n_dim,
    n_array,
    dim_sizes,
    max_reuse_factor_for_arrays,
    hyperplanes_for_arrays,
    return_detail=False,
):
    result = find_bases_for_multi_array_reuse(
        n_dim=n_dim,
        n_array=n_array,
        dim_sizes=dim_sizes,
        max_reuse_factor_for_arrays=max_reuse_factor_for_arrays,
        hyperplanes_for_arrays=hyperplanes_for_arrays,
    )
    base_matrixs = [
        ma_search_status.search_status_per_array[-1].bases
        for ma_search_status in result
    ]
    base_matrixs = [
        padding_for_base_matrix(base_matrix) for base_matrix in base_matrixs
    ]
    schedules = batch_base_to_coor_transform_schedule(base_matrixs)

    if return_detail:
        return schedules, base_matrixs

    return schedules


def parse_operator(domain, access_relations):
    """
    n_dim, n_array, dim_sizes, max_reuse_factor_for_arrays, hyperplanes_for_arrays
    """
    n_dim = domain.dim(isl.dim_type.set)
    n_array = len(access_relations)
    dim_sizes = [domain.dim_max_val(i).get_num_si() + 1 for i in range(n_dim)]

    max_reuse_factor_for_arrays = (max(dim_sizes), max(dim_sizes))

    hyperplanes_for_arrays = []
    for access_relation in access_relations:
        assert (
            access_relation.dim(isl.dim_type.in_) == n_dim
        ), f"{access_relation.dim(isl.dim_type.in_)=}, {n_dim=}"
        pma = access_relation.as_pw_multi_aff()
        assert pma.n_piece() == 1, f"{len(mas)=}"

        mas = []
        pma.foreach_piece(lambda x, y: mas.append(y))
        assert len(mas) == 1
        ma = mas[0]
        hyperplanes = []
        # print(f"{ma=}, {ma.n_piece()=}")
        for aff_idx in range(ma.get_list().n_aff()):
            aff = ma.get_aff(aff_idx)
            coef = []
            for i in range(aff.dim(isl.dim_type.in_)):
                coef.append(int(str(aff.get_coefficient_val(isl.dim_type.in_, i))))
            hyperplanes.append(tuple(coef))
        hyperplanes_for_arrays.append(tuple(hyperplanes))
    return (
        n_dim,
        n_array,
        tuple(dim_sizes),
        tuple(max_reuse_factor_for_arrays),
        tuple(hyperplanes_for_arrays),
    )


def shift_to_positive(op):
    domain = op.domain
    min_val = [
        domain.dim_min_val(i).get_num_si() for i in range(domain.dim(isl.dim_type.set))
    ]
    shift = [0 if val >= 0 else -val for val in min_val]
    # shift = [str(val) for val in shift]
    # shift = ",".join(shift)

    shift_domain = ",".join([f"i{i}" for i in range(domain.dim(isl.dim_type.set))])
    shift_range = ",".join(
        [f"i{i} + {shift[i]}" for i in range(domain.dim(isl.dim_type.set))]
    )
    shift = isl.BasicMap(f"{{ [{shift_domain}] -> [{shift_range}] }}")
    new_op = op.apply_schedule(shift, name="shift_to_positive")

    return new_op


def is_base_trival(point, dim_types):
    assert len(point) == len(dim_types), f"{point=} {dim_types=}"
    use_dim_types = []
    for p, dim_type in zip(point, dim_types):
        if p != 0:
            use_dim_types.append(dim_type)

    if len(use_dim_types) == 1:
        return True
    if (
        len(use_dim_types) == 2
        and (
            "oh_i" in use_dim_types
            or "ow_i" in use_dim_types
            or "oh" in use_dim_types
            or "ow" in use_dim_types
        )
        and ("kh" in use_dim_types or "kw" in use_dim_types)
    ):
        return True
    return False


def cover(corr1, corr2):
    nonzero1 = [int(i != 0) for i in corr1]
    nonzero2 = [int(i != 0) for i in corr2]
    return all(i1 >= i2 for i1, i2 in zip(nonzero1, nonzero2)) and any(
        i1 > i2 for i1, i2 in zip(nonzero1, nonzero2)
    )


def record_points(point, record):
    multi_val = point.get_multi_val()
    if multi_val.is_zero():
        return
    val = [int(str(multi_val.get_val(i))) for i in range(len(multi_val))]
    record.append(val)


def get_nontrival_points(set_):
    points = []
    record_points_fn = partial(record_points, record=points)
    set_.foreach_point(record_points_fn)
    return points


def get_rank_key(base_combination):
    num_non_zero = 0
    for base in base_combination:
        if base.reuse_array_id in [0, 1]:
            nonzero = [int(i != 0) for i in base.corrdinate]
            num_non_zero += sum(nonzero)
    return num_non_zero


def select_full_rank_bases(bases):
    n_dim = len(bases[0].corrdinate)
    matrix = np.array([base.corrdinate for base in bases])

    # full_rank_bases_list = []
    cnt = 0

    def select_single_base(selected_indices, matrix_bases, begin_base_idx):
        selected_matrix = matrix_bases[selected_indices]

        if len(selected_indices) == n_dim:
            if cnt % 10000 == 0:
                print(f"{cnt=}")
            # assert np.linalg.det(selected_matrix) != 0
            # full_rank_bases_list.append(selected_indices)
            cnt += 1
            return

        for i in range(begin_base_idx, len(matrix_bases)):
            # print(f"{selected_matrix.shape=}, {matrix_bases[i].shape=}")
            concated_matrix = np.concatenate(
                (selected_matrix, matrix_bases[i : i + 1, :]), axis=0
            )
            max_rank = min(concated_matrix.shape[0], concated_matrix.shape[1])
            if np.linalg.matrix_rank(concated_matrix) == max_rank:
                select_single_base([*selected_indices, i], matrix_bases, i + 1)

    import pdb

    pdb.set_trace()
    begin_time = time.time()
    select_single_base([], matrix, 0)
    end_time = time.time()
    print(f"time: {end_time - begin_time}")
    import pdb

    pdb.set_trace()
    return full_rank_bases_list


def select_bases(self, bases, operator):
    n_dim = operator.domain.dim(isl.dim_type.set)
    selected_bases = []

    # select_full_rank_bases(bases)

    # Find all combinations of n_dim bases
    base_combinations = itertools.combinations(bases, n_dim)

    for i, combination in enumerate(base_combinations):

        if satisfies_constraints(self, combination, operator):
            if self.prune == False and len(selected_bases) % 10 == 0:
                print(f"{len(selected_bases)=}")
            rank_key = get_rank_key(combination)
            selected_bases.append((rank_key, combination))
    selected_bases = sorted(selected_bases, key=lambda x: x[0])
    selected_bases = [x[1] for x in selected_bases]
    return selected_bases


def satisfies_constraints(self, combination, operator):
    # Extract the coordinates from each item in the combination
    matrix = np.array([item.corrdinate for item in combination])

    # check reuse constraint
    # reuse_ids = {item.reuse_array_id for item in combination}
    # if not (0 in reuse_ids and 1 in reuse_ids):
    #     return False
    reuse_ids = {0: 0, 1: 0}
    reuse_by_skewed_base_ids = {0: 0, 1: 0}
    for item in combination:
        nonzero_count = sum([int(i != 0) for i in item.corrdinate])
        is_skewed_base = nonzero_count >= 2
        if item.reuse_array_id in (0, 1):
            reuse_ids[item.reuse_array_id] += 1
            if is_skewed_base:
                reuse_by_skewed_base_ids[item.reuse_array_id] += 1
    if reuse_ids[0] < 1 or reuse_ids[1] < 1:
        return False
    if reuse_by_skewed_base_ids[0] > 2 or reuse_by_skewed_base_ids[1] > 2:
        return False

    # Check if the matrix is square and invertible
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Calculate the determinant to check if the matrix is invertible
    if np.linalg.det(matrix) == 0:
        return False
    # return True

    if True and self.prune:
        # filter by tiles
        reuse_bases = [base for base in combination if base.reuse_array_id in (0, 1)]
        participate_skewing_dims = set()
        for base in reuse_bases:
            nonzero_dims = [i for i, corr in enumerate(base.corrdinate) if corr != 0]
            if len(nonzero_dims) > 1:
                participate_skewing_dims.update(nonzero_dims)
        tile_sizes = operator.attr["pre_tile_sizes"]

        cur_dim = 0
        tile_check_is_valid = True
        for i, tile_size in enumerate(tile_sizes):
            if len(tile_size) > 1:
                is_valid = False
                for j in range(len(tile_size)):
                    _dim = cur_dim + j
                    if _dim in participate_skewing_dims:
                        is_valid = True
                        break

                if not is_valid:
                    tile_check_is_valid = False
                    break

            cur_dim += len(tile_size)

        if not tile_check_is_valid:
            return False

    return True


def filter_cover_points(points):
    new_points = []
    for point in points:
        can_cover = False
        for other_point in points:
            if point != other_point and cover(point, other_point):
                can_cover = True
                break
        if not can_cover:
            new_points.append(point)
    return new_points


def filter_times_points(points):
    """
    For any two points A and B, if there exists a positive integer k such that A = B*k, then A is not valid.
    """
    filtered_points = []

    for i, point_a in enumerate(points):
        nonzero_elements = [i for i in point_a if i != 0]
        # gcd of nonzero_elements
        gcd = reduce(lambda x, y: math.gcd(x, y), nonzero_elements)
        if gcd == 1:
            filtered_points.append(point_a)
    return filtered_points


def filter_tile_direction_points(points, tile_sizes):

    tile_pair = []
    i_iter = 0
    for tile_size in tile_sizes:
        if len(tile_size) == 2:
            tile_pair.append((i_iter, i_iter + 1))
        elif len(tile_size) == 3:
            tile_pair.append((i_iter, i_iter + 1))
            tile_pair.append((i_iter, i_iter + 2))
            tile_pair.append((i_iter + 1, i_iter + 2))
        i_iter += len(tile_size)

    new_points = []
    for point in points:
        nonzero = [int(i != 0) for i in point]
        is_valid = True
        for tile_out_iter, tile_in_iter in tile_pair:
            if nonzero[tile_out_iter] == 1 and nonzero[tile_in_iter] == 1:
                is_valid = False
                break
        if is_valid:
            new_points.append(point)
    return new_points


def filter_many_direction_points(points):
    new_points = []
    for point in points:
        nonzero = [int(i != 0) for i in point]
        if sum(nonzero) <= 2:
            new_points.append(point)
    return new_points


def extend_scalar_dim_for_operator(operator):
    n_dim = operator.domain.dim(isl.dim_type.set)
    origin_dims = [f"i{i}" for i in range(n_dim)]
    new_dims = ["0", *origin_dims]
    origin_dims_str = ",".join(origin_dims)
    new_dims_str = ",".join(new_dims)
    extend_dim_scheduel = isl.BasicMap(f"{{ [{origin_dims_str}] -> [{new_dims_str}] }}")
    new_op = operator.apply_schedule(extend_dim_scheduel, name="extend_dim")
    return new_op


def extend_scalar_dim_for_bases(bases, reuse_array_id):
    assert reuse_array_id in (0, 1)
    n_dim = len(bases[0].corrdinate)
    new_bases = [Base([1, *([0] * n_dim)], reuse_array_id, True)]
    for base in bases:
        new_corrdinate = tuple([0] + list(base.corrdinate))
        new_bases.append(Base(new_corrdinate, base.reuse_array_id, base.is_trival))
    return new_bases


def create_scalar_axis_for_reuse(bases, operator):
    reuse_cnt = {
        0: len(
            [base for base in bases if base.reuse_array_id == 0 and not base.is_skewed]
        ),
        1: len(
            [base for base in bases if base.reuse_array_id == 1 and not base.is_skewed]
        ),
    }

    if reuse_cnt[0] == 0:
        operator = extend_scalar_dim_for_operator(operator)
        bases = extend_scalar_dim_for_bases(bases, 0)
        operator.set_attr(
            "pre_tile_sizes", ((1,), *operator.attr["pre_tile_sizes"]), overwrite=True
        )
        operator.set_attr(
            "dim_types", ["_"] + operator.attr["dim_types"], overwrite=True
        )

    if reuse_cnt[1] == 0:
        operator = extend_scalar_dim_for_operator(operator)
        bases = extend_scalar_dim_for_bases(bases, 1)
        operator.set_attr(
            "pre_tile_sizes", ((1,), *operator.attr["pre_tile_sizes"]), overwrite=True
        )
        operator.set_attr(
            "dim_types", ["_"] + operator.attr["dim_types"], overwrite=True
        )

    return bases, operator


class AffineSchedule(Schedule):
    def __init__(self, bases):
        super().__init__()
        self.bases = bases

    def dumps(self):
        result = []
        for base in self.bases:
            result.append(
                {
                    "corrdinate": base.corrdinate,
                    "reuse_array_id": base.reuse_array_id,
                    "is_trival": base.is_trival,
                }
            )
        return json.dumps(result)

    def parse(self, data):
        assert isinstance(data, list)
        self.bases = []
        for item in data:
            self.bases.append(
                Base(item["corrdinate"], item["reuse_array_id"], item["is_trival"])
            )


class AffinePass(DepthFirstPass):
    def __init__(
        self,
        args,
        fix_schedule: Optional[AffineSchedule] = None,
        schedule_as_key: bool = False,
        pad: bool = True,
        prune: bool = True,
    ):
        super().__init__(fix_schedule=fix_schedule, schedule_as_key=schedule_as_key)
        self.args = args
        self.pad = pad
        self.prune = prune
        assert self.fix_schedule is None or isinstance(
            self.fix_schedule, AffineSchedule
        )

    def base_construction(self, operator):
        (
            n_dim,
            n_array,
            dim_sizes,
            max_reuse_factor_for_arrays,
            hyperplanes_for_arrays,
        ) = parse_operator(
            domain=operator.domain,
            access_relations=[operator.access_I, operator.access_O],
        )
        bases = OrderedDict()
        for array_id in [0, 1]:
            hyperplanes = hyperplanes_for_arrays[array_id]
            reuse_bases = find_base(
                n_dim=n_dim,
                dim_sizes=dim_sizes,
                min_reuse_factor=1,
                hyperplanes=hyperplanes,
                exclude_null_space_of=None,
                lex_lt_set=None,
            )
            # import pdb; pdb.set_trace()
            points = get_nontrival_points(reuse_bases)
            points = filter_times_points(points)
            if self.prune:
                points = filter_tile_direction_points(
                    points, operator.attr["pre_tile_sizes"]
                )
                points = filter_cover_points(points)
                points = filter_many_direction_points(points)
            # TODO: filter points
            for point in points:
                is_trival = is_base_trival(point, operator.attr["dim_types"])
                base = Base(point, array_id, is_trival)
                bases[base] = None

        for i in range(n_dim):
            point = [0 for d in range(n_dim)]
            point[i] = 1
            base = Base(tuple(point), -1, True)
            bases[base] = None

        bases = list(bases.keys())

        return bases

    def disable_skewing(self, bases, operator):
        if self.args.polycim_disable_affine:
            new_bases = []
            for base in bases:
                num_nonzero = sum([int(i != 0) for i in base.corrdinate])
                if num_nonzero == 1:
                    new_bases.append(base)

            return new_bases
        return bases

    def apply(self, operator):

        if self.fix_schedule is None:
            # 1. base construction
            bases = self.base_construction(operator)

            bases = self.disable_skewing(bases, operator)

            bases, operator = create_scalar_axis_for_reuse(bases, operator)

            # 2. base selection
            # select n_dim base from bases that satisfy:
            #  constraint1: for each array, at least one base is selected to reuse it.
            #  constraint2: the selected bases are linear independent
            # find all selection combinations
            selected_bases_list = select_bases(self, bases, operator)

        else:
            selected_bases_list = [self.fix_schedule.bases]

        # 3. affine transform
        result_list = []
        for selected_bases in selected_bases_list:
            rev_selected_bases = tuple(list(selected_bases)[::-1])
            corrdinates = [base.corrdinate for base in rev_selected_bases]
            matrix = Matrix(corrdinates)

            schedule = base_to_coor_transform_schedule(matrix)

            new_op = operator.apply_schedule(schedule, name="affine")
            # print(new_op.attr.get("affine::bases", "None"))
            # import pdb; pdb.set_trace()
            new_op = shift_to_positive(new_op)
            if self.pad:
                new_op = loop_padding_to_box_all(new_op)

            base_str = ""
            for base in selected_bases:
                base_str += str(base) + "\n"
            new_op.history_schedules.append({"bases": base_str})
            new_op.history_schedules.append({"matrix": matrix})

            new_op.set_attr(
                "AffinePass", {"bases": selected_bases, "schedule": str(schedule)}
            )

            affine_schedule = AffineSchedule(selected_bases)
            result = SchedulePassResult(new_op, affine_schedule)
            result_list.append(result)

        return result_list
