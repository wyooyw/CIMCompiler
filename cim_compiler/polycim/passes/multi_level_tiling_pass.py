import itertools
import json
from typing import Optional

import islpy as isl

import polycim.utils.utils as utils
from polycim.op.base_operator import BasicOperator
from polycim.passes.base import DepthFirstPass, Schedule, SchedulePassResult
from polycim.utils.logger import level_tqdm
from polycim.utils.math import factorize


def multiply(factors):
    result = 1
    for factor in factors:
        result *= factor
    return result


def multi_level_tiling(operator, tiling_level, tiling_factors):
    """
    operator = Operator(
        domain=isl.BasicSet("{ [i,j]: 0 <= i < 8 and 0 <= j < 3 }"),
        acc_rel_out=isl.BasicMap("{ [i,j] -> C[i] }"),
        acc_rel_lhs=isl.BasicMap("{ [i,j] -> A[i + j] }"),
        acc_rel_rhs=isl.BasicMap("{ [i,j] -> B[i + j] }")
    )

    tiling_factors:
    [
        [2,2],
        [1,4]
    ]
    """
    param_names = operator.domain.get_var_names(isl.dim_type.param)

    domain = operator.domain

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set)
    assert (
        len(tiling_factors) == n_iter
    ), f"len(tiling_factors)={len(tiling_factors)} != n_iter={n_iter}"

    domain_shape = utils.get_static_box_shape(domain)

    # do checks
    for i in range(n_iter):
        assert (
            len(tiling_factors[i]) == tiling_level
        ), f"len(tiling_factors[{i}])={len(tiling_factors[i])} != tiling_level={tiling_level}"
        dim_size = domain_shape[i]
        assert (
            multiply(tiling_factors[i]) == dim_size
        ), f"multiply(tiling_factors[{i}])={multiply(tiling_factors[i])} != dim_size={dim_size}"

    # do tiling
    tiling_maps = []
    remain_factors = [*domain_shape]
    for l in range(tiling_level - 1):
        n_keep_iters = l * n_iter
        keep_iter_names = [f"i{i}" for i in range(n_keep_iters)]
        change_iter_names = [
            f"i{i}" for i in range(n_keep_iters, n_keep_iters + n_iter)
        ]

        outer_iters = []
        inner_iters = []
        for i in range(n_iter):
            factor = tiling_factors[i][l]
            remain_factors[i] = remain_factors[i] // factor
            outer_iters.append(f"floor({change_iter_names[i]}/{remain_factors[i]})")
            inner_iters.append(f"{change_iter_names[i]}%{remain_factors[i]}")

        tiling_map_def = f"[{','.join(param_names)}] -> {{ [{','.join(keep_iter_names + change_iter_names)}] -> [{','.join(keep_iter_names + outer_iters + inner_iters)}] }}"
        tiling_map = isl.BasicMap(tiling_map_def)
        # print(f"{tiling_map_def=}")
        tiling_maps.append(tiling_map)

    # import pdb; pdb.set_trace()
    # print("\n")
    tiling_map = tiling_maps[0].intersect_domain(domain)
    for _tiling_map in tiling_maps[1:]:
        tiling_map = tiling_map.apply_range(_tiling_map)

    tiling_map = tiling_map.intersect_domain(domain)

    new_operator = operator.apply_schedule(tiling_map)
    new_operator.history_schedules.append({"tiling_factors": tiling_factors})
    return new_operator


def multi_level_splitting_var_level(operator, tiling_factors, skip_simplify=False):
    """
    tiling_factors:
    [
        [2,2], # for first dim, level = 2
        [4], # for second dim, level = 1
        [2,4,2] # for third dim, level = 3
    ]
    """
    param_names = operator.domain.get_var_names(isl.dim_type.param)

    domain = operator.domain

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set)
    assert (
        len(tiling_factors) == n_iter
    ), f"len(tiling_factors)={len(tiling_factors)} != n_iter={n_iter}"

    domain_shape = utils.get_static_box_shape(domain)

    # do checks
    for i in range(n_iter):
        assert len(tiling_factors[i]) >= 1
        dim_size = domain_shape[i]
        assert (
            multiply(tiling_factors[i]) == dim_size
        ), f"multiply(tiling_factors[{i}])={multiply(tiling_factors[i])} != dim_size={dim_size}"

    # do tiling
    tiling_maps = []
    remain_factors = [*domain_shape]
    split_iter = 0
    total_iter = n_iter
    for i in range(n_iter):
        level = len(tiling_factors[i])
        for l in range(level - 1):
            before_iter_names = [f"i{i}" for i in range(split_iter)]
            split_iter_name = f"i{split_iter}"
            after_iter_names = [f"i{i}" for i in range(split_iter + 1, total_iter)]

            remain_factors[i] = remain_factors[i] // tiling_factors[i][l]
            outer_iter = f"floor({split_iter_name}/{remain_factors[i]})"
            inner_iter = f"{split_iter_name} % {remain_factors[i]}"
            tiling_map_def = f"[{','.join(param_names)}] -> {{ [{','.join( before_iter_names + [split_iter_name] + after_iter_names )}] -> [{','.join( before_iter_names + [outer_iter, inner_iter] + after_iter_names )}] }}"
            tiling_map = isl.BasicMap(tiling_map_def)
            # print(f"{tiling_map_def=}")
            tiling_maps.append(tiling_map)

            split_iter += 1
            total_iter += 1

        split_iter += 1

    if len(tiling_maps) > 0:
        tiling_map = tiling_maps[0].intersect_domain(domain)
        for _tiling_map in tiling_maps[1:]:
            tiling_map = tiling_map.apply_range(_tiling_map)
        tiling_map = tiling_map.intersect_domain(domain)
    else:
        domain_iters = [f"i{i}" for i in range(total_iter)]
        tiling_map = isl.BasicMap(
            f"[{','.join(param_names)}] -> {{ [{','.join(domain_iters)}] -> [{','.join(domain_iters)}] }}"
        )
        tiling_map = tiling_map.intersect_domain(domain)

    new_operator = operator.apply_schedule(
        tiling_map, name="pre_tiling", skip_simplify=True
    )
    new_operator.history_schedules.append({"tiling_factors": tiling_factors})
    return new_operator


def combine_tilesize_by_symmetry_info(dim_factors, symmetry_info):
    """
    symmetry_info: 2d tuple of integers
     ((1, 3), (2, 4))
    """
    # check symmetry_info is a box
    assert type(symmetry_info) == tuple
    assert all(type(group) == tuple for group in symmetry_info)
    assert all(type(dim) == int for group in symmetry_info for dim in group)

    assert len(symmetry_info) >= 2
    assert all(
        len(symmetry_info[i]) == len(symmetry_info[0])
        for i in range(len(symmetry_info[0]))
    )

    # get symmetry dim and non-symmetry dim
    n_dims = len(dim_factors)
    symmetry_dims = [
        symmetry_info[i][j]
        for i in range(len(symmetry_info))
        for j in range(len(symmetry_info[i]))
    ]
    non_symmetry_dims = [i for i in range(n_dims) if i not in symmetry_dims]
    assert len(symmetry_dims) == len(set(symmetry_dims))
    assert len(symmetry_dims) + len(non_symmetry_dims) == n_dims

    # dim_group_product_factors: product of the tile size of dims in one group
    # for example: dim 0 and dim 1 are in the same group,
    #   dim 0 has 3 tile size selections,
    #   dim 1 has 2 tile size selections,
    #   then the product of the tile size of dims in this group is 3*2=6
    # use index to represet the tile size
    dim_indices = symmetry_info[0]
    dim_group_factors = []
    for dim_index in dim_indices:
        dim_group_factors.append(list(range(len(dim_factors[dim_index]))))
    dim_group_product_factors = list(itertools.product(*dim_group_factors))
    n_tilesize_per_group = len(dim_group_product_factors)

    n_groups = len(symmetry_info)
    # assert n_groups==2

    # symmetry tile_size_combinations: combination of tile size of symmetry indices.
    # - element in tile_size_combinations: different tile size strategy
    # - each strategy is a 2d tuple, same shape as symmetry_info
    # - each element in the tuple is an integer, which is the index of the tile size in the dim_factors
    symmetry_tile_size_combinations = []

    def nested_for(depth, depth_end, begin, end, indices):
        if depth == depth_end:
            tile_size_combination = []
            for i in indices:
                tile_size_combination.extend(dim_group_product_factors[i])
            symmetry_tile_size_combinations.append(tuple(tile_size_combination))
            return
        for i in range(begin, end):
            nested_for(depth + 1, depth_end, i, end, [*indices, i])

    nested_for(0, n_groups, 0, n_tilesize_per_group, [])
    # for tile_size_combination_index_0 in range(0, n_tilesize_per_group):
    #     for tile_size_combination_index_1 in range(tile_size_combination_index_0, n_tilesize_per_group):
    #         tile_size_combination = (*dim_group_product_factors[tile_size_combination_index_0], *dim_group_product_factors[tile_size_combination_index_1])
    #         symmetry_tile_size_combinations.append(tile_size_combination)

    # get non-symmetry tile size combinations
    non_symmetry_dim_factor_indices = [
        list(range(len(dim_factors[i]))) for i in non_symmetry_dims
    ]
    non_symmetry_tile_size_combinations = list(
        itertools.product(*non_symmetry_dim_factor_indices)
    )

    # combine symmetry and non-symmetry tile size combinations
    tile_size_combinations = []
    tile_idx_combinations = list(
        itertools.product(
            symmetry_tile_size_combinations, non_symmetry_tile_size_combinations
        )
    )
    for (
        symmetry_tile_size_combination,
        non_symmetry_tile_size_combination,
    ) in tile_idx_combinations:
        tile_size_combination = dict()
        assert len(symmetry_dims) == len(symmetry_tile_size_combination)
        for dim, tile_size_idx in zip(symmetry_dims, symmetry_tile_size_combination):
            tile_size = dim_factors[dim][tile_size_idx]
            tile_size_combination[dim] = tile_size
        for dim, tile_size_idx in zip(
            non_symmetry_dims, non_symmetry_tile_size_combination
        ):
            tile_size = dim_factors[dim][tile_size_idx]
            tile_size_combination[dim] = tile_size
        tile_size_combination = tuple(tile_size_combination[i] for i in range(n_dims))
        tile_size_combinations.append(tile_size_combination)
    assert len(tile_size_combinations) == len(set(tile_size_combinations))

    # check
    all_combinations = list(itertools.product(*dim_factors))
    assert set(tile_size_combinations).issubset(set(all_combinations))

    return tile_size_combinations


def filter_factors(factors):
    """
    filter factor like [1, 1, 4], actually this is not tiling
    """
    new_factors = []
    for factor in factors:
        if factor[0] in (1, 2) or factor[1] in (1, 2):
            new_factors.append(factor)
    return new_factors


def filter_factors_for_3x3_5x5(factors):
    """
    filter factor like [1, 1, 4], actually this is not tiling
    """
    new_factors = []
    for factor in factors:
        if factor[0] in (1, 2, 4) or factor[1] in (1, 2, 4):
            new_factors.append(factor)
    return new_factors


def filter_factors_of_all_axis(combination_list):
    new_combination_list = []
    for all_axis_factor in combination_list:
        count_tiling_axis = 0
        num_axis = len(all_axis_factor)
        for factor in all_axis_factor:
            if factor[0] != 1 and factor[1] != 1:
                count_tiling_axis += 1

        if count_tiling_axis <= num_axis // 2:
            new_combination_list.append(all_axis_factor)
    return new_combination_list


def enumerate_tiling_factors(operator, tiling_factor):
    domain = operator.domain

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set)

    domain_shape = utils.get_static_box_shape(domain)
    dim_factors = []
    for dim_size in domain_shape:
        factors = factorize(dim_size, tiling_factor)
        #
        factors = [factor for factor in factors if factor[-1] != 1 or max(factor) == 1]
        # factors = filter_factors(factors)
        factors = filter_factors_for_3x3_5x5(factors)
        # print(f"{len(factors)=}, {factors=}")
        dim_factors.append(factors)

    combination_list = list(itertools.product(*dim_factors))
    combination_list = filter_factors_of_all_axis(combination_list)
    # import pdb; pdb.set_trace()
    for combination in level_tqdm(combination_list):
        new_operator = multi_level_tiling(operator, tiling_factor, combination)
        yield new_operator


def pre_tiling_pass(op_list):
    new_op_list = []
    for op in op_list:
        new_op_list.append(op)

        for new_op in enumerate_tiling_factors(op, 2):
            new_op_list.append(new_op)
    # new_op_list = new_op_list[:40]
    # print(len(new_op_list))
    # exit()
    return new_op_list


def multi_level_tiling_outer(operator, tiling_level, tiling_factors, inner_level=5):
    """
    operator = Operator(
        domain=isl.BasicSet("{ [i,j]: 0 <= i < 8 and 0 <= j < 3 }"),
        acc_rel_out=isl.BasicMap("{ [i,j] -> C[i] }"),
        acc_rel_lhs=isl.BasicMap("{ [i,j] -> A[i + j] }"),
        acc_rel_rhs=isl.BasicMap("{ [i,j] -> B[i + j] }")
    )

    tiling_factors:
    [
        [2,2],
        [1,4]
    ]
    """
    param_names = operator.domain.get_var_names(isl.dim_type.param)

    domain = operator.domain

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set) - inner_level
    assert (
        len(tiling_factors) == n_iter
    ), f"len(tiling_factors)={len(tiling_factors)} != n_iter={n_iter}"

    domain_shape = utils.get_static_box_shape(domain)

    # do checks
    for i in range(n_iter):
        assert (
            len(tiling_factors[i]) == tiling_level
        ), f"len(tiling_factors[{i}])={len(tiling_factors[i])} != tiling_level={tiling_level}"
        dim_size = domain_shape[i]
        assert (
            multiply(tiling_factors[i]) == dim_size
        ), f"multiply(tiling_factors[{i}])={multiply(tiling_factors[i])} != dim_size={dim_size}"

    # do tiling
    tiling_maps = []
    remain_factors = [*domain_shape]
    for l in range(tiling_level - 1):
        n_keep_iters = l * n_iter
        keep_iter_names = [f"i{i}" for i in range(n_keep_iters)]
        change_iter_names = [
            f"i{i}" for i in range(n_keep_iters, n_keep_iters + n_iter)
        ]
        inner_keep_iter_names = [
            f"i{i}"
            for i in range(n_keep_iters + n_iter, n_keep_iters + n_iter + inner_level)
        ]

        outer_iters = []
        inner_iters = []
        for i in range(n_iter):
            factor = tiling_factors[i][l]
            remain_factors[i] = remain_factors[i] // factor
            outer_iters.append(f"floor({change_iter_names[i]}/{remain_factors[i]})")
            inner_iters.append(f"{change_iter_names[i]}%{remain_factors[i]}")

        tiling_map_def = f"[{','.join(param_names)}] -> {{ [{','.join(keep_iter_names + change_iter_names + inner_keep_iter_names)}] -> [{','.join(keep_iter_names + outer_iters + inner_iters + inner_keep_iter_names)}] }}"
        tiling_map = isl.BasicMap(tiling_map_def)
        # print(f"{tiling_map_def=}")
        tiling_maps.append(tiling_map)

    # print("\n")
    tiling_map = tiling_maps[0].intersect_domain(domain)
    for _tiling_map in tiling_maps[1:]:
        tiling_map = tiling_map.apply_range(_tiling_map)

    tiling_map = tiling_map.intersect_domain(domain)

    new_operator = operator.apply_schedule(tiling_map, skip_simplify=True)
    return new_operator


# def enumerate_tiling_factors_outer(operator, tiling_factor, inner_level=5):
#     domain = operator.domain

#     assert domain.is_box(), f"domain={domain} is not box"
#     n_iter = domain.dim(isl.dim_type.set)

#     domain_shape = utils.get_static_box_shape(domain)[:n_iter-inner_level]
#     dim_factors = []
#     for dim_size in domain_shape:
#         factors = factorize(dim_size, tiling_factor)
#         # factors = filter_factors(factors)
#         factors = [factor for factor in factors if factor[-1]!=1 or max(factor)==1]
#         # print(f"{len(factors)=}, {factors=}")
#         # if len(factors) > 8:
#         #     factors = factors[::4]
#         dim_factors.append(factors)
#     # import pdb; pdb.set_trace()
#     # exit()
#     # dim_factors = dim_factors[::4]
#     for combination in itertools.product(*dim_factors):
#         new_operator = multi_level_tiling_outer(operator, tiling_factor, combination, inner_level)
#         yield new_operator


def remove_all_one_factors(factors):
    new_factors = []
    for factor in factors:
        new_factor = [f for f in factor if f != 1]
        if len(new_factor) == 0:
            new_factor = [1]
        new_factors.append(new_factor)
    return new_factors


def multi_level_splitting_combination(
    operator, max_splitting_level, not_splitting=None
):

    domain = operator.domain
    assert max_splitting_level in (2, 3), f"{max_splitting_level=}"

    assert domain.is_box(), f"domain={domain} is not box"
    n_iter = domain.dim(isl.dim_type.set)

    domain_shape = utils.get_static_box_shape(domain)
    dim_factors = []
    for i, dim_size in enumerate(domain_shape):
        if not_splitting is not None and i in not_splitting:
            dim_factors.append(((dim_size,),))
            continue
        factors = factorize(dim_size, max_splitting_level)
        factors = remove_all_one_factors(factors)
        factors = list({tuple(factor) for factor in factors})
        dim_factors.append(tuple(factors))
    dim_factors = tuple(dim_factors)

    combination_list = list(itertools.product(*dim_factors))

    new_op_list = []
    for idx, combination in enumerate(combination_list):
        new_operator = multi_level_splitting_var_level(
            operator, combination, skip_simplify=True
        )
        new_op_list.append(new_operator)
    return new_op_list


def memory_tiling_pass(op_list, not_splitting=None):
    new_op_list = []
    for op in op_list:
        for new_op in multi_level_splitting_combination(
            op, max_splitting_level=2, not_splitting=not_splitting
        ):
            new_op_list.append(new_op)

    return new_op_list


def change_dim_types_for_pre_tiling(split_factors, dim_types):
    new_dim_types = []
    for dim, dim_factors in enumerate(split_factors):
        if len(dim_factors) == 1:
            new_dim_types.append(dim_types[dim])
        elif len(dim_factors) == 2:
            new_dim_types.append(f"{dim_types[dim]}_o")
            new_dim_types.append(f"{dim_types[dim]}_i")
        elif len(dim_factors) == 3:
            new_dim_types.append(f"{dim_types[dim]}_o")
            new_dim_types.append(f"{dim_types[dim]}_m")
            new_dim_types.append(f"{dim_types[dim]}_i")
        else:
            raise ValueError(f"dim_factors={dim_factors} is not supported")
    return new_dim_types


class PreTilingSchedule(Schedule):
    def __init__(self, tile_sizes):
        super().__init__()
        self.tile_sizes = tile_sizes

    def dumps(self):
        return json.dumps(self.tile_sizes)

    def __hash__(self):
        return hash(self.tile_sizes)

    def __eq__(self, other):
        return self.tile_sizes == other.tile_sizes


class PreTilingPass(DepthFirstPass):
    def __init__(
        self,
        args,
        fix_schedule: Optional[PreTilingSchedule] = None,
        schedule_as_key: bool = False,
        prune: bool = True,
    ):
        super().__init__(fix_schedule=fix_schedule, schedule_as_key=schedule_as_key)
        self.args = args
        assert self.fix_schedule is None or isinstance(
            self.fix_schedule, PreTilingSchedule
        )
        self.prune = prune

    def apply(self, operator):
        symmetry_info = operator.attr.get("symmetry_info", None)
        dim_types = operator.attr.get("dim_types", None)
        max_tiling_level = operator.attr.get("max_tiling_level", 2)
        not_tiling = operator.attr.get("not_tiling", None)

        domain = operator.domain
        assert max_tiling_level in (2, 3), f"{max_tiling_level=}"

        assert domain.is_box(), f"domain={domain} is not box"
        n_iter = domain.dim(isl.dim_type.set)

        domain_shape = utils.get_static_box_shape(domain)
        dim_factors = []
        for i, dim_size in enumerate(domain_shape):
            if not_tiling is not None and i in not_tiling:
                dim_factors.append(((dim_size,),))
                continue
            factors = factorize(dim_size, max_tiling_level)
            # factors = [tuple(factor) for factor in factors if factor[0]!=1 and factor[1]!=1]
            factors = remove_all_one_factors(factors)
            factors = list({tuple(factor) for factor in factors})
            # print(f"{factors=}")
            # factors = reversed(factors)
            # factors = [(dim_size,),*factors]
            dim_factors.append(tuple(factors))
        dim_factors = tuple(dim_factors)

        if symmetry_info is None or self.prune == False:
            combination_list = list(itertools.product(*dim_factors))
        else:
            combination_list = combine_tilesize_by_symmetry_info(
                dim_factors, symmetry_info
            )

        if self.args.polycim_disable_pretile:
            combination_list = [
                combination
                for combination in combination_list
                if all([len(factors) == 1 for factors in combination])
            ]

        if self.fix_schedule is not None:
            combination_list = [self.fix_schedule.tile_sizes]

        result_list = []
        for idx, combination in enumerate(combination_list):
            new_operator = multi_level_splitting_var_level(operator, combination)
            new_dim_types = change_dim_types_for_pre_tiling(combination, dim_types)
            new_operator.set_attr("dim_types", new_dim_types, overwrite=True)
            new_operator.set_attr("pre_tile_sizes", combination)

            schedule = PreTilingSchedule(combination)
            result = SchedulePassResult(new_operator, schedule)
            result_list.append(result)

        return result_list


if __name__ == "__main__":
    op = BasicOperator(
        domain=isl.BasicSet(
            f"{{ [oh,ow,kh,kw]: 0<=oh<8 and 0<=ow<8 and 0<=kh<3 and 0<=kw<3 }}"
        ),
        access_I=isl.BasicMap(f"{{ [oh,ow,kh,kw] -> I[oh + kh, ow + kw] }}"),
        access_O=isl.BasicMap("{ [oh,ow,kh,kw] -> O[oh, ow] }"),
        access_W=isl.BasicMap("{ [oh,ow,kh,kw] -> W[kh, kw] }"),
    )
    tiling_factors = [
        [2, 4],
        [4, 2],
        [2, 4],
        [2, 4],
    ]
    multi_level_tiling()
