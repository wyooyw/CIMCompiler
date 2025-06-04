import copy

import islpy as isl

import polycim.utils.utils as utils
from polycim.config import get_config
from polycim.op.base_operator import BasicOperator


def loop_padding(op, _):

    domain = op.domain
    domain_size = domain.dim(isl.dim_type.set)
    domain_ub = [int(str(domain.dim_max_val(i))) for i in range(domain_size)]
    domain_lb = [int(str(domain.dim_min_val(i))) for i in range(domain_size)]

    cim_cfg = get_config()
    sizes = utils.get_box_hull_shape(domain)
    padding_inner_size = [[0, cim_cfg.n_comp - 1], [0, cim_cfg.n_group_vcol - 1]]
    inner_loop_level = len(padding_inner_size)

    domain_outer = domain.project_out(
        isl.dim_type.set, domain_size - inner_loop_level, inner_loop_level
    )
    domain_padding = domain_outer.insert_dims(
        isl.dim_type.set, domain_size - inner_loop_level, inner_loop_level
    )
    for i in range(inner_loop_level):
        level = domain_size - inner_loop_level + i
        padding_lb = padding_inner_size[i][0]
        padding_ub = padding_inner_size[i][1]
        assert padding_inner_size[i][0] <= domain_lb[level]
        assert padding_inner_size[i][1] >= domain_ub[level]

        domain_padding = domain_padding.set_dim_name(
            isl.dim_type.set, level, domain.get_dim_name(isl.dim_type.set, level)
        )
        domain_padding = domain_padding.lower_bound_val(
            isl.dim_type.set, level, isl.Val(padding_lb)
        )
        domain_padding = domain_padding.upper_bound_val(
            isl.dim_type.set, level, isl.Val(padding_ub)
        )

    return BasicOperator(
        domain=domain_padding,
        access_I=op.access_I,
        access_O=op.access_O,
        access_W=op.access_W,
        history_domains=[*op.history_domains, domain_padding],
        history_schedules=[
            *op.history_schedules,
            {"padding_inner_size": padding_inner_size},
        ],
    )
    # import pdb; pdb.set_trace()
    # pass


def loop_padding_pass(op_list, padding_inner_size):
    new_op_list = []
    for op in op_list:
        new_op = loop_padding(op, padding_inner_size)
        new_op_list.append(new_op)
    return new_op_list


def shift_to_zero(op, skip_simplify=False):
    domain = op.domain
    min_val = [
        domain.dim_min_val(i).get_num_si() for i in range(domain.dim(isl.dim_type.set))
    ]
    shift = [-val for val in min_val]

    shift_domain = ",".join([f"i{i}" for i in range(domain.dim(isl.dim_type.set))])
    shift_range = ",".join(
        [
            f"i{i} + {shift[i]}" if shift[i] >= 0 else f"i{i} - {abs(shift[i])}"
            for i in range(domain.dim(isl.dim_type.set))
        ]
    )
    shift = isl.BasicMap(f"{{ [{shift_domain}] -> [{shift_range}] }}")
    new_op = op.apply_schedule(shift, name="shift_to_zero", skip_simplify=skip_simplify)

    min_vals = [
        new_op.domain.dim_min_val(i).get_num_si()
        for i in range(new_op.domain.dim(isl.dim_type.set))
    ]
    assert all(min_val == 0 for min_val in min_vals), f"{min_vals=}, {shift_range=}"

    return new_op


def loop_padding_to_box_all(op):

    op = shift_to_zero(op, skip_simplify=True)
    domain = op.domain
    n_dim = domain.dim(isl.dim_type.set)
    box_hull_shape = utils.get_box_hull_shape(domain)
    dim_names = [domain.get_dim_name(isl.dim_type.set, i) for i in range(n_dim)]
    constraints = []
    for i in range(n_dim):
        constraint = f"0 <= {dim_names[i]} < {box_hull_shape[i]}"
        constraints.append(constraint)

    domain_padding_str = (
        "{ [" + ", ".join(dim_names) + "]: " + " and ".join(constraints) + "}"
    )
    domain_padding = isl.BasicSet(domain_padding_str)

    op = BasicOperator(
        domain=domain_padding,
        access_I=op.access_I,
        access_O=op.access_O,
        access_W=op.access_W,
        history_domains=[*op.history_domains, domain_padding],
        history_schedules=[*op.history_schedules, {"padding": box_hull_shape}],
        attr=copy.deepcopy(op.attr),
    )
    return op


def loop_padding_dim(op, dim_id, size):
    # op = shift_to_zero(op, skip_simplify=True)
    domain = op.domain
    n_dim = domain.dim(isl.dim_type.set)
    box_hull_shape = utils.get_box_hull_shape(domain)

    new_box_hull_shape = [*box_hull_shape]
    assert 0 <= dim_id and dim_id < n_dim
    assert size >= box_hull_shape[dim_id], f"{size=}, {box_hull_shape=}, {dim_id=}"
    new_box_hull_shape[dim_id] = size

    dim_names = [domain.get_dim_name(isl.dim_type.set, i) for i in range(n_dim)]
    constraints = []
    for i in range(n_dim):
        constraint = f"0 <= {dim_names[i]} < {new_box_hull_shape[i]}"
        constraints.append(constraint)

    domain_padding_str = (
        "{ [" + ", ".join(dim_names) + "]: " + " and ".join(constraints) + "}"
    )
    domain_padding = isl.BasicSet(domain_padding_str)

    op = BasicOperator(
        domain=domain_padding,
        access_I=op.access_I,
        access_O=op.access_O,
        access_W=op.access_W,
        history_domains=[*op.history_domains, domain_padding],
        history_schedules=[
            *op.history_schedules,
            {
                "padding": {
                    "from": box_hull_shape,
                    "to": new_box_hull_shape,
                    "dim_id": dim_id,
                    "size": size,
                }
            },
        ],
        attr=copy.deepcopy(op.attr),
    )
    return op
