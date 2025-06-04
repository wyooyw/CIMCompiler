from typing import Optional

import islpy as isl

import polycim.utils.utils as utils
from polycim.config import CIMConfig
from polycim.op.base_operator import (
    AccessRelation,
    DataMovement,
    DataMovementOperator,
    PartialSumDataMovement,
    TensorAccessRelation,
)
from polycim.passes.base import DepthFirstPass, Schedule, SchedulePassResult


def pwaffs_to_map(affs):
    pw_aff_list = utils.make_pw_affs_to_aff_list(affs)

    assign_buffer_acc_rel = isl.MultiPwAff.from_pw_aff_list(
        affs[0].space.insert_dims(isl.dim_type.out, 0, len(pw_aff_list) - 1),
        pw_aff_list,
    )
    assign_buffer_acc_rel = isl.PwMultiAff.from_multi_pw_aff(assign_buffer_acc_rel)
    assign_buffer_acc_rel = isl.Map.from_pw_multi_aff(assign_buffer_acc_rel)
    return assign_buffer_acc_rel


def transform_access(ori_access, n_inner_level):
    if isinstance(ori_access, AccessRelation):
        access = ori_access.offsets
    else:
        access = ori_access
    assert type(access) in (isl.BasicMap, isl.Map)

    access_domain_size = access.dim(isl.dim_type.in_)
    access_range_size = access.dim(isl.dim_type.out)
    outer_access = access.project_out(
        isl.dim_type.in_, access_domain_size - n_inner_level, n_inner_level
    )
    # if not outer_access.range().is_bounded():
    #     import pdb; pdb.set_trace()
    pwaff_access_sizes = utils.get_dynamic_shape_from_dynamic_map(
        outer_access
    )  # [level:]
    pwaff_access_offsets = [outer_access.dim_min(i) for i in range(access_range_size)]

    access_sizes = pwaffs_to_map(pwaff_access_sizes)
    access_offsets = pwaffs_to_map(pwaff_access_offsets)

    buffer_name = access.get_tuple_name(isl.dim_type.out)
    access_sizes = access_sizes.set_tuple_name(isl.dim_type.out, buffer_name)
    access_offsets = access_offsets.set_tuple_name(isl.dim_type.out, buffer_name)

    return TensorAccessRelation(
        offsets=access_offsets, sizes=access_sizes, memory_name=ori_access.memory_name
    )


def tensorize_cim_compute(op):
    domain = op.domain
    domain_size = domain.dim(isl.dim_type.set)
    n_inner_level = op.attr["n_tensorize_cim_compute_level"]

    outer_domain = domain.project_out(
        isl.dim_type.set, domain_size - n_inner_level, n_inner_level
    )
    access_I = transform_access(op.access_I, n_inner_level)
    access_O = transform_access(op.access_O, n_inner_level)
    access_W = transform_access(op.access_W, n_inner_level)

    return DataMovementOperator(
        domain=outer_domain,
        access_I=access_I,
        access_O=access_O,
        access_W=access_W,
        history_domains=[*op.history_domains, outer_domain],
        history_schedules=[
            *op.history_schedules,
            {"tensorize_cim_compute": n_inner_level},
        ],
        data_movement=op.data_movement,
        attr={key: value for key, value in op.attr.items()},
    )


def vectorize_data_movement(args, data_movement):
    domain = data_movement.domain
    domain_size = domain.dim(isl.dim_type.set)

    # args = get_args()
    if args.data_movement_full_vectorize:
        if data_movement.type_ in ("I", "W"):
            n_access_O_dim = data_movement.access_O.offsets.dim(isl.dim_type.out)
            n_inner_level = n_access_O_dim
        else:
            if isinstance(data_movement, PartialSumDataMovement):
                n_access_I_dim = data_movement.access_I.offsets.dim(isl.dim_type.out)
                n_inner_level = n_access_I_dim - 1
            else:
                n_access_I_dim = data_movement.access_I.offsets.dim(isl.dim_type.out)
                n_inner_level = n_access_I_dim
    else:
        n_inner_level = 1

    outer_domain = domain.project_out(
        isl.dim_type.set, domain_size - n_inner_level, n_inner_level
    )
    access_I = transform_access(data_movement.access_I, n_inner_level)
    access_O = transform_access(data_movement.access_O, n_inner_level)

    if isinstance(data_movement, PartialSumDataMovement):
        domain_partial_sum = data_movement.domain_partial_sum
        domain_partial_sum_size = domain_partial_sum.dim(isl.dim_type.set)
        outer_domain_partial_sum = domain_partial_sum.project_out(
            isl.dim_type.set, domain_partial_sum_size - n_inner_level, n_inner_level
        )
        return PartialSumDataMovement(
            domain=outer_domain,
            domain_partial_sum=outer_domain_partial_sum,
            access_I=access_I,
            access_O=access_O,
            level=data_movement.level,
            type_=data_movement.type_,
        )
    return DataMovement(
        domain=outer_domain,
        access_I=access_I,
        access_O=access_O,
        level=data_movement.level,
        type_=data_movement.type_,
    )


def vectorize_data_movement_for_op(args, op):
    origin_data_movement = op.data_movement

    new_data_movement_dict = dict()
    for array_name, data_movement_list in origin_data_movement.items():
        new_data_movement_list = []
        for data_movement in data_movement_list:
            new_data_movement = vectorize_data_movement(args, data_movement)
            new_data_movement_list.append(new_data_movement)
        new_data_movement_dict[array_name] = new_data_movement_list

    return DataMovementOperator(
        domain=op.domain,
        access_I=op.access_I,
        access_O=op.access_O,
        access_W=op.access_W,
        history_domains=op.history_domains,
        history_schedules=op.history_schedules,
        data_movement=new_data_movement_dict,
        attr={key: value for key, value in op.attr.items()},
    )


def tensorize_pass(args, op_list):
    new_op_list = []
    for op in op_list:

        new_op = tensorize_cim_compute(op)
        new_op = vectorize_data_movement_for_op(args, new_op)
        new_op_list.append(new_op)
    return new_op_list


class TensorizeSchedule(Schedule):
    def __init__(self):
        super().__init__()


class TensorizePass(DepthFirstPass):
    def __init__(
        self,
        args,
        cim_config: CIMConfig,
        fix_schedule: Optional[TensorizeSchedule] = None,
        schedule_as_key: bool = False,
    ):
        super().__init__(fix_schedule=fix_schedule, schedule_as_key=schedule_as_key)
        assert self.fix_schedule is None
        assert self.schedule_as_key is False

        self.args = args
        self.cim_config = cim_config

    def apply(self, operator):
        new_op = tensorize_pass(self.args, [operator])[0]

        result = SchedulePassResult(new_op, TensorizeSchedule())
        return [result]
