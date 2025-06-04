from functools import reduce

import islpy as isl

import polycim.utils.utils as utils
from polycim.config import CIMConfig
from polycim.passes.base import BreadthFirstPass
from polycim.utils.count_minimal_macro import count_minimal_needed_macro
from polycim.utils.timeout import timeout


@timeout(seconds=4)
def count_val(domain):
    return int(str(domain.count_val()))


class UtilizationEvaluatePass(BreadthFirstPass):
    def __init__(
        self,
        args,
        cim_config: CIMConfig,
        pad: bool = True,
        keep_single_best: bool = False,
    ):
        super().__init__()
        self.args = args
        self.cim_config = cim_config
        self.pad = pad
        self.result_value = dict()
        self.result_op = dict()
        self.keep_single_best = keep_single_best

    def get_result(self):
        result = list()
        for key in self.result_op.keys():
            op_list = self.result_op[key]
            if self.keep_single_best:
                op_list = op_list[:1]
            for op in op_list:
                result.append(op)
        return result

    def apply(self, operator):
        domain = operator.domain

        if self.pad:
            # get padding exe time
            box_hull_shape = utils.get_box_hull_shape(domain)
            outer_box_hull_shape = box_hull_shape[:-2]
            exe_time = reduce(lambda x, y: x * y, outer_box_hull_shape)
        else:
            # result.append(op3)
            n_dim = domain.dim(isl.dim_type.set)
            outer_domain = domain.project_out(isl.dim_type.set, n_dim - 2, 2)

            # Use the execute_with_timeout function
            exe_time = count_val(outer_domain)
            if isinstance(exe_time, Exception):
                raise exe_time

            if exe_time is None:
                print(f"timeout")
                return
            else:
                assert isinstance(exe_time, int)

        schedule_key = operator.attr.get("PassManager::schedule_keys")
        # schedule_key = tuple(schedule_key)

        min_compute_times = self.result_value.get(schedule_key, float("inf"))

        if exe_time is not None and exe_time <= min_compute_times:

            # if self.delay_apply:
            #     op3 = op2.apply_schedule(coalescing_schedule, skip_simplify=True, name="coalescing")
            #     op3 = op3.apply_schedule(tiling_schedule, skip_simplify=True, name="tiling")

            need_macro = count_minimal_needed_macro(operator, self.cim_config)
            if self.cim_config.n_macro >= need_macro or (
                not self.args.polycim_disable_weight_rewrite
            ):

                operator.set_attr(
                    "UtilizationEvaluatePass",
                    {
                        "need_macros": need_macro,
                        "compute_ops": exe_time,
                        "utilization": self.get_utilization(operator, exe_time),
                    },
                )

                min_compute_ops = self.result_op.get(schedule_key, list())

                if exe_time < min_compute_times:
                    self.result_value[schedule_key] = exe_time
                    self.result_op[schedule_key] = [operator]
                else:
                    assert exe_time == min_compute_times
                    self.result_value[schedule_key] = exe_time
                    self.result_op[schedule_key].append(operator)

                print(f"min_compute_times={exe_time}, need_macro={need_macro}")

    def apply_all(self):
        pass

    def get_utilization(self, op, compute_ops):
        origin_op = op.attr["origin_op"]
        flops = int(str(origin_op.domain.count_val()))

        flops_per_cim_compute = flops / compute_ops
        peak_flops_per_cim_compute = (
            self.cim_config.n_comp * self.cim_config.n_group_vcol
        )
        use_rate_percent = flops_per_cim_compute / peak_flops_per_cim_compute * 100

        return use_rate_percent
