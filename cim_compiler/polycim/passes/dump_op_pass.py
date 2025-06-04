import json
import os
from collections import OrderedDict
from dataclasses import asdict

from polycim.config import CIMConfig
from polycim.passes.base import BreadthFirstPass
from polycim.utils.draw import extract_frame_info


def dump_schedules(origin_op, new_op, **kwargs):
    cim_cfg = kwargs["cim_cfg"]

    schedule_keys = [
        "pre_tiling",
        "affine",
        "shift_to_positive",
        "coalescing",
        "tiling",
    ]
    comment_keys = ["tiling_factors", "bases", "s2h_mapping"]
    # global dump_index
    schedule_dict = OrderedDict()
    schedule_dict["tiling_factors"] = None
    schedule_dict["pre_tiling"] = None
    schedule_dict["bases"] = None
    schedule_dict["affine"] = None
    schedule_dict["shift_to_positive"] = None
    schedule_dict["shift_to_zero"] = None
    schedule_dict["s2h_mapping"] = None
    schedule_dict["coalescing"] = None
    schedule_dict["tiling"] = None

    for name_schedule in new_op.history_schedules:
        if (
            type(name_schedule) == dict
            and list(name_schedule.keys())[0] in schedule_dict
        ):
            name = list(name_schedule.keys())[0]
            schedule = name_schedule[name]
            schedule_dict[name] = str(schedule)

    dump_code = '"""\n'
    for key, value in kwargs.items():
        dump_code += f"{key} = {value}\n"
    dump_code += '"""\n'
    dump_code += f"import islpy as isl\n"
    dump_code += f"import time\n"
    dump_code += f"from polycim.op.base_operator import BasicOperator\n"
    dump_code += f"from polycim.utils.draw import draw, extract_frame_info\n"
    dump_code += f"from polycim.config import CIMConfig\n"

    cim_config_str = f"""
cim_cfg = CIMConfig(
    n_row={cim_cfg.n_row},
    n_group_vcol={cim_cfg.n_group_vcol},
    n_comp={cim_cfg.n_comp},
    n_group={cim_cfg.n_group},
    n_macro_per_group={cim_cfg.n_macro_per_group},
    n_macro={cim_cfg.n_macro}
)\n
"""
    dump_code += cim_config_str

    origin_op_str = f"""
op = BasicOperator(
    domain = isl.BasicSet(
        \"{origin_op.domain}\"
    ),
    access_I = isl.BasicMap(\"{origin_op.access_I}\"),
    access_O = isl.BasicMap(\"{origin_op.access_O}\"),
    access_W = isl.BasicMap(\"{origin_op.access_W}\"),
)
"""
    # dump_code += f"domain = isl.BasicSet(\"{init_domain}\")\n\n"
    dump_code += origin_op_str
    for key, value in schedule_dict.items():
        if key in schedule_keys:
            dump_code += f'schedule_{key} = isl.BasicMap("{value}")\n'
            # dump_code += f"domain = schedule_{key}.intersect_domain(domain).range()\n\n"
            dump_code += f'op = op.apply_schedule(schedule_{key}, skip_simplify=True, name="{key}")\n\n'
        else:
            dump_code += f'"""\n{key} = \n{value}\n"""\n'

    dump_code += """
domain = op.domain
n_dim = domain.dim(isl.dim_type.set)
begin_time = time.time()
outer_domain = domain.project_out(isl.dim_type.set, n_dim - 2, 2)
val = outer_domain.count_val()
dur_time = time.time() - begin_time
print(f"outer_domain.count_val {val=}, {dur_time=}")
draw(op, cim_cfg)
    """
    # save_dir = "dump_code"
    # os.makedirs(save_dir, exist_ok=True)
    # with open(os.path.join(save_dir, f"dump_code_{dump_index}.py"), "w") as f:
    #     f.write(dump_code)
    # dump_index += 1
    # print("dump_code saved to dump_code.py")
    # exit()
    return dump_code


class DumpOpPass(BreadthFirstPass):
    def __init__(
        self,
        args,
        cim_config: CIMConfig,
        pad: bool = True,
    ):
        super().__init__()
        self.args = args
        self.cim_config = cim_config
        self.pad = pad
        self.op_list = list()

    def get_result(self):
        return self.op_list

    def apply(self, operator):
        self.op_list.append(operator)
        self.dump_op(operator)

    def dump_op(self, op):
        origin_op = op.attr["origin_op"]
        flops = int(str(origin_op.domain.count_val()))

        save_dir = self.args.output_path
        min_compute_times = op.attr["UtilizationEvaluatePass"]["compute_ops"]
        os.makedirs(save_dir, exist_ok=True)
        op_idx = len(self.op_list)

        save_dir_solution = os.path.join(save_dir, f"solution_{op_idx}")
        os.makedirs(save_dir_solution, exist_ok=True)

        # save schedule code
        dump_code = dump_schedules(
            origin_op,
            op,
            min_compute_times=min_compute_times,
            cim_cfg=self.cim_config,
            flops=flops,
        )
        with open(os.path.join(save_dir_solution, f"schedule_code.py"), "w") as f:
            f.write(dump_code)

        # save mapping pictures
        for idx, value in enumerate(
            extract_frame_info(op, self.cim_config, different_weight=True)
        ):
            timestamp, frame_info = value
            frame_str = f"Index: {idx}.    Timestamp: {timestamp}\n"
            frame_str += frame_info.get_str(brief=False)
            picure_save_path = os.path.join(save_dir_solution, f"frame_{idx}.txt")
            with open(picure_save_path, "w") as f:
                f.write(frame_str)
            print(f"mapping pictures to {picure_save_path}")
            break

        # save result
        result_json = self.show_result(min_compute_times, flops, is_print=False)
        result_json["min_compute_op_need_macros"] = op.attr["UtilizationEvaluatePass"][
            "need_macros"
        ]
        with open(os.path.join(save_dir_solution, f"result.json"), "w") as f:
            json.dump(result_json, f, indent=4)

        print(f"op save to {save_dir}")

    def show_result(self, min_compute_times, flops, is_print=True):
        flops_per_cim_compute = flops / min_compute_times
        peak_flops_per_cim_compute = (
            self.cim_config.n_comp * self.cim_config.n_group_vcol
        )
        use_rate_percent = flops_per_cim_compute / peak_flops_per_cim_compute * 100

        result = OrderedDict()
        result["cim_cfg"] = asdict(self.cim_config)
        result["flops"] = flops
        result["min_compute_times"] = min_compute_times
        result["flops_per_cim_compute"] = flops_per_cim_compute
        result["peak_flops_per_cim_compute"] = peak_flops_per_cim_compute
        result["use_rate"] = use_rate_percent

        if is_print:
            print(json.dumps(result, indent=4))
        return result

    def apply_all(self):
        pass

    def get_result(self):
        return self.op_list
