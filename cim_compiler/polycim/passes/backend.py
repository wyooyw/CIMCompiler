import json
import os
import subprocess
from dataclasses import dataclass
from multiprocessing import Pool

from polycim.codegen_.codegen_data_layout_convert import (
    gcc_compile_data_layout_convert_code,
)
from polycim.config import CIMConfig
from polycim.passes.base import BreadthFirstPass


def dump_op_basic_info(op, path):
    n_compute = op.domain.count_val()
    with open(path, "w") as f:
        f.write(f"n_compute: {n_compute}\n")
        f.write(f"domain: {op.domain}\n")
        f.write(f"access_I: {op.access_I}\n")
        f.write(f"access_O: {op.access_O}\n")
        f.write(f"access_W: {op.access_W}\n")


@dataclass
class ProfileResult:
    stats: str
    save_path: int


def backend_compile(op, save_dir, config_file):
    dsl = op.dsl

    os.makedirs(save_dir, exist_ok=True)
    cim_source_file = os.path.join(save_dir, f"code.cim")
    with open(cim_source_file, "w") as f:
        f.write(dsl)

    # run cim compiler
    subprocess.run(
        [
            "cim-compiler",
            "compile",
            "--input-file",
            cim_source_file,
            "--output-dir",
            save_dir,
            "--config-file",
            config_file,
        ],
        check=True,
    )

    # save buffer info
    buffer_manager = op.buffer_manager
    buffer_info_path = os.path.join(save_dir, "buffer_info.json")
    with open(buffer_info_path, "w") as f:
        json.dump(buffer_manager.get_buffer_name_to_info_dict(), f, indent=4)

    # Save origin operand shape
    origin_operand_shape = op.attr["origin_operand_shape"]
    with open(os.path.join(save_dir, "origin_operand_shape.json"), "w") as f:
        json.dump(origin_operand_shape, f, indent=4)

    # run simulator to profile
    code_file = os.path.join(os.path.abspath(save_dir), "final_code.json")
    output_dir = os.path.join(os.path.abspath(save_dir), "output")
    op.set_attr(
        "BackendCompilePass",
        {
            "output_dir": output_dir,
            "code_file": code_file,
        },
    )
    return op


class BackendCompilePass(BreadthFirstPass):
    def __init__(
        self,
        args,
        cim_config: CIMConfig,
        n_workers: int = 1,
        compile_data_layout: bool = True,
    ):
        super().__init__()

        self.args = args
        self.cim_config = cim_config
        self.n_workers = n_workers
        self.compile_data_layout = compile_data_layout

        self.op_list = []

    def _process_single_op(self, args):
        idx, op = args
        backend_compile(
            op,
            save_dir=os.path.join(self.args.output_path, op.attr["name"], str(idx)),
            config_file=self.args.config_path,
        )
        if self.compile_data_layout:
            self.data_layout_compile(op, idx)

    def apply_all(self):
        if self.n_workers == 1:
            # Serial execution
            for idx, op in enumerate(self.op_list):
                self._process_single_op((idx, op))
        else:
            n_workers = min(self.n_workers, len(self.op_list))
            # Parallel execution
            with Pool(processes=self.n_workers) as pool:
                pool.map(self._process_single_op, enumerate(self.op_list))

    def apply(self, operator):
        self.op_list.append(operator)

    def get_result(self):
        return self.op_list

    def data_layout_compile(self, operator, op_idx):
        # save data layout convert code
        data_layout_convert_code = operator.attr["data_layout_convert_code"]
        save_op_dir = os.path.join(
            self.args.output_path, operator.attr["name"], str(op_idx)
        )
        os.makedirs(save_op_dir, exist_ok=True)
        # import pdb; pdb.set_trace()
        for key, value in data_layout_convert_code.items():
            code_path = os.path.join(save_op_dir, f"convert_{key}.cpp")
            with open(code_path, "w") as f:
                f.write(value)
            exe_path = os.path.join(save_op_dir, f"convert_{key}.o")
            gcc_compile_data_layout_convert_code(code_path, exe_path)
