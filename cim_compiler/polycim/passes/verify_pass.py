import json
import os
import subprocess
from functools import reduce

import numpy as np
from cim_compiler.utils.df_layout import tensor_bits_to_int8

from polycim.codegen_.codegen_data_layout_convert import (
    run_data_layout_convert_executable,
)
from polycim.exp.op_list import get_op_list
from polycim.passes.base import BreadthFirstPass
from polycim.utils.logger import get_logger

logger = get_logger(__name__)


def save_and_convert(exe_path, data_path, converted_data_path, data_np):
    with open(data_path, "w") as f:
        np.savetxt(f, data_np.reshape(-1), fmt="%d")

    run_data_layout_convert_executable(exe_path, data_path, converted_data_path)


def get_verify_fn(op_id):
    op_info = get_op_list()[op_id]
    return op_info["verify_fn"]


def verify(temp_dir, cim_cfg_path, op_name, op_id):
    op_dir = os.path.join(temp_dir, op_name, op_id)
    data_dir = os.path.join(op_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # get buffer info
    buffer_info_path = os.path.join(op_dir, "buffer_info.json")
    with open(buffer_info_path, "r") as f:
        buffer_info = json.load(f)

    # get origin operand shape
    origin_operand_shape_path = os.path.join(op_dir, "origin_operand_shape.json")
    with open(origin_operand_shape_path, "r") as f:
        origin_operand_shape = json.load(f)

    # prepare data
    # cim_mask_shape = buffer_info["cim_mask_global"]["shape"]
    # cim_mask_np = np.ones(cim_mask_shape, dtype=np.int8).reshape(-1, 8)
    # cim_mask_np = tensor_bits_to_int8(cim_mask_np)
    # cim_mask_data = bytearray(cim_mask_np)

    input_shape = origin_operand_shape["I"]
    input_np = np.random.randint(-2, 3, size=input_shape, dtype=np.int8)
    # input_np = np.random.randint(-1, 2, size=input_shape, dtype=np.int8)
    # input_np = np.ones(input_shape, dtype=np.int8)

    weight_shape = origin_operand_shape["W"]
    weight_np = np.random.randint(-2, 3, size=weight_shape, dtype=np.int8)
    # weight_np = np.random.randint(-1, 2, size=weight_shape, dtype=np.int8)
    # weight_np = np.ones(weight_shape, dtype=np.int8)

    # convert data
    I_exe_path = os.path.join(op_dir, "convert_I.o")
    I_data_path = os.path.join(data_dir, "I.txt")
    I_converted_data_path = os.path.join(data_dir, "I_converted.txt")
    save_and_convert(I_exe_path, I_data_path, I_converted_data_path, input_np)

    W_exe_path = os.path.join(op_dir, "convert_W.o")
    W_data_path = os.path.join(data_dir, "W.txt")
    W_converted_data_path = os.path.join(data_dir, "W_converted.txt")
    save_and_convert(W_exe_path, W_data_path, W_converted_data_path, weight_np)

    # concat input and weight into global memory image
    I_converted_np = np.loadtxt(I_converted_data_path, dtype=np.int8)
    W_converted_np = np.loadtxt(W_converted_data_path, dtype=np.int8)
    I_converted_data = bytearray(I_converted_np)
    W_converted_data = bytearray(W_converted_np)
    global_memory_image_path = os.path.join(data_dir, "global_memory_image.bin")
    # total_data = cim_mask_data + I_converted_data + W_converted_data
    total_data = I_converted_data + W_converted_data
    with open(global_memory_image_path, "wb") as f:
        f.write(total_data)

    # run the simulator
    code_path = os.path.join(op_dir, "final_code.json")
    sim_output_dir = os.path.join(op_dir, "sim_output")
    subprocess.run(
        [
            "cim-compiler",
            "simulate",
            "--code-file",
            code_path,
            "--data-file",
            global_memory_image_path,
            "--config-file",
            cim_cfg_path,
            "--output-dir",
            sim_output_dir,
            "--code-format",
            "cimflow",
            "--save-stats",
        ],
        check=True,
    )

    # check stats
    stats_path = os.path.join(sim_output_dir, "stats.json")
    assert os.path.exists(stats_path), f"{stats_path=}"
    with open(stats_path, "r") as f:
        stats = json.load(f)

    # get the output
    output_path = os.path.join(sim_output_dir, "image.bin")
    assert os.path.exists(output_path), f"{output_path=}"
    with open(output_path, "rb") as f:
        output_global_image = f.read()

    output_offset = len(total_data)
    # output_size = 28 * 28 * 4
    output_size = reduce(lambda x, y: x * y, buffer_info["O_aligned"]["shape"])
    output_size = output_size * 4  # int32
    output_data = output_global_image[output_offset : output_offset + output_size]
    output_np = np.frombuffer(output_data, dtype=np.int32)
    logger.info(f"{output_np=}")
    # np.savetxt(os.path.join(sim_output_dir, "output.txt"), output_np.reshape(-1), fmt="%d")

    O_exe_path = os.path.join(op_dir, "convert_O.o")
    O_data_path = os.path.join(data_dir, "O.txt")
    O_converted_data_path = os.path.join(data_dir, "O_converted.txt")
    save_and_convert(O_exe_path, O_data_path, O_converted_data_path, output_np)

    # read O_converted
    output_shape = origin_operand_shape["O"]
    O_converted_np = np.loadtxt(O_converted_data_path, dtype=np.int32).reshape(
        output_shape
    )
    logger.info(
        f"output shape={O_converted_np.shape}, dtype={O_converted_np.dtype}, max={O_converted_np.max()}, min={O_converted_np.min()}"
    )
    # import pdb; pdb.set_trace()
    verify_fn = get_verify_fn(op_name)
    golden = verify_fn(input_np, weight_np)
    logger.info(
        f"golden shape={golden.shape}, dtype={golden.dtype}, max={golden.max()}, min={golden.min()}"
    )
    logger.info(f"{temp_dir = } (will be removed after test)")
    # import pdb; pdb.set_trace()
    # assert np.all(O_converted_np==golden), f"{O_converted_np=}, {golden=}"
    check_result = np.all(O_converted_np == golden)

    return stats, check_result


class VerifyPass(BreadthFirstPass):
    def __init__(self, args):
        super().__init__()
        self.op_list = list()
        self.args = args

    def apply(self, operator):
        self.op_list.append(operator)

    def apply_all(self):
        for i, op in enumerate(self.op_list):
            # op_dir = os.path.join(self.args.output_path, op.attr["name"], str(i))
            inst_stats, check_result = verify(
                self.args.output_path, self.args.config_path, op.attr["name"], str(i)
            )
            op.attr["VerifyPass"] = {
                "inst_stats": inst_stats,
                "check_result": check_result,
            }
            # behaviour_simulate(temp_dir, cim_cfg_path, op_id, cim_count):

    def get_result(self):
        return self.op_list
