import json
import math
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from functools import reduce

import numpy as np
from cim_compiler.utils.df_layout import tensor_bits_to_int8

from polycim.codegen_.codegen_data_layout_convert import (
    run_data_layout_convert_executable,
)
from polycim.config import set_raw_config_by_path
from polycim.exp.iccad25.db import DataBase, Experiment, ExperimentResult
from polycim.exp.op_list import get_op_list


def get_verify_fn(op_id):
    op_info = get_op_list()[op_id]
    return op_info["verify_fn"]


def ceil(a, b):
    return int(math.ceil(a / b))


def save_and_convert(exe_path, data_path, converted_data_path, data_np):
    with open(data_path, "w") as f:
        np.savetxt(f, data_np.reshape(-1), fmt="%d")

    run_data_layout_convert_executable(exe_path, data_path, converted_data_path)


def explore(temp_dir, cim_cfg_path, op_id, axis_align):
    cmd = [
        "polycim",
        "explore",
        "--op-id",
        op_id,
        "--config-path",
        cim_cfg_path,
        "--output-path",
        temp_dir,
        "--data-movement-full-vectorize",
    ]
    if axis_align:
        cmd.append("--disable-affine")
    subprocess.run(cmd, check=True)


def verify(temp_dir, cim_cfg_path, op_id, cim_count, axis_align):
    op_dir = os.path.join(temp_dir, op_id, "0")
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
    cim_mask_shape = buffer_info["cim_mask_global"]["shape"]
    cim_mask_np = np.ones(cim_mask_shape, dtype=np.int8).reshape(-1, 8)
    cim_mask_np = tensor_bits_to_int8(cim_mask_np)
    cim_mask_data = bytearray(cim_mask_np)

    input_shape = origin_operand_shape["I"]
    input_np = np.random.randint(-2, 3, size=input_shape, dtype=np.int8)

    weight_shape = origin_operand_shape["W"]
    weight_np = np.random.randint(-2, 3, size=weight_shape, dtype=np.int8)

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
    total_data = cim_mask_data + I_converted_data + W_converted_data
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
    if cim_count >= 0:
        assert (
            stats["CIMComputeInst"] == cim_count
        ), f"{stats['CIMComputeInst']=}, {cim_count=}"
    else:
        print("skip stats check")

    # get the output
    output_path = os.path.join(sim_output_dir, "image.bin")
    assert os.path.exists(output_path), f"{output_path=}"
    with open(output_path, "rb") as f:
        output_global_image = f.read()

    output_offset = len(total_data)
    # output_size = 28 * 28 * 4
    output_size = reduce(lambda x, y: x * y, buffer_info["O_aligned"]["shape"])
    output_data = output_global_image[output_offset : output_offset + output_size]
    output_np = np.frombuffer(output_data, dtype=np.int8)
    print(f"{output_np=}")
    # np.savetxt(os.path.join(sim_output_dir, "output.txt"), output_np.reshape(-1), fmt="%d")

    O_exe_path = os.path.join(op_dir, "convert_O.o")
    O_data_path = os.path.join(data_dir, "O.txt")
    O_converted_data_path = os.path.join(data_dir, "O_converted.txt")
    save_and_convert(O_exe_path, O_data_path, O_converted_data_path, output_np)

    # read O_converted
    output_shape = origin_operand_shape["O"]
    O_converted_np = np.loadtxt(O_converted_data_path, dtype=np.int8).reshape(
        output_shape
    )
    print(
        f"output shape={O_converted_np.shape}, dtype={O_converted_np.dtype}, max={O_converted_np.max()}, min={O_converted_np.min()}"
    )
    # import pdb; pdb.set_trace()
    verify_fn = get_verify_fn(op_id)
    golden = verify_fn(input_np, weight_np)
    print(
        f"golden shape={golden.shape}, dtype={golden.dtype}, max={golden.max()}, min={golden.min()}"
    )
    print(f"{temp_dir = } (will be removed after test)")
    # import pdb; pdb.set_trace()
    assert np.all(O_converted_np == golden), f"{O_converted_np=}, {golden=}"


def profile(temp_dir, pimsim_cfg_path, op_id, axis_align):
    # 1. convert format
    op_dir = os.path.join(temp_dir, op_id, "0")

    cimflow_code_path = os.path.join(op_dir, "final_code.json")
    legacy_code_path = os.path.join(op_dir, "final_code.legacy.json")
    report_path = os.path.join(op_dir, f"pimsim_report.json")

    subprocess.run(
        [
            "cim-compiler",
            "convert",
            "--src-type",
            "cimflow",
            "--dst-type",
            "legacy",
            "--src-file",
            cimflow_code_path,
            "--dst-file",
            legacy_code_path,
            "--filter-out-invalid-instructions",
        ],
        check=True,
    )

    # 2. profile
    """
    pimsim ./pimsim_configs/config-m1g1c32b64.json C1.json -r -c -j ./C1.report.json
    """
    subprocess.run(
        [
            "pimsim",
            pimsim_cfg_path,
            legacy_code_path,
            "-r",
            "-c",
            "-j",
            report_path,
        ],
        check=True,
    )

    # 3. parse report
    # with open(report_path, "r") as f:
    #     report = json.load(f)
    # print(f"{report=}")


def save_data(temp_dir, cim_cfg_name, op_id, axis_align, exp_id, status):

    if axis_align:
        strategy = "im2col"
    else:
        strategy = "polycim"
    time_ms = time.time_ns() // 1_000_000

    if status < 3:

        with DataBase("iccad25.db") as db:

            exp_result = ExperimentResult(
                experiment_id=exp_id,
                op=op_id,
                strategy=strategy,
                status=status,
                save_path=temp_dir,
                macro_ultilization=0,
                macro_compute_times=0,
                flops=0,
                need_minimize_macros=0,
                latency=0,
                energy=0,
                config=cim_cfg_name,
                time=time_ms,
            )
            db.insert_experiment_result(exp_result)

    else:

        op_dir = os.path.join(temp_dir, op_id, "0")
        report_path = os.path.join(op_dir, f"pimsim_report.json")
        with open(report_path, "r") as f:
            report = json.load(f)
        latency = report["latency_"]
        energy = report["total_energy_"]

        solution_path = os.path.join(temp_dir, op_id, "solution_0", "result.json")
        with open(solution_path, "r") as f:
            solution = json.load(f)
        macro_ultilization = solution["use_rate"]
        macro_compute_times = solution["min_compute_times"]
        flops = solution["flops"]
        need_minimize_macros = solution["min_compute_op_need_macros"]

        with DataBase("iccad25.db") as db:

            exp_result = ExperimentResult(
                experiment_id=exp_id,
                op=op_id,
                strategy=strategy,
                status=status,
                save_path=temp_dir,
                macro_ultilization=macro_ultilization,
                macro_compute_times=macro_compute_times,
                flops=flops,
                need_minimize_macros=need_minimize_macros,
                latency=latency,
                energy=energy,
                config=cim_cfg_name,
                time=time_ms,
            )
            db.insert_experiment_result(exp_result)


def explore_verify_profile(
    temp_dir, compiler_cfg_path, pimsim_cfg_path, op_id, cim_count, axis_align
):
    status = 0
    # Explore step
    try:
        explore(temp_dir, compiler_cfg_path, op_id, axis_align)
        status += 1
    except Exception as e:
        print(f"Explore step failed: {str(e)}")
        return status

    # Verify step
    try:
        verify(temp_dir, compiler_cfg_path, op_id, cim_count, axis_align)
        status += 1
    except Exception as e:
        print(f"Verify step failed: {str(e)}")
        return status

    # Profile step
    try:
        profile(temp_dir, pimsim_cfg_path, op_id, axis_align)
        status += 1
    except Exception as e:
        print(f"Profile step failed: {str(e)}")
        return status

    return status


def run_op(cim_cfg_name, op_id, cim_count, axis_align, save_to_db=False, exp_id=None):
    """
    1. iter over all ops
    2. for each ops:
    """
    compiler_cfg_path = os.path.join(
        os.path.dirname(__file__), "compiler_configs", cim_cfg_name
    )
    set_raw_config_by_path(compiler_cfg_path)

    pimsim_cfg_path = os.path.join(
        os.path.dirname(__file__), "pimsim_configs", cim_cfg_name
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        status = explore_verify_profile(
            temp_dir, compiler_cfg_path, pimsim_cfg_path, op_id, cim_count, axis_align
        )

        # Profile step
        if save_to_db:
            save_data(temp_dir, cim_cfg_name, op_id, axis_align, exp_id, status)

        return (op_id, status)


def init_worker():
    """Initialize worker process by redirecting stdout/stderr to devnull"""
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


def run_exp(num_processes=4, exp_message="", config_name=""):
    with DataBase("iccad25.db") as db:
        exp_id = db.insert_experiment(
            Experiment(time=time.time_ns() // 1_000_000, message=exp_message)
        )

    # Create list of all tasks to run
    op_list = [f"C{i}" for i in range(1, 14)]
    tasks = []
    for axis_align in [True, False]:
        for op_id in op_list:
            tasks.append(
                (
                    f"{config_name}.json",  # cim_cfg_name
                    op_id,
                    -1,  # cim_count
                    axis_align,
                    True,  # save_to_db
                    exp_id,
                )
            )

    # Run tasks in parallel with redirected output
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_processes, initializer=init_worker) as pool:
        results = pool.starmap(run_op, tasks)
        print(results)

    return results


def run_importance_of_pretiling(op_id, config_name):
    # with DataBase("iccad25.db") as db:
    # exp_id = db.insert_experiment(Experiment(
    #     time = time.time_ns() // 1_000_000,
    #     message = f"run importance of pretiling for {op_id} with {config_name}"
    # ))
    pass


if __name__ == "__main__":
    config_name = "g8m8c32b64"
    run_exp(
        num_processes=4,
        exp_message=f"run {config_name} C1-C13",
        config_name=config_name,
    )  # Adjust number of processes as needed
