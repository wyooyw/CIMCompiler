import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import reduce
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import math
from cim_compiler.runner.runner import OpRunner

from cim_compiler.polycim.config import (
    get_config,
    get_memory_base,
    get_memory_size,
    set_raw_config_by_path,
)
from cim_compiler.polycim.op import benchmark
from cim_compiler.polycim.op_compiler import run_cimflow


def get_final_code(final_code):
    # final_code = os.path.join(op.attr["BackendCompilePass"]["code_file"])
    if os.path.exists(final_code):
        with open(final_code, "r") as f:
            final_code = json.load(f)
    else:
        final_code = None

    return final_code


def get_code_set_regs(reg_and_value_list):
    code_list = []
    for reg, value in reg_and_value_list:
        # code_li = {"class": 0b10, "type": 0b11, "opcode": 0, "rd": reg, "imm": value}
        code_li = {"opcode": 0b101100, "rd": reg, "imm": value}
        code_list.append(code_li)
    return code_list


def get_code_single_send(src_addr, dst_core, data_size, unqiue_id):
    reg_src_addr = 0
    reg_dst_core = 1
    reg_data_size = 2
    reg_unique_id = 3

    code_set_regs = get_code_set_regs(
        [
            (reg_src_addr, src_addr),
            (reg_dst_core, dst_core),
            (reg_data_size, data_size),
            (reg_unique_id, unqiue_id),
        ]
    )

    code_send = {
        "opcode": 0b110100,
        "rs": reg_src_addr,
        "rt": reg_dst_core,
        "rd": reg_src_addr,
        "re": reg_data_size,
        "rf": reg_unique_id,
    }
    code_list = [*code_set_regs, code_send]
    return code_list


def get_code_single_receive(dst_addr, src_core, data_size, unqiue_id):
    reg_dst_addr = 0
    reg_src_core = 1
    reg_data_size = 2
    reg_unique_id = 3

    code_set_regs = get_code_set_regs(
        [
            (reg_dst_addr, dst_addr),
            (reg_src_core, src_core),
            (reg_data_size, data_size),
            (reg_unique_id, unqiue_id),
        ]
    )

    # code_recv = {
    #     "class": 0b110,
    #     "type": 0b11,
    #     "sync": 0,
    #     "rs1": reg_src_core,
    #     "rs2": 0,
    #     "rd": reg_dst_addr,
    #     "reg_id": reg_unique_id,
    #     "reg_len": reg_data_size,
    # }
    code_recv = {
        "opcode": 0b110110,
        "rs": reg_src_core,
        "rt": reg_dst_addr,
        "rd": reg_dst_addr,
        "re": reg_data_size,
        "rf": reg_unique_id,
    }
    code_list = [*code_set_regs, code_recv]
    return code_list


def get_code_trans(src_addr, dst_addr, data_size):
    reg_src_addr = 0
    reg_dst_addr = 1
    reg_data_size = 2
    code_set_regs = get_code_set_regs(
        [
            (reg_src_addr, src_addr),
            (reg_dst_addr, dst_addr),
            (reg_data_size, data_size),
        ]
    )
    # code_trans = {
    #     "class": 0b110,
    #     "type": 0,
    #     "offset_mask": 0b00,
    #     "rs1": reg_src_addr,
    #     "rs2": reg_data_size,
    #     "rd": reg_dst_addr,
    #     "offset": 0,
    # }
    code_trans = {
        "opcode": 0b110000,
        "rs": reg_src_addr,
        "rt": reg_data_size,
        "rd": reg_dst_addr,
        "imm": 0,
    }
    return [*code_set_regs, code_trans]


def get_code_trans_by_memory_type(src_memory_type, dst_memory_type, data_size):
    src_addr = get_memory_base(src_memory_type)
    dst_addr = get_memory_base(dst_memory_type)
    return get_code_trans(src_addr, dst_addr, data_size)


def get_code_read_global(attr):
    assert "tensor_type" in attr
    if attr["tensor_type"] == "weight":
        dst_memory_type = "macro"
    elif attr["tensor_type"] == "feature":
        dst_memory_type = "input_memory"
    else:
        assert False, attr["tensor_type"]

    return get_code_trans_by_memory_type(
        "global", dst_memory_type, reduce(lambda x, y: x * y, attr["shape"])
    )


def get_code_write_global(attr):
    return get_code_trans_by_memory_type(
        "output_memory", "global", reduce(lambda x, y: x * y, attr["shape"])
    )


def get_code_add(attr):
    data_size = reduce(lambda x, y: x + y, attr["shape"])
    reg_input_1_addr = 1
    reg_input_2_addr = 2
    reg_data_size = 3
    reg_output_addr = 4
    code_set_regs = get_code_set_regs(
        [
            (reg_input_1_addr, get_memory_base("input_memory")),
            (reg_input_2_addr, get_memory_base("input_memory") + data_size),
            (reg_data_size, data_size),
            (reg_output_addr, get_memory_base("output_memory")),
        ]
    )
    # code_add = {
    #     "class": 0b01,
    #     "input_num": 0b01,
    #     "opcode": 0b00,
    #     "rs1": reg_input_1_addr,
    #     "rs2": reg_input_2_addr,
    #     "rs3": reg_data_size,
    #     "rd": reg_output_addr,
    # }
    input_num = 2
    code_add = {
        "opcode": 0b010000 + ((input_num - 1) << 2),
        "rs": reg_input_1_addr,
        "rt": reg_input_2_addr,
        "rd": reg_output_addr,
        "re": reg_data_size,
        "funct": 0b00,
    }
    return [*code_set_regs, code_add]

    # exit()


def calculate_input_memory(X_shape, W_shape, padding):
    cim_cfg = get_config()

    # 计算卷积操作的输入内存占用
    # 这里假设每个元素占用1个单位的内存
    # 你可以根据实际情况调整计算方法
    batch, ic, ih, iw = X_shape
    oc,ic,kh,kw = W_shape
    out_h = ih - kh + 1 + 2 * padding[0]
    out_w = iw - kw + 1 + 2 * padding[1]
    
    
    kernel_size = ic * kh * kw
    kernel_size_pad = math.ceil(kernel_size / (cim_cfg.n_comp * cim_cfg.n_row)) * (cim_cfg.n_comp * cim_cfg.n_row)
    input_memory = batch * out_h * out_w * kernel_size_pad
    
    # input_memory = reduce(lambda x, y: x * y, X_shape) * 3

    return input_memory

@dataclass
class Conv2dConfig:
    batch: int
    in_channel: int
    out_channel: int
    in_hw: int
    out_hw: int
    ker_hw: int
    macro_config: int
    math: int
    n_weight_duplicate_group: int
    n_reduce_group: int

def get_code_conv2d_by_dsl(
        batch,
        in_channel,
        out_channel,
        in_hw,
        out_hw,
        ker_hw,
        args,
        cache_dir
):      
    cim_cfg = get_config()
    if (in_channel % cim_cfg.n_comp != 0) and (cim_cfg.n_comp % in_channel != 0):
        assert in_channel == 3
        in_channel = 4

    kernel_size = in_channel * ker_hw * ker_hw
    n_reduce_group = math.ceil(kernel_size / (cim_cfg.n_comp * cim_cfg.n_row))
    n_weight_duplicate_group = cim_cfg.n_group // n_reduce_group
    op_config = Conv2dConfig(
        batch=batch,
        in_channel=in_channel,
        out_channel=out_channel,
        in_hw=in_hw,
        out_hw=out_hw,
        ker_hw=ker_hw,
        macro_config=cim_cfg,
        math=math,
        n_weight_duplicate_group=n_weight_duplicate_group,
        n_reduce_group=n_reduce_group
    )
    op_path = os.path.join(os.environ["CIMCOMPILER_HOME"], "cim_compiler/op/cimflow/conv2d.cim")
    op_runner = OpRunner(op_path, op_config, args.config_path)
    temp_dir = tempfile.mkdtemp(dir=cache_dir)
    op_runner.run([], [], simulate=False, save_dir=temp_dir)
    code_path = os.path.join(temp_dir, "compiler_output", "final_code.json")
    return SimpleNamespace(
        attr={
            "BackendCompilePass": {
                "code_file": code_path
            }
        },
    )


cache_conv2d_result = dict()


def get_code_conv2d(args, attr, cache_dir):
    global cache_conv2d_result
    """
    'X_shape': [1, 32, 224, 224], 'W_shape': [32, 3, 3, 3], 'padding': [1, 1, 1, 1], 'strides': [2, 2]
    """
    # import pdb; pdb.set_trace()
    batch, in_channel, in_height, in_width = attr["X_shape"]
    out_channel, in_channel, kernel_height, kernel_width = attr["W_shape"]
    padding = attr["padding"][0]
    stride = attr["strides"][0]

    out_h = (in_height + 2 * padding - kernel_height) // stride + 1
    out_w = (in_width + 2 * padding - kernel_width) // stride + 1

    cache_key = (
        batch,
        out_channel,
        in_channel,
        out_h,
        out_w,
        kernel_height,
        kernel_width,
        padding,
        stride,
    )
    if cache_key in cache_conv2d_result:
        result = cache_conv2d_result[cache_key]
    else:
        print(f"Compile Conv {attr=}")
        input_memory = calculate_input_memory(attr["X_shape"], attr["W_shape"], attr["padding"])
        if input_memory > get_memory_size("input_memory"):
            print(f"input_memory: {input_memory} > input_memory_size: {get_memory_size('input_memory')}")
            # import pdb; pdb.set_trace()
            assert stride == 1, f"{stride=}"
            result = get_code_conv2d_by_dsl(
                batch,
                in_channel,
                out_channel,
                in_height,
                out_h,
                kernel_height,
                args,
                cache_dir
            )
            cache_conv2d_result[cache_key] = result
        else:
            # with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = tempfile.mkdtemp(dir=cache_dir)
            operator = benchmark.get_op_conv2d(
                b=batch,
                oc=out_channel,
                ic=in_channel,
                oh=out_h,
                ow=out_w,
                kh=kernel_height,
                kw=kernel_width,
                stride=stride,
                virtual_axis=True,
            )
            operator_compile_args = {
                "config_path": args.config_path,
                "output_path": temp_dir,
                "data_movement_full_vectorize": True,
                "cimflow": True,
                "verify": args.verify,
            }
            operator_compile_args = SimpleNamespace(**operator_compile_args)
            # import pdb; pdb.set_trace()
            operator.set_attr("name", "conv")
            result = run_cimflow(
                args=operator_compile_args, cim_config=get_config(), op=operator
            )
            assert len(result) == 1, f"Fail when generating conv2d code. {attr=}"
            result = result[0]
            if args.verify:
                assert result.attr["VerifyPass"][
                    "check_result"
                ], f"Fail when generating conv2d code. {attr=}"
            cache_conv2d_result[cache_key] = result

    # read code from result
    final_code = get_final_code(
        os.path.join(result.attr["BackendCompilePass"]["code_file"])
    )
    assert final_code is not None, f"Fail when generating conv2d code. {attr=}"

    return final_code


def fill_template(src_path, dst_path, context):

    src_folder, src_file = os.path.split(src_path)

    # 创建 Jinja2 环境和加载器
    env = Environment(loader=FileSystemLoader(src_folder), undefined=StrictUndefined)

    # 加载模板
    template = env.get_template(src_file)

    # 渲染模板
    output = template.render(context)

    with open(dst_path, "w") as f:
        f.write(output)


@dataclass
class ProfileResult:
    stats: str
    save_path: int


cache_dwconv2d_result = dict()


def get_dwcode_conv2d(args, attr):
    global cache_dwconv2d_result
    CIMCOMPILER_HOME = os.environ.get("CIMCOMPILER_HOME")
    template_path = os.path.join(CIMCOMPILER_HOME, "cim_compiler/polycim/template/depthwise_conv.cim")
    temp_dir = tempfile.mkdtemp()
    code_path = os.path.join(temp_dir, "depthwise_conv.cim")

    X_size = reduce(lambda x, y: x * y, attr["X_shape"])
    assert X_size <= get_memory_size("input_memory"), f"{X_size=}"

    batch, _, in_height, in_width = attr["X_shape"]
    in_channel, _, kernel_height, kernel_width = attr["W_shape"]

    padding = 1
    stride = 1

    out_h = (in_height + 2 * padding - kernel_height) // stride + 1
    out_w = (in_width + 2 * padding - kernel_width) // stride + 1

    cache_key = (
        batch,
        in_channel,
        out_h,
        out_w,
        kernel_height,
        kernel_width,
        padding,
        stride,
    )
    if cache_key in cache_dwconv2d_result:
        result = cache_dwconv2d_result[cache_key]
    else:
        context = {
            "INPUT_ROW": in_height,
            "INPUT_COL": in_width,
            "INPUT_CHANNEL": in_channel,
            "OUTPUT_ROW": out_h,
            "OUTPUT_COL": out_w,
            "OUTPUT_CHANNEL": in_channel,
            "KERNEL_SIZE": kernel_height,
            "STRIDE": stride,
            "BATCH": batch,
        }

        fill_template(template_path, code_path, context)

        # backend_compile_cmd = (
        #     f"input_file={code_path} output_path={temp_dir} bash run.sh "
        # )
        # os.system(backend_compile_cmd)
        subprocess.run(
            [
                "cim-compiler",
                "compile",
                "--input-file",
                code_path,
                "--output-dir",
                temp_dir,
                "--config-file",
                args.config_path,
            ],
            check=True,
        )

        result = ProfileResult(stats=None, save_path=temp_dir)
        cache_dwconv2d_result[cache_key] = result

    final_code = get_final_code(os.path.join(result.save_path, "final_code.json"))
    assert final_code is not None
    # import pdb; pdb.set_trace()
    return final_code


def get_core_row_and_col_from_core_name(core_name):
    core_row, core_col = core_name.split("_")[1:]
    return int(core_row), int(core_col)


core_max_row = None
core_max_col = None


def get_max_core_row_col(core_name_list):
    global core_max_row
    global core_max_col
    row, col = get_core_row_and_col_from_core_name(core_name_list[0])
    for core_name in core_name_list:
        core_row, core_col = get_core_row_and_col_from_core_name(core_name)
        row = max(row, core_row)
        col = max(col, core_col)
    core_max_row = row
    core_max_col = col
    return row, col


def get_core_id_from_core_name(core_name):
    global core_max_col
    """
    core_name: core_{row}_{col}
    """
    # parse row and col from core_name
    row, col = core_name.split("_")[1:]
    core_id = int(row) * (core_max_col + 1) + int(col)
    return core_id


unique_name_to_unqiue_id = dict()
max_unique_id = 0


def get_unique_id_from_unique_name(unique_name):
    global unique_name_to_unqiue_id
    global max_unique_id
    if unique_name in unique_name_to_unqiue_id:
        return unique_name_to_unqiue_id[unique_name]
    else:
        unique_name_to_unqiue_id[unique_name] = max_unique_id
        max_unique_id += 1
        return unique_name_to_unqiue_id[unique_name]


def parse_instructions(args, core_name, stage_id, instructions, cache_dir):
    code_list = []
    max_data_size = 0
    max_shape = None
    for idx, instruction in enumerate(instructions):
        # if not (core_name=="core_0_1" and stage_id=="0" and instruction["op"] == "conv" and idx==2):
        #     continue
        # import pdb; pdb.set_trace()
        if instruction["op"] == "read":
            code = get_code_read_global(instruction["attr"])
        elif instruction["op"] == "write":
            code = get_code_write_global(instruction["attr"])
        elif instruction["op"] == "conv":
            code = get_code_conv2d(args, instruction["attr"], cache_dir)
        elif instruction["op"] == "depthwise_conv":
            code = get_dwcode_conv2d(args, instruction["attr"])
        elif instruction["op"] in ("send", "send_ring"):
            data_size = reduce(lambda x, y: x * y, instruction["attr"]["shape"])
            assert data_size <= get_memory_size("output_memory"), f"{data_size=}"
            code = get_code_single_send(
                src_addr=get_memory_base("output_memory"),
                dst_core=get_core_id_from_core_name(
                    instruction["attr"]["dist_core_name"]
                ),
                data_size=reduce(lambda x, y: x * y, instruction["attr"]["shape"]),
                unqiue_id=get_unique_id_from_unique_name(instruction["attr"]["name"]),
            )
        elif instruction["op"] in ("receive", "receive_ring"):
            data_size = reduce(lambda x, y: x * y, instruction["attr"]["shape"])
            assert data_size <= get_memory_size("input_memory"), f"{data_size=}"
            code = get_code_single_receive(
                dst_addr=get_memory_base("input_memory"),
                src_core=get_core_id_from_core_name(
                    instruction["attr"]["src_core_name"]
                ),
                data_size=reduce(lambda x, y: x * y, instruction["attr"]["shape"]),
                unqiue_id=get_unique_id_from_unique_name(instruction["attr"]["name"]),
            )
        elif instruction["op"] == "add":
            code = get_code_add(instruction["attr"])
        else:
            assert False, f"{instruction=}"

        assert code is not None, f"{instruction=}"
        code_list.extend(code)
    return code_list


def get_special_reg_set_code():
    """
    {
        "opcode": 0b101101,
        "rd": inst.reg,
        "imm": inst.value
    }
    #define SPECIAL_REG_SIMD_INPUT_1_BIT_WIDTH 16
    #define SPECIAL_REG_SIMD_INPUT_2_BIT_WIDTH 17
    #define SPECIAL_REG_SIMD_INPUT_3_BIT_WIDTH 18
    #define SPECIAL_REG_SIMD_INPUT_4_BIT_WIDTH 19
    #define SPECIAL_REG_SIMD_OUTPUT_BIT_WIDTH 20
    """

    code_simd_add1 = {"opcode": 0b101101, "rd": 16, "imm": 8}
    code_simd_add2 = {"opcode": 0b101101, "rd": 17, "imm": 8}
    code_simd_out = {"opcode": 0b101101, "rd": 20, "imm": 8}
    return [code_simd_add1, code_simd_add2, code_simd_out]


def parse_noc_tasks(args, json_path, code_save_path, cache_dir):
    """
    {
    "core_0_0": {
        "stages": {
            "0": {
                "cluster_id": "Conv_0_quant",
                "weight_replica_id": 3,
                "instructions": [
                    {
                        "op": "read",
                        "attr": {
                            "shape": [32,3,3,3]
                        }
                    },
                    {
                        "op": "read",
                        "attr": {
                            "shape": [1,4,224,224]
                        }
                    },
                    {
    """
    os.makedirs(code_save_path, exist_ok=True)

    with open(json_path, "r") as f:
        tasks = json.load(f)

    get_max_core_row_col(list(tasks.keys()))

    total_save_files = []
    for core_name, stages in tasks.items():
        code_list = []

        code_list.extend(get_special_reg_set_code())

        for stage_id, stage in stages["stages"].items():
            code = parse_instructions(
                args, core_name, stage_id, stage["instructions"], cache_dir
            )
            code_list.extend(code)

        # padding, forbidden branch code be last codes
        code_list.extend(get_special_reg_set_code())

        # import pdb; pdb.set_trace()

        core_id = get_core_id_from_core_name(core_name)
        core_code_save_path = os.path.join(code_save_path, f"{core_id}.json")
        total_save_files.append(core_code_save_path)
        with open(core_code_save_path, "w") as f:
            json.dump(code_list, f, indent=2)

    return total_save_files


def tidy_json_format(save_path, total_save_files):
    total_code_str = "{"
    for core_idx, file_path in enumerate(total_save_files):
        file_name_with_extension = os.path.basename(file_path)
        file_name_without_extension, _ = os.path.splitext(file_name_with_extension)
        assert str(core_idx) == file_name_without_extension

        with open(file_path, "r") as f:
            code_list = json.load(f)

        code_str = "[\n"
        for idx, code in enumerate(code_list):
            code_str += json.dumps(code, separators=(",", ":"))
            if idx < len(code_list) - 1:
                code_str += ","
            code_str += "\n"
        code_str += "]"

        total_code_str += f'"{core_idx}":' + code_str
        if core_idx < len(total_save_files) - 1:
            total_code_str += ","
        total_code_str += "\n"
    total_code_str += "}"

    with open(save_path, "w") as f:
        f.write(total_code_str)


def parse_cimflow_network_args(subparsers):
    parser = subparsers.add_parser("cimflow_network")
    parser.add_argument("--read-json", "-i", type=str, help="read path")
    parser.add_argument("--save-dir", "-o", type=str, help="save path")
    parser.add_argument("--config-path", "-c", type=str, help="config path")
    parser.add_argument("--verify", action="store_true", help="verify")


def run_cimflow_network(args):
    set_raw_config_by_path(args.config_path)
    each_core_save_dir = os.path.join(args.save_dir, "each_core")
    with tempfile.TemporaryDirectory() as cache_dir:
        try:
            total_save_files = parse_noc_tasks(
                args, args.read_json, each_core_save_dir, cache_dir
            )
        except Exception as e:
            print(f"Error: {e}")
            import pdb; pdb.set_trace()
            pass

    save_file_name = os.path.basename(args.read_json)
    save_file_name = "isa_" + save_file_name
    all_core_code_path = os.path.join(args.save_dir, save_file_name)
    tidy_json_format(
        all_core_code_path,
        total_save_files=total_save_files,
    )

    print("Finish!")
    print(f"{each_core_save_dir = }")
    print(f"{all_core_code_path = }")


# if __name__ == "__main__":
#     main()
