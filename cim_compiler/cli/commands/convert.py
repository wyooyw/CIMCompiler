import argparse
import os
import copy

from cim_compiler.utils.logger import get_logger
from cim_compiler.simulator.inst import (
    LegacyParser,
    LegacyDumper,
    AsmParser,
    AsmDumper,
    CIMFlowParser,
    CIMFlowDumper,
    CIMOutputInst,
    CIMTransferInst,
    PrintInst,
    DebugInst,
    RIInst,
    SIMDInst
)
from cim_compiler.cli.common import show_args, to_abs_path, uniform_parse_code

logger = get_logger(__name__)

try:
    from tqdm import tqdm
except ImportError:      # 若未安装 tqdm，则不启用进度条
    tqdm = None

def parse_convert_args(subparsers):
    parser = subparsers.add_parser('convert')
    parser.add_argument("--src-type", "--st", type=str, choices=["legacy", "cimflow", "asm", "any"], default="any")
    parser.add_argument("--dst-type", "--dt", type=str, choices=["legacy", "cimflow", "asm"], required=True)
    parser.add_argument("--src-file", "--sf", type=str, required=False)
    parser.add_argument("--dst-file", "--df", type=str, required=False)
    parser.add_argument("--filter-out-invalid-instructions", action="store_true", default=True,
                       help="Filter out invalid instructions from the conversion result")
    parser.add_argument("--add-single-core-id", action="store_true",
                       help="Add core id 0 to the conversion result. Only when dst-type is legacy or cimflow can use this.")
    parser.add_argument("--src-dir", type=str, required=False,
                       help="Source directory containing the files to be converted.")
    parser.add_argument("--dst-dir", type=str, required=False,
                       help="Destination directory for the converted files.")

    
def check_args(args):
    if args.src_dir and args.src_file:
        raise ValueError("src-dir and src-file cannot be both provided.")
    if args.dst_dir and args.dst_file:
        raise ValueError("dst-dir and dst-file cannot be both provided.")
    if not args.src_dir and not args.src_file:
        raise ValueError("src-dir or src-file must be provided.")
    if not args.dst_dir and not args.dst_file:
        raise ValueError("dst-dir or dst-file must be provided.")
    
    if args.src_dir and not args.dst_dir:
        raise ValueError("dst-dir must be provided when src-dir is provided.")
    if args.dst_dir and not args.src_dir:
        raise ValueError("src-dir must be provided when dst-dir is provided.")

def run_convert(args):
    check_args(args)

    args.src_dir = to_abs_path(args.src_dir)
    args.dst_dir = to_abs_path(args.dst_dir)
    args.src_file = to_abs_path(args.src_file)
    args.dst_file = to_abs_path(args.dst_file)

    logger.info("Begin to convert.")
    logger.info(show_args(args))

    if args.src_file:
        convert_file(args)
    elif args.src_dir:
        convert_dir(args)
    else:
        raise ValueError("src-file or src-dir must be provided.")

def convert_dir(args):
    """
    Recursively convert every `.json` file inside ``args.src_dir``.
    
    For each source file:
      1. Keep the same relative directory structure under ``args.dst_dir``.
      2. Change the filename extension according to ``args.dst_type``:
         * legacy / cimflow -> .json
         * asm             -> .asm
      3. Call ``convert_file`` to finish the conversion.
      4. Display a progress bar while converting (requires `tqdm`).
    """
    extension_map = {
        "legacy": ".json",
        "cimflow": ".json",
        "asm": ".asm"
    }
    dst_ext = extension_map[args.dst_type]

    # -------- ① 预扫描，收集全部待转换文件 --------
    json_files = [
        os.path.join(dirpath, fn)
        for dirpath, _, filenames in os.walk(args.src_dir)
        for fn in filenames
        if fn.endswith("flat_code.json")
    ]

    if not json_files:
        logger.warning("No .json files found in %s", args.src_dir)
        return

    # -------- ② 选择是否使用进度条 --------
    iterable = tqdm(json_files, desc="Converting", unit="file") if tqdm else json_files

    # -------- ③ 逐文件转换 --------
    for src_path in iterable:
        # 计算相对路径并替换后缀
        rel_path = os.path.relpath(src_path, args.src_dir)
        rel_base, _ = os.path.splitext(rel_path)
        dst_rel_path = rel_base + dst_ext
        dst_path = os.path.join(args.dst_dir, dst_rel_path)

        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 为防止循环中相互干扰，复制一份 args
        file_args = copy.copy(args)
        file_args.src_file = src_path
        file_args.dst_file = dst_path

        logger.info(f"Converting {src_path} -> {dst_path}")
        try:
            convert_file(file_args)
        except Exception as e:
            logger.error(f"Failed to convert {src_path}: {e}")

    # -------- ④ 关闭进度条（若启用） --------
    if tqdm:
        iterable.close()

def convert_file(args):
    data, src_type = uniform_parse_code(args.src_type, args.src_file)

    if args.filter_out_invalid_instructions:
        data = filter_invalid_instructions(data)

    dumper_classes = {
        "legacy": LegacyDumper,
        "asm": AsmDumper,
        "cimflow": CIMFlowDumper
    }
    dumper = dumper_classes[args.dst_type]()
    if args.add_single_core_id:
        # if args.dst_type == "asm":
        #     raise ValueError("ASM does not support adding core id.")
        dumper.dump_to_file(data, args.dst_file, core_id=0)
    else:
        dumper.dump_to_file(data, args.dst_file)
        
    logger.info("Convert done.")

def filter_invalid_instructions(instructions):
    """
    Filter out invalid instructions from the instruction list.
    directly delete instruction will cause jump addresss error.
    so, replace invalid instruction with a no-op instruction.
    """
    new_instructions = []
    no_op_inst = RIInst(
        opcode=0,
        reg_in=0,
        reg_out=0,
        imm=0
    )
    invalid_inst_types = (CIMOutputInst, CIMTransferInst, PrintInst, DebugInst)
    for inst in instructions:
        if isinstance(inst, invalid_inst_types):
            new_instructions.append(no_op_inst)
        elif isinstance(inst, SIMDInst) and inst.opcode == 9:
            new_instructions.append(no_op_inst)
        else:
            new_instructions.append(inst)
    return new_instructions