import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
# sys.path.append(str(os.path.join(project_root, "src")))
os.environ['CIM_COMPILER_BASE'] = str(project_root)

import argparse
from cim_compiler.utils.logger import get_logger
from cim_compiler.cli.commands import *

logger = get_logger("cli/cim_compiler")

def get_project_root():
    """获取项目根目录"""
    # 如果是通过 pip install -e . 安装的，__file__ 会指向实际的源代码位置
    return Path(__file__).parent.parent

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    parse_convert_args(subparsers)
    parse_compile_args(subparsers)
    parse_simulate_args(subparsers)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.command == "convert":
        run_convert(args)
    elif args.command == "compile":
        run_compile(args)
    elif args.command == "simulate":
        run_simulate(args)
    else:
        raise ValueError(f"Invalid command: {args.command}")

if __name__ == "__main__":
    main()