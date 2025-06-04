import sys
from pathlib import Path

from polycim.cli.arguments import set_args

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
# sys.path.append(str(os.path.join(project_root, "src")))
# os.environ['CIM_COMPILER_BASE'] = str(project_root)

import argparse

from polycim.cli.commands import *
from polycim.utils.logger import get_logger

logger = get_logger("cli/polycim")


def get_project_root():
    """获取项目根目录"""
    # 如果是通过 pip install -e . 安装的，__file__ 会指向实际的源代码位置
    return Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    parse_operator_args(subparsers)
    parse_cimflow_network_args(subparsers)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_args(args)

    if args.command == "op":
        run_operator(args)
    elif args.command == "cimflow_network":
        run_cimflow_network(args)
    else:
        raise ValueError(f"Invalid command: {args.command}")


if __name__ == "__main__":
    main()
