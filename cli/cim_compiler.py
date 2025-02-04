import argparse
from utils.logger import get_logger
from cli.commands import *

logger = get_logger("cli/cim_compiler")

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