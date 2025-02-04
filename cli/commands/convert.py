import argparse
import os

from utils.logger import get_logger

logger = get_logger("cli/convert")

def parse_convert_args(subparsers):
    parser = subparsers.add_parser('convert')
    parser.add_argument("--src-type", "--st", type=str, choices=["legacy", "cimflow", "asm"], required=True)
    parser.add_argument("--dst-type", "--dt", type=str, choices=["legacy", "cimflow", "asm"], required=True)
    parser.add_argument("--src-file", "--sf", type=str, required=True)
    parser.add_argument("--dst-file", "--df", type=str, required=True)

def run_convert(args):
    parser_classes = {
        "legacy": LegacyParser,
        "asm": AsmParser,
        "cimflow": CIMFlowParser
    }
    dumper_classes = {
        "legacy": LegacyDumper,
        "asm": AsmDumper,
        "cimflow": CIMFlowDumper
    }
    parser = parser_classes[args.src_type]()
    dumper = dumper_classes[args.dst_type]()
    _, data = parser.parse_file(args.src_file)
    dumper.dump_to_file(data, args.dst_file)
    print(f"Convert from \n{args.src_type}: {args.src_file}\nto \n{args.dst_type}: {args.dst_file}")
