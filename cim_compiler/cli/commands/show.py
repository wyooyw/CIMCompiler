import argparse
import os

from cim_compiler.utils.logger import get_logger
from cim_compiler.simulator.inst import (
    LegacyParser,
    LegacyDumper,
    AsmParser,
    AsmDumper,
    CIMFlowParser,
    CIMFlowDumper
)
from cim_compiler.cli.common import show_args, to_abs_path, uniform_parse_code

logger = get_logger(__name__)

def parse_show_args(subparsers):
    parser = subparsers.add_parser('show')
    parser.add_argument("-i", "--input", type=str, required=True,
                       help="Input file path containing instructions")
    parser.add_argument("--type", type=str, choices=["legacy", "cimflow", "asm", "any"],
                       default="any", help="Input file type")

def run_show(args):
    input_file = to_abs_path(args.input)
    
    logger.info("Begin to show instructions.")
    logger.info(show_args(args))

    # Use uniform_parse_code to automatically detect and parse the input file
    instructions, src_type = uniform_parse_code(args.type, input_file)
    
    # Convert to ASM format and print to console
    dumper = AsmDumper()
    asm_text = dumper.dump_str(instructions)
    
    print(f"Source file type: {src_type}")
    print("Instructions in ASM format:")
    print(asm_text)
    
    logger.info("Show instructions done.")
