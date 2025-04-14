import argparse
import os

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

def parse_convert_args(subparsers):
    parser = subparsers.add_parser('convert')
    parser.add_argument("--src-type", "--st", type=str, choices=["legacy", "cimflow", "asm", "any"], default="any")
    parser.add_argument("--dst-type", "--dt", type=str, choices=["legacy", "cimflow", "asm"], required=True)
    parser.add_argument("--src-file", "--sf", type=str, required=True)
    parser.add_argument("--dst-file", "--df", type=str, required=True)
    parser.add_argument("--filter-out-invalid-instructions", action="store_true", default=True,
                       help="Filter out invalid instructions from the conversion result")
    parser.add_argument("--add-single-core-id", action="store_true", default=True,
                       help="Add core id 0 to the conversion result. Only when dst-type is legacy or cimflow can use this.")



def run_convert(args):
    args.src_file = to_abs_path(args.src_file)
    args.dst_file = to_abs_path(args.dst_file)

    logger.info("Begin to convert.")
    logger.info(show_args(args))

    data, src_type = uniform_parse_code(args.src_type, args.src_file)

    dumper_classes = {
        "legacy": LegacyDumper,
        "asm": AsmDumper,
        "cimflow": CIMFlowDumper
    }
    dumper = dumper_classes[args.dst_type]()

    if isinstance(data, dict) and len(data) > 1:
        # Open the output file once, overwriting any existing content.
        with open(args.dst_file, "w") as outf:
            if args.dst_type != "asm":
                outf.write("{\n")
            # For legacy or cimflow, we need to write the core_id as the key.
            # For asm, we need to write the core_id as a comment line.
            keys = sorted(data.keys())
            for i, core_id in enumerate(keys):
                instructions = data[core_id]
                if args.filter_out_invalid_instructions:
                    instructions = filter_invalid_instructions(instructions)
                # For ASM, add a header comment line for each core.
                if args.dst_type == "asm":
                    dump_str = dumper.dump_str(instructions, core_id)
                else:
                    # For legacy or cimflow, the dumper can take a core_id argument.
                    dump_str = dumper.dump_str(instructions, core_id=core_id, curly=False)
                    if i < len(keys) - 1:
                        dump_str = dump_str[:-1] + ",\n"
                outf.write(dump_str)
            if args.dst_type != "asm":
                outf.write("}\n")
    else:
        if isinstance(data, dict):
            data = list(data.values())[0]
        if args.filter_out_invalid_instructions:
            data = filter_invalid_instructions(data)
        with open(args.dst_file, "w") as outf:
            outf.write("")
        if args.add_single_core_id:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between different instruction formats.",
        epilog="Example usage:\npython convert.py convert --src-type asm --dst-type legacy --src-file input.asm --dst-file output.json"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    parse_convert_args(subparsers)
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        exit(1)
        
    run_convert(args)
