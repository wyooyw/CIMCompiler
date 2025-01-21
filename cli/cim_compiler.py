import argparse
from cli.arguments import parse_args
from simulator.inst import *

def convert(args):
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
    
    

def main():
    args = parse_args()
    if args.command == "convert":
        convert(args)
    else:
        raise ValueError(f"Invalid command: {args.command}")

if __name__ == "__main__":
    main()