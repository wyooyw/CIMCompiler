import argparse

def parse_convert_args(subparsers):
    parser = subparsers.add_parser('convert')
    parser.add_argument("--src-type", "--st", type=str, choices=["legacy", "cimflow", "asm"], required=True)
    parser.add_argument("--dst-type", "--dt", type=str, choices=["legacy", "cimflow", "asm"], required=True)
    parser.add_argument("--src-file", "--sf", type=str, required=True)
    parser.add_argument("--dst-file", "--df", type=str, required=True)
    
def parse_compile_args(subparsers):
    parser = subparsers.add_parser('compile')
    parser.add_argument("--input-file", "-i", type=str, required=True)
    # TODO: 
    # 1. use output_file, save only one file, similar as gcc
    # 2. use '--detail' to save all files in same dir
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    parser.add_argument("--config-file", "-c", type=str, required=True)

def parse_simulate_args(subparsers):
    parser = subparsers.add_parser('simulate')
    parser.add_argument("--code-file", "-i", type=str, required=True)
    parser.add_argument("--data-file", "-d", type=str, required=True)
    parser.add_argument("--config-file", "-c", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)

    parser.add_argument("--code-format", type=str, choices=["legacy", "cimflow", "asm"], required=True)

    parser.add_argument("--save-stats", action="store_true", help="Enable saving stats")
    parser.add_argument("--stats-dir", type=str, required=False)
    parser.add_argument("--save-unrolled-code", action="store_true", help="Enable saving unrolled code")
    parser.add_argument("--unrolled-code-format", type=str, required=False, choices=["legacy", "cimflow", "asm"])
    parser.add_argument("--unrolled-code-file", type=str, required=False, default="unrolled_code.cim")
    
    parser.add_argument("--predict-cimcompute-count", type=int, required=False, default=-1)
    
def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    parse_convert_args(subparsers)
    parse_compile_args(subparsers)
    parse_simulate_args(subparsers)
    args = parser.parse_args()
    return args