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

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    parse_convert_args(subparsers)
    parse_compile_args(subparsers)
    args = parser.parse_args()
    return args