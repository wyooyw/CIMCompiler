import argparse

def parse_convert_args(subparsers):
    parser = subparsers.add_parser('convert')
    parser.add_argument("--src-type", "--st", type=str, choices=["legacy", "cimflow", "asm"], required=True)
    parser.add_argument("--dst-type", "--dt", type=str, choices=["legacy", "cimflow", "asm"], required=True)
    parser.add_argument("--src-file", "--sf", type=str, required=True)
    parser.add_argument("--dst-file", "--df", type=str, required=True)
    

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    parse_convert_args(subparsers)

    
    args = parser.parse_args()
    return args