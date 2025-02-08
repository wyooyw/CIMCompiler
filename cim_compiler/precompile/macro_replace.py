import argparse
from cim_compiler.utils.logger import get_logger

logger = get_logger(__name__)

def replace(macro_mapping, line):
    for key in macro_mapping.keys():
        line = line.replace(key, macro_mapping[key])
    return line

def detect_and_replace_macros(code):
    macro_mapping = dict()
    new_line_list = []
    for line in code.split("\n"):
        line = line.strip()
        if "#define" in line:
            _, name, value = line.strip().split(" ")
            macro_mapping[name] = value
            new_line_list.append("\n")
        else:
            line = replace(macro_mapping, line)
            new_line_list.append(line + "\n")
    return "".join(new_line_list)

def detect_and_replace_macros_from_file(args):
    with open(args.in_file, "r") as f:
        code = f.read()
    new_code = detect_and_replace_macros(code)
    with open(args.out_file, "w") as f:
        f.write(new_code)
    logger.debug(
        f"Macros replaced from file \n{args.in_file}\n and saved to \n{args.out_file}"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Replace macro in a file")
    parser.add_argument("--in-file", "-i", required=True, type=str, help="Input file path")
    parser.add_argument("--out-file", "-o", required=True, type=str, help="Output file path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    detect_and_replace_macros_from_file(args)