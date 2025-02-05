import argparse
from utils.logger import get_logger

logger = get_logger(__name__)

def remove_comments(code):
    lines = code.split("\n")
    new_lines = []
    for line in lines:
        line = line.strip()
        if "//" in line:
            line = line[: line.index("//")]
        line = line + "\n"
        new_lines.append(line)
    return "".join(new_lines)

def remove_comments_from_file(args):
    with open(args.in_file, "r") as f:
        code = f.read()
    new_code = remove_comments(code)
    with open(args.out_file, "w") as f:
        f.write(new_code)
    logger.debug(
        f"Comments removed from file \n{args.in_file}\n and saved to \n{args.out_file}"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Remove comments from a file")
    parser.add_argument(
        "--in-file", 
        "-i", 
        type=str, 
        required=True,
        help="Input file to remove comments from"
    )
    parser.add_argument(
        "--out-file", 
        "-o", 
        type=str, 
        required=True,
        help="Output file to save the processed content"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    remove_comments_from_file(args)