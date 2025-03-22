import argparse
import os
import json
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, required=True)
    parser.add_argument("--output-file", "-o", type=str, required=True)
    return parser.parse_args()

def get_core_ids(input_dir:str):
    core_ids = []
    for core_id in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, core_id)) and core_id.isdigit():
            core_ids.append(int(core_id))
    core_ids = sorted(core_ids)
    assert core_ids[0] == 0, f"Core 0 must be the first core, but got {core_ids[0]}"
    assert core_ids[-1] == len(core_ids) - 1, f"Core {len(core_ids) - 1} must be the last core, but got {core_ids[-1]}"
    return core_ids

def main(args): 
    """
    read args.input_dir/(\d+)/compiler_output/final_code.json
    """
    all_codes = OrderedDict()
    core_ids = get_core_ids(args.input_dir)
    for core_id in core_ids:
        print(f"{core_id=}")
        with open(os.path.join(args.input_dir, str(core_id), "compiler_output", "final_code.json"), "r") as f:
            data = json.load(f)
            all_codes[core_id] = data
    with open(args.output_file, "w") as f:
        json.dump(all_codes, f, indent=4)

    print(f"Combined {len(all_codes)} cores from {args.input_dir} into {args.output_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args)


