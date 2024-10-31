import argparse


def replace(macro_mapping, line):
    for key in macro_mapping.keys():
        line = line.replace(key, macro_mapping[key])
    return line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace macro in a file")
    parser.add_argument("--in-file", type=str, help="")
    parser.add_argument("--out-file", type=str, help="")
    args = parser.parse_args()

    # Read the file and replace the macro
    macro_mapping = dict()
    new_line_list = []
    with open(args.in_file, "r") as f:
        for line in f:
            if "#define" in line:
                _, name, value = line.strip().split(" ")
                macro_mapping[name] = value
                new_line_list.append("\n")
            else:
                line = replace(macro_mapping, line)
                new_line_list.append(line)

    with open(args.out_file, "w") as f:
        f.write("".join(new_line_list))
    print(
        f"Macros replaced from file \n{args.in_file}\n and saved to \n{args.out_file}"
    )
