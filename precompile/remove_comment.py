import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Remove comments from a file')
    parser.add_argument('--in-file', type=str, help='File to remove comments from')
    parser.add_argument('--out-file', type=str, help='File to remove comments from')
    args = parser.parse_args()
    new_line_list = []
    with open(args.in_file, 'r') as f:
        for line in f:
            if '//' in line:
                line = line[:line.index('//')]
                line = line + "\n"
            new_line_list.append(line)
    with open(args.out_file, 'w') as f:
        f.write(''.join(new_line_list))
    print(f'Comments removed from file \n{args.in_file}\n and saved to \n{args.out_file}')