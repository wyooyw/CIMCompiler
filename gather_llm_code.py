import argparse
import json
import os

def gather_json_files(directory, max_core_id):
    combined_data = {}
    for core_id in range(max_core_id + 1):
        file_path = os.path.join(directory, str(core_id), 'compiler_output', 'final_code.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    combined_data[core_id] = data
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {file_path}")
        else:
            print(f"File not found: {file_path}")
    return combined_data

def save_combined_json(combined_data, output_file):
    with open(output_file, 'w') as file:
        json.dump(combined_data, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Combine JSON files from multiple cores.')
    parser.add_argument('--directory', '-i', type=str, help='The directory containing core folders.')
    parser.add_argument('--max_core_id', '-n', type=int, help='The maximum core ID to process.')
    parser.add_argument('--output_file', '-o', type=str, help='The output file to save the combined JSON.')

    args = parser.parse_args()

    combined_data = gather_json_files(args.directory, args.max_core_id)
    save_combined_json(combined_data, args.output_file)

if __name__ == '__main__':
    main()
