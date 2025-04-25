import os
import re
import shutil

def extract_values_from_log(log_path):
    # Initialize a dictionary to store the extracted values
    values = {
        'model_path': None,
        'is_bit_sparse': None,
        'is_value_sparse': None,
        'config': None,
        'model_name': None,
        'macro_count': None
    }
    
    # Define the patterns to search for in the log file
    patterns = {
        'model_name': r'model_name=\'(.+)\'',
        'model_path': r'model_path=(.+)',
        'is_bit_sparse': r'is_bit_sparse=(.+)',
        'is_value_sparse': r'is_value_sparse=(.+)',
        'config': r"os\.environ\.get\('CONFIG_PATH', None\)=(.+)"
    }
    
    # Open and read the log file
    with open(log_path, 'r') as file:
        log_content = file.read()
        
        # Search for each pattern and store the result in the dictionary
        for key, pattern in patterns.items():
            match = re.search(pattern, log_content)
            if match:
                values[key] = match.group(1).strip()
        
        # Extract model name from model_path
        # if values['model_path']:
        #     model_name_match = re.search(r'/models/([^/]+)/', values['model_path'])
        #     if model_name_match:
        #         values['model_name'] = model_name_match.group(1)
        
        # Extract macro count from config
        if values['config']:
            macro_count_match = re.search(r'config_(\d+)macro', values['config'])
            if macro_count_match:
                values['macro_count'] = int(macro_count_match.group(1))
    
    return values

def build_directory_mapping(base_dir):
    # Initialize the main dictionary to store the mapping
    directory_mapping = {}
    
    # Iterate over each subdirectory in the base directory
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        # Check if the path is a directory
        if os.path.isdir(subdir_path):
            log_path = os.path.join(subdir_path, 'model_runner.log')
            
            # Check if the log file exists
            if os.path.exists(log_path):
                # Extract values from the log file
                values = extract_values_from_log(log_path)
                
                # Map the subdirectory name to the extracted values
                directory_mapping[subdir] = values
    
    return directory_mapping

# Define the base directory
base_directory = '.result'
out_dir = 'wyk_result'

# Build the directory mapping
mapping = build_directory_mapping(base_directory)

# Print the mapping
for dir_name, values in mapping.items():
    print(f"Directory: {dir_name}")
    for key, value in values.items():
        print(f"  {key}: {value}")
from tqdm import tqdm
for dir_name, values in tqdm(mapping.items()):
    dense_name = 'dense' if values['is_bit_sparse'] == 'False' else 'bit_sparse'
    src_dir = os.path.join(base_directory, dir_name, values['model_name'], dense_name)
    dst_dir = os.path.join(out_dir, dense_name, str(values['macro_count']), values['model_name'])
    
    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)
    
    # Iterate over each subdirectory in src_dir
    for d in os.listdir(src_dir):
        src_subdir = os.path.join(src_dir, d)
        dst_subdir = os.path.join(dst_dir, d)
        
        # Ensure the destination subdirectory exists
        os.makedirs(dst_subdir, exist_ok=True)
        
        # Define the files to copy
        files_to_copy = ['flat_code.json', 'flat_stats.json', 'global_image']
        # if dense_name == 'bit_sparse':
        #     files_to_copy.append()
        
        for file_name in files_to_copy:
            src_file = os.path.join(src_subdir, file_name)
            dst_file = os.path.join(dst_subdir, file_name)
            
            # Check if the source file exists before copying
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
    