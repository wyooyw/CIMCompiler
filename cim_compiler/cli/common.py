import os
import json
from cim_compiler.simulator.inst import (
    LegacyParser,
    AsmParser,
    CIMFlowParser
)
from cim_compiler.utils.logger import get_logger
import argparse
logger = get_logger(__name__)

def to_abs_path(path, parent=os.getcwd()):
    if path is None:
        return None
    if not os.path.isabs(path):
        return os.path.join(parent, path)
    return path

def show_args(args):
    s = "Arguments:\n"
    for key, value in vars(args).items():
        s += f"  {key}: {value}\n"
    return s

def uniform_parse_code(code_type, code_file_path):
    parser_classes = {
        "legacy": LegacyParser,
        "asm": AsmParser,
        "cimflow": CIMFlowParser
    }

    if code_type == "any":
        for cur_code_type, parser_cls in parser_classes.items():
            try:
                # Use the parse_instructions wrapper to automatically handle
                # both single-core and multi-core file formats.
                data = parse_instructions(code_file_path, parser_cls)
                logger.info(f"Parsed {code_file_path} successfully with {parser_cls.__name__}.")
                return data, cur_code_type
            except Exception as e:
                logger.warning(f"Parser {parser_cls.__name__} failed: {e}")
        raise ValueError("Failed to parse source file.")
    else:
        parser_cls = parser_classes.get(code_type)
        if not parser_cls:
            raise ValueError(f"Unknown code type: {code_type}")
        data = parse_instructions(code_file_path, parser_cls)
        return data, code_type

def parse_instructions(file_path, parser_cls):
    """
    Parse instructions from a file. Handles both JSON formatted files and plain text files for the asm format.
    
    For JSON files, supports both single-core (list) and multi-core (dict) formats.
    For asm format (plain text), expects core IDs specified in comments of the form: "# Core {core_id}".
    
    :param file_path: The file containing instruction data.
    :param parser_cls: The parser class (e.g. LegacyParser, AsmParser, CIMFlowParser).
    :return: Parsed instructions. For multi-core files or asm files, returns a dict mapping core IDs to instructions.
    """
    parser = parser_cls()
    
    if parser_cls.__name__ == "AsmParser":
        # Handle asm plain text format.
        results = {}
        current_core = None
        current_lines = []
        with open(file_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                # Check for a core comment line formatted as "# Core {core_id}"
                if stripped.startswith("# Core"):
                    # If instructions for a previous core have been collected, parse and store them.
                    if current_core is not None and current_lines:
                        results[current_core] = parser.parse(current_lines)
                    # Extract the core id from the comment (expected format: "# Core {core_id}")
                    parts = stripped.split()
                    current_core = parts[2] if len(parts) >= 3 else "0"
                    current_lines = []
                else:
                    if stripped:
                        current_lines.append(line)
        # After reading the file, add the remaining instructions to the current core
        if current_core is not None and current_lines:
            results[current_core] = parser.parse(current_lines)
        elif current_lines:  # If no core comment was found, assume all instructions belong to core "0"
            results["0"] = parser.parse(current_lines)
        return results
    else:
        # Handle JSON formatted files.
        with open(file_path, 'r') as f:
            data = json.load(f)
    
        if isinstance(data, list):
            # Single-core JSON file.
            _, instructions = parser.parse_file(file_path)
            return {"0": instructions}
        elif isinstance(data, dict):
            # Multi-core JSON file.
            results = {}
            for core, instructions_list in data.items():
                results[core] = parser.parse(instructions_list)
            return results
        else:
            raise ValueError("Unexpected file format")

# Example usage for LegacyParser:
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Parse instruction files using LegacyParser')
    parser.add_argument('files', nargs='+', help='Paths to instruction files to parse')
    args = parser.parse_args()
    
    # Process each provided file
    for file_path in args.files:
        print(f"\nProcessing file: {file_path}")
        try:
            instructions = parse_instructions(file_path, LegacyParser)
            print(instructions)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")