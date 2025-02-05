import argparse
import glob
import os
import shutil
import subprocess
import tempfile

from precompile import detect_and_replace_macros, remove_comments
from utils.logger import get_logger

logger = get_logger("cli/compile")


def parse_compile_args(subparsers):
    parser = subparsers.add_parser('compile')
    parser.add_argument("--input-file", "-i", type=str, required=True)
    # TODO: 
    # 1. use output_file, save only one file, similar as gcc
    # 2. use '--detail' to save all files in same dir
    parser.add_argument("--output-dir", "-o", type=str, required=True)
    parser.add_argument("--config-file", "-c", type=str, required=True)

def to_abs_path(path, parent=os.getcwd()):
    if not os.path.isabs(path):
        return os.path.join(parent, path)
    return path

def run_compile(args):
    input_file = to_abs_path(args.input_file)
    output_path = to_abs_path(args.output_dir)
    config_path = to_abs_path(args.config_file)

    os.makedirs(output_path, exist_ok=True)

    with open(input_file, "r") as f:
        code = f.read()
    
    # Precompile
    code = remove_comments(code)
    code = detect_and_replace_macros(code)

    # Copy source file
    origin_code_path = os.path.join(output_path, "origin_code.cim")
    shutil.copy(input_file, origin_code_path)

    # Save precompiled code
    precompile_final_path = os.path.join(output_path, "precompile.cim")
    with open(precompile_final_path, "w") as f:
        f.write(code)

    # ANTLR: code -> ast(json)
    home = os.environ["CIM_COMPILER_BASE"]
    antlr_home = os.path.join(home, "antlr")
    
    with tempfile.TemporaryDirectory() as temp_dir:

        # Generate ANTLR files
        subprocess.run([
            "java", "-cp", os.path.join(antlr_home, "antlr-4.7.1-complete.jar"),
            "org.antlr.v4.Tool", "CIM.g", "-o", temp_dir
        ], check=True)

        # Compile ANTLR Java files
        # TODO: Only compile once
        shutil.copy(os.path.join(antlr_home, "AntlrToJson.java"), temp_dir)
        subprocess.run([
            "javac", 
            "-cp", 
            os.path.join(antlr_home, '*'), 
            *glob.glob(os.path.join(temp_dir, 'CIM*.java')), 
            os.path.join(temp_dir, 'AntlrToJson.java')
        ], check=True)
        
        # Run ANTLR to generate JSON AST
        subprocess.run([
            "java", "-cp", f".:{antlr_home}/antlr-4.7.1-complete.jar:{antlr_home}/gson-2.11.0.jar",
            "AntlrToJson", precompile_final_path, os.path.join(output_path, "ast.json")
        ], cwd=temp_dir, check=True)

    # AST (json) -> MLIR
    subprocess.run([
        "./build/bin/main", os.path.join(output_path, "ast.json"), output_path, config_path
    ], check=True)
    logger.info(f"Compile done. Output in {output_path}")