import argparse
from cli.arguments import parse_args
from simulator.inst import *
import os
import subprocess
from precompile import remove_comments, detect_and_replace_macros
import shutil
import tempfile
import glob

from utils.logger import get_logger

logger = get_logger("cli/cim_compiler")

def run_convert(args):
    parser_classes = {
        "legacy": LegacyParser,
        "asm": AsmParser,
        "cimflow": CIMFlowParser
    }
    dumper_classes = {
        "legacy": LegacyDumper,
        "asm": AsmDumper,
        "cimflow": CIMFlowDumper
    }
    parser = parser_classes[args.src_type]()
    dumper = dumper_classes[args.dst_type]()
    _, data = parser.parse_file(args.src_file)
    dumper.dump_to_file(data, args.dst_file)
    print(f"Convert from \n{args.src_type}: {args.src_file}\nto \n{args.dst_type}: {args.dst_file}")

def to_abs_path(path):
    cwd = os.getcwd()
    if not os.path.isabs(path):
        return os.path.join(cwd, path)
    return path

def run_compile(args):
    input_file = to_abs_path(args.input_file)
    output_path = to_abs_path(args.output_path)
    config_path = to_abs_path(args.config_path)

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


def main():
    args = parse_args()
    if args.command == "convert":
        run_convert(args)
    elif args.command == "compile":
        run_compile(args)
    else:
        raise ValueError(f"Invalid command: {args.command}")

if __name__ == "__main__":
    main()