import argparse
import glob
import os
import shutil
import subprocess
import tempfile
import json

from cim_compiler.utils.logger import get_logger
from cim_compiler.cli.common import show_args, to_abs_path

logger = get_logger(__name__)

def parse_config_args(subparsers):
    parser = subparsers.add_parser('config')
    parser.add_argument("--config-template-path", "-i", type=str, help="config template path")
    parser.add_argument("--config-output-path", "-o", type=str, help="config output path")
    parser.add_argument("--yes", action="store_true", default=False, help="yes")

def calculate_macro_size_byte(config):
    """
    "macro": {
        "n_group": 4,
        "n_macro": 32,
        "n_row": 16,
        "n_comp": 16,
        "n_bcol": 16
    },
    """
    macro = config["macro"]
    macro_size_bit = macro["n_macro"] * macro["n_row"] * macro["n_comp"] * macro["n_bcol"]
    assert macro_size_bit % 8 == 0

    macro_size_byte = macro_size_bit // 8
    return macro_size_byte

def run_config(args):
    """
    1.read config_template.json
    2.calculate each memory's offset, fill into 'offset_byte'
    3.write config.json
    """

    args.config_template_path = to_abs_path(args.config_template_path)
    args.config_output_path = to_abs_path(args.config_output_path)

    logger.info("Begin to make config file.")
    logger.info(show_args(args))

    with open(args.config_template_path, "r") as f:
        config = json.load(f)
    
    # fill macro memory size
    macro_size_byte = calculate_macro_size_byte(config)
    macro_memory_idx = [idx for idx,memory in enumerate(config["memory_list"]) if memory["type"] == "macro"]
    assert len(macro_memory_idx) == 1
    config["memory_list"][macro_memory_idx[0]]["addressing"]["size_byte"] = macro_size_byte
    
    # fill all memory's offset
    last_end = 0
    for memory in config["memory_list"]:
        addressing = memory["addressing"]
        addressing["offset_byte"] = last_end
        last_end += addressing["size_byte"]
    
    # check if config file already exists
    if os.path.exists(args.config_output_path) and (not args.yes):
        make_sure = input(
            f"\nConfig file already exists ({args.config_output_path}).\n Do you want to overwrite it? (y/n)"
        )
        make_sure = True if make_sure.lower() == "y" else False
    else:
        make_sure = True
    
    # write config file
    if make_sure:
        with open(args.config_output_path, "w") as f:
            json.dump(config, f, indent=4)
    
    logger.info("Make config file done.")
