import argparse
import glob
import os
import shutil
import subprocess
import tempfile
import json
from dataclasses import dataclass
from cim_compiler.utils.logger import get_logger
from cim_compiler.cli.common import show_args, to_abs_path

logger = get_logger(__name__)

def parse_cfg_pimsim_args(subparsers):
    parser = subparsers.add_parser('cfg_pimsim')
    parser.add_argument("--config-template-path", "-i", type=str, help="config template path")
    parser.add_argument("--config-output-path", "-o", type=str, help="config output path")

def calculate_macro_size_byte(pim_unit_config):
    """
    "pim_unit_config": {
        "macro_total_cnt": 1,
        "macro_group_size": 1,
        "macro_size": {
          "compartment_cnt_per_macro": 32,
          "element_cnt_per_compartment": 8,
          "row_cnt_per_element": 1,
          "bit_width_per_row": 8
        },
        "address_space": {
          "offset_byte": 0,
          "size_byte": 256
        },
    """
    n_macro = pim_unit_config["macro_total_cnt"]
    n_row = pim_unit_config["macro_size"]["row_cnt_per_element"]
    n_comp = pim_unit_config["macro_size"]["compartment_cnt_per_macro"]
    n_bit_per_element = pim_unit_config["macro_size"]["bit_width_per_row"]
    n_vcol = pim_unit_config["macro_size"]["element_cnt_per_compartment"]
    macro_size_byte = n_macro * n_row * n_comp * n_vcol * n_bit_per_element // 8
    return macro_size_byte

@dataclass
class MemoryInfo:
    name: str
    offset_byte: int
    size_byte: int

def parse_memory_list(config_path):
    """

    macro_total_cnt": 1,
        "macro_group_size": 1,
        "macro_size": {
          "compartment_cnt_per_macro": 32,
          "element_cnt_per_compartment": 8,
          "row_cnt_per_element": 1,
          "bit_width_per_row": 8
        },
        
    return: list of MemoryInfo:
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    is_pimsim_config = "chip_config" in config
    is_compiler_config = "macro" in config
    memory_info_list = []
    if is_pimsim_config:
        # parse macro size
        pim_unit_config = config["chip_config"]["core_config"]["pim_unit_config"]
        n_macro = pim_unit_config["macro_total_cnt"]
        n_row = pim_unit_config["macro_size"]["row_cnt_per_element"]
        n_comp = pim_unit_config["macro_size"]["compartment_cnt_per_macro"]
        n_bit_per_element = pim_unit_config["macro_size"]["bit_width_per_row"]
        n_vcol = pim_unit_config["macro_size"]["element_cnt_per_compartment"]
        size_byte = n_macro * n_row * n_comp * n_vcol * n_bit_per_element // 8
        memory_info_list.append(MemoryInfo(name="macro", offset_byte=0, size_byte=size_byte))

        # parse local memory size
        local_memory_list = pim_unit_config["local_memory_unit_config"]["local_memory_list"]
        for local_memory in local_memory_list:
            memory_info_list.append(MemoryInfo(
                name=local_memory["name"], 
                offset_byte=0, 
                size_byte=local_memory["addressing"]["size_byte"]
            ))
    elif is_compiler_config:
        n_macro = config["macro"]["n_macro"]
        n_row = config["macro"]["n_row"]
        n_comp = config["macro"]["n_comp"]
        n_bcol = config["macro"]["n_bcol"]
        



def run_cfg_pimsim(args):
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
    
    # parse macro memory size
    pim_unit_config = config["chip_config"]["core_config"]["pim_unit_config"]
    macro_size_byte = calculate_macro_size_byte(pim_unit_config)

    # parse local memory size
    local_memory_list = config["chip_config"]["core_config"]["local_memory_unit_config"]["local_memory_list"]
    last_end = macro_size_byte
    memory_info_list = []
    for local_memory in local_memory_list:
        memory_info_list.append(MemoryInfo(
            name=local_memory["name"], 
            offset_byte=last_end, 
            size_byte=local_memory["addressing"]["size_byte"]
        ))
        last_end += local_memory["addressing"]["size_byte"]

    # fill macro memory offset
    pim_unit_config["address_space"]["offset_byte"] = 0
    pim_unit_config["address_space"]["size_byte"] = macro_size_byte
    
    # fill local memory size
    for local_memory_cfg, local_memory_info in zip(local_memory_list, memory_info_list):
        local_memory_cfg["addressing"]["offset_byte"] = local_memory_info.offset_byte

    # fill global memory size
    config["chip_config"]["global_memory_config"]["addressing"]["offset_byte"] = last_end
    
    # check if config file already exists
    if os.path.exists(args.config_output_path):
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
