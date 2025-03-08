import argparse

from cim_compiler.cli.commands.compile import parse_compile_args, run_compile
from cim_compiler.cli.commands.convert import parse_convert_args, run_convert
from cim_compiler.cli.commands.simulate import parse_simulate_args, run_simulate, parse_multi_core_simulate_args, run_multi_core_simulate
from cim_compiler.cli.commands.config import parse_config_args, run_config
from cim_compiler.cli.commands.cfg_pimsim import parse_cfg_pimsim_args, run_cfg_pimsim
from cim_compiler.cli.commands.show import parse_show_args, run_show

__all__ = [
    "parse_convert_args", 
    "run_convert", 
    "parse_compile_args", 
    "run_compile", 
    "parse_simulate_args", 
    "run_simulate", 
    "parse_config_args", 
    "run_config", 
    "parse_cfg_pimsim_args", 
    "run_cfg_pimsim",
    "parse_show_args",
    "run_show",
    "parse_multi_core_simulate_args",
    "run_multi_core_simulate"
]
