import argparse

from cim_compiler.cli.commands.compile import parse_compile_args, run_compile
from cim_compiler.cli.commands.convert import parse_convert_args, run_convert
from cim_compiler.cli.commands.simulate import parse_simulate_args, run_simulate
from cim_compiler.cli.commands.config import parse_config_args, run_config

__all__ = ["parse_convert_args", "run_convert", "parse_compile_args", "run_compile", "parse_simulate_args", "run_simulate", "parse_config_args", "run_config"]
