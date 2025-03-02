import argparse
import os
from cim_compiler.simulator.simulator import Simulator
from cim_compiler.utils.logger import get_logger

from cim_compiler.precompile import detect_and_replace_macros, remove_comments
from cim_compiler.simulator.inst import *
from cim_compiler.simulator.simulator import Memory, MemorySpace, SpecialReg
from cim_compiler.cli.common import show_args, to_abs_path, uniform_parse_code

logger = get_logger(__name__)

def parse_simulate_args(subparsers):
    parser = subparsers.add_parser('simulate')
    parser.add_argument("--code-file", "-i", type=str, required=True)
    parser.add_argument("--data-file", "-d", type=str, required=False)
    parser.add_argument("--config-file", "-c", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)

    parser.add_argument("--code-format", "-f", type=str, choices=["legacy", "cimflow", "asm", "any"], required=False, default="any")

    parser.add_argument("--save-stats", action="store_true", default=True, help="Enable saving stats")
    parser.add_argument("--stats-dir", type=str, required=False)
    parser.add_argument("--save-unrolled-code", action="store_true", default=True, help="Enable saving unrolled code")
    parser.add_argument("--unrolled-code-format", type=str, required=False, choices=["legacy", "cimflow", "asm"])
    parser.add_argument("--unrolled-code-file", type=str, required=False, default="unrolled_code.json")
    
    parser.add_argument("--predict-cimcompute-count", type=int, required=False, default=-1)


def run_simulate(args):
    
    # update args
    args.code_file = to_abs_path(args.code_file)
    args.data_file = to_abs_path(args.data_file)
    args.output_dir = to_abs_path(args.output_dir)
    args.config_file = to_abs_path(args.config_file)

    logger.info("Begin to simulate.")
    logger.info(show_args(args))

    os.makedirs(args.output_dir, exist_ok=True)

    code, code_format = uniform_parse_code(args.code_format, args.code_file)
    
    if args.save_unrolled_code and args.unrolled_code_format is None:
        args.unrolled_code_format = code_format

    if args.save_stats and args.stats_dir is None:
        args.stats_dir = args.output_dir

    if args.save_unrolled_code:
        args.unrolled_code_file = to_abs_path(args.unrolled_code_file, parent=args.output_dir)

    if args.predict_cimcompute_count == -1:
        pimcompute_count = None
    else:
        pimcompute_count = args.predict_cimcompute_count

    simulator = Simulator.from_config(args.config_file)

    # load data to global memory
    # TODO: support load data into other memory space
    # read data
    if args.data_file is not None:
        with open(args.data_file, "rb") as file:
            data = file.read()
        data = bytearray(data)
        global_memory_base = simulator.memory_space.get_base_of("global")
        simulator.memory_space.write(data, global_memory_base, len(data))

    # run code
    status, stats, flat = simulator.run_code(
        code, total_pim_compute_count=pimcompute_count
    )
    if status != simulator.FINISH:
        raise ValueError(f"Simulator failed: {status=}")
    if args.save_stats:
        stats.dump(args.stats_dir)
    if args.save_unrolled_code:
        flat_code = flat.get_flat_code()
        dumper = {
            "legacy": LegacyDumper,
            "asm": AsmDumper,
            "cimflow": CIMFlowDumper
        }[args.unrolled_code_format]()
        dumper.dump_to_file(flat_code, args.unrolled_code_file, core_id=0)

    # get image of global memory
    # TODO: support get image from other memory space
    output_image = simulator.memory_space.get_memory_by_name("global").read_all()
    with open(os.path.join(args.output_dir, "image.bin"), "wb") as f:
        f.write(output_image)

    logger.info(f"Simulate finished.")