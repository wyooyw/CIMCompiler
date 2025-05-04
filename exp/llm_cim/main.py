import argparse
import os
import time
import logging
from pathlib import Path
from cim_compiler.op.llm.helper import AttnDecodeCPConfig, SplitStageConfig
from cim_compiler.simulator.macro_utils import MacroConfig
from test.op.test_reduce.test_reduce import get_reduce_config
from test.base import SPMDOpRunner
import math
from datetime import datetime
from cim_compiler.simulator.simulator import MemorySpace

def parse_args():
    parser = argparse.ArgumentParser(description="LLM CIM Configuration")
    
    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--n-head", type=int, required=True, help="Number of attention heads")
    model_group.add_argument("--hidden-size", type=int, required=True, help="Hidden dimension size")
    model_group.add_argument("--seqlen", type=int, required=True, help="Sequence length")
    
    # Mapping parameters
    mapping_group = parser.add_argument_group("Mapping Parameters")
    mapping_group.add_argument("--mapping-cp-sizes", type=int, nargs="+", required=True, 
                             help="CP sizes for each computation")
    
    # Configuration parameters
    config_group = parser.add_argument_group("Configuration Parameters")
    config_group.add_argument("--world-size", type=int, required=True, help="Number of cores")
    config_group.add_argument("--config-path", type=str, required=True, help="Path to configuration file")
    config_group.add_argument("--distributed-dram-type", type=str, default="default", 
                            choices=["split", "multiply", "default"], 
                            help="Memory distribution type")
    
    # Experiment parameters
    exp_group = parser.add_argument_group("Experiment Parameters")
    exp_group.add_argument("--split-stages", action="store_true", 
                          help="Whether to divide into 5 stages and generate code for each")
    exp_group.add_argument("--verify", action="store_true", 
                         help="Verify correctness with data")
    exp_group.add_argument("--debug", action="store_true", 
                         help="Enable debug logs, save files and logs to .debug/time directory")
    exp_group.add_argument("--save-dir", type=str, 
                         default=f"result/{datetime.now().strftime('%Y%m%d%H%M%S')}/", 
                         help="Save directory")
    return parser.parse_args()

def setup_logging(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if debug_mode:
        debug_dir = Path(f".debug/{int(time.time())}")
        debug_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(debug_dir / "debug.log")
        file_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(file_handler)
        return debug_dir
    return None

def main():
    args = parse_args()
    debug_dir = setup_logging(args.debug)
    
    # Your main code logic here
    if args.debug:
        logging.debug(f"Debug mode enabled. Files will be saved to: {debug_dir}")
    
    logging.info(f"Model parameters: n_head={args.n_head}, hidden_size={args.hidden_size}, seqlen={args.seqlen}")
    logging.info(f"Mapping CP sizes: {args.mapping_cp_sizes}")
    logging.info(f"World size: {args.world_size}, Distributed DRAM type: {args.distributed_dram_type}")
    
    # Add the rest of your application logic here
    cim_config = MacroConfig.from_config(args.config_path)
    cim_config.set_default_bit_width(16)

    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "cim_compiler/op/llm/attn_decode_tp_cp.cim")
    
    n_head = 0
    for cp_size in args.mapping_cp_sizes:
        assert args.world_size % cp_size == 0, f"World size {args.world_size} must be divisible by CP size {cp_size}"
        n_head_per_round = args.world_size // cp_size
        n_head += n_head_per_round
    assert n_head == args.n_head, f"n_head {n_head} must be equal to {args.n_head}"

    assert args.hidden_size % args.n_head == 0, f"hidden_size {args.hidden_size} must be divisible by n_head {args.n_head}"
    hidden_size_per_head = args.hidden_size // args.n_head

    # check capacity of input memory
    for i, cp_size in enumerate(args.mapping_cp_sizes):
        k_local_capacity = hidden_size_per_head * (args.seqlen // cp_size) * 2
        input_memory_capacity = MemorySpace.from_memory_config(args.config_path).get_memory_by_name("input_memory").size
        assert k_local_capacity <= input_memory_capacity, f"k_local_capacity {k_local_capacity} more than input_memory_capacity {input_memory_capacity} when CP size is {cp_size}. Please use greater CP sizes."

    for i, cp_size in enumerate(args.mapping_cp_sizes):
        n_head_this_round = args.world_size // cp_size
        print(f"CP size: {cp_size}, n_head: {n_head_this_round}, hidden_size_per_head: {hidden_size_per_head}")
        
        if args.split_stages:
            split_stage_configs = [
                SplitStageConfig(run_step=i, run_all_steps=False) for i in range(5)
            ]
        else:
            split_stage_configs = [SplitStageConfig(run_step=0, run_all_steps=True)]
        
        for stage_idx, split_stage_config in enumerate(split_stage_configs):
            op_config = AttnDecodeCPConfig(
                head_hidden=hidden_size_per_head,
                seqlen=args.seqlen,
                macro_config=cim_config,
                transpose_row=16,
                transpose_col=128,
                reduce_config=get_reduce_config(args.config_path),
                math=math,
                split_stage_config=split_stage_config
            )

            def config_cp_group(rank, op_config):
                op_config.cp_group_offset = (rank // cp_size) * cp_size
                op_config.cp_group_stride = 1
                op_config.cp_group_size = cp_size

            op_runner = SPMDOpRunner(
                op_path, 
                op_config, 
                args.config_path, 
                args.world_size,
                config_for_each_core=config_cp_group,
            )

            op_runner.run(simulate=False, save_dir=os.path.join(args.save_dir, f"round_{i}", f"attn_stage_{stage_idx}"))

    # TODO: add the rest of the logic here

if __name__ == "__main__":
    main()
