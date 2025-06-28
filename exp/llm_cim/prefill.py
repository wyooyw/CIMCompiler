import argparse
import os
import time
import logging
from pathlib import Path
from cim_compiler.op.llm.helper import AttnDecodeCPConfig, SplitStageConfig, GELUOpConfig, LayerNormOpConfig, ResAddOpConfig, ModifyConfigSplitGlobalMemory
from cim_compiler.simulator.macro_utils import MacroConfig
from test.op.test_reduce.test_reduce import get_reduce_config, get_reduce_max_config
from test.base import SPMDOpRunner,OpRunner
import math
from datetime import datetime
from cim_compiler.simulator.simulator import MemorySpace
from cim_compiler.simulator.simd_utils import SIMDConfig
from cim_compiler.simulator.reduce_utils import ReduceConfig
import shutil
import tarfile
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description="LLM CIM Prefill Configuration")
    
    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--hidden-size", type=int, required=True, help="Hidden dimension size")
    model_group.add_argument("--seqlen", type=int, required=True, help="Sequence length")
    
    # Configuration parameters
    config_group = parser.add_argument_group("Configuration Parameters")
    config_group.add_argument("--world-size", type=int, required=True, help="Number of cores")
    config_group.add_argument("--config-path", type=str, required=True, help="Path to configuration file")
    config_group.add_argument("--distributed-dram-type", type=str, default="default", 
                            choices=["split", "multiply", "default"], 
                            help="Memory distribution type")
    
    # Experiment parameters
    exp_group = parser.add_argument_group("Experiment Parameters")
    exp_group.add_argument("--save-dir", type=str, 
                         default=None, 
                         help="Save directory")
    exp_group.add_argument("--debug", action="store_true", 
                         help="Enable debug logs, save files and logs to .debug/time directory")
    exp_group.add_argument("--name-prefix", type=str, default="", help="Name prefix for the experiment")
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

def config_seqlen(rank, op_config, cp_size, total_seqlen):
    cp_local_rank = rank % cp_size 
    core_seqlen = total_seqlen // cp_size
    if cp_local_rank < (total_seqlen % cp_size):
        core_seqlen += 1
    op_config.seqlen = core_seqlen
    op_config.local_seqlen = core_seqlen

def config_global_memory_name(rank, op_config):
    op_config.global_memory_name = f"__GLOBAL_{rank}__"

def config_cp_group(rank, op_config, cp_size, total_seqlen):
    op_config.cp_group_offset = (rank // cp_size) * cp_size
    op_config.cp_group_stride = 1
    op_config.cp_group_size = cp_size
    config_global_memory_name(rank, op_config)
    config_seqlen(rank, op_config, cp_size, total_seqlen)
    
def main():
    args = parse_args()
    debug_dir = setup_logging(args.debug)

    datetime_str = datetime.now().strftime('%Y%m%d_%H%M')
    args.datetime_str = datetime_str

    if args.save_dir is None:
        args.save_dir = f"result/{args.name_prefix}_{datetime_str}"
    print(f"Save directory: {args.save_dir}")
    
    # Your main code logic here
    if args.debug:
        logging.debug(f"Debug mode enabled. Files will be saved to: {debug_dir}")
    
    logging.info(f"Model parameters: hidden_size={args.hidden_size}, seqlen={args.seqlen}")
    logging.info(f"World size: {args.world_size}, Distributed DRAM type: {args.distributed_dram_type}")

    with ModifyConfigSplitGlobalMemory(args.config_path, "global", args.world_size) as m:
        args.split_config_path = m.modified_config_path
        _main_impl(args)

    print(f"Save directory: {args.save_dir}")

def _main_impl(args):
    # Add the rest of your application logic here
    cim_config = MacroConfig.from_config(args.config_path)
    cim_config.set_default_bit_width(16)

    simd_config = SIMDConfig.from_config(args.config_path)
    reduce_config = ReduceConfig.from_config(args.config_path)

    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    # if args.op == "layernorm":
    #     op_path = os.path.join(cim_compiler_home, "test/op/llm/layernorm/test_batch_layernorm_single_token.cim")
    # elif args.op == "resadd":
    #     op_path = os.path.join(cim_compiler_home, "test/op/llm/resadd/test_batch_resadd.cim")
    # elif args.op == "gelu":
    #     op_path = os.path.join(cim_compiler_home, "test/op/llm/gelu/test_batch_gelu.cim")
    # else:
    #     raise ValueError(f"Invalid operator: {args.op}")
    
    # check capacity of input memory
    # for i, cp_size in enumerate(args.mapping_cp_sizes):
    #     k_local_capacity = hidden_size_per_head * (args.seqlen // cp_size) * 2
    #     input_memory_capacity = MemorySpace.from_memory_config(args.config_path).get_memory_by_name("input_memory").size
    #     assert k_local_capacity <= input_memory_capacity, f"k_local_capacity {k_local_capacity} more than input_memory_capacity {input_memory_capacity} when CP size is {cp_size}. Please use greater CP sizes."

    collect_dir = os.path.join(args.save_dir, "code")
    os.makedirs(collect_dir, exist_ok=True)

    # resadd
    # resadd_config = ResAddOpConfig(
    #     hidden=args.hidden_size,
    #     simd=simd_config,
    #     world_size=args.world_size,
    # )
    # resadd_path = os.path.join(cim_compiler_home, "test/op/llm/resadd/test_resadd_prefill.cim")
    # resadd_runner = SPMDOpRunner(
    #     resadd_path, 
    #     resadd_config, 
    #     args.split_config_path,
    #     args.world_size,
    #     config_for_each_core=partial(config_cp_group, cp_size=args.world_size, total_seqlen=args.seqlen),
    # )
    # resadd_runner.run(simulate=False, save_dir=os.path.join(args.save_dir, "resadd"), gather_multicore_code=True)
    # shutil.copy(os.path.join(args.save_dir, "resadd", "multi_core_code.json"), os.path.join(collect_dir, "resadd.json"))

    
    # layernorm
    ln_config = LayerNormOpConfig(
        hidden=args.hidden_size,
        reduce_config=get_reduce_config(args.config_path),
        math=math,
        simd=simd_config,
        reduce=reduce_config,
        world_size=args.world_size,
    )
    ln_path = os.path.join(cim_compiler_home, "test/op/llm/layernorm/test_layernorm_prefill.cim")
    ln_runner = SPMDOpRunner(
        ln_path,
        ln_config, 
        args.split_config_path,
        args.world_size,
        config_for_each_core=partial(config_cp_group, cp_size=args.world_size, total_seqlen=args.seqlen),
    )
    ln_runner.run(simulate=False, save_dir=os.path.join(args.save_dir, "layernorm"), gather_multicore_code=True)
    shutil.copy(os.path.join(args.save_dir, "layernorm", "multi_core_code.json"), os.path.join(collect_dir, "layernorm.json"))


    # # # gelu
    # gelu_config = GELUOpConfig(
    #     hidden=args.hidden_size // args.world_size,
    #     simd=simd_config,
    #     world_size=args.world_size,
    # )   
    # gelu_path = os.path.join(cim_compiler_home, "test/op/llm/gelu/test_gelu_prefill.cim")
    # gelu_runner = SPMDOpRunner(
    #     gelu_path,
    #     gelu_config, 
    #     args.split_config_path,
    #     args.world_size,
    #     config_for_each_core=partial(config_cp_group, cp_size=args.world_size, total_seqlen=args.seqlen),
    # )
    # gelu_runner.run(simulate=False, save_dir=os.path.join(args.save_dir, "gelu"), gather_multicore_code=True)
    # shutil.copy(os.path.join(args.save_dir, "gelu", "multi_core_code.json"), os.path.join(collect_dir, "gelu.json"))

    
    shutil.copy(args.config_path, os.path.join(collect_dir, "config.json"))
    shutil.copy(args.split_config_path, os.path.join(collect_dir, "split_config.json"))
    create_tar_gz(collect_dir, os.path.join(args.save_dir, f"{args.name_prefix}_{args.datetime_str}_code.tar.gz"))
    
    
    
def delete_unwanted_files(save_dir, keep_files):
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if file not in keep_files:
                os.remove(os.path.join(root, file))

def create_tar_gz(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

if __name__ == "__main__":
    main()
