from cim_compiler.simulator.macro_utils import MacroConfig
from dataclasses import dataclass
import json
import tempfile
import os

@dataclass
class AttnDecodeConfig:
    head_hidden: int = None
    seqlen: int = None
    macro_config: MacroConfig = None
    transpose_row: int = None
    transpose_col: int = None
    reduce_config: str = None
    math: str = None

@dataclass
class SplitStageConfig:
    run_step: int = None
    run_all_steps: bool = None

@dataclass
class AttnDecodeCPConfig(AttnDecodeConfig):
    cp_group_offset: int = None
    cp_group_stride: int = None
    cp_group_size: int = None
    core_id: int = None
    world_size: int = None
    split_stage_config:SplitStageConfig = None
    global_memory_name: str = None

@dataclass
class LayerNormOpConfig:
    hidden: int
    reduce_config: int
    math: int

@dataclass
class GELUOpConfig:
    hidden: int = 0
    core_id: int = None
    world_size: int = None


def split_global_memory(num_split, src_config_path, dst_config_path, global_memory_name):
    with open(src_config_path, "r") as f:
        src_config = json.load(f)
    memory_list = src_config["memory_list"]
    global_index = -1
    for i, memory in enumerate(memory_list):
        if memory["name"] == global_memory_name:
            global_index = i
            break
    assert global_index != -1, f"{global_memory_name} not found in {src_config_path}"
    global_memory = memory_list[global_index]
    global_size = global_memory["addressing"]["size_byte"]
    global_offset = global_memory["addressing"]["offset_byte"]
    global_memory_list = []
    
    assert global_size % num_split == 0, f"{global_size} must be divisible by {num_split}"
    size_per_split = global_size // num_split
    for i in range(num_split):
        global_memory_list.append({
            "name": f"{global_memory_name}_{i}",
            "type": global_memory["type"],
            "addressing": {
                "size_byte": size_per_split,
                "offset_byte": global_offset + i * size_per_split
            }
        })
    new_memory_list = memory_list[:global_index] + global_memory_list + memory_list[global_index+1:]
    src_config["memory_list"] = new_memory_list
    with open(dst_config_path, "w") as f:
        json.dump(src_config, f, indent=4)
    return src_config

class ModifyConfigSplitGlobalMemory:
    def __init__(self, base_config_path, global_memory_name, num_split):
        self.base_config_path = base_config_path
        self.global_memory_name = global_memory_name
        self.num_split = num_split
    
    def __enter__(self):
        self.modified_config_path = tempfile.mktemp()
        self.modified_config = split_global_memory(self.num_split, self.base_config_path, self.modified_config_path, self.global_memory_name)
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        os.remove(self.modified_config_path)
        return False  # 返回 False 表示异常未处理，会继续传播