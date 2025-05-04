from cim_compiler.simulator.macro_utils import MacroConfig
from dataclasses import dataclass

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
