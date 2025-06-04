import os
import numpy as np
from dataclasses import dataclass
from cim_compiler.simulator.macro_utils import MacroConfig
from cim_compiler.simulator.simd_utils import SIMDConfig
import math
from test.base import OpRunner


@dataclass
class Conv2dConfig:
    batch: int
    in_channel: int
    out_channel: int
    in_hw: int
    out_hw: int
    ker_hw: int
    macro_config: MacroConfig
    math: int
    n_weight_duplicate_group: int
    n_reduce_group: int

def test_conv():
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "cim_compiler/op/cimflow/conv2d.cim")
    cim_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    cim_config = MacroConfig.from_config(cim_config_path)
    cim_config.set_default_bit_width(8)
    op_config = Conv2dConfig(
        batch=2,
        in_channel=16,
        out_channel=16,
        in_hw=34,
        out_hw=32,
        ker_hw=3,
        macro_config=cim_config,
        math=math,
        n_weight_duplicate_group=2,
        n_reduce_group=2
    )

    op_runner = OpRunner(op_path, op_config, cim_config_path)
    op_runner.run([], [])

if __name__ == "__main__":
    test_conv()