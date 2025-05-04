import os
import numpy as np
from dataclasses import dataclass
from test.base import OpRunner, SIMDOpConfig, SPMDOpRunner
from test.op.test_reduce.test_reduce import get_reduce_config
import math
import pytest
import torch
import torch.nn.functional as F
from cim_compiler.op.llm.helper import GELUOpConfig
from cim_compiler.simulator.simd_utils import SIMDConfig
def gelu(x):
    torch_tensor = torch.tensor(x.astype(np.float32))
    gelu_output = F.gelu(torch_tensor)
    output_data = gelu_output.numpy().astype(x.dtype)
    return output_data

def config_global_memory_name(rank, op_config):
    op_config.global_memory_name = f"__GLOBAL_{rank}__"

@pytest.mark.parametrize(
    "hidden, world_size",
    [
        (hidden, world_size)
        for hidden in [128, 512, 1024]
        for world_size in [1, 4, 32]
    ],
)
def test_gelu(hidden, world_size):
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "test/op/llm/gelu/test_gelu.cim")
    cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
    op_config = GELUOpConfig(
        hidden=hidden // world_size,
        simd=SIMDConfig.from_config(cim_config_path),
        global_memory_name=f"__GLOBAL__"
    )

    op_runner = SPMDOpRunner(op_path, op_config, cim_config_path, world_size)

    """
    x_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
    score_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
    """
    assert hidden % world_size == 0, f"{hidden=} {world_size=}"
    x = np.random.randint(-16, 17, (hidden,)).astype(np.float16)
    # x = np.ones((seqlen,), dtype=np.float16)
    input_list = np.array_split(x, world_size)
    print(f"{input_list=}")
    input_list = [[item] for item in input_list]
    output_list = [[np.zeros((hidden // world_size,), dtype=np.float16)] for _ in range(world_size)]
    
    golden = gelu(x)
    golden_list = np.array_split(golden, world_size)
    op_runner.run(input_list, output_list)

    for core_id in range(world_size):
        print(f"{output_list[core_id][0]=}")
    print(" ")
    for core_id in range(world_size):
        print(f"{golden_list[core_id]=}")
    # print(f"{output_list.shape=}")
    # print(f"{output=}")
    # print(f"{golden.shape=}")
    # print(f"{golden=}")

    # 设置相对误差和绝对误差阈值
    rtol = 1e-2  # 相对误差：0.1%
    atol = 1e-2  # 绝对误差：0.001
    for core_id in range(world_size):
        allclose = np.allclose(output_list[core_id][0], golden_list[core_id], rtol=rtol, atol=atol)
        assert allclose, f"{output_list[core_id][0]=} {golden_list[core_id]=}"
    print("done")

if __name__=="__main__":
    test_gelu(128, 1)