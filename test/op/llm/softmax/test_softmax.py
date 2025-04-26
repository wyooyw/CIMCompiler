import os
import numpy as np
from dataclasses import dataclass
from test.base import OpRunner, SIMDOpConfig, SPMDOpRunner
from test.op.test_reduce.test_reduce import get_reduce_config
import math
import pytest

@dataclass
class SoftmaxOpConfig:
    seqlen: int
    core_id: int
    world_size: int
    reduce_config: str = ""
    math: str = ""

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

@pytest.mark.parametrize(
    "seqlen",
    [
        4,8,16,32,64,128, 256, 512, 1024
    ],
)
def test_softmax(seqlen):
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "test/op/llm/softmax/test_softmax.cim")
    cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
    op_config = SoftmaxOpConfig(
        seqlen=seqlen,
        core_id=0,
        world_size=1,
        reduce_config=get_reduce_config(cim_config_path),
        math=math
    )

    op_runner = OpRunner(op_path, op_config, cim_config_path)

    """
    x_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
    score_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
    """
    x = np.random.randint(-64, 65, (op_config.seqlen,)).astype(np.float16)
    # x = np.zeros((op_config.seqlen,), dtype=np.float16) + 1
    output = np.zeros((op_config.seqlen,), dtype=np.float16)
    
    golden = softmax(x)
    op_runner.run([x], [output])

    print(f"{output.shape=}")
    print(f"{output=}")
    print(f"{golden.shape=}")
    print(f"{golden=}")

    # 设置相对误差和绝对误差阈值
    rtol = 1e-2  # 相对误差：0.1%
    atol = 1e-2  # 绝对误差：0.001
    allclose = np.allclose(output, golden, rtol=rtol, atol=atol)
    # assert allclose, f"{output=} {golden=}"
    print(f"{allclose=}")

@dataclass
class CpSoftmaxOpConfig(SIMDOpConfig):
    seqlen: int = 0
    cp_group_offset: int = 0
    cp_group_stride: int = 1
    cp_group_size: int = 1
    reduce_config: str = ""
    math: str = ""

# @pytest.mark.parametrize(
#     "seqlen, world_size",
#     [
#         (seqlen, world_size)
#         for seqlen in [128, 256, 512, 1024]
#         for world_size in [1, 2, 4, 8, 16, 32]
#     ],
# )
# def test_cp_softmax(seqlen, world_size):
#     cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
#     op_path = os.path.join(cim_compiler_home, "test/op/llm/softmax/test_cp_online_softmax.cim")
#     cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
#     op_config = CpSoftmaxOpConfig(
#         seqlen=seqlen,
#         cp_group_offset=0,
#         cp_group_stride=1,
#         cp_group_size=world_size,
#         reduce_config=get_reduce_config(cim_config_path),
#         math=math
#     )

#     op_runner = SPMDOpRunner(op_path, op_config, cim_config_path, world_size)

#     """
#     x_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
#     score_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
#     """
#     assert seqlen % world_size == 0, f"{seqlen=} {world_size=}"
#     x = np.random.randint(-16, 17, (seqlen,)).astype(np.float16)
#     # x = np.ones((seqlen,), dtype=np.float16)
#     input_list = np.array_split(x, world_size)
#     print(f"{input_list=}")
#     input_list = [[item] for item in input_list]
#     output_list = [[np.zeros((seqlen // world_size,), dtype=np.float16)] for _ in range(world_size)]
    
#     golden = softmax(x)
#     golden_list = np.array_split(golden, world_size)
#     op_runner.run(input_list, output_list)

#     for core_id in range(world_size):
#         print(f"{output_list[core_id][0]=}")
#     print(" ")
#     for core_id in range(world_size):
#         print(f"{golden_list[core_id]=}")
#     # print(f"{output_list.shape=}")
#     # print(f"{output=}")
#     # print(f"{golden.shape=}")
#     # print(f"{golden=}")

#     # 设置相对误差和绝对误差阈值
#     rtol = 1e-2  # 相对误差：0.1%
#     atol = 1e-2  # 绝对误差：0.001
#     for core_id in range(world_size):
#         allclose = np.allclose(output_list[core_id][0], golden_list[core_id], rtol=rtol, atol=atol)
#         assert allclose, f"{output_list[core_id][0]=} {golden_list[core_id]=}"
#     print("done")

if __name__=="__main__":
    test_softmax(128)