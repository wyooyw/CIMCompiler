import os
import numpy as np
from dataclasses import dataclass
from test.base import OpRunner, SIMDOpConfig, SPMDOpRunner
import math
import pytest
from cim_compiler.simulator.simd_utils import SIMDConfig
from cim_compiler.simulator.reduce_utils import ReduceConfig
from test.op.test_reduce.test_reduce import get_reduce_config, get_reduce_max_config


def softmax_each_row(x, axis=-1):
    ori_shape = x.shape
    assert len(x.shape)==2
    all_result = []
    for i in range(x.shape[0]):
        x_row = x[i]
        exp_x = np.exp(x_row - np.max(x_row, axis=axis, keepdims=True))
        result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        all_result.append(result)
    all_result = np.vstack(all_result)
    assert all_result.shape==ori_shape, f"{all_result.shape=} {ori_shape=}"
    return all_result

@dataclass
class SoftmaxPrefillOpConfig(SIMDOpConfig):
    seqlen: int = 0
    local_seqlen: int = 0
    reduce_config: str = ""
    simd: SIMDConfig = None
    reduce: ReduceConfig = None
    math: str = ""
    debug_mode: bool = True
    reduce_max_config: str = None


# @pytest.mark.parametrize(
#     "seqlen, world_size",
#     [
#         (seqlen, world_size)
#         for seqlen in [128, 256, 512, 1024]
#         for world_size in [1, 2, 4, 8, 16, 32]
#     ],
# )
def test_softmax_prefill(seqlen, world_size, cp_world_size):
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "test/op/llm/softmax/test_softmax_prefill.cim")
    cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
    op_config = SoftmaxPrefillOpConfig(
        seqlen=seqlen,
        local_seqlen=seqlen // cp_world_size,
        reduce_config=get_reduce_config(cim_config_path),
        simd=SIMDConfig.from_config(cim_config_path),
        reduce=ReduceConfig.from_config(cim_config_path),
        reduce_max_config=get_reduce_max_config(cim_config_path),
        math=math,
        debug_mode=True,
    )

    op_runner = SPMDOpRunner(op_path, op_config, cim_config_path, world_size)

    """
    x_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
    score_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
    """
    assert seqlen % cp_world_size == 0, f"{seqlen=} {world_size=}"
    n_head = tp_size = world_size // cp_world_size
    x = np.random.randint(-2, 2, (n_head, seqlen, seqlen,)).astype(np.float16)
    local_seqlen = seqlen // cp_world_size

    input_list = []
    output_list = []
    golden_list = []
    for t in range(tp_size):
        for c in range(cp_world_size):
            input_data = x[t, c * local_seqlen:(c + 1) * local_seqlen, :]
            input_list.append([input_data])
            output_list.append([np.zeros((local_seqlen, seqlen,), dtype=np.float16)])
            golden_list.append(softmax_each_row(input_data))
    
    op_runner.run(input_list, output_list)

    # for core_id in range(world_size):
    #     print(f"{output_list[core_id][0]=}")
    # print(" ")
    # for core_id in range(world_size):
    #     print(f"{golden_list[core_id]=}")
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
    test_softmax_prefill(512, 1, 1)