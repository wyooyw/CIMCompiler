import os
import numpy as np
from dataclasses import dataclass
from test.base import OpRunner
from test.op.test_reduce.test_reduce import get_reduce_config
import math
import pytest
from cim_compiler.op.llm.helper import LayerNormOpConfig, ResAddOpConfig
from cim_compiler.simulator.simd_utils import SIMDConfig

def layernorm(x, eps, a, b):
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    y = a * x_normalized + b
    print(f"{x=}, {x_mean=}, {x_var=}, {x_normalized=}, {y=}")
    print(f"{(x - x_mean)=}, {np.sqrt(x_var + eps)=}")
    return y

@pytest.mark.parametrize(
    "hidden",
    [
        4, 8, 16, 32, 64, 128, 256, 512, 1024
    ],
)
def test_resadd(hidden):
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "test/op/llm/resadd/test_resadd.cim")
    cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
    op_config = ResAddOpConfig(
        hidden=hidden,
        simd=SIMDConfig.from_config(cim_config_path)
    )

    op_runner = OpRunner(op_path, op_config, cim_config_path)

    """
    x_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
    score_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
    """
    x1 = np.random.randint(-1, 2, (op_config.hidden,)).astype(np.float16)
    x2 = np.random.randint(-1, 2, (op_config.hidden,)).astype(np.float16)
    # x = np.zeros((op_config.seqlen,), dtype=np.float16) + 1
    output = np.zeros((op_config.hidden,), dtype=np.float16)
    golden = x1 + x2
    op_runner.run([x1, x2], [output])

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

if __name__=="__main__":
    test_resadd(32)