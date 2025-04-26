import os
import numpy as np
from dataclasses import dataclass
from test.base import OpRunner
from test.op.test_reduce.test_reduce import get_reduce_config
import math
import pytest


@dataclass
class LayerNormOpConfig:
    hidden: int
    reduce_config: int
    math: int

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
def test_layernorm_single_token(hidden):
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "test/op/llm/layernorm/test_layernorm_single_token.cim")
    cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
    op_config = LayerNormOpConfig(
        hidden=hidden,
        reduce_config=get_reduce_config(cim_config_path),
        math=math
    )

    op_runner = OpRunner(op_path, op_config, cim_config_path)

    """
    x_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
    score_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
    """
    x = np.random.randint(-1, 2, (op_config.hidden,)).astype(np.float16)
    eps = np.array([1e-5], dtype=np.float16)
    a = np.array([1], dtype=np.float16)
    b = np.array([0], dtype=np.float16)
    d = np.array([op_config.hidden], dtype=np.float16)
    # x = np.zeros((op_config.seqlen,), dtype=np.float16) + 1
    output = np.zeros((op_config.hidden,), dtype=np.float16)
    golden = layernorm(x, eps, a, b)
    op_runner.run([x, d, eps, a, b], [output])

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
    test_layernorm_single_token(32)