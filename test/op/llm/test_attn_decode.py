import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined
import numpy as np
import tempfile
import subprocess
from dataclasses import dataclass
from cim_compiler.simulator.macro_utils import MacroConfig
from cim_compiler.utils.df_layout import tensor_bits_to_int8
import pytest
from test.base import OpRunner

@dataclass
class OpConfig:
    head_hidden: int
    seqlen: int
    N_ROW: int
    N_COMP: int
    N_GROUP_VCOL: int
    N_MACRO_PER_GROUP: int
    N_GROUP: int
    transpose_row: int
    transpose_col: int

def make_cimset_mask(length: int):
    assert length % 8 == 0, f"{length} is not divisible by 8"
    mask = np.ones(length, dtype=np.int8)
    mask = mask.reshape(-1, 8)
    mask = tensor_bits_to_int8(mask)
    return mask

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

@pytest.mark.parametrize(
    "head_hidden, seqlen",
    [
        (128, 8192),
        (128, 4096),
        (128, 2048),
        (128, 1024),
        (128, 512),
        (128, 256),
        (128, 128),
        # hidden_size != 128 not support now.
        # (256, 4096),
        # (256, 2048),
        # (256, 1024),
    ],
)
def test_attn_decode(head_hidden, seqlen):
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "cim_compiler/op/llm/attn_decode.cim")
    cim_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    cim_config = MacroConfig.from_config(cim_config_path)
    op_config = OpConfig(
        head_hidden=head_hidden, 
        seqlen=seqlen, 
        N_ROW=cim_config.n_row, 
        N_COMP=cim_config.n_comp, 
        N_GROUP_VCOL=cim_config.n_group_vcol(16),
        N_MACRO_PER_GROUP=cim_config.n_macro_per_group,
        N_GROUP=cim_config.n_group,
        transpose_row=16,
        transpose_col=128
    )

    op_runner = OpRunner(op_path, op_config, cim_config_path)

    """
    q_global = Buffer(<128>, fp16, __GLOBAL__);
    v_global = Buffer(<4096, 128>, fp16, __GLOBAL__);
    k_T_global = Buffer(<128, 4096>, fp16, __GLOBAL__);
    output_global = Buffer(<128>, fp16, __GLOBAL__);
    """
    cimset_mask = make_cimset_mask(op_config.N_GROUP_VCOL)
    query = np.random.randint(-1, 2, (op_config.head_hidden,)).astype(np.float16)
    key = np.random.randint(-1, 2, ( op_config.seqlen, op_config.head_hidden)).astype(np.float16)
    value = np.random.randint(-1, 2, (op_config.seqlen, op_config.head_hidden)).astype(np.float16)
    # query = np.ones((op_config.head_hidden,), dtype=np.float16)
    # key = np.ones((op_config.seqlen, op_config.head_hidden), dtype=np.float16)
    # value = np.ones((op_config.seqlen, op_config.head_hidden), dtype=np.float16)
    
    golden = np.dot(softmax(np.dot(query, np.transpose(key))), value).reshape(-1)

    output = np.zeros(op_config.head_hidden, dtype=np.float16)
    op_runner.run([cimset_mask, query, key, value], [output])

    # print(f"{output=}")
    # print(f"{golden=}")
    # 设置相对误差和绝对误差阈值
    rtol = 1e-2  # 相对误差：0.1%
    atol = 1e-2  # 绝对误差：0.001
    allclose = np.allclose(output, golden, rtol=rtol, atol=atol)
    assert allclose, f"{output=} {golden=}"
    
if __name__=="__main__":
    test_attn_decode(128, 1024)