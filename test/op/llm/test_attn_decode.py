import os
import numpy as np
from dataclasses import dataclass
from cim_compiler.simulator.macro_utils import MacroConfig
from cim_compiler.utils.df_layout import tensor_bits_to_int8
import pytest
from test.base import OpRunner, SIMDOpConfig, SPMDOpRunner

@dataclass
class AttnDecodeConfig:
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
    op_config = AttnDecodeConfig(
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

@dataclass
class AttnDecodeCPConfig(SIMDOpConfig, AttnDecodeConfig):
    cp_group_offset: int = -1
    cp_group_stride: int = -1
    cp_group_size: int = -1


@pytest.mark.parametrize(
    "head_hidden, seqlen, world_size, cp_group_size",
    [
        (128, 4096, 8, 1),
        (128, 4096, 8, 2),
        (128, 4096, 8, 4),
        (128, 4096, 8, 8),
        (128, 4096, 16, 1),
        (128, 4096, 16, 2),
        (128, 4096, 16, 4),
        (128, 4096, 16, 8),
        (128, 4096, 16, 16),
    ],
)
def test_attn_decode_cp(head_hidden, seqlen, world_size, cp_group_size):
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "cim_compiler/op/llm/attn_decode_tp_cp.cim")
    cim_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    cim_config = MacroConfig.from_config(cim_config_path)
    op_config = AttnDecodeCPConfig(
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

    def config_cp_group(rank, op_config):
        op_config.cp_group_offset = (rank // cp_group_size) * cp_group_size
        op_config.cp_group_stride = 1
        op_config.cp_group_size = cp_group_size

    op_runner = SPMDOpRunner(
        op_path, 
        op_config, 
        cim_config_path, 
        world_size,
        config_for_each_core=config_cp_group
    )

    """
    q_global = Buffer(<{{head_hidden}}>, fp16, __GLOBAL__);
    k_global = Buffer(<{{seqlen // cp_group_size}}, {{head_hidden}}>, fp16, __GLOBAL__);
    v_global = Buffer(<{{seqlen // cp_group_size}}, {{head_hidden}}>, fp16, __GLOBAL__);
    output_global = Buffer(<{{head_hidden}}>, fp16, __GLOBAL__);
    """
    cimset_mask = make_cimset_mask(op_config.N_GROUP_VCOL)
    num_head = tp_size = world_size // cp_group_size
    query = np.random.randint(-1, 2, (num_head, op_config.head_hidden,)).astype(np.float16)
    key = np.random.randint(-1, 2, ( num_head, op_config.seqlen, op_config.head_hidden)).astype(np.float16)
    value = np.random.randint(-1, 2, (num_head, op_config.seqlen, op_config.head_hidden)).astype(np.float16)
    output = np.zeros((tp_size, cp_group_size, op_config.head_hidden), dtype=np.float16)
    # query = np.ones((num_head, op_config.head_hidden,)).astype(np.float16)
    # key = np.ones(( num_head, op_config.seqlen, op_config.head_hidden)).astype(np.float16)
    # value = np.ones((num_head, op_config.seqlen, op_config.head_hidden)).astype(np.float16)
    

    golden = np.zeros((num_head, op_config.head_hidden), dtype=np.float16)
    for h in range(num_head):
        golden_head = np.dot(softmax(np.dot(query[h], np.transpose(key[h]))), value[h])
        golden[h] = golden_head
    
    inputs = []
    outputs = []
    key_cp = key.reshape(num_head, cp_group_size, op_config.seqlen // cp_group_size, op_config.head_hidden)
    value_cp = value.reshape(num_head, cp_group_size, op_config.seqlen // cp_group_size, op_config.head_hidden)
    for tp_rank in range(tp_size):
        for cp_rank in range(cp_group_size):
            rank = tp_rank * cp_group_size + cp_rank
            inputs.append([
                cimset_mask, 
                query[tp_rank], 
                key_cp[tp_rank, cp_rank], 
                value_cp[tp_rank, cp_rank]
            ])
            outputs.append([output[tp_rank, cp_rank]])

    op_runner.run(inputs, outputs)

    # print(f"{output=}")
    # print(f"{golden=}")
    # 设置相对误差和绝对误差阈值
    rtol = 1e-2  # 相对误差：0.1%
    atol = 1e-2  # 绝对误差：0.001
    for tp_rank in range(tp_size):
        for cp_rank in range(cp_group_size):
            rank = tp_rank * cp_group_size + cp_rank
            allclose = np.allclose(outputs[rank][0], golden[tp_rank], rtol=rtol, atol=atol)
            # print("")
            # print(f"{tp_rank=}, {cp_rank=}, {allclose=}")
            # print(outputs[rank][0])
            
            assert allclose, f"{outputs[rank][0]=} {golden[tp_rank]=}"

if __name__=="__main__":
    test_attn_decode_cp(
        head_hidden=128, 
        seqlen=4096,
        world_size=8,
        cp_group_size=2,
    )