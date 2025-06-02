import os
import numpy as np
from dataclasses import dataclass
from cim_compiler.simulator.macro_utils import MacroConfig
from cim_compiler.simulator.simd_utils import SIMDConfig
from cim_compiler.simulator.reduce_utils import ReduceConfig

from cim_compiler.utils.df_layout import tensor_bits_to_int8
import pytest
from test.base import OpRunner, SIMDOpConfig, SPMDOpRunner
import math
from test.op.test_reduce.test_reduce import get_reduce_config, get_reduce_max_config
from cim_compiler.op.llm.helper import AttnDecodeConfig, AttnDecodeCPConfig, SplitStageConfig

def make_cimset_mask(length: int):
    assert length % 8 == 0, f"{length} is not divisible by 8"
    mask = np.ones(length, dtype=np.int8)
    mask = mask.reshape(-1, 8)
    mask = tensor_bits_to_int8(mask)
    return mask

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# @pytest.mark.parametrize(
#     "head_hidden, seqlen",
#     [
#         (128, 8192),
#         (128, 4096),
#         (128, 2048),
#         (128, 1024),
#         (128, 512),
#         (128, 256),
#         (128, 128),
#         # hidden_size != 128 not support now.
#         # (256, 4096),
#         # (256, 2048),
#         # (256, 1024),
#     ],
# )
# def test_attn_decode(head_hidden, seqlen):
#     cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
#     op_path = os.path.join(cim_compiler_home, "cim_compiler/op/llm/attn_decode.cim")
#     cim_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
#     cim_config = MacroConfig.from_config(cim_config_path)
#     cim_config.set_default_bit_width(16)
#     op_config = AttnDecodeConfig(
#         head_hidden=head_hidden, 
#         seqlen=seqlen, 
#         macro_config=cim_config,
#         transpose_row=16,
#         transpose_col=128,
#         reduce_config=get_reduce_config(cim_config_path),
#         math=math
#     )

#     op_runner = OpRunner(op_path, op_config, cim_config_path)

#     """
#     q_global = Buffer(<128>, fp16, __GLOBAL__);
#     v_global = Buffer(<4096, 128>, fp16, __GLOBAL__);
#     k_T_global = Buffer(<128, 4096>, fp16, __GLOBAL__);
#     output_global = Buffer(<128>, fp16, __GLOBAL__);
#     """
#     # cimset_mask = make_cimset_mask(op_config.macro_config.n_group_vcol)
#     query = np.random.randint(-1, 2, (op_config.head_hidden,)).astype(np.float16)
#     key = np.random.randint(-1, 2, ( op_config.seqlen, op_config.head_hidden)).astype(np.float16)
#     value = np.random.randint(-1, 2, (op_config.seqlen, op_config.head_hidden)).astype(np.float16)
#     # query = np.ones((op_config.head_hidden,), dtype=np.float16)
#     # key = np.ones((op_config.seqlen, op_config.head_hidden), dtype=np.float16)
#     # value = np.ones((op_config.seqlen, op_config.head_hidden), dtype=np.float16)
    
#     golden = np.dot(softmax(np.dot(query, np.transpose(key))), value).reshape(-1)

#     output = np.zeros(op_config.head_hidden, dtype=np.float16)
#     op_runner.run([query, key, value], [output])

#     # print(f"{output=}")
#     # print(f"{golden=}")
#     # 设置相对误差和绝对误差阈值
#     rtol = 1e-2  # 相对误差：0.1%
#     atol = 1e-2  # 绝对误差：0.001
#     allclose = np.allclose(output, golden, rtol=rtol, atol=atol)
#     assert allclose, f"{output=} {golden=}"

@pytest.mark.parametrize(
    "head_hidden, seqlen, world_size, cp_group_size",
    [
        (128, 1024, 8, 1),
        (128, 2048, 8, 2),
        (128, 4096, 8, 4),
        (128, 4096, 8, 8),
        (128, 1024, 16, 1),
        (128, 2048, 16, 2),
        (128, 4096, 16, 4),
        (128, 4096, 16, 8),
        (128, 4096, 16, 16),
    ],
)
def test_attn_decode_cp(head_hidden, seqlen, world_size, cp_group_size, load_k_stages):
    check_result = os.environ.get("CHECK_RESULT", "1") == "1"
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "cim_compiler/op/llm/attn_decode_tp_cp_flex.cim")
    cim_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    cim_config = MacroConfig.from_config(cim_config_path)
    cim_config.set_default_bit_width(16)
    assert head_hidden <= 128 and (head_hidden % 16) == 0, f"{head_hidden=} must be divisible by 16"
    pad_head_hidden = 128
    pad_seqlen = seqlen
    op_config = AttnDecodeCPConfig(
        head_hidden=pad_head_hidden, 
        head_hidden_real=head_hidden,
        seqlen=seqlen // cp_group_size, 
        seqlen_real=seqlen // cp_group_size,
        macro_config=cim_config,
        transpose_row=16,
        transpose_col=128,
        reduce_config=get_reduce_config(cim_config_path),
        reduce_max_config=get_reduce_max_config(cim_config_path),
        math=math,
        split_stage_config=SplitStageConfig(run_step=0, run_all_steps=True),
        global_memory_name=f"__GLOBAL__",
        simd=SIMDConfig.from_config(cim_config_path),
        reduce=ReduceConfig.from_config(cim_config_path),
        load_k_stages=load_k_stages,
        n_activate_core=world_size
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
    # cimset_mask = make_cimset_mask(op_config.macro_config.n_group_vcol)
    num_head = tp_size = world_size // cp_group_size
    query = np.random.randint(-1, 2, (num_head, head_hidden,)).astype(np.float16)
    key = np.random.randint(-1, 2, ( num_head, seqlen, head_hidden)).astype(np.float16)
    value = np.random.randint(-1, 2, (num_head, seqlen, head_hidden)).astype(np.float16)
    output = np.zeros((tp_size, cp_group_size, head_hidden), dtype=np.float16)
    attn_out1_global = np.zeros((tp_size, cp_group_size, seqlen // cp_group_size), dtype=np.float16)
    score_global = np.zeros((tp_size, cp_group_size, seqlen // cp_group_size), dtype=np.float16)
    attn_out2_global = np.zeros((tp_size, cp_group_size, head_hidden), dtype=np.float16)
    # query = np.ones((num_head, op_config.head_hidden,)).astype(np.float16)
    # key = np.ones(( num_head, op_config.seqlen, op_config.head_hidden)).astype(np.float16)
    # value = np.ones((num_head, op_config.seqlen, op_config.head_hidden)).astype(np.float16)
    # Pad to op_config.head_hidden by adding zeros
    pad_query = np.zeros((num_head, pad_head_hidden), dtype=np.float16)
    pad_key = np.zeros((num_head, pad_seqlen, pad_head_hidden), dtype=np.float16)
    pad_value = np.zeros((num_head, pad_seqlen, pad_head_hidden), dtype=np.float16)

    # Copy original data to the beginning of the padded tensors
    pad_query[:, :head_hidden] = query
    pad_key[:, :, :head_hidden] = key
    pad_value[:, :, :head_hidden] = value

    golden = np.zeros((num_head, head_hidden), dtype=np.float16)
    for h in range(num_head):
        golden_head = np.dot(softmax(np.dot(query[h], np.transpose(key[h]))), value[h])
        golden[h] = golden_head
    
    golden_attn_out1= np.zeros((num_head, seqlen // cp_group_size), dtype=np.float16)
    for h in range(num_head):
        golden_attn_out1_head = np.dot(query[h], np.transpose(key[h]))
        golden_attn_out1[h] = golden_attn_out1_head

    golden_score= np.zeros((num_head, seqlen // cp_group_size), dtype=np.float16)
    for h in range(num_head):
        golden_score_head = softmax(np.dot(query[h], np.transpose(key[h])))
        golden_score[h] = golden_score_head
    
    golden_attn_out2 = np.zeros((num_head, head_hidden), dtype=np.float16)
    for h in range(num_head):
        golden_attn_out2_head = np.dot(softmax(np.dot(query[h], np.transpose(key[h]))), value[h])
        golden_attn_out2[h] = golden_attn_out2_head
    
    inputs = []
    outputs = []
    key_cp = pad_key.reshape(num_head, cp_group_size, op_config.seqlen, op_config.head_hidden)
    value_cp = pad_value.reshape(num_head, cp_group_size, op_config.seqlen, op_config.head_hidden)
    for tp_rank in range(tp_size):
        for cp_rank in range(cp_group_size):
            rank = tp_rank * cp_group_size + cp_rank
            inputs.append([
                pad_query[tp_rank], 
                key_cp[tp_rank, cp_rank], 
                value_cp[tp_rank, cp_rank]
            ])
            outputs.append([
                output[tp_rank, cp_rank],
                attn_out1_global[tp_rank, cp_rank],
                score_global[tp_rank, cp_rank],
                attn_out2_global[tp_rank, cp_rank]
            ])

    op_runner.run(inputs, outputs, simulate=check_result)

    print("===== Check attn_out1 =====")
    if check_result:
        rtol = 1e-2  # 相对误差：0.1%
        atol = 1e-2  # 绝对误差：0.001
        for tp_rank in range(tp_size):
            for cp_rank in range(cp_group_size):
                rank = tp_rank * cp_group_size + cp_rank
                single_golden_attn_out1 = golden_attn_out1[tp_rank]
                single_output_attn_out1 = outputs[rank][1]
                print(f"{tp_rank=}, {cp_rank=}, golden.shape={single_golden_attn_out1.shape}, golden=\n{single_golden_attn_out1}\n")
                print(f"{tp_rank=}, {cp_rank=}, output.shape={single_output_attn_out1.shape}, output=\n{single_output_attn_out1}\n")
                allclose = np.allclose(single_output_attn_out1, single_golden_attn_out1, rtol=rtol, atol=atol)
                print(f"{allclose=}")

    print("===== Check score =====")
    if check_result:
        rtol = 1e-2  # 相对误差：0.1%
        atol = 1e-2  # 绝对误差：0.001
        for tp_rank in range(tp_size):
            for cp_rank in range(cp_group_size):
                rank = tp_rank * cp_group_size + cp_rank
                single_golden_score = golden_score[tp_rank]
                single_output_score = outputs[rank][2]
                print(f"{tp_rank=}, {cp_rank=}, golden.shape={single_golden_score.shape}, golden=\n{single_golden_score}\n")
                print(f"{tp_rank=}, {cp_rank=}, output.shape={single_output_score.shape}, output=\n{single_output_score}\n")
                allclose = np.allclose(single_output_score, single_golden_score, rtol=rtol, atol=atol)
                print(f"{allclose=}")

    print("===== Check attn_out2 =====")
    if check_result:
        rtol = 1e-2  # 相对误差：0.1%
        atol = 1e-2  # 绝对误差：0.001
        for tp_rank in range(tp_size):
            for cp_rank in range(cp_group_size):
                rank = tp_rank * cp_group_size + cp_rank
                single_golden_attn_out2 = golden_attn_out2[tp_rank]
                single_output_attn_out2 = outputs[rank][3]
                print(f"{tp_rank=}, {cp_rank=}, golden.shape={single_golden_attn_out2.shape}, golden=\n{single_golden_attn_out2}\n")
                print(f"{tp_rank=}, {cp_rank=}, output.shape={single_output_attn_out2.shape}, output=\n{single_output_attn_out2}\n")
                allclose = np.allclose(single_output_score, single_golden_score, rtol=rtol, atol=atol)
                print(f"{allclose=}")
    # exit()
    # print(f"{output=}")
    # print(f"{golden=}")
    # 设置相对误差和绝对误差阈值
    if check_result:
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
    # seqlen = 2048
    # cp_group_size = 32
    # for cp_group_size in [2]:
    test_attn_decode_cp(
        head_hidden=80, 
        seqlen=80,
        world_size=1,
        cp_group_size=1,
        load_k_stages=1
    )
    # test_attn_decode_cp(
    #     head_hidden=80, 
    #     seqlen=64,
    #     world_size=1,
    #     cp_group_size=2,
    #     load_k_stages=1
    # )
    # test_attn_decode(128, 128)