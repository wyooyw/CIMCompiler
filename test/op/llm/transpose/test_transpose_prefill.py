import os
import numpy as np
import pytest
from cim_compiler.op.llm.helper import TransposePrefillOpConfig
from cim_compiler.simulator.simd_utils import SIMDConfig
from test.base import SPMDOpRunner
from cim_compiler.simulator.macro_utils import MacroConfig


# @pytest.mark.parametrize(
#     "hidden",
#     [
#         4, 8, 16, 32, 64, 128, 256, 512, 1024
#     ],
# )
def test_transpose_prefill(head_hidden, seqlen, world_size, cp_group_size):
    check_result = os.environ.get("CHECK_RESULT", "1") == "1"
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "test/op/llm/transpose/test_transpose_prefill.cim")
    cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
    cim_config = MacroConfig.from_config(cim_config_path)
    cim_config.set_default_bit_width(16)
    pad_head_hidden = 128
    real_head_hidden = head_hidden
    op_config = TransposePrefillOpConfig(
        head_hidden=pad_head_hidden,
        head_hidden_real=head_hidden,
        simd=SIMDConfig.from_config(cim_config_path),
        global_memory_name="__GLOBAL__",
        debug_mode=True,
        # core_id: int = None
        # world_size: int = None
        macro_config=cim_config,
        transpose_row=16,
        transpose_col=128,
    )

    def config_cp_group(rank, op_config):
        local_seqlen = seqlen // cp_group_size
        cp_local_rank = rank % cp_group_size
        if cp_local_rank < (seqlen % cp_group_size):
            local_seqlen += 1
        op_config.local_seqlen = local_seqlen

    op_runner = SPMDOpRunner(
        op_path, 
        op_config, 
        cim_config_path, 
        world_size,
        config_for_each_core=config_cp_group
    )

    """
    k_global = Buffer(<{{local_seqlen}}, {{head_hidden}}>, fp16, __GLOBAL__);
    """
    num_head = tp_size = world_size // cp_group_size
    key = np.random.randint(-10, 10, (num_head, seqlen, real_head_hidden)).astype(np.float16)
    for i in range(num_head):
        for j in range(seqlen):
            key[i, j] = np.arange(real_head_hidden) +1
    pad_key = np.zeros((num_head, seqlen, pad_head_hidden), dtype=np.float16)
    pad_key[:, :, :real_head_hidden] = key
    output = np.zeros((num_head, pad_head_hidden, seqlen), dtype=np.float16)
    # import pdb; pdb.set_trace()
    
    inputs = []
    outputs = []
    goldens = []
    # import pdb; pdb.set_trace()
    key_cp = pad_key.reshape(num_head, cp_group_size, seqlen // cp_group_size, op_config.head_hidden)
    output_cp = output.reshape(num_head, op_config.head_hidden, cp_group_size, seqlen // cp_group_size)
    # value_cp = pad_value.reshape(num_head, cp_group_size, op_config.seqlen, op_config.head_hidden)
    # import pdb; pdb.set_trace()
    for tp_rank in range(tp_size):
        for cp_rank in range(cp_group_size):
            rank = tp_rank * cp_group_size + cp_rank
            inputs.append([
                key_cp[tp_rank, cp_rank, :, :], 
            ])
            outputs.append([
                output_cp[tp_rank, :, cp_rank, :],
            ])
            goldens.append(key_cp[tp_rank, cp_rank, :, :].transpose(1,0))

    op_runner.run(inputs, outputs, simulate=check_result)

    print("===== Check final output =====")
    if check_result:
        rtol = 1e-2  # 相对误差：0.1%
        atol = 1e-2  # 绝对误差：0.001
        for tp_rank in range(tp_size):
            for cp_rank in range(cp_group_size):
                rank = tp_rank * cp_group_size + cp_rank
                allclose = np.allclose(outputs[rank][0], goldens[rank], rtol=rtol, atol=atol)
                print(f"{tp_rank=}, {cp_rank=}, {allclose=}")
                print(f"{outputs[rank][0].shape=} {goldens[rank].shape=}")
                assert allclose, f"{outputs[rank][0]=} {goldens[rank]=}"

if __name__=="__main__":
    # test_transpose_prefill(head_hidden=128, seqlen=32, world_size=1, cp_group_size=1)
    # test_transpose_prefill(head_hidden=128, seqlen=1024, world_size=1, cp_group_size=1)
    # test_transpose_prefill(head_hidden=64, seqlen=512, world_size=1, cp_group_size=1)
    test_transpose_prefill(head_hidden=128, seqlen=512, world_size=4, cp_group_size=2)