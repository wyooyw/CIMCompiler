import os
import math
from cim_compiler.cli.commands.cimflow_network import Conv2dConfig
from cim_compiler.cli.commands.cimflow_network import get_config, set_raw_config_by_path
from cim_compiler.cli.commands.cimflow_network import OpRunner
import tempfile
import numpy as np
from cim_compiler.poly.op.calculate import conv2d
import pytest

def pad_to_multiple(x, dim, multiple):
    pad_to = math.ceil(x.shape[dim] / multiple) * multiple
    pad_width = [(0, 0)] * len(x.shape)
    pad_width[dim] = (0, pad_to - x.shape[dim])
    return np.pad(x, pad_width, mode="constant", constant_values=0)


def transform_weight_to_macro_layout(weight, cim_cfg, n_reduce_group, n_weight_duplicate_group):
    out_channel, in_channel, ker_hw, _ = weight.shape
    weight = np.transpose(weight, (0,2,3,1))
    weight = weight.reshape(out_channel, -1)
    assert out_channel <= cim_cfg.n_group_vcol
    n_reduce = weight.shape[1]
    n_use_row = math.ceil(n_reduce / (cim_cfg.n_comp * n_reduce_group))
    weight = pad_to_multiple(weight, 0, cim_cfg.n_group_vcol)
    weight = pad_to_multiple(weight, 1, n_reduce_group * cim_cfg.n_comp)
    weight = weight.reshape(1, cim_cfg.n_group_vcol, -1, n_reduce_group, cim_cfg.n_comp)
    weight = np.repeat(weight, repeats=n_weight_duplicate_group, axis=0)
    assert weight.shape == (n_weight_duplicate_group, cim_cfg.n_group_vcol, n_use_row, n_reduce_group, cim_cfg.n_comp)
    weight = np.transpose(weight, (2,4,0,3,1))
    weight = pad_to_multiple(weight, 0, cim_cfg.n_row)
    return weight

def calc_conv(inputs, weight, padding, stride):
    inputs = np.transpose(inputs, (0,3,1,2))
    conv_output = conv2d(inputs, weight, stride=stride)
    conv_output = np.transpose(conv_output, (0,2,3,1))
    return conv_output

"""
batch=1, 
in_channel=in_channel,
in_hw=16, 
ker_hw=3,
out_hw=16,
out_channel=32, 
stride=1, 
padding=1, 
config_path="/app/CIMCompiler/CIMCompiler/config/dac25/config_gs_4.json"
"""
@pytest.mark.parametrize(
    "batch, in_channel, in_hw, ker_hw, out_hw, out_channel, stride, padding, config_path",
    [
        *[(
            1, 
            in_channel, 
            in_hw, 
            3, 
            in_hw,
            out_channel,
            1,
            1,
            "/app/CIMCompiler/CIMCompiler/config/dac25/config_gs_4.json"
        ) for in_channel in [3, 8, 16, 24, 32, 40, 64] for out_channel in [8,16,24,32] for in_hw in [4, 8, 16]]
    ],
)
def test_dsl_conv(batch, in_channel, in_hw, ker_hw, out_hw, out_channel, stride, padding, config_path):
    set_raw_config_by_path(config_path)
    cim_cfg = get_config()
    kernel_size = in_channel * ker_hw * ker_hw
    n_reduce_group = math.ceil(kernel_size / (cim_cfg.n_comp * cim_cfg.n_row))
    n_weight_duplicate_group = cim_cfg.n_group // n_reduce_group
    pad_in_hw = in_hw + 2 * padding
    op_config = Conv2dConfig(
        batch=batch,
        in_channel=in_channel,
        out_channel=out_channel,
        in_hw=pad_in_hw,
        out_hw=out_hw,
        ker_hw=ker_hw,
        stride=stride,
        macro_config=cim_cfg,
        math=math,
        n_weight_duplicate_group=n_weight_duplicate_group,
        n_reduce_group=n_reduce_group,
        test_mode=True
    )
    op_path = os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/cimflow/conv2d.cim")
    op_runner = OpRunner(op_path, op_config, config_path)
    inputs = np.random.randint(-2, 3, (batch, in_hw, in_hw, in_channel), dtype=np.int8)
    weight = np.random.randint(-2, 3, (out_channel, in_channel, ker_hw, ker_hw), dtype=np.int8)
    # inputs = np.ones((batch, in_hw, in_hw, in_channel), dtype=np.int8)
    # weight = np.ones((out_channel, in_channel, ker_hw, ker_hw), dtype=np.int8)
    # weight[:,0,:,:] = 1
    # weight[:,1,:,:] = 2
    # weight[:,2,:,:] = 3
    # weight[:,3,:,:] = 4
    outputs = np.zeros((batch, out_hw, out_hw, out_channel), dtype=np.int32)

    inputs = np.pad(inputs, ((0,0), (padding, padding), (padding,padding), (0,0)), mode="constant", constant_values=0)
    macros = transform_weight_to_macro_layout(weight, cim_cfg, n_reduce_group, n_weight_duplicate_group)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = "test_result"
        os.makedirs(temp_dir, exist_ok=True)
        op_runner.run([inputs, macros], [outputs], simulate=True)

    golden = calc_conv(inputs, weight, padding, stride)
    assert outputs.shape == golden.shape, f"{outputs.shape=} != {golden.shape=}"
    print(f"golden:\n{golden}")
    print(f"outputs:\n{outputs}")
    num_all_close = np.sum(np.abs(outputs - golden) <= 0.01)
    correct_rate_str = f"{num_all_close}/{outputs.size} ({num_all_close/outputs.size*100:.2f}%)"
    print(f"Correct Rate: {correct_rate_str}")
    assert np.allclose(outputs, golden), f"Correct Rate: {correct_rate_str}"

if __name__ == "__main__":
    test_dsl_conv(
        batch=1, 
        in_channel=32, 
        in_hw=8, 
        ker_hw=3, 
        out_hw=8, 
        out_channel=8, 
        stride=1, 
        padding=1, 
        config_path="/app/CIMCompiler/CIMCompiler/config/dac25/config_gs_4.json"
    )