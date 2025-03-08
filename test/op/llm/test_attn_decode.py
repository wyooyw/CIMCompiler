import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined
import numpy as np
import tempfile
import subprocess
from dataclasses import dataclass
from cim_compiler.simulator.macro_utils import MacroConfig
from cim_compiler.utils.df_layout import tensor_bits_to_int8
import pytest

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

class OpRunner:
    def __init__(self, op_path, op_config, cim_config_path):
        self.op_path = op_path
        self.op_config = op_config
        self.cim_config_path = cim_config_path

    def run(self, input_list:list[np.ndarray], output_list:list[np.ndarray]):
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # tmp_dir = "/home/wangyiou/project/CIMCompiler/.temp"
            op_code_path = os.path.join(tmp_dir, "op_code.cim")
            final_code_dir = os.path.join(tmp_dir, "compiler_output")
            simulator_output_dir = os.path.join(tmp_dir, "simulator_output")
            image_path = os.path.join(tmp_dir, "image.bin")
            self.fill_template(self.op_path, self.op_config, op_code_path)
            self.compile(op_code_path, final_code_dir)
            self.make_image(input_list, image_path)
            self.simulate(image_path, final_code_dir, simulator_output_dir)
            self.get_output(simulator_output_dir, input_list, output_list)

    def get_output(self, simulator_output_dir:str, input_list:list[np.ndarray], output_list:list[np.ndarray]):
        # Calculate total size of input arrays in bytes
        input_size = sum(arr.nbytes for arr in input_list)
        
        # Read the output data from image.bin
        image_path = os.path.join(simulator_output_dir, "image.bin")
        with open(image_path, 'rb') as f:
            # Skip input data
            f.seek(input_size)
            
            # Read output data for each output array
            for output_arr in output_list:
                output_bytes = f.read(output_arr.nbytes)
                # Copy bytes directly into the output array
                output = np.frombuffer(output_bytes, dtype=output_arr.dtype).reshape(output_arr.shape)
                output_arr[:] = output

    def simulate(self, image_path:str, final_code_dir:str, simulator_output_dir:str):
        subprocess.run([
            "cim-compiler", "simulate",
            "--code-file", os.path.join(final_code_dir, "final_code.json"),
            "--data-file", image_path,
            "--config-file", self.cim_config_path,
            "--output-dir", simulator_output_dir,
            "--code-format", "cimflow",
            "--save-stats"
        ], check=True)

    def make_image(self, input_list:list[np.ndarray], image_path:str):
        # 将所有输入张量转换为字节数组并拼接
        image_byte_array = bytearray()
        for tensor in input_list:
            tensor_byte_array = bytearray(tensor)
            image_byte_array += tensor_byte_array
        
        # 将拼接后的字节数组写入文件
        with open(image_path, 'wb') as f:
            f.write(image_byte_array)

    def compile(self, op_code_path:str, final_code_dir:str):
        subprocess.run([
            "cim-compiler", "compile",
            "--input-file", op_code_path,
            "--output-dir", final_code_dir,
            "--config-file", self.cim_config_path
        ], check=True)

    def fill_template(self, src_path:str, context:OpConfig, dst_path:str=None):
        src_folder, src_file = os.path.split(src_path)

        env = Environment(
            loader=FileSystemLoader([
                src_folder, 
                os.environ["CIM_COMPILER_BASE"],
                os.environ.get(os.environ["CIM_COMPILER_BASE"], "cim_compiler")
            ]),
            undefined=StrictUndefined
        )
        template = env.get_template(src_file)
        output = template.render(context.__dict__)

        if dst_path:
            with open(dst_path, "w") as f:
                f.write(output)
        
        return output

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