import os
import json
import numpy as np
from dataclasses import dataclass
from test.base import OpRunner
import math
import tempfile
import shutil
import pytest

@dataclass
class ReduceOpConfig:
    reduce_num: int
    reduce_len: int

@dataclass
class ReduceTestConfig:
    reduce_config: ReduceOpConfig
    vector_len: int
    math: str

def get_reduce_config(cim_config_path):
    with open(cim_config_path, "r") as f:
        config = json.load(f)
    return ReduceOpConfig(
        reduce_num=config["reduce"]["reduce_num"],
        reduce_len=config["reduce"]["reduce_len"]
    )

class ModifyReduceConfig:
    def __init__(self, base_config_path, reduce_len, reduce_num):
        self.base_config_path = base_config_path
        self.reduce_len = reduce_len
        self.reduce_num = reduce_num

    def __enter__(self):
        # read base config
        with open(self.base_config_path, "r") as f:
            self.base_config = json.load(f)
        
        # modify reduce config
        self.modified_config = self.base_config.copy()
        self.modified_config["reduce"] = {}
        self.modified_config["reduce"]["reduce_len"] = self.reduce_len
        self.modified_config["reduce"]["reduce_num"] = self.reduce_num

        # create temp config file in a temp dir
        self.modified_config_path = tempfile.mktemp()
        with open(self.modified_config_path, "w") as f:
            json.dump(self.modified_config, f)
        return self  # 可以返回一个对象，供 `as` 关键字使用

    def __exit__(self, exc_type, exc_value, traceback):
        # clear modified_config_path
        os.remove(self.modified_config_path)
        return False  # 返回 False 表示异常未处理，会继续传播

@pytest.mark.parametrize(
    "vector_len, reduce_num, reduce_len",
    [
        (vector_len, reduce_num, reduce_len)
        for vector_len in [1, 2, 4, 8, 16, 32, 64, 2048]
        for reduce_num in [1, 2, 4, 8, 16, 32]
        for reduce_len in [2, 4, 8, 16, 32]
    ],
)
def test_reduce_inplace(vector_len, reduce_num, reduce_len):
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "test/op/test_reduce/test_reduce_inplace.cim")
    cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
    with ModifyReduceConfig(cim_config_path, reduce_len, reduce_num) as m:

        op_config = ReduceTestConfig(
            vector_len=vector_len,
            reduce_config=get_reduce_config(m.modified_config_path),
            math=math
        )

        op_runner = OpRunner(op_path, op_config, m.modified_config_path)

        """
        x_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
        """
        x = np.random.randint(-2,3, (op_config.vector_len,)).astype(np.float16)
        # x = np.zeros((op_config.vector_len,), dtype=np.float16) + 1
        output = np.zeros((1,), dtype=np.float16)
        
        golden = x.sum()
        op_runner.run([x], [output])

    allclose = np.allclose(output, golden, rtol=1e-2, atol=1e-2)
    print(f"{output=}")
    print(f"{golden=}")
    print(f"{allclose=}")

@pytest.mark.parametrize(
    "vector_len, reduce_num, reduce_len",
    [
        (vector_len, reduce_num, reduce_len)
        for vector_len in [1, 2, 4, 32, 512, 1024, 2048]
        for reduce_num in [1, 2, 4, 8, 16, 32]
        for reduce_len in [2, 4, 8, 16, 32]
    ],
)
def test_reduce(vector_len, reduce_num, reduce_len):
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "test/op/test_reduce/test_reduce.cim")
    cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
    with ModifyReduceConfig(cim_config_path, reduce_len, reduce_num) as m:

        op_config = ReduceTestConfig(
            vector_len=vector_len,
            reduce_config=get_reduce_config(m.modified_config_path),
            math=math
        )

        op_runner = OpRunner(op_path, op_config, m.modified_config_path)

        """
        x_global = Buffer(<{{seqlen}}>, fp16, __GLOBAL__);
        """
        x = np.random.randint(-2,3, (op_config.vector_len,)).astype(np.float16)
        # x = np.zeros((op_config.vector_len,), dtype=np.float16) + 1
        output = np.zeros((1,), dtype=np.float16)
        
        golden = x.sum()
        op_runner.run([x], [output])

    allclose = np.allclose(output, golden, rtol=1e-2, atol=1e-2)
    print(f"{output=}")
    print(f"{golden=}")
    print(f"{allclose=}")

if __name__=="__main__":
    test_reduce(32, 4, 2)