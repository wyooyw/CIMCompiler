from test.base import SPMDOpRunner, SIMDOpConfig
import os
from dataclasses import dataclass
import numpy as np



def test_send_recv():
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "test/compiler/comm/test_send_recv.cim")
    cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
    runner = SPMDOpRunner(
        op_path=op_path,
        op_config=SIMDOpConfig(),
        cim_config_path=cim_config_path,
        num_cores=2
    )
    input_0 = np.arange(16, dtype=np.int32)
    input_1 = np.arange(16, dtype=np.int32) + 16
    output_0 = np.zeros(16, dtype=np.int32)
    output_1 = np.zeros(16, dtype=np.int32)
    runner.run(
        [[input_0], [input_1]], 
        [[output_0], [output_1]]
    )
    assert np.all(output_1 == input_0), f"output_1: {output_1}, input_0: {input_0}"
    assert np.all(output_0 == input_1), f"output_0: {output_0}, input_1: {input_1}"

@dataclass
class AllGatherTestConfig(SIMDOpConfig):
    data_size: int = 0

def test_all_gather(data_size, world_size):
    cim_compiler_home = os.environ["CIM_COMPILER_BASE"]
    op_path = os.path.join(cim_compiler_home, "test/compiler/comm/test_all_gather.cim")
    cim_config_path = os.path.join(cim_compiler_home, "test/op/llm/config.json")
    runner = SPMDOpRunner(
        op_path=op_path,
        op_config=AllGatherTestConfig(data_size=data_size),
        cim_config_path=cim_config_path,
        num_cores=world_size
    )
    
    inputs = []
    for i in range(world_size):
        inputs.append([np.arange(data_size, dtype=np.float16) + i * data_size])
    outputs = []
    for i in range(world_size):
        outputs.append([np.zeros((world_size, data_size), dtype=np.float16)])

    runner.run(
        inputs, 
        outputs
    )

    golden = np.concatenate(inputs, axis=0)
    for i in range(world_size):
        assert np.all(outputs[i][0] == golden), f"output_{i}: {outputs[i][0]}, golden: {golden}"

if __name__ == "__main__":
    # test_send_recv()
    test_all_gather(data_size=4, world_size=2)