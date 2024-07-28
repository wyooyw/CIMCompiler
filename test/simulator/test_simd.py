import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
import numpy as np
def init_memory_space():
    memory_space = MemorySpace()
    global_memory = Memory("global_memory", "dram", 0, 128)
    local_memory = Memory("local_memory", "sram", 128, 128)
    input_buffer = Memory("pim_input_reg_buffer", "rf", 256, 64)
    output_buffer = Memory("pim_output_reg_buffer", "rf", 320, 64)
    macro = Memory("macro", "macro", 384, 2*4*4*16//8)
    memory_space.add_memory(global_memory)
    memory_space.add_memory(local_memory)
    memory_space.add_memory(input_buffer)
    memory_space.add_memory(output_buffer)
    memory_space.add_memory(macro)
    return memory_space

def init_macro_config():
    macro_config = MacroConfig(n_macro=2, n_row=4, n_comp=4, n_bcol=16)
    return macro_config

def init_mask_config():
    mask_config = MaskConfig(n_from=8, n_to=4) # 8选4
    return mask_config

class TestSimulatorSIMD:

    @classmethod
    def setup_class(cls):
        cls.inst_util = InstUtil()
        cls.memory_space = init_memory_space()
        cls.macro_config = init_macro_config()
        cls.mask_config = init_mask_config()
        cls.simulator = Simulator(cls.memory_space , cls.macro_config, cls.mask_config)

    def setup_method(self):
        self.simulator.clear()

    def test_vvadd(self):
        """
        z = x + y
        x: int32, shape=[4], memory=local, addr=LOCAL_BASE, size=16
        y: int32, shape=[4], memory=local, addr=LOCAL_BASE+16, size=16
        z: int32, shape=[4], memory=local, addr=LOCAL_BASE+32, size=16
        """

        local_memory_base = self.simulator.memory_space.get_base_of("local_memory")

        input1_addr = local_memory_base
        input2_addr = local_memory_base + 16
        input_size = 16
        output_addr = local_memory_base + 32

        inst_list = [
            # set general register
            self.inst_util.general_li(0, input1_addr), # input1 addr
            self.inst_util.general_li(1, input2_addr), # input2 addr
            self.inst_util.general_li(2, input_size), # input size
            self.inst_util.general_li(3, output_addr), # output addr

            # set special register
            self.inst_util.special_li(SpecialReg.SIMD_INPUT_1_BIT_WIDTH, 32),
            self.inst_util.special_li(SpecialReg.SIMD_INPUT_2_BIT_WIDTH, 32),
            self.inst_util.special_li(SpecialReg.SIMD_OUTPUT_BIT_WIDTH, 32),

            self.inst_util.simd_vvadd(
                rs1=0, # input1 addr
                rs2=1, # input2 addr
                rs3=2, # input size
                rd=3,  # output addr
            )
        ]
        input1 = np.arange(0,4, dtype=np.int32)
        input2 = np.arange(4,8, dtype=np.int32)
        output_golden = input1 + input2

        self.simulator.memory_space.write(input1, input1_addr, input_size)
        self.simulator.memory_space.write(input2, input2_addr, input_size)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, 16, np.int32)
        assert (output==output_golden).all()

    def test_vsadd(self):
        """
        z = a + x
        a: int32, scalar
        x: int32, shape=[4], memory=local, addr=LOCAL_BASE, size=16
        z: int32, shape=[4], memory=local, addr=LOCAL_BASE+32, size=16
        """

        local_memory_base = self.simulator.memory_space.get_base_of("local_memory")

        scalar_value = 1111
        input_addr = local_memory_base
        input_size = 16
        output_addr = local_memory_base + 16

        inst_list = [
            # set general register
            self.inst_util.general_li(0, input_addr), # scalar value
            self.inst_util.general_li(1, scalar_value), # input vector addr
            self.inst_util.general_li(2, input_size), # activate row
            self.inst_util.general_li(3, output_addr), # output addr

            # set special register
            self.inst_util.special_li(SpecialReg.SIMD_INPUT_1_BIT_WIDTH, 32),
            self.inst_util.special_li(SpecialReg.SIMD_INPUT_2_BIT_WIDTH, 32),
            self.inst_util.special_li(SpecialReg.SIMD_OUTPUT_BIT_WIDTH, 32),

            self.inst_util.simd_vsadd(
                rs1=0, # scalar value
                rs2=1, # input vector addr
                rs3=2, # input size
                rd=3,  # output addr
            )
        ]
        input_vec = np.arange(4,8, dtype=np.int32)
        output_golden = input_vec + scalar_value

        self.simulator.memory_space.write(input_vec, input_addr, input_size)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, 16, np.int32)
        assert (output==output_golden).all()

if __name__=="__main__":
    TestSimulatorSIMD.setup_class()
    test_simulator = TestSimulatorSIMD()

    test_simulator.setup_method()
    test_simulator.test_vvadd()

    test_simulator.setup_method()
    test_simulator.test_vsadd()