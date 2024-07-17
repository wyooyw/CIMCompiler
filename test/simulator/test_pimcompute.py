import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
import numpy as np
def init_memory_space():
    memory_space = MemorySpace()
    global_memory = Memory("global_memory", "dram", 0, 128)
    local_memory = Memory("local_memory", "sram", 128, 128)
    input_buffer = Memory("input_buffer", "rf", 256, 64)
    output_buffer = Memory("output_buffer", "rf", 320, 64)
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

class TestSimulatorPIMCompute:

    @classmethod
    def setup_class(cls):
        cls.inst_util = InstUtil()
        cls.memory_space = init_memory_space()
        cls.macro_config = init_macro_config()
        cls.simulator = Simulator(cls.memory_space , cls.macro_config)

    def setup_method(self):
        self.simulator.clear()

    def test_pimcompute_dense_single_group(self):
        """
        y = xA
        x: int8, shape=[4], memory=local, addr=LOCAL_BASE, size=4
        A: int8, shape=[4, 4], memory=macro, addr=MACRO_BASE, size=16
        y: int32, shape=[4], memory=local, addr=LOCAL_BASE+4, size=16
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("input_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("output_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")

        input_addr = input_buffer_base
        input_size = 4
        output_addr = output_buffer_base

        inst_list = [
            # set general register
            self.inst_util.general_li(0, input_addr), # input addr
            self.inst_util.general_li(1, input_size), # input size
            self.inst_util.general_li(2, 0), # activate row
            self.inst_util.general_li(3, output_addr), # output addr

            # set special register
            self.inst_util.special_li(SpecialReg.INPUT_BIT_WIDTH, 8),
            self.inst_util.special_li(SpecialReg.WEIGHT_BIT_WIDTH, 8),
            self.inst_util.special_li(SpecialReg.OUTPUT_BIT_WIDTH, 32),
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 4),

            self.inst_util.pimcompute_dense_single_group(
                0, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            )
        ]
        input = np.arange(4, dtype=np.int8)
        weight = np.arange(16, dtype=np.int8).reshape(4,4)
        output_golden = np.dot(input.astype(np.int32), weight.astype(np.int32))

        self.simulator.memory_space.write(input, input_addr, 4)
        self.simulator.memory_space.write(weight, macro_base, 16)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, 16, np.int32)
        assert (output==output_golden).all()

    def test_pimcompute_dense_single_group_accumulate(self):
        """
        y = xA + xB
        x: int8, shape=[4], memory=local, addr=LOCAL_BASE, size=4
        A: int8, shape=[4, 4], memory=macro, addr=MACRO_BASE, size=16
        B: int8, shape=[4, 4], memory=macro, addr=MACRO_BASE+16, size=16
        y: int32, shape=[4], memory=local, addr=LOCAL_BASE+4, size=16
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("input_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("output_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")

        input_addr = input_buffer_base
        input_size = 4
        output_addr = output_buffer_base

        inst_list = [
            # set general register
            self.inst_util.general_li(0, input_addr), # input addr
            self.inst_util.general_li(1, input_size), # input size
            self.inst_util.general_li(2, 0), # activate row
            self.inst_util.general_li(3, output_addr), # output addr

            # set special register
            self.inst_util.special_li(SpecialReg.INPUT_BIT_WIDTH, 8),
            self.inst_util.special_li(SpecialReg.WEIGHT_BIT_WIDTH, 8),
            self.inst_util.special_li(SpecialReg.OUTPUT_BIT_WIDTH, 32),
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 4),

            self.inst_util.pimcompute_dense_single_group(
                0, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            ),

            self.inst_util.general_li(2, 1), # activate row
            self.inst_util.pimcompute_dense_single_group(
                1, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            )
        ]
        input = np.arange(4, dtype=np.int8)
        weight_A = np.arange(16, dtype=np.int8).reshape(4,4)
        weight_B = np.arange(16,32, dtype=np.int8).reshape(4,4)
        output_golden = np.dot(input.astype(np.int32), weight_A.astype(np.int32)) + np.dot(input.astype(np.int32), weight_B.astype(np.int32))

        self.simulator.memory_space.write(input, input_addr, 4)
        self.simulator.memory_space.write(weight_A, macro_base, 16)
        self.simulator.memory_space.write(weight_B, macro_base+16, 16)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, 16, np.int32)
        assert (output==output_golden).all()
    


if __name__=="__main__":
    TestSimulatorPIMCompute.setup_class()
    test_simulator = TestSimulatorPIMCompute()
    test_simulator.test_pimcompute_dense_single_group()
    test_simulator.test_pimcompute_dense_single_group_accumulate()