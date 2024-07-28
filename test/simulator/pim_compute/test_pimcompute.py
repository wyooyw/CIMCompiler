import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
import numpy as np

def init_macro_config():
    macro_config = MacroConfig(n_macro=2, n_row=4, n_comp=4, n_bcol=16)
    return macro_config

def init_mask_config():
    mask_config = MaskConfig(n_from=128, n_to=16)
    return mask_config

def init_memory_space(macro_config):
    memory_space = MemorySpace()
    global_memory = Memory("global_memory", "dram", 0, 128)
    local_memory = Memory("local_memory", "sram", 128, 128)
    input_buffer = Memory("pim_input_reg_buffer", "rf", 256, 64)
    output_buffer = Memory("pim_output_reg_buffer", "rf", 320, 64)
    macro = Memory("macro", "macro", 384, macro_config.total_size())
    memory_space.add_memory(global_memory)
    memory_space.add_memory(local_memory)
    memory_space.add_memory(input_buffer)
    memory_space.add_memory(output_buffer)
    memory_space.add_memory(macro)
    return memory_space

class TestSimulatorPIMCompute:

    @classmethod
    def setup_class(cls):
        cls.inst_util = InstUtil()
        cls.macro_config = init_macro_config()
        cls.mask_config = init_mask_config()
        cls.memory_space = init_memory_space(cls.macro_config)
        cls.simulator = Simulator(cls.memory_space , cls.macro_config, cls.mask_config)

    def setup_method(self):
        self.simulator.clear()

    def test_pimcompute_dense_single_group(self):
        """
        y = xA
        x: int8, shape=[4], memory=local, addr=LOCAL_BASE, size=4
        A: int8, shape=[4, 4], memory=macro, addr=MACRO_BASE, size=16
        y: int32, shape=[4], memory=local, addr=LOCAL_BASE+4, size=16
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
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
            
            self.inst_util.special_li(SpecialReg.GROUP_SIZE, self.macro_config.n_macro),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, 1),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, 0),

            self.inst_util.pimcompute_dense_single_group(
                0, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            ),
            self.inst_util.pim_output_dense(3)
        ]
        input = np.arange(4, dtype=np.int8)
        weight = np.arange(16, dtype=np.int8).reshape(4,4)
        output_golden = np.dot(input.astype(np.int32), weight.astype(np.int32))

        self.simulator.memory_space.write(input, input_addr, 4)
        self.simulator.memory_space.write(weight, macro_base, 16)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, 16, np.int32)
        assert (output==output_golden).all(), f"{output=}, {output_golden=}"

    def test_pimcompute_dense_single_group_accumulate(self):
        """
        y = xA + xB
        x: int8, shape=[4], memory=local, addr=LOCAL_BASE, size=4
        A: int8, shape=[4, 4], memory=macro, addr=MACRO_BASE, size=16
        B: int8, shape=[4, 4], memory=macro, addr=MACRO_BASE+16, size=16
        y: int32, shape=[4], memory=local, addr=LOCAL_BASE+4, size=16
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
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

            self.inst_util.special_li(SpecialReg.GROUP_SIZE, self.macro_config.n_macro),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, 1),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, 0),

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
            ),

            self.inst_util.pim_output_dense(3)
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
    
    def test_pimcompute_dense_single_group_part(self):
        """
        y = xA
        x: int8, shape=[3], memory=local, addr=input_buffer_base, size=3
        A: int8, shape=[3, 3], memory=macro. addr=
            A[0,:]: MACRO_BASE, size=3,
            A[1,:]: MACRO_BASE + 4, size=3,
            A[2,:]: MACRO_BASE + 8, size=3,
        y: int32, shape=[3], memory=local, addr=output_buffer_base, size=3
        """
        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")

        input_addr = input_buffer_base
        input_size = 3
        output_addr = output_buffer_base
        output_size = 3

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
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 3),

            self.inst_util.special_li(SpecialReg.GROUP_SIZE, self.macro_config.n_macro),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, 1),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, 0),

            self.inst_util.pimcompute_dense_single_group(
                0, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            ),

            self.inst_util.pim_output_dense(3)
        ]
        input = np.arange(4, dtype=np.int8)
        weight = np.arange(16, dtype=np.int8).reshape(4, 4)
        output_golden = np.dot(input[:3].astype(np.int32), weight[:3,:3].astype(np.int32))

        self.simulator.memory_space.write(input, input_addr, 4)
        self.simulator.memory_space.write(weight, macro_base, 16)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, 32, np.int32)
        print(output)
        print(output_golden)
        assert (output[:3]==output_golden).all()
        assert output[3]==0

    def test_pimcompute_dense_multi_group_fix_step(self):
        """
        batched mvm
        \forall b: y[b,:] = x[b,:] \cdot A[b,:,:]
        x: int8, shape=[2, 4], memory=local, addr=LOCAL_BASE, size=8
        A: int8, shape=[4, 2, 2], memory=macro, addr=MACRO_BASE, size=16
        y: int32, shape=[2, 2], memory=local, addr=LOCAL_BASE+8, size=16
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")

        input_addr = input_buffer_base
        input_size = 4
        weight_addr = macro_base
        weight_size = 16
        output_addr = output_buffer_base

        group_num = 2

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
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 2),

            self.inst_util.special_li(SpecialReg.GROUP_SIZE, self.macro_config.n_macro // group_num),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, group_num),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, input_size),

            self.inst_util.pimcompute_dense_single_group(
                0, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            ),

            self.inst_util.pim_output_dense(3)
        ]
        group_input = np.arange(8, dtype=np.int8).reshape(2,4)
        group_weight = np.arange(16, dtype=np.int8).reshape(4,2,2) # [comp, group, macro_per_group]
        group_output_golden = []
        for group_id in range(group_num):
            input = group_input[group_id]
            weight = group_weight[:,group_id,:]
            output_golden = np.dot(input.astype(np.int32), weight.astype(np.int32))
            group_output_golden.append(output_golden)

        self.simulator.memory_space.write(group_input, input_addr, input_size * group_num)
        self.simulator.memory_space.write(group_weight, weight_addr, weight_size)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, 16, np.int32)
        group_output_golden = np.concatenate(group_output_golden, axis=0).reshape(-1)
        assert (output==group_output_golden).all(), f"{output=}, {group_output_golden=}"

    def test_pimcompute_dense_multi_group_accumulate_fix_step(self):
        """
        batched mvm
        \forall b: y[b,:] = x[b,:] \cdot A[b,:,:] + x[b,:] \cdot B[b,:,:]
        x: int8, shape=[2, 4], memory=local, addr=LOCAL_BASE, size=8
        A: int8, shape=[4, 2, 2], memory=macro, addr=MACRO_BASE, size=16
        B: int8, shape=[4, 2, 2], memory=macro, addr=MACRO_BASE+16, size=16
        y: int32, shape=[2, 2], memory=local, addr=LOCAL_BASE+8, size=16
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")

        input_addr = input_buffer_base
        input_size = 4
        weight_addr = macro_base
        weight_size = 32
        output_addr = output_buffer_base

        group_num = 2

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
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 2),

            self.inst_util.special_li(SpecialReg.GROUP_SIZE, self.macro_config.n_macro // group_num),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, group_num),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, input_size),

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
            ),

            self.inst_util.pim_output_dense(3)
        ]
        group_input = np.arange(8, dtype=np.int8).reshape(2,4)
        group_weight = np.arange(32, dtype=np.int8).reshape(2,4,2,2) # [row, comp, group, macro_per_group]
        group_output_golden = []
        for group_id in range(group_num):
            input = group_input[group_id]
            weight0 = group_weight[0,:,group_id,:]
            weight1 = group_weight[1,:,group_id,:]
            output0 = np.dot(input.astype(np.int32), weight0.astype(np.int32))
            output1 = np.dot(input.astype(np.int32), weight1.astype(np.int32))
            output_golden = output0 + output1
            group_output_golden.append(output_golden)

        self.simulator.memory_space.write(group_input, input_addr, input_size * group_num)
        self.simulator.memory_space.write(group_weight, weight_addr, weight_size)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, 16, np.int32)
        group_output_golden = np.concatenate(group_output_golden, axis=0).reshape(-1)
        assert (output==group_output_golden).all(), f"{output=}, {group_output_golden=}"

if __name__=="__main__":
    TestSimulatorPIMCompute.setup_class()
    test_simulator = TestSimulatorPIMCompute()
    test_simulator.setup_method()
    test_simulator.test_pimcompute_dense_single_group()
    # test_simulator.test_pimcompute_dense_multi_group_accumulate_fix_step()
    # test_simulator.test_pimcompute_dense_single_group_accumulate()
    # test_simulator.test_pimcompute_dense_single_group_part()