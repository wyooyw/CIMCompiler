import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
import numpy as np

def init_macro_config():
    macro_config = MacroConfig(n_macro=4, n_row=4, n_comp=4, n_bcol=16)
    return macro_config

def init_mask_config():
    mask_config = MaskConfig(n_from=8, n_to=4) # 8选4
    return mask_config

def init_memory_space(macro_config):
    memory_space = MemorySpace()
    global_memory = Memory("global_memory", "dram", 0, 128)
    local_memory = Memory("local_memory", "sram", 128, 128)
    input_buffer = Memory("pim_input_reg_buffer", "rf", 256, 64)
    output_buffer = Memory("pim_output_reg_buffer", "rf", 320, 64)
    macro = Memory("macro", "macro", output_buffer.end, macro_config.total_size())
    mask = Memory("mask", "rf", macro.end, 64)
    memory_space.add_memory(global_memory)
    memory_space.add_memory(local_memory)
    memory_space.add_memory(input_buffer)
    memory_space.add_memory(output_buffer)
    memory_space.add_memory(macro)
    memory_space.add_memory(mask)
    return memory_space

class TestSimulatorPIMComputeValueSparse:

    @classmethod
    def setup_class(cls):
        cls.inst_util = InstUtil()
        cls.macro_config = init_macro_config()
        cls.mask_config = init_mask_config()
        cls.memory_space = init_memory_space(cls.macro_config)
        cls.simulator = Simulator(cls.memory_space , cls.macro_config, cls.mask_config)

    def setup_method(self):
        self.simulator.clear()

    def test_pimcompute_value_sparse_single_group(self):
        """
        y = xA
        x: int8, shape=[8], memory=local, addr=INPUT_BUFFER_BASE, size=8
        mask: [
            [1,1,1,1,0,0,0,0], for macro 0
            [0,1,1,1,1,0,0,0]  for macro 1
            [0,0,1,1,1,1,0,0]  for macro 2
            [0,0,0,1,1,1,1,0]  for macro 3
        ]
        A: int8, shape=[4, 8], memory=macro, addr=MACRO_BASE, size=32
        y: int32, shape=[8], memory=local, addr=OUTPUT_BUFFER_BASE, size=32
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")
        mask_base = self.simulator.memory_space.get_base_of("mask")

        input_addr = input_buffer_base
        input_size = 8
        output_addr = output_buffer_base
        output_size = 8 * 4
        weight_addr = macro_base
        weight_size = 32
        mask_addr = mask_base
        mask = np.array([
            [1,1,1,1,0,0,0,0],
            [0,1,1,1,1,0,0,0],
            [0,0,1,1,1,1,0,0],
            [0,0,0,1,1,1,1,0]
        ], dtype=bool)

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
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 8),
            
            self.inst_util.special_li(SpecialReg.GROUP_SIZE, self.macro_config.n_macro),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, 1),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, 0),

            self.inst_util.special_li(SpecialReg.VALUE_SPARSE_MASK_ADDR, mask_addr),

            self.inst_util.pimcompute_value_sparse(
                0, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            ),
            self.inst_util.pim_output_dense(3)
        ]
        input = np.arange(input_size, dtype=np.int8)
        weight = np.arange(weight_size, dtype=np.int8).reshape(4,4,2)

        output_list = []
        for macro_id in range(self.macro_config.n_macro):
            macro_mask = mask[macro_id]
            macro_input = input[macro_mask]
            macro_weight = weight[:,macro_id,:]
            macro_output = np.dot(macro_input.astype(np.int32), macro_weight.astype(np.int32))
            output_list.append(macro_output)
        output_golden = np.concatenate(output_list)

        self.simulator.memory_space.write(input, input_addr, input_size)
        self.simulator.memory_space.write(weight, macro_base, weight_size)
        self.simulator.memory_space.write(mask, mask_base, mask.size)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, output_size, np.int32)
        assert (output==output_golden).all(), f"{output=}, {output_golden=}"

    def test_pimcompute_value_sparse_single_group_accumulate(self):
        """
        y = xA + xB
        x: int8, shape=[8], memory=local, addr=INPUT_BUFFER_BASE, size=8
        mask1: [
            [1,1,1,1,0,0,0,0], for macro 0
            [1,1,1,1,0,0,0,0]  for macro 1
            [1,1,1,1,0,0,0,0]  for macro 2
            [1,1,1,1,0,0,0,0]  for macro 3
        ]
        mask2: [
            [0,0,0,0,1,1,1,1], for macro 0
            [0,0,0,0,1,1,1,1]  for macro 1
            [0,0,0,0,1,1,1,1]  for macro 2
            [0,0,0,0,1,1,1,1]  for macro 3
        ]
        A: int8, shape=[4, 8], memory=macro, addr=MACRO_BASE, size=32
        B: int8, shape=[4, 8], memory=macro, addr=MACRO_BASE+32, size=32
        y: int32, shape=[8], memory=local, addr=OUTPUT_BUFFER_BASE, size=32
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")
        mask_base = self.simulator.memory_space.get_base_of("mask")

        input_addr = input_buffer_base
        input_size = 8
        output_addr = output_buffer_base
        output_size = 8 * 4
        weight_addr = macro_base
        weight_size = 64
        mask_addr = mask_base
        mask = np.array([[
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0]
        ],[
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1],
        ]], dtype=bool)

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
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 8),
            
            self.inst_util.special_li(SpecialReg.GROUP_SIZE, self.macro_config.n_macro),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, 1),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, 0),

            self.inst_util.special_li(SpecialReg.VALUE_SPARSE_MASK_ADDR, mask_addr),

            self.inst_util.pimcompute_value_sparse(
                0, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            ),

            self.inst_util.general_li(2, 1), # activate row
            self.inst_util.special_li(SpecialReg.VALUE_SPARSE_MASK_ADDR, mask_addr + 32),
            self.inst_util.pimcompute_value_sparse(
                1, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            ),
            self.inst_util.pim_output_dense(3)
        ]
        input = np.arange(input_size, dtype=np.int8)
        weight = np.arange(weight_size, dtype=np.int8).reshape(2,4,4,2)

        output_list = []
        for macro_id in range(self.macro_config.n_macro):
            macro_mask = mask[0, macro_id]
            macro_input = input[macro_mask]
            macro_weight = weight[0,:,macro_id,:]
            macro_output = np.dot(macro_input.astype(np.int32), macro_weight.astype(np.int32))
            output_list.append(macro_output)
        for macro_id in range(self.macro_config.n_macro):
            macro_mask = mask[1, macro_id]
            macro_input = input[macro_mask]
            macro_weight = weight[1,:,macro_id,:]
            macro_output = np.dot(macro_input.astype(np.int32), macro_weight.astype(np.int32))
            output_list[macro_id] += macro_output
        output_golden = np.concatenate(output_list)

        self.simulator.memory_space.write(input, input_addr, input_size)
        self.simulator.memory_space.write(weight, macro_base, weight_size)
        self.simulator.memory_space.write(mask, mask_base, mask.size)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, output_size, np.int32)
        assert (output==output_golden).all(), f"{output=}, {output_golden=}"

    def test_pimcompute_value_sparse_single_group_notalign(self):
        """
        y = xA
        x: int8, shape=[8], memory=local, addr=INPUT_BUFFER_BASE, size=8
        mask: [
            [1,1,1,1,0,0,0,0], for macro 0
            [0,0,1,1,1,0,0,0]  for macro 1
            [0,0,1,1,0,0,0,0]  for macro 2
            [0,0,0,1,0,0,0,0]  for macro 3
        ]
        A: int8, shape=[4, 8], memory=macro, addr=MACRO_BASE, size=32
        y: int32, shape=[8], memory=local, addr=OUTPUT_BUFFER_BASE, size=32
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")
        mask_base = self.simulator.memory_space.get_base_of("mask")

        input_addr = input_buffer_base
        input_size = 8
        output_addr = output_buffer_base
        output_size = 8 * 4
        weight_addr = macro_base
        weight_size = 32
        mask_addr = mask_base
        mask = np.array([
            [1,0,1,1,0,0,0,0], # 3 input
            [0,0,1,1,0,0,0,0], # 2 input
            [0,0,0,0,0,1,0,0], # 1 input
            [0,0,0,0,0,0,0,0], # 0 input
        ], dtype=bool)

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
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 8),
            
            self.inst_util.special_li(SpecialReg.GROUP_SIZE, self.macro_config.n_macro),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, 1),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, 0),

            self.inst_util.special_li(SpecialReg.VALUE_SPARSE_MASK_ADDR, mask_addr),

            self.inst_util.pimcompute_value_sparse(
                0, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            ),
            self.inst_util.pim_output_dense(3)
        ]
        input = np.arange(input_size, dtype=np.int8)
        weight = np.arange(weight_size, dtype=np.int8).reshape(4,4,2)

        output_list = []
        for macro_id in range(self.macro_config.n_macro):
            macro_mask = mask[macro_id]
            macro_input = input[macro_mask]
            macro_input = np.pad(macro_input, (0, self.macro_config.n_comp - macro_input.size), mode='constant', constant_values=0)
            macro_weight = weight[:,macro_id,:]
            macro_output = np.dot(macro_input.astype(np.int32), macro_weight.astype(np.int32))
            output_list.append(macro_output)
        output_golden = np.concatenate(output_list)

        self.simulator.memory_space.write(input, input_addr, input_size)
        self.simulator.memory_space.write(weight, macro_base, weight_size)
        self.simulator.memory_space.write(mask, mask_base, mask.size)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, output_size, np.int32)
        assert (output==output_golden).all(), f"{output=}, {output_golden=}"

    def test_pimcompute_value_sparse_single_group_accumulate_notalign(self):
        """
        y = xA + xB
        x: int8, shape=[8], memory=local, addr=INPUT_BUFFER_BASE, size=8
        mask1: [
            [1,1,1,1,0,0,0,0], for macro 0
            [1,1,1,1,0,0,0,0]  for macro 1
            [1,1,1,1,0,0,0,0]  for macro 2
            [1,1,1,1,0,0,0,0]  for macro 3
        ]
        mask2: [
            [0,0,0,0,1,1,1,1], for macro 0
            [0,0,0,0,1,1,1,1]  for macro 1
            [0,0,0,0,1,1,1,1]  for macro 2
            [0,0,0,0,1,1,1,1]  for macro 3
        ]
        A: int8, shape=[4, 8], memory=macro, addr=MACRO_BASE, size=32
        B: int8, shape=[4, 8], memory=macro, addr=MACRO_BASE+32, size=32
        y: int32, shape=[8], memory=local, addr=OUTPUT_BUFFER_BASE, size=32
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")
        mask_base = self.simulator.memory_space.get_base_of("mask")

        input_addr = input_buffer_base
        input_size = 8
        output_addr = output_buffer_base
        output_size = 8 * 4
        weight_addr = macro_base
        weight_size = 64
        mask_addr = mask_base
        mask = np.array([[
            [1,1,1,1,0,0,0,0],
            [0,1,1,1,1,0,0,0],
            [0,0,1,1,1,1,0,0],
            [0,0,0,1,1,1,1,0]
        ],[
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,0,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0],
        ]], dtype=bool)

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
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 8),
            
            self.inst_util.special_li(SpecialReg.GROUP_SIZE, self.macro_config.n_macro),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, 1),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, 0),

            self.inst_util.special_li(SpecialReg.VALUE_SPARSE_MASK_ADDR, mask_addr),

            self.inst_util.pimcompute_value_sparse(
                0, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            ),

            self.inst_util.general_li(2, 1), # activate row
            self.inst_util.special_li(SpecialReg.VALUE_SPARSE_MASK_ADDR, mask_addr + 32),
            self.inst_util.pimcompute_value_sparse(
                1, # accumulate
                0, # rs1 input addr
                1, # rs2 input size
                2, # rs3 activate row
                3, # rd output addr
            ),
            self.inst_util.pim_output_dense(3)
        ]
        input = np.arange(input_size, dtype=np.int8)
        weight = np.arange(weight_size, dtype=np.int8).reshape(2,4,4,2)

        output_list = []
        for macro_id in range(self.macro_config.n_macro):
            macro_mask = mask[0, macro_id]
            macro_input = input[macro_mask]
            macro_input = np.pad(macro_input, (0, self.macro_config.n_comp - macro_input.size), mode='constant', constant_values=0)
            macro_weight = weight[0,:,macro_id,:]
            macro_output = np.dot(macro_input.astype(np.int32), macro_weight.astype(np.int32))
            output_list.append(macro_output)
        for macro_id in range(self.macro_config.n_macro):
            macro_mask = mask[1, macro_id]
            macro_input = input[macro_mask]
            macro_input = np.pad(macro_input, (0, self.macro_config.n_comp - macro_input.size), mode='constant', constant_values=0)
            macro_weight = weight[1,:,macro_id,:]
            macro_output = np.dot(macro_input.astype(np.int32), macro_weight.astype(np.int32))
            output_list[macro_id] += macro_output
        output_golden = np.concatenate(output_list)

        self.simulator.memory_space.write(input, input_addr, input_size)
        self.simulator.memory_space.write(weight, macro_base, weight_size)
        self.simulator.memory_space.write(mask, mask_base, mask.size)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, output_size, np.int32)
        assert (output==output_golden).all(), f"{output=}, {output_golden=}"

if __name__=="__main__":
    TestSimulatorPIMComputeValueSparse.setup_class()
    test_simulator = TestSimulatorPIMComputeValueSparse()
    test_simulator.setup_method()
    test_simulator.test_pimcompute_value_sparse_single_group_accumulate_notalign()
    # test_simulator.test_pimcompute_dense_multi_group_accumulate_fix_step()
    # test_simulator.test_pimcompute_dense_single_group_accumulate()
    # test_simulator.test_pimcompute_dense_single_group_part()