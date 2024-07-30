import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
import numpy as np
from utils.df_layout import tensor_bits_to_int8
from utils.bit_sparse_weight_transform import generate_valid_weight, weight_transform

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

class TestSimulatorPIMComputeBitSparse:

    @classmethod
    def setup_class(cls):
        cls.inst_util = InstUtil()
        cls.config_path = "/home/wangyiou/project/cim_compiler_frontend/playground/config/config.json"
        cls.simulator = Simulator.from_config(cls.config_path)
        cls.memory_space = cls.simulator.memory_space
        cls.macro_config = cls.simulator.macro_config
        cls.mask_config = cls.simulator.mask_config

    def setup_method(self):
        self.simulator.clear()

    def test_pimcompute_bit_sparse_multi_group(self):
        """
        y = xA
        x: int8, shape=[4, 16], memory=local, addr=INPUT_BUFFER_BASE, size=4*16
        mask: [
            [1,1,1,1,0,0,0,0], for macro 0
            [0,1,1,1,1,0,0,0]  for macro 1
            [0,0,1,1,1,1,0,0]  for macro 2
            [0,0,0,1,1,1,1,0]  for macro 3
        ]
        A: int8, shape=[16, 4, 128], dtype=bit, memory=macro, addr=MACRO_BASE, size = 16*4*256 / 8 = 2048
        y: int32, shape=[4, 128], memory=local, addr=OUTPUT_BUFFER_BASE, size=4*256*4 = 4096
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")
        meta_base = self.simulator.memory_space.get_base_of("meta")

        input_addr = input_buffer_base
        input_size = 8
        output_addr = output_buffer_base
        output_size = 8 * 4
        weight_addr = macro_base
        weight_size = 32
        mask_addr = mask_base
        
        # prepare data
        input = np.arange(64).reshape(4,16).astype(np.int8)
        weight = generate_valid_weight([128, 8, 2, 2], threshold=2)
        from types import SimpleNamespace
        op_cfg = SimpleNamespace(
            out_channel = 128
        )
        cim_cfg = SimpleNamespace(
            bits_column=self.macro_config.n_bcol
            n_macro=self.macro_config.n_macro // 4 # n_macro_per_group
            n_group=4
        )
        bit_sparse_weight, info, fold = weight_transform_group(weight, cim_cfg, op_cfg)
        self.simulator.memory_space.write(input, input_addr, input.size)
        self.simulator.memory_space.write(bit_sparse_weight, weight_addr, bit_sparse_weight.size)
        self.simulator.memory_space.write(info, meta_base, info.size)

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
        self.simulator.memory_space.write(mask_bits, mask_base, mask_bits.size)
        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, output_size, np.int32)
        assert (output==output_golden).all(), f"{output=}, {output_golden=}"

if __name__=="__main__":
    TestSimulatorPIMComputeValueSparse.setup_class()
    test_simulator = TestSimulatorPIMComputeValueSparse()
    test_simulator.setup_method()
    # test_simulator.test_pimcompute_value_sparse_single_group()
    test_simulator.test_pimcompute_value_sparse_single_group_accumulate()
    # test_simulator.test_pimcompute_dense_single_group_accumulate()
    # test_simulator.test_pimcompute_dense_single_group_part()