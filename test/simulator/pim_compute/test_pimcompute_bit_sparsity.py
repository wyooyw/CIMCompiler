import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
import numpy as np
from utils.df_layout import tensor_bits_to_int8
from utils.bit_sparse_weight_transform import generate_valid_weight, weight_transform, weight_transform_group, parse_out_mask, outsum_mask_to_transfer_mask

def init_macro_config():
    macro_config = MacroConfig(n_macro=4, n_row=4, n_comp=4, n_bcol=16)
    return macro_config

def init_mask_config():
    mask_config = MaskConfig(n_from=8, n_to=4) # 8选4
    return mask_config

def init_memory_space(macro_config):
    memory_space = MemorySpace()
    global_memory = Memory("global_memory", "dram", 0, 128)
    local_memory = Memory("local_memory", "sram", global_memory.end, 512)
    input_buffer = Memory("pim_input_reg_buffer", "rf", local_memory.end, 64)
    output_buffer = Memory("pim_output_reg_buffer", "rf", input_buffer.end, 512)
    macro = Memory("macro", "macro", output_buffer.end, macro_config.total_size())
    mask = Memory("pim_meta_data_reg_buffer", "rf", macro.end, 1024)
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
        # cls.config_path = "/home/wangyiou/project/cim_compiler_frontend/playground/config/config.json"
        
        cls.macro_config = init_macro_config()
        cls.mask_config = init_mask_config()
        cls.memory_space = init_memory_space(cls.macro_config)
        cls.simulator = Simulator(cls.memory_space , cls.macro_config, cls.mask_config)


    def setup_method(self):
        self.simulator.clear()

    def test_pimcompute_bit_sparse_multi_group(self):
        """
        y = xA
        x: int8, shape=[2, 4], memory=local, addr=INPUT_BUFFER_BASE, size=2*4=8
        A: int8, shape=[4, 2, 32], dtype=bit, memory=macro, addr=MACRO_BASE, size = 4*2*32 / 8 = 32
        y: int32, shape=[2, 32], memory=local, addr=OUTPUT_BUFFER_BASE, size=2*32*4 = 256
        """

        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")
        meta_base = self.simulator.memory_space.get_base_of("pim_meta_data_reg_buffer")

        input_addr = input_buffer_base
        input_size = 8
        output_addr = output_buffer_base
        output_size = 256
        weight_addr = macro_base
        weight_size = 32
        
        # prepare data
        input = np.arange(8).reshape(2,4).astype(np.int8)
        weight = generate_valid_weight([32, 4, 1, 1], threshold=1)
        from types import SimpleNamespace
        op_cfg = SimpleNamespace(
            out_channel = 32
        )
        cim_cfg = SimpleNamespace(
            bits_column=self.macro_config.n_bcol,
            n_macro=self.macro_config.n_macro // 2,   # n_macro_per_group
            n_group=2
        )
        
        bit_sparse_weight, info, fold = weight_transform_group(weight, cim_cfg, op_cfg)
        # print(f"{weight=}")
        # print(f"{info=}")
        # print(f"{fold=}")
        # import pdb; pdb.set_trace()
        self.simulator.memory_space.write(input, input_addr, input.size)
        self.simulator.memory_space.write(bit_sparse_weight, weight_addr, bit_sparse_weight.size)
        self.simulator.memory_space.write(info, meta_base, info.size)

        inst_list = [
            # set general register
            self.inst_util.general_li(0, input_addr), # input addr
            self.inst_util.general_li(1, input_size // 2), # input size
            self.inst_util.general_li(2, 0), # activate row
            self.inst_util.general_li(3, output_addr), # output addr

            # set special register
            self.inst_util.special_li(SpecialReg.INPUT_BIT_WIDTH, 8),
            self.inst_util.special_li(SpecialReg.WEIGHT_BIT_WIDTH, 1),
            self.inst_util.special_li(SpecialReg.OUTPUT_BIT_WIDTH, 32),
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 32),
            
            self.inst_util.special_li(SpecialReg.GROUP_SIZE, 2),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, 2),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, 4),

            self.inst_util.special_li(SpecialReg.BIT_SPARSE_META_ADDR, meta_base),

            self.inst_util.pimcompute(
                value_sparse=0, 
                bit_sparse=1, 
                group=1, 
                group_input_mode=0, 
                accumulate=1, 
                rs1=0, 
                rs2=1, 
                rs3=2, 
                rd=3
            ),
            self.inst_util.pim_output_dense(3)
        ]
        # input = np.arange(input_size, dtype=np.int8)
        # weight = np.arange(weight_size, dtype=np.int8).reshape(4,4,2)
        # print(f"{weight}")
        output_list = []
        for group_id in range(2):
            input_group = input[group_id]
            weight_group = weight.reshape(weight.shape[0], -1)
            weight_group = np.transpose(weight_group, [1,0])
            output_group = np.dot(input_group.astype(np.int32), weight_group.astype(np.int32))
            output_list.append(output_group)
        output_golden = np.concatenate(output_list)
        output_golden = output_golden.reshape(-1)

        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(output_addr, output_size, np.int32)
        print(output==output_golden)
        # assert output.shape==output_golden.shape, f"{output.shape=}, {output_golden.shape=}"
        assert np.array_equal(output,output_golden), f"\n{output=}, \n{output_golden=}"


    def test_pim_transfer(self):
        input_addr = self.memory_space.get_base_of("pim_output_reg_buffer")
        input_size = 16
        transfer_mask_addr = self.memory_space.get_base_of("local_memory")
        output_addr = transfer_mask_addr + input_size // 8
        inst_list = [
            # set general register
            self.inst_util.general_li(0, input_addr), # input addr
            self.inst_util.general_li(1, input_size), # input size
            self.inst_util.general_li(2, transfer_mask_addr), # transfer_mask_addr
            self.inst_util.general_li(3, output_addr), # output addr

            # set special register
            self.inst_util.special_li(SpecialReg.OUTPUT_BIT_WIDTH, 32),

            self.inst_util.pim_transfer(0, 1, 2, 3)
        ]
        
        input_data = np.arange(input_size, dtype=np.int32)
        mask_data = np.array([0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0], dtype=np.int8).reshape(-1,8)
        mask_data_bits = tensor_bits_to_int8(mask_data)
        golden_data = np.array([1,3,5,7,8,10,12,14], dtype=np.int32)

        self.simulator.memory_space.write(input_data, input_addr, input_data.size * 4)
        self.simulator.memory_space.write(mask_data_bits, transfer_mask_addr, mask_data_bits.size)
        status = self.simulator.run_code(inst_list)
        assert status==self.simulator.FINISH
        output_data = self.simulator.memory_space.read_as(output_addr, mask_data.sum() * 4, np.int32)
        assert np.array_equal(output_data, golden_data)

    def test_pimcompute_bit_sparse_multi_group_threshold2(self):
        """
        y = xA
        x: int8, shape=[2, 4], memory=local, addr=INPUT_BUFFER_BASE, size=2*4=8
        A: int8, shape=[4, 2, 32], dtype=bit, memory=macro, addr=MACRO_BASE, size = 4*2*32 / 8 = 32
        y: int32, shape=[2, 32], memory=local, addr=OUTPUT_BUFFER_BASE, size=2*32*4 = 256
        """
        local_base = self.simulator.memory_space.get_base_of("local_memory")
        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")
        meta_base = self.simulator.memory_space.get_base_of("pim_meta_data_reg_buffer")

        input_addr = input_buffer_base
        input_size = 8
        output_addr = output_buffer_base
        output_size = 256
        weight_addr = macro_base
        weight_size = 32
        
        # prepare data
        # input = np.arange(8).reshape(2,4).astype(np.int8)
        input = np.ones((8,)).reshape(2,4).astype(np.int8)
        weight = generate_valid_weight([16, 4, 1, 1], threshold=2)
        from types import SimpleNamespace
        op_cfg = SimpleNamespace(
            out_channel = 16
        )
        cim_cfg = SimpleNamespace(
            bits_column=self.macro_config.n_bcol,
            n_macro=self.macro_config.n_macro // 2,   # n_macro_per_group
            n_group=2
        )
        
        bit_sparse_weight, info, fold = weight_transform_group(weight, cim_cfg, op_cfg)
        outsum_mask = parse_out_mask(fold[0])
        transfer_mask = outsum_mask_to_transfer_mask(outsum_mask)
        print(f"{weight=}")
        print(f"{info=}")
        print(f"{fold=}")
        print(f"{outsum_mask=}")
        print(f"{transfer_mask=}")
        # import pdb; pdb.set_trace()
        outsum_mask = np.array(outsum_mask, dtype=np.int8)
        transfer_mask = np.array(transfer_mask, dtype=np.int8)
        assert outsum_mask.size % 8 == 0
        outsum_mask = tensor_bits_to_int8(outsum_mask.reshape(-1, 8))
        transfer_mask = tensor_bits_to_int8(transfer_mask.reshape(-1, 8))
        outsum_base = local_base
        transfer_base = outsum_base + outsum_mask.size
        transfer_output_addr = transfer_base + transfer_mask.size

        self.simulator.memory_space.write(input, input_addr, input.size)
        self.simulator.memory_space.write(bit_sparse_weight, weight_addr, bit_sparse_weight.size)
        self.simulator.memory_space.write(info, meta_base, info.size)
        self.simulator.memory_space.write(outsum_mask, outsum_base, outsum_mask.size)
        self.simulator.memory_space.write(transfer_mask, transfer_base, transfer_mask.size)

        inst_list = [
            # set general register
            self.inst_util.general_li(0, input_addr), # input addr
            self.inst_util.general_li(1, input_size // 2), # input size
            self.inst_util.general_li(2, 0), # activate row
            self.inst_util.general_li(3, output_addr), # output addr

            # set special register
            self.inst_util.special_li(SpecialReg.INPUT_BIT_WIDTH, 8),
            self.inst_util.special_li(SpecialReg.WEIGHT_BIT_WIDTH, 1),
            self.inst_util.special_li(SpecialReg.OUTPUT_BIT_WIDTH, 32),
            self.inst_util.special_li(SpecialReg.ACTIVATION_ELEMENT_COL_NUM, 32),
            
            self.inst_util.special_li(SpecialReg.GROUP_SIZE, 2),
            self.inst_util.special_li(SpecialReg.ACTIVATION_GROUP_NUM, 2),
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, 4),

            self.inst_util.special_li(SpecialReg.BIT_SPARSE_META_ADDR, meta_base),

            self.inst_util.pimcompute(
                value_sparse=0, 
                bit_sparse=1, 
                group=1, 
                group_input_mode=0, 
                accumulate=1, 
                rs1=0, 
                rs2=1, 
                rs3=2, 
                rd=3
            ),

            # outsum-move
            self.inst_util.general_li(4, outsum_base), # output_mask_addr
            self.inst_util.general_li(5, 32), # out_n
            self.inst_util.pim_output(
                outsum_move=0, 
                outsum=1 ,
                out_n=5, 
                output_mask_addr=4, 
                output_addr=3
            ),

            # pim-transfer for group 0
            self.inst_util.general_li(6, transfer_base), # transfer_mask_addr
            self.inst_util.general_li(7, transfer_output_addr), # output addr
            self.inst_util.pim_transfer(
                src_addr=3, 
                src_size=5, 
                transfer_mask=6, 
                dst_addr=7
            ),

            # pim-transfer for group 1
            self.inst_util.general_li(3, output_addr + 128), # src_addr
            self.inst_util.general_li(6, transfer_base), # transfer_mask_addr
            self.inst_util.general_li(7, transfer_output_addr + 64), # output addr
            self.inst_util.pim_transfer(
                src_addr=3, 
                src_size=5, 
                transfer_mask=6, 
                dst_addr=7
            )
        ]

        output_list = []
        for group_id in range(2):
            input_group = input[group_id]
            weight_group = weight.reshape(weight.shape[0], -1)
            weight_group = np.transpose(weight_group, [1,0])
            # import pdb; pdb.set_trace()
            output_group = np.dot(input_group.astype(np.int32), weight_group.astype(np.int32))
            output_list.append(output_group)
        output_golden = np.concatenate(output_list)
        output_golden = output_golden.reshape(-1)

        status = self.simulator.run_code(inst_list)

        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(transfer_output_addr, output_size//2, np.int32)
        print(output==output_golden)
        # assert output.shape==output_golden.shape, f"{output.shape=}, {output_golden.shape=}"
        assert np.array_equal(output,output_golden), f"\n{output=}, \n{output_golden=}"
if __name__=="__main__":
    TestSimulatorPIMComputeBitSparse.setup_class()
    test_simulator = TestSimulatorPIMComputeBitSparse()
    test_simulator.setup_method()
    test_simulator.test_pimcompute_bit_sparse_multi_group_threshold2()