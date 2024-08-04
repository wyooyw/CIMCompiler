import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
import numpy as np
from utils.df_layout import tensor_bits_to_int8
from utils.bit_sparse_weight_transform import generate_valid_weight, weight_transform, weight_transform_group, parse_out_mask, outsum_mask_to_transfer_mask
from utils.bit_value_sparse_weight_transform import convert_value_bit_sparse_conv2d_weight
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
    meta = Memory("pim_meta_data_reg_buffer", "rf", macro.end, 1024)
    mask = Memory("pim_mask_data_reg_buffer", "rf", meta.end, 1024)
    memory_space.add_memory(global_memory)
    memory_space.add_memory(local_memory)
    memory_space.add_memory(input_buffer)
    memory_space.add_memory(output_buffer)
    memory_space.add_memory(macro)
    memory_space.add_memory(meta)
    memory_space.add_memory(mask)
    return memory_space

class TestSimulatorPIMComputeValueBitSparse:

    @classmethod
    def setup_class(cls):
        cls.inst_util = InstUtil()
        # cls.config_path = "/home/wangyiou/project/cim_compiler_frontend/playground/config/config.json"
        
        cls.macro_config = init_macro_config()
        cls.mask_config = init_mask_config()
        cls.memory_space = init_memory_space(cls.macro_config)
        cls.simulator = Simulator(cls.memory_space , cls.macro_config, cls.mask_config, mask_memory_name="pim_mask_data_reg_buffer")


    def setup_method(self):
        self.simulator.clear()

    def test_pimcompute_value_bit_sparse_multi_group_threshold2(self):
        """
        y = xA
        x: int8, shape=[2, 8], memory=local, addr=INPUT_BUFFER_BASE, size=2*8=16
        A: int8, shape=[8, 2, 16], dtype=bit, memory=macro, addr=MACRO_BASE, size = 8*2*16 / 8 = 32
        mask: [
            [1,1,1,1,0,0,0,0], for macro 0
            [0,1,1,1,1,0,0,0]  for macro 1
        ]
        y: int32, shape=[2, 16], memory=local, addr=OUTPUT_BUFFER_BASE, size=2*16*4 = 256
        """
        local_base = self.simulator.memory_space.get_base_of("local_memory")
        input_buffer_base = self.simulator.memory_space.get_base_of("pim_input_reg_buffer")
        output_buffer_base = self.simulator.memory_space.get_base_of("pim_output_reg_buffer")
        macro_base = self.simulator.memory_space.get_base_of("macro")
        meta_base = self.simulator.memory_space.get_base_of("pim_meta_data_reg_buffer")
        mask_base = self.simulator.memory_space.get_base_of("pim_mask_data_reg_buffer")

        input_addr = input_buffer_base
        input_size = 16
        output_addr = output_buffer_base
        output_size = 256
        weight_addr = macro_base
        weight_size = 32
        mask_addr = mask_base
        # mask = mask = np.array([
        #     [1,1,1,1,0,0,0,0],
        #     [0,1,1,1,1,0,0,0]
        # ], dtype=bool)
        # mask_bits = tensor_bits_to_int8(mask)
        
        # prepare data
        # input = np.arange(8).reshape(2,4).astype(np.int8)
        input = np.arange(0,16).reshape(2,8).astype(np.int8)
        weight = generate_valid_weight([16, 1, 1, 8], threshold=2)
        print(weight[:8,0,0,:].T)
        
        mask_ = np.zeros([16, 1, 1, 8], dtype=np.int8)
        mask_[0:8,0,0,0:4] = 1
        mask_[8:16,0,0,1:5] = 1
        weight = weight * mask_

        macro_config = {
            "n_row": self.macro_config.n_row,
            "n_bcol": self.macro_config.n_bcol,
            # "n_vcol": 2,
            "n_group": 2,
            "n_macro": self.macro_config.n_macro,
            "n_comp": self.macro_config.n_comp,
            "n_value_sparse_from": self.mask_config.n_from,
            "n_value_sparse_to": self.mask_config.n_to
        }
        
        value_bit_sparse_result = convert_value_bit_sparse_conv2d_weight(weight, macro_config)
        value_sparse_result = value_bit_sparse_result["value_sparse_result"]
        bit_sparse_result = value_bit_sparse_result["bit_sparse_result"]

        # save data into memory
        value_bit_sparse_weight = value_bit_sparse_result["value_bit_sparse_weight"]

        # value sparse
        value_sparse_mask = value_sparse_result["mask"]
        assert value_sparse_mask.size % 8 == 0
        value_sparse_mask = tensor_bits_to_int8(value_sparse_mask.reshape(-1, 8))
        value_sparse_mask_base = mask_base
        
        # bit sparse
        bis_sparse_meta = bit_sparse_result["meta"]
        outsum_mask = bit_sparse_result["outsum_mask"]
        transfer_mask = bit_sparse_result["transfer_mask"]
        assert outsum_mask.size % 8 == 0
        assert transfer_mask.size % 8 == 0
        outsum_mask = tensor_bits_to_int8(outsum_mask.reshape(-1, 8))
        transfer_mask = tensor_bits_to_int8(transfer_mask.reshape(-1, 8))

        bis_sparse_meta_base = meta_base
        outsum_base = local_base
        transfer_base = outsum_base + outsum_mask.size

        transfer_output_addr = transfer_base + transfer_mask.size

        self.simulator.memory_space.write(input, input_addr, input.size)
        self.simulator.memory_space.write(value_bit_sparse_weight, weight_addr, value_bit_sparse_weight.size)
        # value sparse
        self.simulator.memory_space.write(value_sparse_mask, value_sparse_mask_base, value_sparse_mask.size)
        # bit sparse
        self.simulator.memory_space.write(bis_sparse_meta, bis_sparse_meta_base, bis_sparse_meta.size)
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
            self.inst_util.special_li(SpecialReg.GROUP_INPUT_STEP, input_size // 2),

            self.inst_util.special_li(SpecialReg.BIT_SPARSE_META_ADDR, bis_sparse_meta_base),
            self.inst_util.special_li(SpecialReg.VALUE_SPARSE_MASK_ADDR, value_sparse_mask_base),

            self.inst_util.pimcompute(
                value_sparse=1, 
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
        # import pdb; pdb.set_trace()

        output_list = []
        for group_id in range(2):
            input_group = input[group_id]
            weight_group = weight.reshape(weight.shape[0], -1)
            weight_group = np.transpose(weight_group, [1,0])
            # import pdb; pdb.set_trace()
            print(f"{input_group.shape=}, {weight_group.shape=}")
            output_group = np.dot(input_group.astype(np.int32), weight_group.astype(np.int32))
            output_list.append(output_group)
        output_golden = np.concatenate(output_list)
        output_golden = output_golden.reshape(-1)
        
        status = self.simulator.run_code(inst_list)
        import pdb; pdb.set_trace()
        assert status==self.simulator.FINISH
        
        output = self.simulator.memory_space.read_as(transfer_output_addr, output_size//2, np.int32)
        print(output)
        print(output==output_golden)
        # assert output.shape==output_golden.shape, f"{output.shape=}, {output_golden.shape=}"
        assert np.array_equal(output,output_golden), f"\n{output=}, \n{output_golden=}"
if __name__=="__main__":
    TestSimulatorPIMComputeValueBitSparse.setup_class()
    test_simulator = TestSimulatorPIMComputeValueBitSparse()
    test_simulator.setup_method()
    test_simulator.test_pimcompute_value_bit_sparse_multi_group_threshold2()