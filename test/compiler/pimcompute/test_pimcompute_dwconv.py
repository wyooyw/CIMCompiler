import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
import numpy as np
import subprocess
import os
import json
from functools import partial
import numpy as np
from utils.predict_pimcompute_count import predict_pimcompute_count_for_conv2d_dense
def debug_hook(simulator, helper):
    input_rf_base = simulator.memory_space.get_base_of("pim_input_reg_buffer")
    output_rf_base = simulator.memory_space.get_base_of("pim_output_reg_buffer")
    macro_util = simulator.macro_util
    def see_macro(row):
        macro_data = macro_util.get_macro_data(
            activate_row=row,
            data_type=np.int8,
            group_num=4,
            activate_element_row_num=16,
            activate_element_col_num=32,
            activate_group_num=4
        )
        print(f"{macro_data.shape=} (n_comp, n_group, n_vcol_per_group)")
        print(macro_data)
    def see_input_rf(double_buffer_id, group):
        offset = input_rf_base + double_buffer_id * 512
        group_in_buffer_size = 128
        group_offset = offset + group * group_in_buffer_size
        input_for_group = simulator.memory_space.read_as(group_offset, group_in_buffer_size, np.int8)
        memory_type = simulator.memory_space.get_memory_by_address(group_offset).name
        print(f"see begin:{group_offset}, size:{group_in_buffer_size}, type:{memory_type}")
        print(f"{input_for_group.shape=}")
        print(input_for_group)
    def see_output_rf(group):
        offset = output_rf_base
        group_out_buffer_size = 128 # 32 * 4

        group_offset = offset + group * group_out_buffer_size
        output_per_group = simulator.memory_space.read_as(group_offset, group_out_buffer_size, np.int32)

        memory_type = simulator.memory_space.get_memory_by_address(group_offset).name
        print(f"see begin:{group_offset}, size:{group_out_buffer_size}, type:{memory_type}")
        print(f"{output_per_group.shape=}")
        print(output_per_group)

    def see_golden(row, col):
        input = helper.input_data[row:row+3,col:col+3,:].reshape(-1,1)
        weight = helper.weight_data
        print(f"{weight.shape=}, {input.shape=}")
        golden = np.matmul(weight.astype(np.int32), input.astype(np.int32))
        print(f"{row=}, {col=}, {golden.shape=}:")
        print(golden.reshape(-1))

    def see_output_memory():
        output_memory = simulator.memory_space.get_memory_by_name("output_memory")
        output = simulator.memory_space.read_as(output_memory.offset, 6*6*32*4, np.int32)
        output = output.reshape(6,6,32)
        print(output)

    def see_output_and_golden():
        golden = helper._calculate_golden()

        output_memory = simulator.memory_space.get_memory_by_name("output_memory")
        output = simulator.memory_space.read_as(output_memory.offset, 6*6*32*4, np.int32)
        output = output.reshape(6,6,32)

        print("output:\n", output)
        print("golden:\n", golden)
        print(f"{np.array_equal(output, golden)=}")

    # import pdb; pdb.set_trace()
    pass

class TestPIMComputeValueSparse:

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

    @pytest.mark.parametrize('casename',[
        # quantify
        'dense/dense_dwconv_group_quantify' ,
        ])
    @pytest.mark.parametrize('op_config',[
        {"out_channel":16, "in_channel": 16, "ker_size": 3, "in_hw": 8, "out_hw": 6},
        {"out_channel":32, "in_channel": 32, "ker_size": 3, "in_hw": 8, "out_hw": 6},
        {"out_channel":64, "in_channel": 64, "ker_size": 3, "in_hw": 8, "out_hw": 6},
        {"out_channel":384, "in_channel": 384, "ker_size": 3, "in_hw": 8, "out_hw": 6}, 
        {"out_channel":16, "in_channel": 16, "ker_size": 3, "in_hw": 8, "out_hw": 4, "padding":1, "stride":2},
        {"out_channel":32, "in_channel": 32, "ker_size": 3, "in_hw": 8, "out_hw": 4, "padding":1, "stride":2},
        {"out_channel":64, "in_channel": 64, "ker_size": 3, "in_hw": 8, "out_hw": 4, "padding":1, "stride":2},
        {"out_channel":384, "in_channel": 384, "ker_size": 3, "in_hw": 8, "out_hw": 4, "padding":1, "stride":2},      
        ])
    def test_pim_compute(self, casename, op_config):
        case_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), casename)
        assert os.path.exists(case_dir), f"{case_dir} not exists"
        assert os.path.isdir(case_dir), f"{case_dir} is not a directory"

        # Prepare path
        input_template_path = os.path.join(case_dir, "code_template.cim")
        input_path = os.path.join(case_dir, "code.cim")
        test_helper_path = os.path.join(case_dir, "helper.py")
        assert os.path.exists(input_template_path), f"{input_template_path} not exists"
        assert os.path.exists(test_helper_path), f"{test_helper_path} not exists"
        
        output_folder = os.path.join(case_dir, ".result")
        os.makedirs(output_folder, exist_ok=True)

        # If there is already files in .result, remove them, to make sure not execute old codes.
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # return
        # Get helper
        with open(test_helper_path, 'r') as file:
            code = file.read()
        local_namespace = {}
        exec(code, {}, local_namespace)
        Helper = local_namespace["TestHelper"]
        helper = Helper(op_config)

        # load image
        image = helper.get_image(self.simulator)
        global_memory_base = self.simulator.memory_space.get_base_of("global")
        self.simulator.memory_space.write(image, global_memory_base, len(image))

        # fill code template
        helper.fill_template(input_template_path, input_path, self.simulator)
        # return

        # register debug hook
        self.simulator.debug_hook = partial(debug_hook, helper=helper)

        # run compiler
        cmd = f"bash compile.sh isa {input_path} {output_folder} {self.config_path}"
        result = subprocess.run(cmd.split(" "), capture_output=True, text=True)
        print('输出:', result.stdout)
        print('错误:', result.stderr)
        assert result.returncode==0
        # return
        # get output code
        output_path = os.path.join(output_folder, "final_code.json")
        with open(output_path, "r") as f:
            code = json.load(f)

        # run code in simulator

        pimcompute_count = predict_pimcompute_count_for_conv2d_dense(self.macro_config, op_config, group_size=16)
        status = self.simulator.run_code(code, total_pim_compute_count = pimcompute_count)
        assert status==self.simulator.FINISH

        # check result
        # print_record = self.simulator.print_record
        helper.check_image(self.simulator.memory_space)

    # def test_memory_with_image(self):
    #     pass

    # @pytest.mark.parametrize('casename',[
    #     'print_in_loop','count_in_loop','accumulate_in_loop', 
    #     'print_in_double_loop', 'count_in_double_loop', 'accumulate_in_double_loop',
    #     'fibonacci'
    #     ])
    # def test_memory(self, casename):
    #     case_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), casename)
    #     assert os.path.exists(case_dir), f"{case_dir} not exists"
    #     assert os.path.isdir(case_dir), f"{case_dir} is not a directory"

    #     # Prepare path
    #     input_path = os.path.join(case_dir, "code.cim")
    #     golden_path = os.path.join(case_dir, "golden.txt")
    #     memory_image_path = os.path.join(case_dir, "memory_image.txt")
    #     assert os.path.exists(input_path), f"{input_path} not exists"
    #     assert os.path.exists(golden_path), f"{golden_path} not exists"
    #     assert os.path.exists(memory_image_path), f"{golden_path} not exists"
        
    #     output_folder = os.path.join(case_dir, ".result")
    #     os.makedirs(output_folder, exist_ok=True)

    #     # run compiler
    #     cmd = f"bash compile.sh isa {input_path} {output_folder}"
    #     result = subprocess.run(cmd.split(" "), capture_output=True, text=True)
    #     print('输出:', result.stdout)
    #     print('错误:', result.stderr)
    #     assert result.returncode==0

    #     # get output code
    #     output_path = os.path.join(output_folder, "final_code.json")
    #     with open(output_path, "r") as f:
    #         code = json.load(f)

    #     # run code in simulator
    #     self.simulator.load_memory_image(memory_image_path)
    #     status = self.simulator.run_code(code)
    #     assert status==self.simulator.FINISH

    #     # check result
    #     print_record = self.simulator.print_record
    #     with open(golden_path, "r") as f:
    #         golden = f.read().split()
    #         golden = [int(x.strip()) for x in golden]
    #     assert print_record==golden, f"{print_record=}, {golden=}"

if __name__=="__main__":
    TestPIMComputeValueSparse.setup_class()
    tester = TestPIMComputeValueSparse()
    tester.setup_method()
    tester.test_pim_compute('dense/dense_dwconv_group_quantify', 
        {"out_channel":4, "in_channel": 4, 
        "ker_size": 3, "in_hw": 4, "out_hw": 2, 
        "padding": 1,
        "stride":2,
        "input_buffer_size_per_group": 128
        }
    )