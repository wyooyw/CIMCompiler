import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
import numpy as np
import subprocess
import os
import json
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

def init_mask_config():
    mask_config = MaskConfig(n_from=8, n_to=4) # 8选4
    return mask_config

class TestPIMCompute:

    @classmethod
    def setup_class(cls):
        cls.inst_util = InstUtil()
        cls.memory_space = init_memory_space()
        cls.macro_config = init_macro_config()
        cls.mask_config = init_mask_config()
        cls.simulator = Simulator(cls.memory_space , cls.macro_config, cls.mask_config)

    def setup_method(self):
        self.simulator.clear()

    @pytest.mark.parametrize('casename',[
        'dense'
        ])
    def test_pim_compute(self, casename):
        case_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "print", casename)
        assert os.path.exists(case_dir), f"{case_dir} not exists"
        assert os.path.isdir(case_dir), f"{case_dir} is not a directory"

        # Prepare path
        input_path = os.path.join(case_dir, "code.cim")
        test_helper_path = os.path.join(case_dir, "helper.py")
        assert os.path.exists(input_path), f"{input_path} not exists"
        assert os.path.exists(test_helper_path), f"{test_helper_path} not exists"
        
        output_folder = os.path.join(case_dir, ".result")
        os.makedirs(output_folder, exist_ok=True)

        # Get helper
        with open(test_helper_path, 'r') as file:
            code = file.read()
        local_namespace = {}
        exec(code, {}, local_namespace)
        Helper = local_namespace["TestHelper"]
        helper = Helper()

        # load image
        image = helper.get_image()
        global_memory = self.simulator.memory_space.get_base_of("global_memory")
        global_memory.write(image, 0, len(image))

        # run compiler
        cmd = f"bash compile.sh isa {input_path} {output_folder}"
        result = subprocess.run(cmd.split(" "), capture_output=True, text=True)
        print('输出:', result.stdout)
        print('错误:', result.stderr)
        assert result.returncode==0

        # get output code
        output_path = os.path.join(output_folder, "final_code.json")
        with open(output_path, "r") as f:
            code = json.load(f)

        # run code in simulator
        status = self.simulator.run_code(code)
        assert status==self.simulator.FINISH

        # check result
        print_record = self.simulator.print_record
        helper.check_image(global_memory.read_all())

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
    TestPIMCompute.setup_class()
    test_simd = TestPIMCompute()
    test_simd.setup_method()
    test_simd.test_pim_compute("dense")