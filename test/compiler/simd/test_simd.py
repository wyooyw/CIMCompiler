import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
import numpy as np
import subprocess
import os
import json

def init_macro_config():
    macro_config = MacroConfig(n_macro=2, n_row=4, n_comp=4, n_bcol=16)
    return macro_config

def init_mask_config():
    mask_config = MaskConfig(n_from=8, n_to=4) # 8选4
    return mask_config

class TestSIMD:

    @classmethod
    def setup_class(cls):
        cls.inst_util = InstUtil()
        cls.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        cls.memory_space = MemorySpace.from_memory_config(cls.config_path)
        cls.macro_config = init_macro_config()
        cls.mask_config = init_mask_config()
        cls.simulator = Simulator(cls.memory_space , cls.macro_config, cls.mask_config)

    def setup_method(self):
        self.simulator.clear()

    @pytest.mark.parametrize('casename',[
        'vvadd'
        ])
    def test_simd(self, casename):
        case_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), casename)
        assert os.path.exists(case_dir), f"{case_dir} not exists"
        assert os.path.isdir(case_dir), f"{case_dir} is not a directory"

        # Prepare path
        input_path = os.path.join(case_dir, "code.cim")
        golden_path = os.path.join(case_dir, "golden.txt")
        assert os.path.exists(input_path), f"{input_path} not exists"
        assert os.path.exists(golden_path), f"{golden_path} not exists"
        
        output_folder = os.path.join(case_dir, ".result")
        os.makedirs(output_folder, exist_ok=True)

        # run compiler
        cmd = f"bash compile.sh isa {input_path} {output_folder} {self.config_path}"
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
        with open(golden_path, "r") as f:
            golden = f.read().split()
            golden = [int(x.strip()) for x in golden]
        assert print_record==golden, f"{print_record=}, {golden=}"

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
    TestSIMD.setup_class()
    test_simd = TestSIMD()
    test_simd.setup_method()
    test_simd.test_simd("vvadd")