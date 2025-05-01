import json
import os
import subprocess

import numpy as np
import pytest

from cim_compiler.simulator.macro_utils import MacroConfig
from cim_compiler.simulator.mask_utils import MaskConfig
from cim_compiler.simulator.simulator import Memory, MemorySpace, Simulator, SpecialReg
from cim_compiler.simulator.inst import CIMFlowParser
from jinja2 import Environment, FileSystemLoader, StrictUndefined
import tempfile

def init_macro_config():
    macro_config = MacroConfig(n_group=1, n_macro=2, n_row=4, n_comp=4, n_bcol=16)
    return macro_config


def init_mask_config():
    mask_config = MaskConfig(n_from=8, n_to=4)  # 8选4
    return mask_config

def fill_template(src_path, dst_path):

    src_folder, src_file = os.path.split(src_path)

    # 创建 Jinja2 环境和加载器
    env = Environment(
        loader=FileSystemLoader([
            src_folder, 
            os.environ["CIM_COMPILER_BASE"],
            os.environ.get(os.environ["CIM_COMPILER_BASE"], "cim_compiler")
        ]),
        undefined=StrictUndefined
    )

    # 加载模板
    template = env.get_template(src_file)

    context = {}

    # 渲染模板
    output = template.render(context)

    with open(dst_path, "w") as f:
        f.write(output)

class TestBase:

    @classmethod
    def setup_class(cls):
        cls.config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.json"
        )
        cls.memory_space = MemorySpace.from_memory_config(cls.config_path)
        cls.macro_config = init_macro_config()
        cls.mask_config = init_mask_config()
        cls.simulator = Simulator(cls.memory_space, cls.macro_config, cls.mask_config)

    def setup_method(self):
        self.simulator.clear()

    def run_test(self, casename):
        case_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), casename)
        assert os.path.exists(case_dir), f"{case_dir} not exists"
        assert os.path.isdir(case_dir), f"{case_dir} is not a directory"

        # Prepare path
        origin_input_path = os.path.join(case_dir, "code.cim")
        golden_path = os.path.join(case_dir, "golden.txt")
        assert os.path.exists(origin_input_path), f"{origin_input_path} not exists"
        assert os.path.exists(golden_path), f"{golden_path} not exists"

        with tempfile.TemporaryDirectory() as output_folder:

            # fill template
            input_path = os.path.join(output_folder, "code.cim")
            fill_template(origin_input_path, input_path)

            # run compiler
            subprocess.run([
                "cim-compiler", "compile",
                "--input-file", input_path,
                "--output-dir", output_folder,
                "--config-file", self.config_path
            ], check=True)

            # get output code
            _, code = CIMFlowParser().parse_file(
                os.path.join(output_folder, "final_code.json")
            )

            # run code in simulator
            status, stats, _ = self.simulator.run_code(code)
            assert status == self.simulator.FINISH

            # check result
            print_record = self.simulator.print_record
            with open(golden_path, "r") as f:
                golden = f.read().split()
                golden = [int(x.strip()) for x in golden]
            assert print_record == golden, f"{print_record=}, {golden=}"

    def run_test_with_image(self, casename):
        case_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), casename
        )
        assert os.path.exists(case_dir), f"{case_dir} not exists"
        assert os.path.isdir(case_dir), f"{case_dir} is not a directory"

        # Prepare path
        origin_input_path = os.path.join(case_dir, "code.cim")
        test_helper_path = os.path.join(case_dir, "helper.py")
        assert os.path.exists(origin_input_path), f"{origin_input_path} not exists"
        assert os.path.exists(test_helper_path), f"{test_helper_path} not exists"

        with tempfile.TemporaryDirectory() as output_folder:

            # fill template
            input_path = os.path.join(output_folder, "code.cim")
            fill_template(origin_input_path, input_path)

            # Get helper
            with open(test_helper_path, "r") as file:
                code = file.read()
            local_namespace = {}
            exec(code, {}, local_namespace)
            Helper = local_namespace["TestHelper"]
            helper = Helper()

            # load image
            image = helper.get_image()
            global_memory = self.simulator.memory_space.get_memory_by_name("global")
            global_memory.write(image, global_memory.offset, len(image))

            # run compiler
            subprocess.run([
                "cim-compiler", "compile",
                "--input-file", input_path,
                "--output-dir", output_folder,
                "--config-file", self.config_path
            ], check=True)

            # get output code
            _, code = CIMFlowParser().parse_file(
                os.path.join(output_folder, "final_code.json")
            )

            # run code in simulator
            status, stats, _ = self.simulator.run_code(code)
            assert status == self.simulator.FINISH

            # check result
            helper.check_image(global_memory.read_all())

if __name__ == "__main__":
    TestForLoop.setup_class()
    test_for_loop = TestForLoop()
    test_for_loop.setup_class()
    test_for_loop.setup_method()
    test_for_loop.test_control_flow("if")
