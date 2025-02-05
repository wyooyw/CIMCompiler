import json
import os
import subprocess
from functools import partial
from test.simulator.utils import InstUtil

import numpy as np
import pytest

from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
from simulator.simulator import Memory, MemorySpace, Simulator, SpecialReg
from utils.predict_pimcompute_count import predict_pimcompute_count_for_conv2d_dense
from simulator.inst import CIMFlowParser

import tempfile

class TestBase:

    @classmethod
    def setup_class(cls):
        cls.config_path = (
            os.path.join(os.environ["CIM_COMPILER_BASE"], "config/config.json")
        )
        cls.simulator = Simulator.from_config(cls.config_path)

    def setup_method(self):
        self.simulator.clear()

    def run_op_test(self, casename, op_config, pimcompute_count=None):
        op_base_dir = os.path.join(os.environ["CIM_COMPILER_BASE"], "op")
        case_dir = os.path.join(op_base_dir, casename)
        code_template_path = os.path.join(case_dir, "code_template.cim")
        test_helper_path = os.path.join(case_dir, "helper.py")
        assert os.path.exists(code_template_path), f"{code_template_path} not exists"
        assert os.path.exists(test_helper_path), f"{test_helper_path} not exists"

        with tempfile.TemporaryDirectory() as output_folder:

            code_path = os.path.join(output_folder, "code.cim")

            # Get helper
            with open(test_helper_path, "r") as file:
                code = file.read()
            local_namespace = {}
            exec(code, {}, local_namespace)
            Helper = local_namespace["TestHelper"]
            helper = Helper(op_config)

            # load image
            image = helper.get_image(self.simulator)
            with open(os.path.join(output_folder, "global_image"), "wb") as file:
                file.write(image)
            with open(os.path.join(output_folder, "global_image"), "rb") as file:
                image = bytearray(file.read())
            global_memory_base = self.simulator.memory_space.get_base_of("global")
            self.simulator.memory_space.write(image, global_memory_base, len(image))

            # fill code template
            helper.fill_template(code_template_path, code_path, self.simulator)
            # return

            # register debug hook
            # self.simulator.debug_hook = partial(debug_hook, helper=helper)

            # run compiler
            cmd = f"bash compile.sh isa {code_path} {output_folder} {self.config_path}"
            result = subprocess.run(cmd.split(" "), text=True)
            # print("输出:", result.stdout)
            # print("错误:", result.stderr)
            assert result.returncode == 0
            # return
            # get output code
            output_path = os.path.join(output_folder, "final_code.json")
            with open(output_path, "r") as f:
                code = json.load(f)
            cimflow_parser = CIMFlowParser()
            code = cimflow_parser.parse(code)

            # run code in simulator
            # pimcompute_count = predict_pimcompute_count_for_conv2d_dense(
            #     self.simulator.macro_config, op_config, group_size=16
            # )
            status, stats, flat = self.simulator.run_code(
                code, total_pim_compute_count=pimcompute_count
            )
            assert status == self.simulator.FINISH

            # check result
            # print_record = self.simulator.print_record
            stats.dump(output_folder)
            flat.dump(output_folder)
            helper.check_image(self.simulator.memory_space)

            # run flat code
            flat_code = flat.get_flat_code()
            self.simulator.clear()
            self.simulator.memory_space.write(image, global_memory_base, len(image))
            # self.simulator._read_reg_value_directly = True
            status, stats, flat = self.simulator.run_code(
                flat_code, total_pim_compute_count=pimcompute_count, record_flat=False
            )
            assert status == self.simulator.FINISH
            stats.dump(output_folder, prefix="flat_")
            helper.check_image(self.simulator.memory_space)