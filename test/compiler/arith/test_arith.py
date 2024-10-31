import json
import os
import subprocess
from test.simulator.utils import InstUtil

import numpy as np
import pytest

from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
from simulator.simulator import Memory, MemorySpace, Simulator, SpecialReg


def init_macro_config():
    macro_config = MacroConfig(n_macro=2, n_row=4, n_comp=4, n_bcol=16)
    return macro_config


def init_mask_config():
    mask_config = MaskConfig(n_from=8, n_to=4)  # 8选4
    return mask_config


class TestArith:

    @classmethod
    def setup_class(cls):
        cls.inst_util = InstUtil()
        cls.config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.json"
        )
        cls.memory_space = MemorySpace.from_memory_config(cls.config_path)
        cls.macro_config = init_macro_config()
        cls.mask_config = init_mask_config()
        cls.simulator = Simulator(cls.memory_space, cls.macro_config, cls.mask_config)

    def setup_method(self):
        self.simulator.clear()

    def test_basic_arith(self):
        input_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "case0.cim"
        )
        output_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".result"
        )
        os.makedirs(output_folder, exist_ok=True)
        cmd = f"bash compile.sh isa {input_path} {output_folder} {self.config_path}"
        result = subprocess.run(cmd.split(" "), capture_output=True, text=True)
        assert result.returncode == 0

        output_path = os.path.join(output_folder, "final_code.json")
        with open(output_path, "r") as f:
            code = json.load(f)

        status = self.simulator.run_code(code)
        assert status == self.simulator.FINISH

        # check result
        print_record = self.simulator.print_record
        assert len(print_record) == 6, f"{len(print_record)=}"
        assert print_record[0] == 10, f"{print_record[0]=}"
        assert print_record[1] == 6, f"{print_record[1]=}"
        assert print_record[2] == 16, f"{print_record[2]=}"
        assert print_record[3] == 4, f"{print_record[3]=}"
        assert print_record[4] == 2, f"{print_record[4]=}"
        assert print_record[5] == 2, f"{print_record[5]=}"


if __name__ == "__main__":
    TestArith.setup_class()
    test_arith = TestArith()
    test_arith.setup_method()
    test_arith.test_basic_arith()
