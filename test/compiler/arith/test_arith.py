import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
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

class TestArith:

    @classmethod
    def setup_class(cls):
        cls.inst_util = InstUtil()
        cls.memory_space = init_memory_space()
        cls.macro_config = init_macro_config()
        cls.simulator = Simulator(cls.memory_space , cls.macro_config)

    def setup_method(self):
        self.simulator.clear()

    def test_basic_arith(self):
        input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "case0.cim")
        output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".result")
        os.makedirs(output_folder, exist_ok=True)
        cmd = f"bash compile.sh isa {input_path} {output_folder}"
        result = subprocess.run(cmd.split(" "), capture_output=True, text=True)
        assert result.returncode==0

        output_path = os.path.join(output_folder, "final_code.json")
        with open(output_path, "r") as f:
            code = json.load(f)
        
        status = self.simulator.run_code(code)
        assert status==self.simulator.FINISH

        # check result
        print_record = self.simulator.print_record
        assert len(print_record)==4, f"{len(print_record)=}"
        assert print_record[0]==10, f"{print_record[0]=}"
        assert print_record[1]==6, f"{print_record[1]=}"
        assert print_record[2]==16, f"{print_record[2]=}"
        assert print_record[3]==4, f"{print_record[3]=}"

if __name__=="__main__":
    TestArith.setup_class()
    test_arith = TestArith()
    test_arith.setup_method()
    test_arith.test_basic_arith()