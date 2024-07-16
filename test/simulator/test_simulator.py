import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator
from simulator.macro_utils import MacroConfig
import numpy as np
def init_memory_space():
    memory_space = MemorySpace()
    global_memory = Memory("global_memory", "dram", 0, 128)
    local_memory = Memory("local_memory", "sram", 128, 128)
    macro = Memory("macro", "macro", 256, 4*64*16*16//8)
    memory_space.add_memory(global_memory)
    memory_space.add_memory(local_memory)
    memory_space.add_memory(macro)
    return memory_space

def init_macro_config():
    macro_config = MacroConfig(n_macro=4, n_row=64, n_comp=16, n_bcol=16)

class TestSimulator:
    def __init__(self):
        self.inst_util = InstUtil()
        self.memory_space = init_memory_space()
        self.macro_config = init_macro_config()

    def test_general_li(self):
        inst_list = [
            self.inst_util.general_li(i, i) for i in range(32)
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        simulator.run_code(inst_list)
        for i in range(32):
            assert simulator.general_rf[i] == i, f"{simulator.general_rf[i]=}, {i=}"

    def test_special_li(self):
        inst_list = [
            self.inst_util.special_li(i, i) for i in range(32)
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        simulator.run_code(inst_list)
        for i in range(32):
            assert simulator.special_rf[i] == i, f"{simulator.special_rf[i]=}, {i=}"
    
    def test_rr_scalar(self):
        inst_list = [
            self.inst_util.general_li(0, 8),
            self.inst_util.general_li(1, 2),
            self.inst_util.scalar_rr(0, 1, rd=2, opcode="add"),
            self.inst_util.scalar_rr(0, 1, rd=3, opcode="sub"),
            self.inst_util.scalar_rr(0, 1, rd=4, opcode="mul"),
            self.inst_util.scalar_rr(0, 1, rd=5, opcode="div")
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        simulator.run_code(inst_list)

        assert simulator.general_rf[2] == 10
        assert simulator.general_rf[3] == 6
        assert simulator.general_rf[4] == 16
        assert simulator.general_rf[5] == 4

    def test_general_to_special(self):
        inst_list_1 = [
            self.inst_util.general_li(i, i) for i in range(32)
        ]
        inst_list_2 = [
            self.inst_util.general_to_special(i, i) for i in range(32)
        ]
        inst_list = [*inst_list_1, *inst_list_2]
        simulator = Simulator(self.memory_space , self.macro_config)
        simulator.run_code(inst_list)
        for i in range(32):
            assert simulator.special_rf[i] == i, f"{simulator.special_rf[i]=}, {i=}"

    def test_special_to_general(self):
        inst_list_1 = [
            self.inst_util.special_li(i, i) for i in range(32)
        ]
        inst_list_2 = [
            self.inst_util.special_to_general(i, i) for i in range(32)
        ]
        inst_list = [*inst_list_1, *inst_list_2]
        simulator = Simulator(self.memory_space , self.macro_config)
        simulator.run_code(inst_list)
        for i in range(32):
            assert simulator.general_rf[i] == i, f"{simulator.general_rf[i]=}, {i=}"

    def test_jump(self):
        inst_list = [
            self.inst_util.general_li(0, 1),
            self.inst_util.jump(2),
            self.inst_util.general_li(0, 2)
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        simulator.run_code(inst_list)
        assert simulator.general_rf[0] == 1, f"{simulator.general_rf[0]=}"

    def test_jump_back(self):
        inst_list = [
            self.inst_util.general_li(0, 1),
            self.inst_util.general_li(0, 2),
            self.inst_util.jump(-1)
        ]
        simulator = Simulator(self.memory_space , self.macro_config, safe_time = 100)
        status = simulator.run_code(inst_list)
        assert status == simulator.TIMEOUT, f"{status=}"

    def test_branch_eq(self):
        inst_list = [
            self.inst_util.general_li(0, 1),
            self.inst_util.general_li(1, 2),
            self.inst_util.branch("beq", 0, 1, 2), # not jump
            self.inst_util.general_li(2, 1),

            self.inst_util.general_li(3, 0),
            self.inst_util.general_li(0, 2),
            self.inst_util.general_li(1, 2),
            self.inst_util.branch("beq", 0, 1, 2), # jump
            self.inst_util.general_li(3, 1)
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        simulator.run_code(inst_list)
        assert simulator.general_rf[2] == 1, f"{simulator.general_rf[1]=}"
        assert simulator.general_rf[3] == 0, f"{simulator.general_rf[3]=}"
    
    def test_branch_ne(self):
        inst_list = [
            self.inst_util.general_li(2, 0),
            self.inst_util.general_li(0, 1),
            self.inst_util.general_li(1, 2),
            self.inst_util.branch("bne", 0, 1, 2), # jump
            self.inst_util.general_li(2, 1),

            self.inst_util.general_li(3, 0),
            self.inst_util.general_li(0, 2),
            self.inst_util.general_li(1, 2),
            self.inst_util.branch("bne", 0, 1, 2), # not jump
            self.inst_util.general_li(3, 1)
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        simulator.run_code(inst_list)
        assert simulator.general_rf[2] == 0, f"{simulator.general_rf[2]=}"
        assert simulator.general_rf[3] == 1, f"{simulator.general_rf[3]=}"

    def test_branch_lt(self):
        inst_list = [
            self.inst_util.general_li(2, 0),
            self.inst_util.general_li(0, 1),
            self.inst_util.general_li(1, 2),
            self.inst_util.branch("blt", 0, 1, 2), # jump
            self.inst_util.general_li(2, 1),

            self.inst_util.general_li(3, 0),
            self.inst_util.general_li(0, 2),
            self.inst_util.general_li(1, 2),
            self.inst_util.branch("blt", 0, 1, 2), # not jump
            self.inst_util.general_li(3, 1),

            self.inst_util.general_li(4, 0),
            self.inst_util.general_li(0, 2),
            self.inst_util.general_li(1, 1),
            self.inst_util.branch("blt", 0, 1, 2), # not jump
            self.inst_util.general_li(4, 1)
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        simulator.run_code(inst_list)
        assert simulator.general_rf[2] == 0, f"{simulator.general_rf[2]=}"
        assert simulator.general_rf[3] == 1, f"{simulator.general_rf[3]=}"
        assert simulator.general_rf[4] == 1, f"{simulator.general_rf[4]=}"

    def test_branch_gt(self):
        inst_list = [
            self.inst_util.general_li(2, 0),
            self.inst_util.general_li(0, 1),
            self.inst_util.general_li(1, 2),
            self.inst_util.branch("bgt", 0, 1, 2), # jump
            self.inst_util.general_li(2, 1),

            self.inst_util.general_li(3, 0),
            self.inst_util.general_li(0, 2),
            self.inst_util.general_li(1, 2),
            self.inst_util.branch("bgt", 0, 1, 2), # not jump
            self.inst_util.general_li(3, 1),

            self.inst_util.general_li(4, 0),
            self.inst_util.general_li(0, 2),
            self.inst_util.general_li(1, 1),
            self.inst_util.branch("bgt", 0, 1, 2), # not jump
            self.inst_util.general_li(4, 1)
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        simulator.run_code(inst_list)
        assert simulator.general_rf[2] == 1, f"{simulator.general_rf[2]=}"
        assert simulator.general_rf[3] == 1, f"{simulator.general_rf[3]=}"
        assert simulator.general_rf[4] == 0, f"{simulator.general_rf[4]=}"

        
    def test_branch_back(self):
        """
        test:
        sum = 0;
        for(i=0;i<11;i++){
            sum += i;
        }
        """
        inst_list = [
            self.inst_util.general_li(0, 0), # i
            self.inst_util.general_li(1, 0), # sum
            self.inst_util.general_li(2, 11), # upperbound: 11
            self.inst_util.general_li(3, 1), # step: 1
            self.inst_util.scalar_rr(0,1,1,opcode="add"),
            self.inst_util.scalar_rr(0,3,0,opcode="add"),
            self.inst_util.branch("blt", 0, 2, -2), # jump
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        status = simulator.run_code(inst_list)
        assert status==simulator.FINISH
        assert simulator.general_rf[1] == 55, f"{simulator.general_rf[1]=}"

    def test_trans(self):
        inst_list = [
            self.inst_util.general_li(0, 0), # src addr
            self.inst_util.general_li(1, 64), # dst addr
            self.inst_util.general_li(2, 64), # size
            self.inst_util.trans(0, 1, 2)
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        data = np.arange(64, dtype=np.int8)
        simulator.memory_space.write(data, 0, 64)
        status = simulator.run_code(inst_list)

        assert status==simulator.FINISH
        assert (simulator.memory_space.read_as(64, 64, np.int8)==data).all()

    def test_trans_across_memory(self):
        inst_list = [
            self.inst_util.general_li(0, 0), # src addr
            self.inst_util.general_li(1, 128), # dst addr
            self.inst_util.general_li(2, 64), # size
            self.inst_util.trans(0, 1, 2)
        ]
        simulator = Simulator(self.memory_space , self.macro_config)
        data = np.arange(64, dtype=np.int8)
        simulator.memory_space.write(data, 0, 64)
        status = simulator.run_code(inst_list)

        assert status==simulator.FINISH
        assert (simulator.memory_space.read_as(128, 64, np.int8)==data).all()

    def test_pim_compute(self):
        pass
    


if __name__=="__main__":
    test_simulator = TestSimulator()
    test_simulator.test_general_li()
    test_simulator.test_special_li()
    test_simulator.test_rr_scalar()
    test_simulator.test_general_to_special()
    test_simulator.test_special_to_general()
    test_simulator.test_jump()
    test_simulator.test_jump_back()
    test_simulator.test_branch_eq()
    test_simulator.test_branch_ne()
    test_simulator.test_branch_lt()
    test_simulator.test_branch_gt()
    test_simulator.test_branch_back()
    test_simulator.test_trans()
    test_simulator.test_trans_across_memory()