import json
from simulator.inst.instruction import *

class CIMFlowDumper:
    def __init__(self):
        pass

    def dump(self, instructions):
        data = []
        
        for inst in instructions:
            inst_dict = self._convert_to_dict(inst)
            data.append(inst_dict)

        return data

    def dump_to_file(self, instructions, path):
        with open(path, 'w') as file:
            data = self.dump(instructions)
            json.dump(data, file, indent=4)

    def _convert_to_dict(self, inst):
        if isinstance(inst, GeneralLiInst):
            return {
                "opcode": 0b101100,
                "rd": inst.reg,
                "imm": inst.value
            }
        elif isinstance(inst, SpecialLiInst):
            return {
                "opcode": 0b101101,
                "rd": inst.reg,
                "imm": inst.value
            }
        elif isinstance(inst, ArithInst):
            return {
                "opcode": 0b100000,
                "rs": inst.reg_lhs,
                "rt": inst.reg_rhs,
                "rd": inst.reg_out,
                "funct": inst.opcode
            }
        elif isinstance(inst, RIInst):
            return {
                "opcode": 0b100100,
                "rs": inst.reg_in,
                "rd": inst.reg_out,
                "funct": inst.opcode,
                "imm": inst.imm
            }
        elif isinstance(inst, SIMDInst):
            assert inst.input_num >=1 and inst.input_num <= 4
            return {
                "opcode": 0b010000 + (inst.input_num - 1 << 2),
                "rs": inst.reg_in1,
                "rt": inst.reg_in2,
                "rd": inst.reg_out,
                "re": inst.reg_size,
                "funct": inst.opcode
            }
        elif isinstance(inst, TransInst):
            return {
                "opcode": 0b110000,
                "rs": inst.reg_in,
                "rt": inst.reg_size,
                "rd": inst.reg_out,
                "imm": 0b0
            }
        elif isinstance(inst, LoadInst):
            return {
                "opcode": 0b101000,
                "rs": inst.reg_addr,
                "rt": inst.reg_value,
                "imm": inst.offset
            }
        elif isinstance(inst, StoreInst):
            return {
                "opcode": 0b101001,
                "rs": inst.reg_addr,
                "rt": inst.reg_value,
                "imm": inst.offset
            }
        elif isinstance(inst, PrintInst):
            return {
                "opcode": -1,
                "rs": inst.reg
            }
        elif isinstance(inst, DebugInst):
            return {
                "opcode": -2
            }
        elif isinstance(inst, BranchInst):
            assert inst.compare >=0  and inst.compare <= 3
            return {
                "opcode": 0b111000 + inst.compare,
                "rs": inst.reg_lhs,
                "rt": inst.reg_rhs,
                "imm": inst.offset
            }
        elif isinstance(inst, JumpInst):
            return {
                "opcode": 0b111100,
                "imm": inst.offset
            }
        elif isinstance(inst, GeneralToSpecialAssignInst):
            return {
                "opcode": 0b101110,
                "rs": inst.reg_general,
                "rd": inst.reg_special
            }
        elif isinstance(inst, SpecialToGeneralAssignInst):
            return {
                "opcode": 0b101111,
                "rs": inst.reg_special,
                "rd": inst.reg_general
            }
        else:
            raise ValueError(f"Unknown instruction type: {type(inst)}")
        