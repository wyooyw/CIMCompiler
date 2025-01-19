import json
from simulator.inst.instruction import *
from simulator.inst.legacy.dumper import LegacyDumper
from simulator.inst.asm.dumper import AsmDumper
from simulator.inst.asm.parser import AsmParser
class CIMFlowParser:
    def __init__(self):
        pass

    def parse_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data, self.parse(data)

    def parse(self, data):
        
        instructions = []
        for item in data:
            inst = self._parse_inst(item)
            instructions.append(inst)
        return instructions
    
    def _parse_inst(self, inst):
        opcode = inst.get("opcode")

        if opcode == 0b101100:
            return GeneralLiInst(
                reg=inst["rd"],
                value=inst["imm"]
            )
        elif opcode == 0b101101:
            return SpecialLiInst(
                reg=inst["rd"],
                value=inst["imm"]
            )
        elif opcode == 0b100000:
            return ArithInst(
                opcode=inst["funct"],
                reg_lhs=inst["rs"],
                reg_rhs=inst["rt"],
                reg_out=inst["rd"]
            )
        elif opcode == 0b100100:
            return RIInst(
                opcode=inst["funct"],
                reg_in=inst["rs"],
                reg_out=inst["rd"],
                imm=inst["imm"]
            )
        elif opcode in (
            0b010000,
            0b010100,
            0b011000,
            0b011100,
        ):
            input_num = ((opcode & 0b1100) >> 2) + 1
            return SIMDInst(
                opcode=inst["funct"],
                reg_in1=inst["rs"],
                reg_in2=inst["rt"],
                reg_size=inst["re"],
                reg_out=inst["rd"],
                input_num=input_num
            )
        elif opcode == 0b110000:
            return TransInst(
                reg_in=inst["rs"],
                reg_size=inst["rt"],
                reg_out=inst["rd"]
            )
        elif opcode == 0b101000:
            return LoadInst(
                reg_addr=inst["rs"],
                reg_value=inst["rt"],
                offset=inst["imm"]
            )
        elif opcode == 0b101001:
            return StoreInst(
                reg_addr=inst["rs"],
                reg_value=inst["rt"],
                offset=inst["imm"]
            )
        elif opcode == -1:
            return PrintInst(
                reg=inst["rs"]
            )
        elif opcode == -2:
            return DebugInst()
        elif opcode >= 0b111000 and opcode <= 0b111011:
            compare = opcode - 0b111000
            return BranchInst(
                compare=compare,
                reg_lhs=inst["rs"],
                reg_rhs=inst["rt"],
                offset=inst["imm"]
            )
        elif opcode == 0b111100:
            return JumpInst(
                offset=inst["imm"]
            )
        elif opcode == 0b101110:
            return GeneralToSpecialAssignInst(
                reg_general=inst["rs"],
                reg_special=inst["rd"]
            )
        elif opcode == 0b101111:
            return SpecialToGeneralAssignInst(
                reg_special=inst["rs"],
                reg_general=inst["rd"]
            )
        else:
            raise ValueError(f"Unknown opcode: {opcode}")
        