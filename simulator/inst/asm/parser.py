import json
from simulator.inst.instruction import *
from simulator.inst.asm.op_name_mapping import (
    mapping_name_to_sc_funct,
    mapping_name_to_simd_funct,
    mapping_name_to_branch_compare
)

class AsmParser:
    def __init__(self):
        pass

    def parse_file(self, file_path):
        with open(file_path, 'r') as file:
            data = [line.strip() for line in file.readlines()]
        return data, self.parse(data)

    def parse(self, data):
        instructions = []
        for line in data:
            line = line.strip()
            if line:  # Skip empty lines
                inst = self._parse_inst(line)
                instructions.append(inst)
        return instructions

    def _parse_inst(self, inst_str):
        # Split the instruction into op_name and args
        parts = inst_str.split(maxsplit=1)
        op_name = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        # Split args by comma and strip spaces
        args = [arg.strip() for arg in args.split(',')]

        if op_name == "G_LI":
            reg, value = map(int, args)
            return GeneralLiInst(reg=reg, value=value)
        elif op_name == "S_LI":
            reg, value = map(int, args)
            return SpecialLiInst(reg=reg, value=value)
        elif op_name == "GS_MOV":
            reg_general, reg_special = map(int, args)
            return GeneralToSpecialAssignInst(reg_special=reg_special, reg_general=reg_general)
        elif op_name == "SG_MOV":
            reg_special, reg_general = map(int, args)
            return SpecialToGeneralAssignInst(reg_general=reg_general, reg_special=reg_special)
        elif op_name.startswith("SC_") and not op_name.endswith("I"):
            reg_out, reg_lhs, reg_rhs = map(int, args)
            opcode = self._name_to_arith_funct(op_name)
            return ArithInst(opcode=opcode, reg_lhs=reg_lhs, reg_rhs=reg_rhs, reg_out=reg_out)
        elif op_name.startswith("SC_") and op_name.endswith("I"):
            reg_out, reg_in, imm = map(int, args)
            opcode = self._name_to_arith_funct(op_name[:-1], is_ri=True)
            return RIInst(opcode=opcode, reg_in=reg_in, reg_out=reg_out, imm=imm)
        elif op_name.startswith("VEC_"):
            reg_out, reg_in1, reg_in2, reg_size, input_num = map(int, args)
            opcode = self._name_to_simd_funct(op_name)
            return SIMDInst(opcode=opcode, reg_in1=reg_in1, reg_in2=reg_in2, reg_size=reg_size, reg_out=reg_out, input_num=input_num)
        elif op_name == "MEM_CPY":
            reg_out, reg_in, reg_size = map(int, args)
            return TransInst(reg_out=reg_out, reg_in=reg_in, reg_size=reg_size)
        elif op_name == "SC_LD":
            reg_value, offset_reg_addr = args
            reg_value = int(reg_value)
            offset, reg_addr = map(int, offset_reg_addr.strip('()').split(','))
            return LoadInst(reg_addr=reg_addr, reg_value=reg_value, offset=offset)
        elif op_name == "SC_ST":
            reg_value, offset_reg_addr = args
            reg_value = int(reg_value)
            offset, reg_addr = map(int, offset_reg_addr.strip('()').split(','))
            return StoreInst(reg_addr=reg_addr, reg_value=reg_value, offset=offset)
        elif op_name == "PRINT":
            reg = int(args[0])
            return PrintInst(reg=reg)
        elif op_name == "DEBUG":
            return DebugInst()
        elif op_name in ["BEQ", "BNE", "BGT", "BLT"]:
            reg_lhs, reg_rhs, offset = map(int, args)
            compare = self._name_to_branch_compare(op_name)
            return BranchInst(compare=compare, reg_lhs=reg_lhs, reg_rhs=reg_rhs, offset=offset)
        elif op_name == "JMP":
            offset = int(args[0])
            return JumpInst(offset=offset)
        else:
            raise ValueError(f"Unknown instruction name: {op_name}")

    def _name_to_arith_funct(self, name, is_ri=False):
        op_name = mapping_name_to_sc_funct[name]
        return op_name

    def _name_to_simd_funct(self, name):
        op_name = mapping_name_to_simd_funct[name]
        return op_name

    def _name_to_branch_compare(self, name):
        op_name = mapping_name_to_branch_compare[name]
        return op_name
        