import json
from simulator.inst.instruction import *
from simulator.inst.asm.op_name_mapping import (
    mapping_sc_funct_to_name,
    mapping_simd_funct_to_name,
    mapping_branch_compare_to_name
)

class AsmDumper:
    def __init__(self):
        pass

    def dump(self, instructions):
        data = []
        for inst in instructions:
            inst_dict = self._convert_to_dict(inst)
            data.append(inst_dict)
        return data

    def dump_to_file(self, instructions, file_path):
        with open(file_path, 'w') as file:
            for inst in self.dump(instructions):
                file.write(inst + "\n")

    def _arith_funct_to_name(self, funct, is_ri=False):
        """
        SC_ADD(I) 000000, SC_SUB(I) 000001, SC_MUL(I) 000010, SC_DIV(I) 000011, SC_SLL(I) 000100, SC_SRL(I) 000101, SC_SRA(I) 000110, SC_MOD(I) 000111, SC_MIN(I) 001000
        """
        op_name = mapping_sc_funct_to_name[funct]
        if is_ri:
            op_name += "I"
        return op_name

    def _simd_funct_to_name(self, funct):
        op_name = mapping_simd_funct_to_name[funct]
        return op_name

    def _branch_compare_to_name(self, compare):
        """
        BEQ 00, BNE 01, BGT 10, BLT 11
        """
        op_name = mapping_branch_compare_to_name[compare]
        return op_name

    def _convert_to_dict(self, inst):
        if isinstance(inst, GeneralLiInst):
            return f"G_LI {inst.reg}, {inst.value}"
        elif isinstance(inst, SpecialLiInst):
            return f"S_LI {inst.reg}, {inst.value}"
        elif isinstance(inst, GeneralToSpecialAssignInst):
            return f"GS_MOV {inst.reg_special}, {inst.reg_general}"
        elif isinstance(inst, SpecialToGeneralAssignInst):
            return f"SG_MOV {inst.reg_general}, {inst.reg_special}"
        elif isinstance(inst, ArithInst):
            op_name = self._arith_funct_to_name(inst.opcode)
            return f"{op_name} {inst.reg_out}, {inst.reg_lhs}, {inst.reg_rhs}"
        elif isinstance(inst, RIInst):
            op_name = self._arith_funct_to_name(inst.opcode, is_ri=True)
            return f"{op_name} {inst.reg_out}, {inst.reg_in}, {inst.imm}"
        elif isinstance(inst, SIMDInst):
            op_name = self._simd_funct_to_name(inst.opcode)
            return f"{op_name} {inst.reg_out}, {inst.reg_in1}, {inst.reg_in2}, {inst.reg_size}, {inst.input_num}"
        elif isinstance(inst, TransInst):
            return f"MEM_CPY {inst.reg_out}, {inst.reg_in}, {inst.reg_size}"
        elif isinstance(inst, LoadInst):
            return f"SC_LD {inst.reg_value}, {inst.offset}({inst.reg_addr})"
        elif isinstance(inst, StoreInst):
            return f"SC_ST {inst.reg_value}, {inst.offset}({inst.reg_addr})"
        elif isinstance(inst, PrintInst):
            return f"PRINT {inst.reg}"
        elif isinstance(inst, DebugInst):
            return f"DEBUG"
        elif isinstance(inst, BranchInst):
            op_name = self._branch_compare_to_name(inst.compare)
            return f"{op_name} {inst.reg_lhs}, {inst.reg_rhs}, {inst.offset}"
        elif isinstance(inst, JumpInst):
            return f"JMP {inst.offset}"
        else:
            raise ValueError(f"Unknown instruction type: {type(inst)}")
        
