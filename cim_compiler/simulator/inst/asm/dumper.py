import json
from cim_compiler.simulator.inst.instruction import *
from cim_compiler.simulator.inst.asm.op_name_mapping import (
    mapping_sc_funct_to_name,
    mapping_simd_funct_to_name,
    mapping_branch_compare_to_name,
    mapping_special_reg_to_name
)

class AsmDumper:
    def __init__(self):
        pass

    def dump(self, instructions):
        data = []
        for inst in instructions:
            inst_dict = self._convert_to_asm(inst)
            data.append(inst_dict)
        return data

    def dump_str(self, instructions, core_id=None):
        data = self.dump(instructions)
        return "\n".join(data)

    def dump_to_file(self, instructions, file_path, core_id=None):
        with open(file_path, 'w') as file:
            data_str = self.dump_str(instructions)
            file.write(data_str)

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

    def _convert_to_asm(self, inst):
        if isinstance(inst, GeneralLiInst):
            return f"G_LI r{inst.reg}, {inst.value}"
        elif isinstance(inst, SpecialLiInst):
            special_reg_name = mapping_special_reg_to_name[inst.reg]
            return f"S_LI {special_reg_name}, {inst.value}"
        elif isinstance(inst, GeneralToSpecialAssignInst):
            special_reg_name = mapping_special_reg_to_name[inst.reg_special]
            return f"GS_MOV {special_reg_name}, r{inst.reg_general}"
        elif isinstance(inst, SpecialToGeneralAssignInst):
            special_reg_name = mapping_special_reg_to_name[inst.reg_special]
            return f"SG_MOV r{inst.reg_general}, {special_reg_name}"
        elif isinstance(inst, RRInst):
            op_name = self._arith_funct_to_name(inst.opcode)
            return f"{op_name} r{inst.reg_out}, r{inst.reg_lhs}, r{inst.reg_rhs}"
        elif isinstance(inst, RIInst):
            op_name = self._arith_funct_to_name(inst.opcode, is_ri=True)
            return f"{op_name} r{inst.reg_out}, r{inst.reg_in}, {inst.imm}"
        elif isinstance(inst, SIMDInst):
            op_name = self._simd_funct_to_name(inst.opcode)
            input_num = inst.input_num - 1
            return f"{op_name} r{inst.reg_out}, r{inst.reg_in1}, r{inst.reg_in2}, r{inst.reg_size}, {input_num}"
        elif isinstance(inst, TransInst):
            terms = [f"r{inst.reg_out}", f"r{inst.reg_in}", f"r{inst.reg_size}", f"{inst.offset}"]
            if inst.flag_src_offset:
                terms.append("SRC_O")
            if inst.flag_dst_offset:
                terms.append("DST_O")
            return f"MEM_CPY {', '.join(terms)}"
        elif isinstance(inst, LoadInst):
            return f"SC_LD r{inst.reg_value}, {inst.offset}(r{inst.reg_addr})"
        elif isinstance(inst, StoreInst):
            return f"SC_ST r{inst.reg_value}, {inst.offset}(r{inst.reg_addr})"
        elif isinstance(inst, PrintInst):
            return f"PRINT r{inst.reg}"
        elif isinstance(inst, DebugInst):
            return f"DEBUG"
        elif isinstance(inst, BranchInst):
            op_name = self._branch_compare_to_name(inst.compare)
            return f"{op_name} r{inst.reg_lhs}, r{inst.reg_rhs}, {inst.offset}"
        elif isinstance(inst, JumpInst):
            return f"JMP {inst.offset}"
        elif isinstance(inst, CIMComputeInst):
            terms = [f"r{inst.reg_input_addr}", f"r{inst.reg_input_size}", f"r{inst.reg_activate_row}"]
            if inst.flag_group:
                terms.append("GRP")
            if inst.flag_group_input_mode:
                terms.append("GRP_I")
            if inst.flag_value_sparse:
                terms.append("SP_V")
            if inst.flag_bit_sparse:
                terms.append("SP_B")
            return f"CIM_MVM {', '.join(terms)}"
        elif isinstance(inst, CIMConfigInst):
            terms = [f"r{inst.reg_single_group_id}", f"r{inst.reg_mask_addr}"]
            if inst.flag_group_broadcast:
                terms.append("GRP_B")
            return f"CIM_CFG {', '.join(terms)}"
        elif isinstance(inst, CIMOutputInst):
            terms = [f"r{inst.reg_out_n}", f"r{inst.reg_out_mask_addr}", f"r{inst.reg_out_addr}"]
            if inst.flag_outsum:
                terms.append("OSUM")
            if inst.flag_outsum_move:
                terms.append("OSUM_MOVE")
            return f"CIM_OUT {', '.join(terms)}"
        else:
            raise ValueError(f"Unknown instruction type: {type(inst)}")
        
