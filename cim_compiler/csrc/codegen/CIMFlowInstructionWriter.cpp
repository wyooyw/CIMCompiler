#include "codegen/InstructionWriter.h"
// Implement a concrete InstructionWriter
Inst CIMFlowInstructionWriter::getGeneralLIInst(int reg, int value) {
    return {
        {"opcode", 0b101100}, 
        {"rd", reg},
        {"imm", value}
    };
}

Inst CIMFlowInstructionWriter::getSpecialLIInst(int reg, int value) {
    return {
        {"opcode", 0b101101}, 
        {"rd", reg},
        {"imm", value}
    };
}

Inst CIMFlowInstructionWriter::getRRInst(int funct, int reg_in1, int reg_in2, int reg_out) {
    return {
        {"opcode", 0b100000}, 
        {"rs", reg_in1}, 
        {"rt", reg_in2}, 
        {"rd", reg_out},
        {"funct", funct}
    };
}

Inst CIMFlowInstructionWriter::getRIInst(int funct, int reg_in, int reg_out, int imm) {
    return {
        {"opcode", 0b100100}, 
        {"rs", reg_in},      
        {"rd", reg_out}, 
        {"funct", funct},
        {"imm", imm}
    };
}

Inst CIMFlowInstructionWriter::getSIMDInst(int funct, int input_num, int reg_in1, int reg_in2, int reg_size, int reg_out) {
    return {
        {"opcode", 0b010000 + ((input_num - 1) << 2)}, 
        {"rs", reg_in1}, 
        {"rt", reg_in2}, 
        {"rd", reg_out},
        {"re", reg_size},
        {"funct", funct}
    };
}

Inst CIMFlowInstructionWriter::getTransInst(int reg_addr_in, int reg_addr_out, int reg_size, int imm, bool src_offset_flag, bool dst_offset_flag) {
    return {
        {"opcode", 0b110000 + (int(src_offset_flag) << 1) + int(dst_offset_flag)}, 
        {"rs", reg_addr_in}, 
        {"rt", reg_size}, 
        {"rd", reg_addr_out}, 
        {"imm", imm},
    };
}

Inst CIMFlowInstructionWriter::getLoadInst(int reg_addr, int reg_value, int offset) {
    return {
        {"opcode", 0b101000}, 
        {"rs", reg_addr},
        {"rt", reg_value},
        {"imm", offset}
    };
}

Inst CIMFlowInstructionWriter::getStoreInst(int reg_addr, int reg_value, int offset) {
    return {
        {"opcode", 0b101001}, 
        {"rs", reg_addr},
        {"rt", reg_value},
        {"imm", offset}
    };
}

Inst CIMFlowInstructionWriter::getPrintInst(int reg) {
    return {
        {"opcode", -1}, {"rs", reg}
    };
}

Inst CIMFlowInstructionWriter::getDebugInst() {
    return {
        {"opcode", -2}
    };
}

Inst CIMFlowInstructionWriter::getBranchInst(int compare, int reg_lhs, int reg_rhs, int offset) {
    return {
        {"opcode", 0b111000 + compare}, 
        {"rs", reg_lhs}, 
        {"rt", reg_rhs}, 
        {"imm", offset}
    };
}

Inst CIMFlowInstructionWriter::getJumpInst(int offset) {
    return {
        {"opcode", 0b111100}, 
        {"imm", offset}
    };
}

Inst CIMFlowInstructionWriter::getGeneralToSpecialAssignInst(int reg_general, int reg_special) {
    return {
        {"opcode", 0b101110},
        {"rs", reg_general},
        {"rd", reg_special}
    };
}

Inst CIMFlowInstructionWriter::getSpecialToGeneralAssignInst(int reg_general, int reg_special) {
    return {
        {"opcode", 0b101111},
        {"rs", reg_special},
        {"rd", reg_general}
    };
}

Inst CIMFlowInstructionWriter::getCIMComputeInst(int reg_input_addr, int reg_input_size, int reg_activate_row, int reg_batch_size, int flag_accumulate, int flag_value_sparse, int flag_bit_sparse, int flag_group, int flag_group_input_mode, int flag_batch) {
    return {
        {"opcode", 0b000000},
        {"rs", reg_input_addr},
        {"rt", reg_input_size},
        {"re", reg_activate_row},
        {"rf", reg_batch_size},
        {"SP_V", bool(flag_value_sparse)},
        {"SP_B", bool(flag_bit_sparse)},
        {"GRP", bool(flag_group)},
        {"GRP_I", bool(flag_group_input_mode)},
        {"BATCH", bool(flag_batch)}
    };
}

Inst CIMFlowInstructionWriter::getCIMSetInst(int reg_single_group_id, int reg_mask_addr, int flag_group_broadcast ) {
    return {
        {"opcode", 0b000100},
        {"rs", reg_single_group_id}, 
        {"rt", reg_mask_addr},
        {"GRP_B", bool(flag_group_broadcast)}
    };
}

Inst CIMFlowInstructionWriter::getCIMOutputInst(int reg_out_n, int reg_out_mask_addr, int reg_out_addr, int flag_outsum, int flag_outsum_move ) {
    return {
        {"opcode", 0b001000},
        {"rs", reg_out_n}, 
        {"rt", reg_out_mask_addr},
        {"rd", reg_out_addr},
        {"OSUM", bool(flag_outsum)},
        {"OSUM_MOV", bool(flag_outsum_move)}
    };
}

Inst CIMFlowInstructionWriter::getCIMTransferInst(int reg_src_addr, int reg_out_n, int reg_out_mask_addr, int reg_buffer_addr, int reg_dst_addr) {
    return {
        {"opcode", 0b001100}, 
        {"rs", reg_src_addr}, 
        {"rt", reg_out_n}, 
        {"re", reg_out_mask_addr},
        {"rf", reg_buffer_addr},
        {"rd", reg_dst_addr}
    };
}

Inst CIMFlowInstructionWriter::getSendInst(int reg_src_addr, int reg_dst_addr, int reg_size, int reg_core_id, int reg_transfer_id) {
    return {
        {"opcode", 0b110100}, 
        {"rs", reg_src_addr}, 
        {"rt", reg_core_id}, 
        {"rd", reg_dst_addr},
        {"re", reg_size},
        {"rf", reg_transfer_id}
    };
}

Inst CIMFlowInstructionWriter::getRecvInst(int reg_src_addr, int reg_dst_addr, int reg_size, int reg_core_id, int reg_transfer_id) {
    return {
        {"opcode", 0b110110}, 
        {"rs", reg_core_id}, 
        {"rt", reg_src_addr}, 
        {"rd", reg_dst_addr},
        {"re", reg_size},
        {"rf", reg_transfer_id}
    };
}

void CIMFlowInstructionWriter::setJumpOffset(Inst &inst, int offset) {
    inst["imm"] = offset;
}

void CIMFlowInstructionWriter::setBranchOffset(Inst &inst, int offset) {
    inst["imm"] = offset;
}


bool CIMFlowInstructionWriter::isGeneralToSpecialAssign(Inst &inst) {
  return inst.count("opcode") && std::holds_alternative<int>(inst["opcode"]) && std::get<int>(inst["opcode"]) == 0b101110;
}

bool CIMFlowInstructionWriter::isSpecialToGeneralAssign(Inst &inst) {
  return inst.count("opcode") && std::holds_alternative<int>(inst["opcode"]) && std::get<int>(inst["opcode"]) == 0b101111;
}

bool CIMFlowInstructionWriter::isGeneralReg(Inst &inst, std::string key) {
  if (isGeneralToSpecialAssign(inst)) {
    return key == "rs";
  }else if (isSpecialToGeneralAssign(inst)) {
    return key == "rd";
  }
  return key=="rs" || key=="rt" || key=="re" || key=="rf" || key=="rd";
}

bool CIMFlowInstructionWriter::isSpecialLi(Inst &inst) {
  return inst.count("opcode") && std::holds_alternative<int>(inst["opcode"]) && std::get<int>(inst["opcode"]) == 0b101101;
}

