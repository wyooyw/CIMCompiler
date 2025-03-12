#include "codegen/InstructionWriter.h"
// Implement a concrete InstructionWriter
Inst LegacyInstructionWriter::getGeneralLIInst(int reg, int value) {
    return {
        {"class", 0b10}, 
        {"type", 0b11}, 
        {"opcode", 0b00}, 
        {"rd", reg}, 
        {"imm", value}
    };
}

Inst LegacyInstructionWriter::getSpecialLIInst(int reg, int value) {
    return {
        {"class", 0b10}, 
        {"type", 0b11}, 
        {"opcode", 0b01}, 
        {"rd", reg}, 
        {"imm", value}
    };
}

Inst LegacyInstructionWriter::getRRInst(int opcode, int reg_in1, int reg_in2, int reg_out) {
    return {
        {"class", 0b10}, 
        {"type", 0b00}, 
        {"opcode", opcode}, 
        {"rs1", reg_in1}, 
        {"rs2", reg_in2}, 
        {"rd", reg_out}
    };
}

Inst LegacyInstructionWriter::getRIInst(int opcode, int reg_in, int reg_out, int imm) {
    return {
        {"class", 0b10}, 
        {"type", 0b01}, 
        {"opcode", opcode},
        {"rs", reg_in},      
        {"rd", reg_out},     
        {"imm", imm}
    };
}

Inst LegacyInstructionWriter::getSIMDInst(int opcode, int input_num, int in1_reg, int in2_reg, int size_reg, int out_reg) {
    return {
        {"class", 0b01}, 
        {"input_num", input_num - 1}, 
        {"opcode", opcode}, 
        {"rs1", in1_reg}, 
        {"rs2", in2_reg}, 
        {"rs3", size_reg}, 
        {"rd", out_reg}
    };
}

Inst LegacyInstructionWriter::getTransInst(int reg_addr_in, int reg_addr_out, int size) {
    return {
        {"class", 0b110}, 
        {"type", 0b0}, 
        {"source_offset_mask", 0b0}, 
        {"destination_offset_mask", 0b0}, 
        {"rs1", reg_addr_in}, 
        {"rd", reg_addr_out}, 
        {"offset", 0b0}, 
        {"rs2", size}
    };
}

Inst LegacyInstructionWriter::getLoadInst(int reg_addr, int reg_value, int offset) {
    return {
        {"class", 0b10}, 
        {"type", 0b10}, 
        {"opcode", 0b00},
        {"rs1", reg_addr},
        {"rs2", reg_value},
        {"offset", offset}
    };
}

Inst LegacyInstructionWriter::getStoreInst(int reg_addr, int reg_value, int offset) {
    return {
        {"class", 0b10}, {"type", 0b10}, {"opcode", 0b01},
        {"rs1", reg_addr},    {"rs2", reg_value},   {"offset", offset},
    };
}

Inst LegacyInstructionWriter::getPrintInst(int reg) {
    return {
        {"class", -1}, {"type", 0}, {"rs", reg}
    };
}

Inst LegacyInstructionWriter::getDebugInst() {
    return {
        {"class", -1}, {"type", 1}
    };
}

Inst LegacyInstructionWriter::getBranchInst(int compare, int reg_lhs, int reg_rhs, int offset) {
    return {
        {"class", 0b111}, {"type", compare}, {"rs1", reg_lhs}, {"rs2", reg_rhs}, {"offset", offset}
    };
}

Inst LegacyInstructionWriter::getJumpInst(int offset) {
    return {
        {"class", 0b111}, {"type", 0b100}, {"offset", offset}
    };
}

Inst LegacyInstructionWriter::getGeneralToSpecialAssignInst(int reg_general, int reg_special) {
    return {
        {"class", 0b10}, {"type", 0b11}, {"opcode", 0b10}, {"rs1", reg_general}, {"rs2", reg_special}
    };
}

Inst LegacyInstructionWriter::getSpecialToGeneralAssignInst(int reg_general, int reg_special) {
    return {
        {"class", 0b10}, {"type", 0b11}, {"opcode", 0b11}, {"rs1", reg_general}, {"rs2", reg_special}
    };
}

void LegacyInstructionWriter::setJumpOffset(Inst &inst, int offset) {
    inst["offset"] = offset;
}

void LegacyInstructionWriter::setBranchOffset(Inst &inst, int offset) {
    inst["offset"] = offset;
}

bool LegacyInstructionWriter::isGeneralToSpecialAssign(Inst &inst) {
  if ((inst.count("class") && std::holds_alternative<int>(inst["class"]) && std::get<int>(inst["class"]) == 0b10) &&
      (inst.count("type") && std::holds_alternative<int>(inst["type"]) && std::get<int>(inst["type"]) == 0b11) &&
      (inst.count("opcode") && std::holds_alternative<int>(inst["opcode"]) && std::get<int>(inst["opcode"]) == 0b10)) {
    return true;
  }
  return false;
}

bool LegacyInstructionWriter::isSpecialToGeneralAssign(Inst &inst) {
  if ((inst.count("class") && std::holds_alternative<int>(inst["class"]) && std::get<int>(inst["class"]) == 0b10) &&
      (inst.count("type") && std::holds_alternative<int>(inst["type"]) && std::get<int>(inst["type"]) == 0b11) &&
      (inst.count("opcode") && std::holds_alternative<int>(inst["opcode"]) && std::get<int>(inst["opcode"]) == 0b11)) {
    return true;
  }
  return false;
}

static bool isPrefix(const std::string &str, const std::string &prefix) {
  // 检查前缀长度是否大于字符串长度
  if (prefix.length() > str.length()) {
    return false;
  }

  // 获取字符串的子串，长度等于前缀的长度，从字符串的开始位置
  std::string strPrefix = str.substr(0, prefix.length());

  // 比较子串和前缀是否相等
  return strPrefix == prefix;
}

bool LegacyInstructionWriter::isGeneralReg(Inst &inst, std::string key) {
  bool is_gen_to_spec_assign = isGeneralToSpecialAssign(inst);
  bool is_spec_to_gen_assign = isSpecialToGeneralAssign(inst);
  bool is_special_assign = is_gen_to_spec_assign || is_spec_to_gen_assign;
  bool is_reg_general =
      (is_special_assign && key == "rs1") ||
      ((!is_special_assign) && (isPrefix(key, "rs") || isPrefix(key, "rd")));
  return is_reg_general;
}

bool LegacyInstructionWriter::isSpecialLi(Inst &inst) {
  if ((inst.count("class") && std::holds_alternative<int>(inst["class"]) && std::get<int>(inst["class"]) == 0b10) &&
      (inst.count("type") && std::holds_alternative<int>(inst["type"]) && std::get<int>(inst["type"]) == 0b11) &&
      (inst.count("opcode") && std::holds_alternative<int>(inst["opcode"]) && std::get<int>(inst["opcode"]) == 0b01)) {
    return true;
  }
  return false;
}

Inst LegacyInstructionWriter::getCIMComputeInst(int reg_input_addr, int reg_input_size, int reg_activate_row, int flag_accumulate, int flag_value_sparse, int flag_bit_sparse, int flag_group, int flag_group_input_mode) {
    return {
        {"class", 0b00}, 
        {"type", 0b0}, 
        {"value_sparse", flag_value_sparse}, 
        {"bit_sparse", flag_bit_sparse}, 
        {"group", flag_group}, 
        {"group_input_mode", flag_group_input_mode}, 
        {"accumulate", flag_accumulate}, 
        {"rs1", reg_input_addr}, 
        {"rs2", reg_input_size}, 
        {"rs3", reg_activate_row}, 
    };
}

Inst LegacyInstructionWriter::getCIMSetInst(int reg_single_group_id, int reg_mask_addr, int flag_group_broadcast ) {
    return {
        {"class", 0b00}, 
        {"type", 0b01}, 
        {"group_broadcast", flag_group_broadcast}, 
        {"rs1", reg_single_group_id}, 
        {"rs2", reg_mask_addr}
    };
}

Inst LegacyInstructionWriter::getCIMOutputInst(int reg_out_n, int reg_out_mask_addr, int reg_out_addr, int flag_outsum, int flag_outsum_move ) {
    return {
        {"class", 0b00}, 
        {"type", 0b10}, 
        {"outsum_move", flag_outsum_move}, 
        {"outsum", flag_outsum}, 
        {"rs1", reg_out_n}, 
        {"rs2", reg_out_mask_addr}, 
        {"rd", reg_out_addr}
    };
}

Inst LegacyInstructionWriter::getCIMTransferInst(int reg_src_addr, int reg_out_n, int reg_out_mask_addr, int reg_buffer_addr, int reg_dst_addr) {
    return {
        {"class", 0b00}, 
        {"type", 0b11}, 
        {"rs1", reg_src_addr}, 
        {"rs2", reg_out_n}, 
        {"rs3", reg_out_mask_addr}, 
        {"rs4", reg_buffer_addr}, 
        {"rd", reg_dst_addr}
    };
}

Inst LegacyInstructionWriter::getSendInst(int reg_src_addr, int reg_dst_addr, int reg_size, int reg_core_id, int reg_transfer_id) {
    return {
        {"opcode", 0},
    };
}

Inst LegacyInstructionWriter::getRecvInst(int reg_src_addr, int reg_dst_addr, int reg_size, int reg_core_id, int reg_transfer_id) {
    return {
        {"opcode", 0},
    };
}