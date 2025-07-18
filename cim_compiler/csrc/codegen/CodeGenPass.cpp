#include "cim/Dialect.h"
#include "cim/Passes.h"
#include "cim/ShapeInferenceInterface.h"
#include "cimisa/Dialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <stdexcept>


#include "common/macros.h"
#include "codegen/InstructionWriter.h"

#define SPECIAL_REG_INPUT_BIT_WIDTH 0
#define SPECIAL_REG_OUTPUT_BIT_WIDTH 1
#define SPECIAL_REG_WEIGHT_BIT_WIDTH 2

#define SPECIAL_REG_SIMD_INPUT_1_BIT_WIDTH 16
#define SPECIAL_REG_SIMD_INPUT_2_BIT_WIDTH 17
#define SPECIAL_REG_SIMD_INPUT_3_BIT_WIDTH 18
#define SPECIAL_REG_SIMD_INPUT_4_BIT_WIDTH 19
#define SPECIAL_REG_SIMD_OUTPUT_BIT_WIDTH 20

using namespace mlir;
using namespace cim;

namespace {
using namespace std;

static const boost::property_tree::ptree &
get_item(const boost::property_tree::ptree &ast, int index) {
  auto it = ast.begin();
  std::advance(it, index);
  return it->second;
}

template <typename Ty>
Ty safe_get_as(const boost::property_tree::ptree &ast, const std::string &key) {
  if (ast.count(key)) {
    return ast.get<Ty>(key);
  } else {
    std::cerr << "[safe_get_] Key error: " << key << std::endl;
    std::exit(1);
  }
}
const boost::property_tree::ptree &
safe_get_child(const boost::property_tree::ptree &ast, const std::string &key) {
  if (ast.count(key)) {
    return ast.get_child(key);
  } else {
    std::cerr << "[safe_get_child] Key error: " << key << std::endl;
    std::exit(1);
    return ast;
  }
}
static std::map<std::string, int> memory_addr_list;
static std::map<std::string, int> memory_size_list;
static void getMemoryAddrList(std::string config_path) {
  boost::property_tree::ptree ast;
  boost::property_tree::read_json(config_path, ast);

  LOG_DEBUG << "getMemoryAddrList";
  auto json_memory_list = safe_get_child(ast, "memory_list");
  for (const auto &pair : json_memory_list) {
    auto json_memory = pair.second;
    std::string name = safe_get_as<std::string>(json_memory, "identifier");
    auto json_address = safe_get_child(json_memory, "addressing");
    int offset = safe_get_as<int>(json_address, "offset_byte");
    int size = safe_get_as<int>(json_address, "size_byte");

    memory_addr_list[name] = offset;
    memory_size_list[name] = size;
    LOG_DEBUG << "name: " << name << " offset: " << offset << " size: " << size;
  }

}

static int getReg(std::unordered_map<llvm::hash_code, int> &regmap,
                  mlir::Value value) {
  if (regmap.count(mlir::hash_value(value))) {
    return regmap[mlir::hash_value(value)];
  } else {
    std::cerr << "error: can't find register for " << mlir::hash_value(value)
              << std::endl;
    return -1;
  }
}

class InstList {
  private:
    std::vector<std::vector<Inst>> block_inst_list;
    std::vector<Block *> blocks;
    std::map<Block*, Block*> jump_block_to_block;
    std::map<Block *, int> block_to_index;
    int current_block_idx = -1;
  public:
    InstList(std::vector<Block *> blocks) {
      block_inst_list.resize(blocks.size());
      this->blocks = blocks;
      for (int i = 0; i < (int)blocks.size(); i++) {
        block_to_index[blocks[i]] = i;
      }
    }
    std::vector<Block *> getBlocks() {
      return blocks;
    }
    std::vector<Inst> getAllInst() {
      std::vector<Inst> all_inst;
      for (int i = 0; i < (int)block_inst_list.size(); i++) {
        all_inst.insert(all_inst.end(), block_inst_list[i].begin(), block_inst_list[i].end());
      }
      return all_inst;
    }
    void setJumpBlockToBlock(Block* from_block, Block* to_block) {
      jump_block_to_block[from_block] = to_block;
    }
    Block* getJumpToBlock(Block* from_block) {
      return jump_block_to_block[from_block];
    }
    void setCurrentBlockIdx(int block_index) {
      current_block_idx = block_index;
    }
    void setCurrentBlock(Block* block) {
      setCurrentBlockIdx(block_to_index[block]);
    }
    void pushInst(Inst inst, int block_index) {
      assert(block_index >= 0 && block_index < (int)block_inst_list.size());
      block_inst_list[block_index].push_back(inst);
    }
    void push_back(Inst inst) {
      pushInst(inst, current_block_idx);
    }
    std::pair<int, int> getRelPosByAbsPos(int abs_pos) {
      int block_index = 0;
      for (int i = 0; i < (int)block_inst_list.size(); i++) {
        if (abs_pos < block_inst_list[i].size()) {
          return std::pair<int, int>(i, abs_pos);
        }
        abs_pos -= block_inst_list[i].size();
      }
      std::cerr << "error: can't find inst by abs pos " << abs_pos << std::endl;
      std::exit(1);
    }
    void insertInst(Inst inst, int abs_pos) {
      std::pair<int, int> block_idx_and_rel_pos = getRelPosByAbsPos(abs_pos);
      int block_idx = block_idx_and_rel_pos.first;
      int rel_pos = block_idx_and_rel_pos.second;
      block_inst_list[block_idx].insert(block_inst_list[block_idx].begin() + rel_pos, inst);
    }
    Inst& getInst(int abs_pos) {
      std::pair<int, int> block_idx_and_rel_pos = getRelPosByAbsPos(abs_pos);
      int block_idx = block_idx_and_rel_pos.first;
      int rel_pos = block_idx_and_rel_pos.second;
      return block_inst_list[block_idx][rel_pos];
    }
    void setInst(Inst inst, int abs_pos) {
      std::pair<int, int> block_idx_and_rel_pos = getRelPosByAbsPos(abs_pos);
      int block_idx = block_idx_and_rel_pos.first;
      int rel_pos = block_idx_and_rel_pos.second;
      block_inst_list[block_idx][rel_pos] = inst;
    }
    int size(){
      int size = 0;
      for (int i = 0; i < (int)block_inst_list.size(); i++) {
        size += block_inst_list[i].size();
      }
      return size;
    }

    std::map<Block *, int> getBlockBeginToLine() {
      std::map<Block *, int> block_begin_to_line;
      int line = 0;
      for (int i = 0; i < (int)blocks.size(); i++) {
        block_begin_to_line[blocks[i]] = line;
        line += block_inst_list[i].size();
      }
      return block_begin_to_line;
    }
    std::map<Block *, int> getBlockEndToLine() {
      std::map<Block *, int> block_end_to_line;
      int line = 0;
      for (int i = 0; i < (int)blocks.size(); i++) {
        line += block_inst_list[i].size();
        block_end_to_line[blocks[i]] = line - 1;
      }
      return block_end_to_line;
    }
};

static void codeGen(mlir::arith::ConstantOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int value = cast<IntegerAttr>(op.getValueAttr()).getInt();
  int reg = getReg(regmap, op.getResult());
  def.insert(reg);
  Inst inst = writer.getGeneralLIInst(reg, value);
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::GeneralRegLiOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int64_t value = op.getValue().getSExtValue();
  int reg = getReg(regmap, op.getResult());
  def.insert(reg);
  Inst inst = writer.getGeneralLIInst(reg, value);
  instr_list.push_back(inst);
}

template <typename Ty>
static void codeGenArith(Ty op,
                         InstructionWriter &writer,
                         std::unordered_map<llvm::hash_code, int> &regmap,
                         InstList &instr_list, std::set<int> &def,
                         std::set<int> &use) {
  int rs1 = getReg(regmap, op.getOperand(0));
  int rs2 = getReg(regmap, op.getOperand(1));
  int rd = getReg(regmap, op.getResult());
  def.insert(rd);
  use.insert(rs1);
  use.insert(rs2);

  int opcode = 0b000;
  if constexpr (std::is_same<Ty, mlir::arith::AddIOp>::value) {
    opcode = 0b000;
  } else if constexpr (std::is_same<Ty, mlir::arith::SubIOp>::value) {
    opcode = 0b001;
  } else if constexpr (std::is_same<Ty, mlir::arith::MulIOp>::value) {
    opcode = 0b010;
  } else if constexpr (std::is_same<Ty, mlir::arith::DivSIOp>::value) {
    opcode = 0b011;
  } else if constexpr (std::is_same<Ty, mlir::arith::RemSIOp>::value) {
    opcode = 0b111;
  } else if constexpr (std::is_same<Ty, mlir::arith::MinSIOp>::value) {
    opcode = 0b1000;
  } else if constexpr (std::is_same<Ty, mlir::arith::MaxSIOp>::value) {
    opcode = 0b1001;
 
  // Logical
  } else if constexpr (std::is_same<Ty, mlir::arith::AndIOp>::value) {
    opcode = 0b1010;
  } else if constexpr (std::is_same<Ty, mlir::arith::OrIOp>::value) {
    opcode = 0b1011;
  } else if constexpr (std::is_same<Ty, mlir::arith::CmpIOp>::value) {
    auto predicate = op.getPredicate();
    if (predicate == arith::CmpIPredicate::eq) {
      opcode = 0b1100;
    } else if (predicate == arith::CmpIPredicate::ne) {
      opcode = 0b1101;
    } else if (predicate == arith::CmpIPredicate::sgt) {
      opcode = 0b1110;
    } else if (predicate == arith::CmpIPredicate::slt) {
      opcode = 0b1111;
    } else {
      std::cerr << "error: unsupport predicate" << std::endl;
      std::exit(1);
    }  
  } else {
    std::cerr << "Unsupport arith op!" << std::endl;
    std::exit(1);
  }

  Inst inst = writer.getRRInst(opcode, rs1, rs2, rd);
  instr_list.push_back(inst);
}

template <typename Ty>
static void codeGenRI(Ty op,
                      InstructionWriter &writer,
                      std::unordered_map<llvm::hash_code, int> &regmap,
                      InstList &instr_list, std::set<int> &def,
                      std::set<int> &use) {
  int rs = getReg(regmap, op.getOperand());
  int rd = getReg(regmap, op.getResult());
  int64_t imm = op.getConstant().getSExtValue();

  def.insert(rd);
  use.insert(rs);

  int opcode = 0b000;
  if constexpr (std::is_same<Ty, mlir::cimisa::RIAddIOp>::value) {
    opcode = 0b000;
  } else if constexpr (std::is_same<Ty, mlir::cimisa::RISubIOp>::value) {
    opcode = 0b001;
  } else if constexpr (std::is_same<Ty, mlir::cimisa::RIMulIOp>::value) {
    opcode = 0b010;
  } else if constexpr (std::is_same<Ty, mlir::cimisa::RIDivSIOp>::value) {
    opcode = 0b011;
  } else if constexpr (std::is_same<Ty, mlir::cimisa::RIRemSIOp>::value) {
    opcode = 0b111;
  } else if constexpr (std::is_same<Ty, mlir::cimisa::RIMinSIOp>::value) {
    opcode = 0b1000;
  } else {
    std::cerr << "Unsupport arith ri op!" << std::endl;
    std::exit(1);
  }

  Inst inst = writer.getRIInst(opcode, rs, rd, imm);

  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::SIMDOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {

  int opcode = static_cast<int>(op.getOpCode());
  int numInputs = static_cast<int>(op.getNumInputs());

  int num_operands = op.getNumOperands();
  int in1_reg = getReg(regmap, op.getOperand(0));
  int out_reg = getReg(regmap, op.getOperand(num_operands - 2));
  int size_reg = getReg(regmap, op.getOperand(num_operands - 1));
  use.insert(in1_reg);
  use.insert(out_reg);
  use.insert(size_reg);

  int in2_reg = 0;
  if (numInputs > 1) {
    in2_reg = getReg(regmap, op.getOperand(1));
    use.insert(in2_reg);
  }
  Inst inst = writer.getSIMDInst(opcode, numInputs, in1_reg, in2_reg, size_reg, out_reg);
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::ReduceOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {

  int opcode = static_cast<int>(op.getOpCode());

  int src_reg = getReg(regmap, op.getOperand(0));
  int dst_reg = getReg(regmap, op.getOperand(1));
  int size_reg = getReg(regmap, op.getOperand(2));
  use.insert(src_reg);
  use.insert(dst_reg);
  use.insert(size_reg);

  Inst inst = writer.getReduceInst(opcode, src_reg, dst_reg, size_reg);
  instr_list.push_back(inst);
}

/*
  PrintOp
*/

static void codeGen(mlir::cim::PrintOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {

  int rs = getReg(regmap, op.getOperand());
  use.insert(rs);
  Inst inst = writer.getPrintInst(rs);
  instr_list.push_back(inst);
}

static void codeGen(mlir::cim::DebugOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  Inst inst = writer.getDebugInst();
  instr_list.push_back(inst);
}

/*
  Memory
  TransOp, LoadOp, StoreOp
*/

static void codeGen(mlir::cimisa::TransOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int rs = getReg(regmap, op.getOperand(0));
  int rd = getReg(regmap, op.getOperand(1));
  int size = getReg(regmap, op.getOperand(2));
  int32_t imm = op.getImm();
  bool src_offset_flag = op.getSrcOffsetFlag();
  bool dst_offset_flag = op.getDstOffsetFlag();
  use.insert(rs);
  use.insert(rd);
  use.insert(size);
  Inst inst = writer.getTransInst(rs, rd, size, imm, src_offset_flag, dst_offset_flag);
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::LoadOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int rs1 = getReg(regmap, op.getOperand());
  int rs2 = getReg(regmap, op.getResult());
  int64_t imm = op.getConstant().getSExtValue();
  use.insert(rs1);
  def.insert(rs2);
  Inst inst = writer.getLoadInst(rs1, rs2, imm);
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::StoreOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int rs1 = getReg(regmap, op.getOperand(0));
  int rs2 = getReg(regmap, op.getOperand(1));
  int64_t imm = op.getConstant().getSExtValue();
  use.insert(rs1);
  use.insert(rs2);
  Inst inst = writer.getStoreInst(rs1, rs2, imm);
  instr_list.push_back(inst);
}

/*
  CIMCompute
*/

static void codeGen(mlir::cimisa::CIMComputeOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int input_addr_reg = getReg(regmap, op.getOperand(0));
  // int output_addr_reg = getReg(regmap, op.getOperand(1));
  int activate_row_reg = getReg(regmap, op.getOperand(1));
  int input_size_reg = getReg(regmap, op.getOperand(2));
  Value batch_size_value = op.getBatchSize();
  int batch_flag = static_cast<int>(op.getBatchFlag());

  use.insert(input_addr_reg);
  use.insert(activate_row_reg);
  use.insert(input_size_reg);

  int batch_size_reg = 1024;
  if (batch_flag==1){
    batch_size_reg = getReg(regmap, batch_size_value);
    use.insert(batch_size_reg);
  }
  
  Inst inst = writer.getCIMComputeInst(
    input_addr_reg, 
    input_size_reg, 
    activate_row_reg,
    batch_size_reg,
    static_cast<int>(op.getAccFlag()),
    static_cast<int>(op.getValueSparseFlag()),
    static_cast<int>(op.getBitSparseFlag()),
    1,
    0,
    static_cast<int>(op.getBatchFlag())
  );
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::CIMOutputOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int out_n = getReg(regmap, op.getOperand(0));
  int out_mask_addr = getReg(regmap, op.getOperand(1));
  int output_addr_reg = getReg(regmap, op.getOperand(2));
  use.insert(out_n);
  use.insert(out_mask_addr);
  use.insert(output_addr_reg);
  Inst inst = writer.getCIMOutputInst(
    /*reg_out_n=*/ out_n,
    /*reg_out_mask_addr=*/ out_mask_addr,
    /*reg_out_addr=*/ output_addr_reg,
    /*flag_outsum=*/ 0,
    /*flag_outsum_move=*/ 0
  );
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::CIMOutputSumOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int out_n = getReg(regmap, op.getOperand(0));
  int out_mask_addr = getReg(regmap, op.getOperand(1));
  int output_addr_reg = getReg(regmap, op.getOperand(2));
  use.insert(out_n);
  use.insert(out_mask_addr);
  use.insert(output_addr_reg);
  Inst inst = writer.getCIMOutputInst(
    /*reg_out_n=*/ out_n,
    /*reg_out_mask_addr=*/ out_mask_addr,
    /*reg_out_addr=*/ output_addr_reg,
    /*flag_outsum=*/ 1,
    /*flag_outsum_move=*/ 0
  );
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::CIMTransferOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int src_addr = getReg(regmap, op.getOperand(0));
  int output_number = getReg(regmap, op.getOperand(1));
  int output_mask_addr = getReg(regmap, op.getOperand(2));
  int buffer_addr = getReg(regmap, op.getOperand(3));
  int dst_addr = getReg(regmap, op.getOperand(4));
  use.insert(src_addr);
  use.insert(output_number);
  use.insert(output_mask_addr);
  use.insert(buffer_addr);
  use.insert(dst_addr);
  Inst inst = writer.getCIMTransferInst(
    /*reg_src_addr=*/ src_addr,
    /*reg_out_n=*/ output_number,
    /*reg_out_mask_addr=*/ output_mask_addr,
    /*reg_buffer_addr=*/ buffer_addr,
    /*reg_dst_addr=*/ dst_addr
  );
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::CIMSetOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int mask_addr = getReg(regmap, op.getOperand());
  use.insert(mask_addr);
  Inst inst = writer.getCIMSetInst(
    /*reg_single_group_id=*/ 0, 
    /*reg_mask_addr=*/ mask_addr, 
    /*flag_group_broadcast=*/ 1
  );
  instr_list.push_back(inst);
}

/*
  ControlFlow
  BranchOp, CondBranchOp
*/

static void
codeGen(mlir::cf::BranchOp op, 
        InstructionWriter &writer,
        std::unordered_map<llvm::hash_code, int> &regmap,
        std::unordered_map<llvm::hash_code, int> &block_args_special_reg_map,
        InstList &instr_list, std::set<int> &def, std::set<int> &use) {
  Block *dest_block = op.getDest();
  auto dest_args = dest_block->getArguments();
  auto dest_operands = op.getDestOperands();
  int args_size = dest_args.size();
  int operands_size = dest_operands.size();
  if (args_size != operands_size) {
    std::cerr << "error: args_size != operands_size" << args_size << " vs "
              << operands_size << std::endl;
    std::exit(1);
  }

  for (int i = 0; i < args_size; i++) {
    mlir::Value arg = llvm::cast<mlir::Value>(dest_args[i]);
    mlir::Value operand = dest_operands[i];

    int operand_reg = getReg(regmap, operand);
    if (block_args_special_reg_map.count(mlir::hash_value(arg))) {
      int arg_special_reg = getReg(block_args_special_reg_map, arg);
      // assign into special register
      Inst to_special_inst = writer.getGeneralToSpecialAssignInst(operand_reg, arg_special_reg);
      instr_list.push_back(to_special_inst);
      use.insert(operand_reg);
    } else {
      int arg_reg = getReg(regmap, arg);
      // we don't have a move instruct between general register, so we use addi
      // zero instead
      Inst add_zero_inst = writer.getRIInst(0b00, operand_reg, arg_reg, 0);
      // assign args into special register
      instr_list.push_back(add_zero_inst);
      use.insert(operand_reg);
      def.insert(arg_reg);
    }

    // use.insert(arg_reg);
  }
  Inst inst = writer.getJumpInst(-1);
  instr_list.push_back(inst);
}

static void codeGen(mlir::cf::CondBranchOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {

  arith::CmpIOp cmpi_op = op.getOperand(0).getDefiningOp<arith::CmpIOp>();
  if (!cmpi_op) {
    std::cerr << "cmpi_op is null!" << std::endl;
  }
  auto predicate = cmpi_op.getPredicate();
  int compare = 0;
  if (predicate == arith::CmpIPredicate::eq) {
    compare = 0;
  } else if (predicate == arith::CmpIPredicate::ne) {
    compare = 1;
  } else if (predicate == arith::CmpIPredicate::sgt) {
    compare = 2;
  } else if (predicate == arith::CmpIPredicate::slt) {
    compare = 3;
  } else {
    std::cerr << "error: unsupport predicate" << std::endl;
    std::exit(1);
  }
  mlir::Value lhs = cmpi_op.getLhs();
  mlir::Value rhs = cmpi_op.getRhs();

  int lhs_reg = getReg(regmap, cmpi_op.getLhs());
  int rhs_reg = getReg(regmap, cmpi_op.getRhs());
  Inst inst = writer.getBranchInst(compare, lhs_reg, rhs_reg, -1);
  instr_list.push_back(inst);
  use.insert(lhs_reg);
  use.insert(rhs_reg);
}

static void codeGen(mlir::cimisa::SpecialRegLiOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {

  int special_reg = static_cast<int>(op.getSpecialReg());
  int set_value = static_cast<int>(op.getSetValue());
  Inst inst = writer.getSpecialLIInst(special_reg, set_value);
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::SpecialRegAssignOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {

  int special_reg = static_cast<int>(op.getSpecialReg());
  int from_general_reg = getReg(regmap, op.getOperand());
  use.insert(from_general_reg);
  Inst inst = writer.getGeneralToSpecialAssignInst(from_general_reg, special_reg);
  instr_list.push_back(inst);
}

/*
Communication
 */

static void codeGen(mlir::cimisa::SendOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int src_addr_reg = getReg(regmap, op.getOperand(0));
  int dst_addr_reg = getReg(regmap, op.getOperand(1));
  int size_reg = getReg(regmap, op.getOperand(2));
  int core_reg = getReg(regmap, op.getOperand(3));
  int transfer_id_reg = getReg(regmap, op.getOperand(4));
 
  use.insert(src_addr_reg);
  use.insert(dst_addr_reg);
  use.insert(size_reg);
  use.insert(core_reg);
  use.insert(transfer_id_reg);

  Inst inst = writer.getSendInst(
    src_addr_reg, 
    dst_addr_reg, 
    size_reg,
    core_reg,
    transfer_id_reg
  );
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::RecvOp op,
                    InstructionWriter &writer,
                    std::unordered_map<llvm::hash_code, int> &regmap,
                    InstList &instr_list, std::set<int> &def,
                    std::set<int> &use) {
  int src_addr_reg = getReg(regmap, op.getOperand(0));
  int dst_addr_reg = getReg(regmap, op.getOperand(1));
  int size_reg = getReg(regmap, op.getOperand(2));
  int core_reg = getReg(regmap, op.getOperand(3));
  int transfer_id_reg = getReg(regmap, op.getOperand(4));
 
  use.insert(src_addr_reg);
  use.insert(dst_addr_reg);
  use.insert(size_reg);
  use.insert(core_reg);
  use.insert(transfer_id_reg);

  Inst inst = writer.getRecvInst(
    src_addr_reg, 
    dst_addr_reg, 
    size_reg,
    core_reg,
    transfer_id_reg
  );
  instr_list.push_back(inst);
}
/*
  CodeGen For Operator Finish!
*/

static void block_dfs(Block *block, std::vector<Block *> &blocks,
                      std::unordered_map<Block *, bool> &blocks_completed) {
  blocks.push_back(block);
  blocks_completed[block] = true;

  auto terminator = block->getTerminator();
  if (auto _op = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
    if (blocks_completed[_op.getFalseDest()]) {
      std::cerr << "Error: false-dest block already completed" << std::endl;
      std::exit(1);
    } else {
      block_dfs(_op.getFalseDest(), blocks, blocks_completed);
    }

    if (!blocks_completed[_op.getTrueDest()]) {
      block_dfs(_op.getTrueDest(), blocks, blocks_completed);
    }
  }
}

static std::vector<Block *> getBlockList(mlir::func::FuncOp func) {
  LOG_DEBUG << "getBlockList begin";
  auto regions = func->getRegions();
  if (regions.size() > 1) {
    LOG_ERROR << "regions.size()" << regions.size();
    std::exit(1);
  }
  Region &region = regions.front();
  std::vector<Block *> blocks;
  std::unordered_map<Block *, bool> blocks_completed;
  for (Block &block : region.getBlocks()) {
    blocks_completed[&block] = false;
  }

  int block_cnt = 0;
  int total_block_cnt = region.getBlocks().size();
  while (blocks.size() < total_block_cnt) {
    Block *selected_block;
    if (block_cnt == 0) {

      // find the block with no predeccessor
      for (Block &block : region.getBlocks()) {
        int num_predecessors = 0;
        for (auto *b : block.getPredecessors())
          num_predecessors++;
        if (num_predecessors == 0) {
          selected_block = &block;
          break;
        }
      }
      block_cnt = 1;

    } else {

      // find the block with no false-dest predecessor
      int find = 0;
      for (Block &block : region.getBlocks()) {
        if (blocks_completed[&block])
          continue;
        int flag = 1;
        for (auto *b : block.getPredecessors()) {
          auto terminator = b->getTerminator();
          if (auto _op = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
            if (_op.getFalseDest() == &block) {
              flag = 0;
              break;
            }
          }
        }
        if (flag) {
          find = 1;
          selected_block = &block;
          break;
        }
      }
      if (!find) {
        LOG_ERROR << "can't find block with no false-dest predecessor";
        std::exit(1);
      }
      block_cnt++;

    } // end if block_cnt==0

    block_dfs(selected_block, blocks, blocks_completed);

    // False-dest chain
    // while(true){
    //   auto terminator = selected_block->getTerminator();
    //   if (auto _op = dyn_cast<mlir::cf::CondBranchOp>(terminator)){
    //     if (blocks_completed[_op.getFalseDest()]){
    //       std::cerr << "Error: false-dest block already completed" <<
    //       std::endl; std::exit(1);
    //     }
    //     blocks.push_back(_op.getFalseDest());
    //     blocks_completed[_op.getFalseDest()] = true;
    //     selected_block = _op.getFalseDest();
    //     block_cnt++;
    //   }else{
    //     break;
    //   }
    // }
  }
  LOG_DEBUG << "getBlockList end";
  return blocks;
}

static void codeGenForBlockArgs(
    Block *block, InstructionWriter &writer,
    std::unordered_map<llvm::hash_code, int> &general_reg_map,
    std::unordered_map<llvm::hash_code, int> &block_args_special_reg_map,
    InstList &instr_list, 
    std::set<int> &write, std::set<int> &read) {

  auto args = block->getArguments();
  for (int i = 0; i < args.size(); i++) {
    mlir::Value arg = llvm::cast<mlir::Value>(args[i]);
    if (block_args_special_reg_map.count(mlir::hash_value(arg))) {
      int special_reg = getReg(block_args_special_reg_map, arg);
      int general_reg = getReg(general_reg_map, arg);
      Inst inst = writer.getSpecialToGeneralAssignInst(general_reg, special_reg);
      instr_list.push_back(inst);
      write.insert(general_reg);
    }
  }
}

static void
codeGen(std::vector<Block *> &blocks,
        InstructionWriter &writer,
        std::unordered_map<llvm::hash_code, int> &regmap,
        std::unordered_map<llvm::hash_code, int> &block_args_special_reg_map,
        InstList &instr_list, 
        std::map<Block *, std::set<int>> &def,
        std::map<Block *, std::set<int>> &use, std::map<int, int> &twin_reg) {

  for (Block *block : blocks) {
    // iter all Operation in this block
    instr_list.setCurrentBlock(block);

    std::set<int> _write;
    std::set<int> _read;

    codeGenForBlockArgs(block, writer, regmap, block_args_special_reg_map, instr_list,
                        _write, _read);
    for (Operation &op_obj : block->getOperations()) {
      Operation *op = &op_obj;

      if (auto _op = dyn_cast<mlir::arith::ConstantOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::GeneralRegLiOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);

        // RR
      } else if (auto _op = dyn_cast<mlir::arith::AddIOp>(op)) {
        codeGenArith<mlir::arith::AddIOp>(_op, writer, regmap, instr_list, _write,
                                          _read);
      } else if (auto _op = dyn_cast<mlir::arith::SubIOp>(op)) {
        codeGenArith<mlir::arith::SubIOp>(_op, writer, regmap, instr_list, _write,
                                          _read);
      } else if (auto _op = dyn_cast<mlir::arith::MulIOp>(op)) {
        codeGenArith<mlir::arith::MulIOp>(_op, writer, regmap, instr_list, _write,
                                          _read);
      } else if (auto _op = dyn_cast<mlir::arith::DivSIOp>(op)) {
        codeGenArith<mlir::arith::DivSIOp>(_op, writer, regmap, instr_list, _write,
                                           _read);
      } else if (auto _op = dyn_cast<mlir::arith::RemSIOp>(op)) {
        codeGenArith<mlir::arith::RemSIOp>(_op, writer, regmap, instr_list, _write,
                                           _read);
      } else if (auto _op = dyn_cast<mlir::arith::MinSIOp>(op)) {
        codeGenArith<mlir::arith::MinSIOp>(_op, writer, regmap, instr_list, _write,
                                           _read);
      } else if (auto _op = dyn_cast<mlir::arith::MaxSIOp>(op)) {
        codeGenArith<mlir::arith::MaxSIOp>(_op, writer, regmap, instr_list, _write,
                                           _read);
      } else if (auto _op = dyn_cast<mlir::arith::AndIOp>(op)) {
        codeGenArith<mlir::arith::AndIOp>(_op, writer, regmap, instr_list, _write,
                                           _read);
      } else if (auto _op = dyn_cast<mlir::arith::OrIOp>(op)) {
        codeGenArith<mlir::arith::OrIOp>(_op, writer, regmap, instr_list, _write,
                                           _read);
      } else if (auto _op = dyn_cast<mlir::arith::CmpIOp>(op)) {
        codeGenArith<mlir::arith::CmpIOp>(_op, writer, regmap, instr_list, _write,
                                           _read);

        // RI
      } else if (auto _op = dyn_cast<mlir::cimisa::RIAddIOp>(op)) {
        codeGenRI<mlir::cimisa::RIAddIOp>(_op, writer, regmap, instr_list, _write,
                                          _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::RISubIOp>(op)) {
        codeGenRI<mlir::cimisa::RISubIOp>(_op, writer, regmap, instr_list, _write,
                                          _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::RIMulIOp>(op)) {
        codeGenRI<mlir::cimisa::RIMulIOp>(_op, writer, regmap, instr_list, _write,
                                          _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::RIDivSIOp>(op)) {
        codeGenRI<mlir::cimisa::RIDivSIOp>(_op, writer, regmap, instr_list, _write,
                                           _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::RIRemSIOp>(op)) {
        codeGenRI<mlir::cimisa::RIRemSIOp>(_op, writer, regmap, instr_list, _write,
                                           _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::RIMinSIOp>(op)) {
        codeGenRI<mlir::cimisa::RIMinSIOp>(_op, writer, regmap, instr_list, _write,
                                           _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::SIMDOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::ReduceOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::CIMComputeOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::CIMOutputOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::CIMOutputSumOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::CIMTransferOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::CIMSetOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::TransOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::LoadOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::StoreOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cim::PrintOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cim::DebugOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cf::CondBranchOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
        // jump2line[op] = instr_list.size() - 1;
        Block *dest_block = _op.getTrueDest();
        instr_list.setJumpBlockToBlock(block, dest_block);
      } else if (auto _op = dyn_cast<mlir::cf::BranchOp>(op)) {
        codeGen(_op, writer, regmap, block_args_special_reg_map, instr_list, _write,
                _read);
        // jump2line[op] = instr_list.size() - 1;
        Block *dest_block = _op.getDest();
        instr_list.setJumpBlockToBlock(block, dest_block);
      } else if (auto _op = dyn_cast<mlir::cimisa::SpecialRegLiOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::SpecialRegAssignOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::SendOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::cimisa::RecvOp>(op)) {
        codeGen(_op, writer, regmap, instr_list, _write, _read);
      } else if (auto _op = dyn_cast<mlir::func::ReturnOp>(op)) {
        // do nothing
      } else {
        std::cerr << "error: unsupport operator: "
                  << op->getName().getStringRef().str() << std::endl;
      }
    }

    std::set<int> _def;
    std::set<int> _use;
    _def.insert(_write.begin(), _write.end());
    std::set_difference(_read.begin(), _read.end(), _write.begin(),
                        _write.end(), std::inserter(_use, _use.begin()));

    def[block] = _def;
    use[block] = _use;
  }
}

static void
mapValueAsRegister(mlir::Value &value,
                   std::unordered_map<llvm::hash_code, int> &mapping,
                   int &reg_cnt) {
  llvm::hash_code hash_code = mlir::hash_value(value);
  if (!mapping.count(hash_code)) {
    mapping[hash_code] = reg_cnt++;
  } else {
    LOG_DEBUG << "register already allocted! " << mapping[hash_code];
  }
}

template <typename Ty>
static void
mapResultAsRegister(Ty op, std::unordered_map<llvm::hash_code, int> &mapping,
                    int &reg_cnt) {
  mlir::Value result = op.getResult();
  mapValueAsRegister(result, mapping, reg_cnt);
}

static void _getRegisterMappingAliasBetweenBasicBlock(
    mlir::func::FuncOp func, std::unordered_map<llvm::hash_code, int> &mapping,
    int &reg_cnt) {
  auto regions = func->getRegions();
  LOG_DEBUG << "regions.size()" << regions.size();
  for (Region &region : regions) {
    // for each block
    for (Block &block : region.getBlocks()) {

      auto block_arguments = block.getArguments();
      for (int arg_i = 0; arg_i < block_arguments.size(); arg_i++) {
        std::vector<mlir::Value> alias_values;

        BlockArgument block_arg = block_arguments[arg_i];
        mlir::Value block_arg_val = llvm::cast<mlir::Value>(block_arg);
        alias_values.push_back(block_arg_val);

        // get all predecessor block, record args
        for (Block *pred : block.getPredecessors()) {
          TypeSwitch<Operation *>(pred->getTerminator())
              .Case<cf::BranchOp>([&](auto branch) {
                Value caller_operand = branch.getOperand(arg_i);
                alias_values.push_back(caller_operand);
              });
        }
        LOG_DEBUG << "reg = " << reg_cnt
                  << ", alias_values.size() = " << alias_values.size();
        // map all alias to same
        for (mlir::Value &alias : alias_values) {
          int _reg_cnt = reg_cnt;
          mapValueAsRegister(alias, mapping, _reg_cnt);
        }
        reg_cnt++;
      }
    }
  }
}

struct BlockWithLifeTime {
  Block *block;
  int lifetime;
};
static void _getRegisterMappingForBlockArgs(
    mlir::func::FuncOp func, std::unordered_map<llvm::hash_code, int> &mapping,
    std::unordered_map<llvm::hash_code, int> &special_reg_mapping,
    int &reg_cnt) {
  std::vector<Block *> blocks = getBlockList(func);
  std::map<Block *, int> block2index;
  // get block2index
  for (int i = 0; i < blocks.size(); i++) {
    block2index[blocks[i]] = i;
  }

  std::vector<BlockWithLifeTime> block_with_lifetime;

  for (Block *block : blocks) {
    auto block_arguments = block->getArguments();
    if (block_arguments.size() == 0) {
      BlockWithLifeTime block_with_lifetime_obj = {block, -1};
      block_with_lifetime.push_back(block_with_lifetime_obj);
      continue;
    }

    std::vector<Block *> predecessors;
    for (auto *b : block->getPredecessors())
      predecessors.push_back(b);
    if (predecessors.size() == 0) {
      BlockWithLifeTime block_with_lifetime_obj = {block, -1};
      block_with_lifetime.push_back(block_with_lifetime_obj);
      continue;
    }

    int lifetime_begin = block2index[block];
    int lifetime_end = block2index[block];
    for (Block *pred : predecessors) {
      lifetime_begin = std::min(lifetime_begin, block2index[pred]);
      lifetime_end = std::max(lifetime_end, block2index[pred]);
    }
    int lifetime = lifetime_end - lifetime_begin;
    BlockWithLifeTime block_with_lifetime_obj = {block, lifetime};
    block_with_lifetime.push_back(block_with_lifetime_obj);
  }

  // sort block_with_lifetime by lifetime
  std::sort(block_with_lifetime.begin(), block_with_lifetime.end(),
            [](const BlockWithLifeTime &a, const BlockWithLifeTime &b) {
              return a.lifetime > b.lifetime;
            });

  std::vector<int> vec = {9,  10, 11, 12, 13, 14, 15, 21, 22,
                          23, 24, 25, 26, 27, 28, 29, 30, 31};
  std::queue<int> usable_special_regs;
  // for (int &i : vec) usable_special_regs.push(i);
  // std::queue<int> usable_special_regs =
  // std::queue<int> usable_special_regs = {};

  for (BlockWithLifeTime &block_with_lifetime_obj : block_with_lifetime) {
    Block *block = block_with_lifetime_obj.block;
    auto block_arguments = block->getArguments();
    LOG_DEBUG << "block->lifetime = " << block_with_lifetime_obj.lifetime
              << ", num args: " << block_arguments.size();
    for (int arg_i = 0; arg_i < block_arguments.size(); arg_i++) {
      BlockArgument block_arg = block_arguments[arg_i];
      mlir::Value block_arg_val = llvm::cast<mlir::Value>(block_arg);
      int special_reg = -1;
      if (!usable_special_regs.empty()) {
        special_reg = usable_special_regs.front();
        usable_special_regs.pop();
        mapValueAsRegister(block_arg_val, special_reg_mapping, special_reg);
      }
      mapValueAsRegister(block_arg_val, mapping, reg_cnt);
      LOG_DEBUG << "    block arg: logical reg: " << reg_cnt - 1
                << ", special reg: " << special_reg;
    }
  }

}

static void
_getRegisterMappingGeneral(mlir::func::FuncOp func,
                           std::unordered_map<llvm::hash_code, int> &mapping,
                           int &reg_cnt) {

  func.walk([&](mlir::Operation *op) {
    if (auto _op = dyn_cast<mlir::arith::ConstantOp>(op)) {
      mapResultAsRegister<mlir::arith::ConstantOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::cimisa::GeneralRegLiOp>(op)) {
      mapResultAsRegister<mlir::cimisa::GeneralRegLiOp>(_op, mapping, reg_cnt);

      // RR
    } else if (auto _op = dyn_cast<mlir::arith::AddIOp>(op)) {
      mapResultAsRegister<mlir::arith::AddIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::arith::SubIOp>(op)) {
      mapResultAsRegister<mlir::arith::SubIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::arith::MulIOp>(op)) {
      mapResultAsRegister<mlir::arith::MulIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::arith::DivSIOp>(op)) {
      mapResultAsRegister<mlir::arith::DivSIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::arith::RemSIOp>(op)) {
      mapResultAsRegister<mlir::arith::RemSIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::arith::MinSIOp>(op)) {
      mapResultAsRegister<mlir::arith::MinSIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::arith::MaxSIOp>(op)) {
      mapResultAsRegister<mlir::arith::MaxSIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::arith::AndIOp>(op)) {
      mapResultAsRegister<mlir::arith::AndIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::arith::OrIOp>(op)) {
      mapResultAsRegister<mlir::arith::OrIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::arith::CmpIOp>(op)) {
      mapResultAsRegister<mlir::arith::CmpIOp>(_op, mapping, reg_cnt);

      // RI
    } else if (auto _op = dyn_cast<mlir::cimisa::RIAddIOp>(op)) {
      mapResultAsRegister<mlir::cimisa::RIAddIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::cimisa::RISubIOp>(op)) {
      mapResultAsRegister<mlir::cimisa::RISubIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::cimisa::RIMulIOp>(op)) {
      mapResultAsRegister<mlir::cimisa::RIMulIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::cimisa::RIDivSIOp>(op)) {
      mapResultAsRegister<mlir::cimisa::RIDivSIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::cimisa::RIRemSIOp>(op)) {
      mapResultAsRegister<mlir::cimisa::RIRemSIOp>(_op, mapping, reg_cnt);
    } else if (auto _op = dyn_cast<mlir::cimisa::RIMinSIOp>(op)) {
      mapResultAsRegister<mlir::cimisa::RIMinSIOp>(_op, mapping, reg_cnt);

    // Other
    } else if (auto _op = dyn_cast<mlir::cimisa::LoadOp>(op)) {
      mapResultAsRegister<mlir::cimisa::LoadOp>(_op, mapping, reg_cnt);
    }
  });
  LOG_DEBUG << "_getRegisterMappingGeneral finish";
}
static std::pair<std::unordered_map<llvm::hash_code, int>,
                 std::unordered_map<llvm::hash_code, int>>
getRegisterMapping(mlir::func::FuncOp func) {
  /*
    Step 1: get special register mapping
    Step 2: get alias rf between basic block
    Step 3: get other rf
  */
  std::unordered_map<llvm::hash_code, int> mapping;
  std::unordered_map<llvm::hash_code, int> block_args_special_reg_map;
  int reg_cnt = 1024;

  // _getRegisterMappingAliasBetweenBasicBlock(func, mapping, reg_cnt);
  _getRegisterMappingForBlockArgs(func, mapping, block_args_special_reg_map,
                                  reg_cnt);
  LOG_DEBUG << "_getRegisterMappingForBlockArgs:" << reg_cnt;
  _getRegisterMappingGeneral(func, mapping, reg_cnt);
  LOG_DEBUG << "getRegisterMapping finish";
  return std::make_pair(mapping, block_args_special_reg_map);
}

static string instToStr(Inst &inst) {
  std::string json = "{";
  for (auto it = inst.begin(); it != inst.end();) {
    std::string value = "";
    if (std::holds_alternative<int>(it->second)) {
      value = std::to_string(std::get<int>(it->second));
    } else if (std::holds_alternative<bool>(it->second)) {
      value = std::get<bool>(it->second) ? "true" : "false";
    } else {
      std::cerr << "error: unknown type" << std::endl;
      std::exit(1);
    }
    json += "\"" + it->first + "\": " + value;
    if ((++it) != inst.end()) {
      json += ", ";
    }
  }
  json += "}";
  return json;
}

static void fillJumpBranchOffset(mlir::func::FuncOp func,
                                 InstructionWriter &writer,
                                 InstList &instr_list
                                 ) {
  std::map<Block *, int> block2line = instr_list.getBlockBeginToLine();
  std::map<Block *, int> block2line_end = instr_list.getBlockEndToLine();
  for (Block *block : instr_list.getBlocks()) {
    if (auto _op = dyn_cast<mlir::cf::BranchOp>(block->getTerminator())) {
      Block *dest_block = _op.getDest();
      int target_line = block2line[dest_block];
      int current_line = block2line_end[block];
      int offset = target_line - current_line;
      // instr_list[current_line]["offset"] = offset;
      writer.setJumpOffset(instr_list.getInst(current_line), offset);
    } else if (auto _op = dyn_cast<mlir::cf::CondBranchOp>(block->getTerminator())) {
      Block *dest_block = _op.getTrueDest();
      int target_line = block2line[dest_block];
      int current_line = block2line_end[block];
      int offset = target_line - current_line;
      writer.setBranchOffset(instr_list.getInst(current_line), offset);
    }
  }
}

static void liveVariableAnalysis(std::vector<Block *> blocks,
                                 std::map<Block *, std::set<int>> &def,
                                 std::map<Block *, std::set<int>> &use,
                                 std::map<Block *, std::set<int>> &in,
                                 std::map<Block *, std::set<int>> &out) {
  Block *exit_block = blocks.back();
  if (exit_block->getSuccessors().size() != 0) {
    std::cerr << "error: exit block should have no successor" << std::endl;
    std::exit(1);
  }
  for (Block *block : blocks) {
    in[block] = {};
  }
  int change;
  do {
    change = 0;
    for (int i = 0; i < blocks.size() - 1;
         i++) { // TODO: 必须保证blocks有一个单独的exit block,且位于最后一个位置
      Block *block = blocks[i];
      std::set<int> _in = in[block];
      std::set<int> _out = out[block];
      std::set<int> _def = def[block];
      std::set<int> _use = use[block];
      std::set<int> _new_in;
      std::set<int> _new_out;

      // out[B] = \union_{S:successor of B} in[B]
      for (Block *successor : block->getSuccessors()) {
        std::set<int> _succ_in = in[successor];
        _new_out.insert(_succ_in.begin(), _succ_in.end());
      }

      // in[B] = use[B] \union (out[B] - def[B])
      std::set<int> difference;
      std::set_difference(_new_out.begin(), _new_out.end(), _def.begin(),
                          _def.end(),
                          std::inserter(difference, difference.begin()));
      std::set_union(difference.begin(), difference.end(), _use.begin(),
                     _use.end(), std::inserter(_new_in, _new_in.begin()));

      bool is_equal_in = (_in == _new_in);
      bool is_equal_out = (_out == _new_out);
      bool is_equal = is_equal_in && is_equal_out;
      if (!is_equal)
        change = 1;

      in[block] = _new_in;
      out[block] = _new_out;
    }
    // std::cout << "------" << change << std::endl;
  } while (change);
}

static bool isPrefix(const std::string &str, const std::string &prefix) {
  if (prefix.length() > str.length()) {
    return false;
  }

  std::string strPrefix = str.substr(0, prefix.length());

  return strPrefix == prefix;
}


static std::pair<int, int> get_twin_physical_reg(
    std::priority_queue<int, std::vector<int>, std::greater<int>>
        &physical_regs) {
  std::vector<int> temp_save;
  std::pair<int, int> twin;
  bool find = false;
  while (!physical_regs.empty()) {
    int top = physical_regs.top();
    if (temp_save.size() >= 1 && top == temp_save.back() + 1) {
      twin.first = temp_save.back();
      twin.second = top;
      find = true;
      temp_save.pop_back();
      physical_regs.pop();
      break;
    }
    temp_save.push_back(top);
    physical_regs.pop();
  }
  if (!find) {
    std::cerr << "error: can't find twin physical register" << std::endl;
    std::exit(1);
  }
  // move temp_save back to physical_regs
  for (int i = 0; i < temp_save.size(); i++) {
    physical_regs.push(temp_save[i]);
  }
  LOG_DEBUG << "get_twin_physical_reg: " << twin.first << " " << twin.second;
  return twin;
}

struct Interval {
  int logical_reg_id;
  int begin;
  int end;
};

struct CompareIntervalByBegin {
    bool operator()(const Interval& p1, const Interval& p2) {
        return p1.begin < p2.begin;
    }
};
struct CompareIntervalByEnd {
    bool operator()(const Interval& p1, const Interval& p2) const {
        if (p1.end != p2.end) return p1.end < p2.end;
        return p1.logical_reg_id < p2.logical_reg_id;
    }
};

static void mappingRegisterLogicalToPhysical(
    InstList &instr_list, 
    InstructionWriter &writer,
    std::map<Block *, std::set<int>> &in,
    std::map<Block *, std::set<int>> &out, 
    std::map<int, int> &twin_reg) {

  // show twin_reg
  for (const auto &[key, value] : twin_reg) {
    LOG_DEBUG << "twin_reg: " << key << " -> " << value;
  }
  std::map<int, int> two_way_twin_reg;
  for (const auto &[key, value] : twin_reg) {
    two_way_twin_reg[key] = value;
    two_way_twin_reg[value] = key;
  }

  // Step 1: get life cycle of each logical register
  std::map<int, int> logic_reg_life_begin;
  std::map<int, int> logic_reg_life_end;
  std::set<int> set_logical_regs;
  std::vector<int> logical_regs;
  for (int inst_id = 0; inst_id < instr_list.size(); inst_id++) {
    Inst inst = instr_list.getInst(inst_id);
    // skip special register instruction
    if (writer.isSpecialLi(inst))
      continue;

    for (const auto &[key, value] : inst) {
      // bool is_gen_to_spec_assign = isGeneralToSpecialAssign(inst);
      // bool is_spec_to_gen_assign = isSpecialToGeneralAssign(inst);
      // bool is_special_assign = is_gen_to_spec_assign ||
      // is_spec_to_gen_assign; bool is_reg_general = (is_special_assign &&
      // key=="rs1") || ((!is_special_assign) && (isPrefix(key, "rs") ||
      // isPrefix(key, "rd")));
      if (writer.isGeneralReg(inst, key)) {
        int reg_id = std::get<int>(value);
        if (!logic_reg_life_begin.count(reg_id)) {
          logic_reg_life_begin[reg_id] = inst_id;
          logic_reg_life_end[reg_id] = inst_id;
        } else {
          logic_reg_life_end[reg_id] = inst_id;
        }
        set_logical_regs.insert(reg_id);
      }
    }
    // LOG_DEBUG << std::endl;
  }
  std::map<Block *, int> block2line = instr_list.getBlockBeginToLine();
  std::map<Block *, int> block2line_end = instr_list.getBlockEndToLine();
  for (const auto &[block, regs] : in) {
    for (auto reg_id : regs) {
      if (!(logic_reg_life_begin.count(reg_id) &&
            logic_reg_life_end.count(reg_id))) {
        LOG_ERROR << "error: reg_id not in logic_reg_life_begin";
        std::exit(1);
      }
      int old_begin = logic_reg_life_begin[reg_id];
      int old_end = logic_reg_life_end[reg_id];
      logic_reg_life_begin[reg_id] = min(old_begin, block2line[block]);
      logic_reg_life_end[reg_id] = max(old_end, block2line_end[block]);
    }
  }
  for (const auto &[block, regs] : out) {
    for (auto reg_id : regs) {
      if (!(logic_reg_life_begin.count(reg_id) &&
            logic_reg_life_end.count(reg_id))) {
        std::cerr << "error: reg_id not in logic_reg_life_begin" << std::endl;
        std::exit(1);
      }
      int old_begin = logic_reg_life_begin[reg_id];
      int old_end = logic_reg_life_end[reg_id];
      logic_reg_life_begin[reg_id] = min(old_begin, block2line[block]);
      logic_reg_life_end[reg_id] = max(old_end, block2line_end[block]);
    }
  }
  logical_regs.assign(set_logical_regs.begin(), set_logical_regs.end());
  std::sort(logical_regs.begin(), logical_regs.end(),
            [&](int a, int b) { return a < b; });

  // Step 2: Construct a mapping from logical register to physical register
  int num_logical_regs = logic_reg_life_begin.size();
  int num_physical_regs = 31;
  std::priority_queue<int, std::vector<int>, std::greater<int>> physical_regs;
  std::map<int, int> logical_to_physical_mapping;
  int max_physical_reg_used = 0;
  std::set<int> twin_have_allocated;
  for (int i = 0; i < num_physical_regs; i++)
    physical_regs.push(i);
  
  std::set<Interval,  CompareIntervalByEnd> active;
  std::vector<Interval> intervals;
  for (int logical_reg_id : logical_regs) {
    Interval interval = {
      logical_reg_id,  // Assuming logical_reg_id is a variable
      logic_reg_life_begin[logical_reg_id],
      logic_reg_life_end[logical_reg_id]
    };
    intervals.push_back(interval);
  }
  std::sort(intervals.begin(), intervals.end(), CompareIntervalByBegin());

  std::vector<int> spill_logical_regs;
  for (int i = 0; i < intervals.size(); i++) {
    Interval interval = intervals[i];
    while (!active.empty() && active.begin()->end <= interval.begin) {
      Interval top_interval = *active.begin();
      active.erase(active.begin());
      int physical_reg = logical_to_physical_mapping[top_interval.logical_reg_id];
      physical_regs.push(physical_reg);
    }
    if (physical_regs.empty()) {
      Interval top_interval = *active.rbegin();
      if (top_interval.end > interval.end) {
        int physical_reg = logical_to_physical_mapping[top_interval.logical_reg_id];
        logical_to_physical_mapping[interval.logical_reg_id] = physical_reg;
        logical_to_physical_mapping.erase(top_interval.logical_reg_id);
        LOG_DEBUG << "erase logical_reg_id:" << top_interval.logical_reg_id;
        auto end = active.end();
        --end;
        active.erase(end);
        active.insert(interval);
        spill_logical_regs.push_back(top_interval.logical_reg_id);
      } else {
        spill_logical_regs.push_back(interval.logical_reg_id);
      }
    } else {
      int physical_reg = physical_regs.top();
      logical_to_physical_mapping[interval.logical_reg_id] = physical_reg;
      physical_regs.pop();
      active.insert(interval);
      max_physical_reg_used = max(max_physical_reg_used, physical_reg);
    }
  }

  LOG_DEBUG << "max_physical_reg_used: " << max_physical_reg_used;
  for (int logical_reg_id : logical_regs) {
    if (logical_to_physical_mapping.find(logical_reg_id)!=logical_to_physical_mapping.end()) {
      LOG_DEBUG << "logical_reg: " << logical_reg_id << " -> physical_reg: "
                << logical_to_physical_mapping.at(logical_reg_id);
    }
  }
  for (int logical_reg_id : logical_regs) {
    LOG_DEBUG << "logical_reg:" << logical_reg_id << " begin: "
              << logic_reg_life_begin[logical_reg_id] << " end: "
              << logic_reg_life_end[logical_reg_id];
  }
  LOG_DEBUG << "num of spill_logical_regs: " << spill_logical_regs.size();
  for (int logical_reg_id : spill_logical_regs) {
    LOG_DEBUG << "spill: " << logical_reg_id;
  }
  // return;
  // Step 3: replace logical register to physical register
  for (int inst_id = 0; inst_id < instr_list.size(); inst_id++) {
    Inst inst = instr_list.getInst(inst_id);
    // skip special register instruction
    if (writer.isSpecialLi(inst))
      continue;

    std::unordered_map<string, int> replace;
    for (const auto &[key, value] : inst) {
      // bool is_gen_to_spec_assign = isGeneralToSpecialAssign(inst);
      // bool is_spec_to_gen_assign = isSpecialToGeneralAssign(inst);
      // bool is_special_assign = is_gen_to_spec_assign ||
      // is_spec_to_gen_assign; bool is_reg_general = (is_special_assign &&
      // key=="rs1") || ((!is_special_assign) && (isPrefix(key, "rs") ||
      // isPrefix(key, "rd")));

      if (writer.isGeneralReg(inst, key) && logical_to_physical_mapping.count(std::get<int>(value))) {
        replace[key] = logical_to_physical_mapping.at(std::get<int>(value));
      }
    }
    for (const auto &[key, value] : replace) {
      inst[key] = value;
    }
    instr_list.setInst(inst, inst_id);
  }

  // Step 4: spill logical register
  LOG_DEBUG << "Begin to do spill";
  int spill_base_addr = memory_addr_list.at("spill_memory");
  LOG_DEBUG << "spill_base_addr:" <<spill_base_addr;
  int temp_save_memory_base_addr = spill_base_addr;// addr = spill_memory_base_addr + spill_offset
  int spill_memory_base_addr = spill_base_addr + 8 * 4;// addr = spill_memory_base_addr + spill_offset
  std::map<int, int> spill_to_offset_mapping;
  for (int i = 0; i < spill_logical_regs.size(); i++) {
    int logical_reg_id = spill_logical_regs[i];
    spill_to_offset_mapping[logical_reg_id] = spill_memory_base_addr + i * 4;
  }
  std::vector<std::unordered_map<string, int>> replace_list;
  int inst_size = instr_list.size();
  std::set<std::string> spill_inst_and_key;
  for (int inst_id = inst_size - 1; inst_id >= 0; inst_id--) {
    Inst ori_inst = instr_list.getInst(inst_id);
    if (writer.isSpecialLi(ori_inst))
      continue;
    std::vector<std::pair<string, int>> spill_read;
    std::vector<std::pair<string, int>> spill_write;
    std::unordered_map<string, int> origin_use_physical_reg_map;
    std::set<int> origin_use_physical_reg;
    for (const auto &[key, value] : ori_inst) {
      if (writer.isGeneralReg(ori_inst, key) ) {
        if (std::get<int>(value) < 1024) { // phsical regs
          origin_use_physical_reg.insert(std::get<int>(value));
          origin_use_physical_reg_map[key] = std::get<int>(value);
          // continue;
        // if (logical_to_physical_mapping.count(std::get<int>(value))) {
        //   origin_use_physical_reg.insert(logical_to_physical_mapping.at(std::get<int>(value)));
        //   origin_use_physical_reg_map[key] = logical_to_physical_mapping.at(std::get<int>(value));
        } else {
          if (logical_to_physical_mapping.count(std::get<int>(value))) {
            std::cerr << "Should not happen!" << std::endl;
            std::exit(1);
          }
          spill_inst_and_key.insert(std::to_string(std::get<int>(ori_inst["opcode"])) + "_" + key);
          if (writer.isWriteGeneralReg(ori_inst, key)) {
            int spill_addr = spill_to_offset_mapping.at(std::get<int>(value));
            spill_write.push_back({key, spill_addr});
          } else {
            spill_read.push_back({key, spill_to_offset_mapping.at(std::get<int>(value))});
          }
        }
      }
    }
    std::vector<int> can_use_phy_regs;
    for (int i = 0; i < num_physical_regs; i++) {
      if (!origin_use_physical_reg.count(i)) {
        can_use_phy_regs.push_back(i);
      }
    }

    
    int spill_reg_num = 0;
    int addr_reg = num_physical_regs;
    std::map<std::string, int> key_to_reg;
    std::vector<Inst> insert_before;
    for (auto [key, addr] : spill_read) {
      insert_before.push_back(writer.getGeneralLIInst(addr_reg, temp_save_memory_base_addr + can_use_phy_regs[spill_reg_num] * 4));
      insert_before.push_back(writer.getStoreInst(addr_reg, can_use_phy_regs[spill_reg_num], 0));
      insert_before.push_back(writer.getGeneralLIInst(addr_reg, addr));
      insert_before.push_back(writer.getLoadInst(addr_reg, can_use_phy_regs[spill_reg_num], 0));
      key_to_reg[key] = can_use_phy_regs[spill_reg_num];
      spill_reg_num += 1; 
    }
    for (auto [key, addr] : spill_write) {
      insert_before.push_back(writer.getGeneralLIInst(addr_reg, temp_save_memory_base_addr + can_use_phy_regs[spill_reg_num] * 4));
      insert_before.push_back(writer.getStoreInst(addr_reg, can_use_phy_regs[spill_reg_num], 0));
      key_to_reg[key] = can_use_phy_regs[spill_reg_num];
      spill_reg_num += 1;
    }

    std::vector<Inst> insert_after;
    // save output reg
    for (auto [key, addr] : spill_write) {
      int reg = key_to_reg.at(key);
      insert_after.push_back(writer.getGeneralLIInst(addr_reg, addr));
      insert_after.push_back(writer.getStoreInst(addr_reg, reg, 0));  
    }
    // recover origin regs
    for (auto [key, addr] : spill_read) {
      int reg = key_to_reg.at(key);
      insert_after.push_back(writer.getGeneralLIInst(addr_reg, temp_save_memory_base_addr + reg * 4));
      insert_after.push_back(writer.getLoadInst(addr_reg, reg, 0));
    }
    for (auto [key, addr] : spill_write) {
      int reg = key_to_reg.at(key);
      insert_after.push_back(writer.getGeneralLIInst(addr_reg, temp_save_memory_base_addr + reg * 4));
      insert_after.push_back(writer.getLoadInst(addr_reg, reg, 0));
    }
    std::reverse(insert_before.begin(), insert_before.end());
    std::reverse(insert_after.begin(), insert_after.end());
    // do the insersion
    for (const auto &[key, value] : spill_read) {
      ori_inst[key] = key_to_reg.at(key);
    }
    for (const auto &[key, value] : spill_write) {
      ori_inst[key] = key_to_reg.at(key);
    }
    instr_list.setInst(ori_inst, inst_id);
    for (auto inst : insert_after) {
      instr_list.insertInst(inst, inst_id + 1);
    }
    for (auto inst : insert_before) {
      instr_list.insertInst(inst, inst_id);
    }
    if (insert_before.size() > 0) {
      LOG_DEBUG << "insert_before.size()=" <<insert_before.size() << ", insert_after.size()=" <<insert_after.size();
    }
    if (insert_after.size() > 0) {
      LOG_DEBUG << "insert_after.size()=" <<insert_after.size();
    }
  }
  LOG_DEBUG << "Finish spill!";
}

struct CodeGenerationPass
    : public mlir::PassWrapper<CodeGenerationPass,
                               OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CodeGenerationPass)

  std::string outputFilePath;
  std::string config_path;

  void runOnOperation() override {
    LOG_DEBUG << "run on operation";
    LOG_DEBUG << "code generation pass!";
    getMemoryAddrList(config_path);
    auto f = getOperation();
    if (f.getName() != "main") {
      return;
    }
    LOG_DEBUG << "code generation pass, run on main!";

    std::pair<std::unordered_map<llvm::hash_code, int>,
              std::unordered_map<llvm::hash_code, int>>
        regmaps = getRegisterMapping(f);
    std::unordered_map<llvm::hash_code, int> regmap = regmaps.first;
    std::unordered_map<llvm::hash_code, int> block_args_special_reg_map =
        regmaps.second;
    LOG_DEBUG << "getRegisterMapping finish!";

    // LegacyInstructionWriter writer;
    CIMFlowInstructionWriter writer;

    // std::map<Operation *, int> jump2line;
    std::map<Block *, std::set<int>> def;
    std::map<Block *, std::set<int>> use;
    std::map<int, int> twin_reg;
    std::vector<Block *> blocks = getBlockList(f);
    InstList instr_list(blocks);
    codeGen(blocks, writer, regmap, block_args_special_reg_map, instr_list, 
             def, use, twin_reg);
    LOG_DEBUG << "codegen finish!";

    std::map<Block *, std::set<int>> in;
    std::map<Block *, std::set<int>> out;
    liveVariableAnalysis(blocks, def, use, in, out);
    LOG_DEBUG << "live variable analysis finish!";

    mappingRegisterLogicalToPhysical(instr_list, writer, in, out, 
                                     twin_reg);

    fillJumpBranchOffset(f, writer, instr_list);

    // std::string filename = "result.json";
    std::ofstream file(outputFilePath);
    if (!file.is_open()) {
      std::cerr << "Unable to open file: " << outputFilePath << std::endl;
    } else {
      file << "[\n";
      std::vector<Inst> all_inst = instr_list.getAllInst();
      for (auto it = all_inst.begin(); it != all_inst.end();) {
        file << instToStr(*it);
        if ((++it) != all_inst.end()) {
          file << ",";
        }
        file << "\n";
      }
      file << "]";

      file.close();
    }
    LOG_DEBUG << "Generated code was saved to " << outputFilePath;
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
mlir::cim::createCodeGenerationPass(std::string outputFilePath, std::string config_path) {
  auto pass = std::make_unique<CodeGenerationPass>();
  pass->outputFilePath = outputFilePath;
  pass->config_path = config_path;
  return pass;
}
