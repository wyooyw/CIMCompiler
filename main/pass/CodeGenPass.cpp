//===- ShapeInferencePass.cpp - Shape Inference ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a Function level pass performing interprocedural
// propagation of array shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "cim/Dialect.h"
#include "cim/Passes.h"
#include "cim/ShapeInferenceInterface.h"
#include "cimisa/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"

#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

#define DEBUG_TYPE "shape-inference"

#define SPECIAL_REG_INPUT_BIT_WIDTH 0
#define SPECIAL_REG_OUTPUT_BIT_WIDTH 1
#define SPECIAL_REG_WEIGHT_BIT_WIDTH 2

using namespace mlir;
using namespace cim;

/// Include the auto-generated definitions for the shape inference interfaces.

namespace {
/// The ShapeInferencePass is a pass that performs intra-procedural
/// shape inference.
///
///    Algorithm:
///
///   1) Build a worklist containing all the operations that return a
///      dynamically shaped tensor: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the argument types.
///   3) If the worklist is empty, the algorithm succeeded.
///
using namespace std;

static int getReg(std::unordered_map<llvm::hash_code, int > &regmap, mlir::Value value){
  if (regmap.count(mlir::hash_value(value))){
    return regmap[mlir::hash_value(value)];
  }else{
    std::cerr << "error: can't find register for " << mlir::hash_value(value) << std::endl;
    return -1;
  }
  
}

typedef map<string, int> Inst;

static void codeGen(mlir::arith::ConstantOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list){
  /*
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为11
    - [27, 26]，2bit：opcode，指令操作码，值为00
    - [25, 21]，5bit：rd，通用寄存器编号，即要赋值的通用寄存器
    - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
  */
  int value = cast<IntegerAttr>(op.getValueAttr()).getInt();
  int reg = getReg(regmap, op.getResult());
  Inst inst = {
    {"class", 0b10},
    {"type", 0b11},
    {"opcode", 0b00},
    {"rd", reg},
    {"imm", value}
  };
  instr_list.push_back(inst);
}

template <typename Ty>
static void codeGenArith(Ty op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list){
  /*
  - [31, 30]，2bit：class，指令类别码，值为10
  - [29, 28]，2bit：type，指令类型码，值为00
  - [27, 26]，2bit：reserve，保留字段
  - [25, 21]，5bit：rs1，通用寄存器1，表示运算数1的值
  - [20, 16]，5bit：rs2，通用寄存器2，表示运算数2的值
  - [15, 11]，5bit：rd，通用寄存器3，即运算结果写回的寄存器
  - [10, 3]，8bit：reserve，保留字段
  - [2, 0]，3bit：opcode，操作类别码，表示具体计算的类型
    - 000：add，整型加法
    - 001：sub，整型减法
    - 010：mul，整型乘法，结果寄存器仅保留低32位
    - 011：div，整型除法，结果寄存器仅保留商
    - 100：sll，逻辑左移
    - 101：srl，逻辑右移
    - 110：sra，算数右移
  */
  int rs1 = getReg(regmap, op.getOperand(0));
  int rs2 = getReg(regmap, op.getOperand(1));
  int rd = getReg(regmap, op.getResult());

  int opcode = 0b000; // 默认值
  if constexpr (std::is_same<Ty, mlir::arith::AddIOp>::value) {
      opcode = 0b000; // Ty1 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::arith::SubIOp>::value) {
      opcode = 0b001; // Ty2 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::arith::MulIOp>::value) {
      opcode = 0b010; // Ty2 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::arith::DivSIOp>::value) {
      opcode = 0b011; // Ty2 的 opcode
  }
  
  Inst inst = {
    {"class", 0b10},
    {"type", 0b00},
    {"opcode", opcode},
    {"rs1", rs1},
    {"rs2", rs2},
    {"rd", rd}
  };
  instr_list.push_back(inst);
}

static void codeGen(mlir::cim::PrintOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list){
  /*
  - [31, 29]，3bit：class，指令类别码，值为110
  - [28, 28]，1bit：type，指令类型码，值为0
  - [27, 26]，1bit：offset mask，偏移值掩码，0表示该地址不使用偏移值，1表示使用偏移值
    - [27]，1bit：source offset mask，源地址偏移值掩码
    - [26]，1bit：destination offset mask，目的地址偏移值掩码
  - [25, 21]，5bit：rs，通用寄存器1，表示传输源地址的基址
  - [20, 16]，5bit：rd，通用寄存器2，表示传输目的地址的基址
  - [15, 0]，16bit：offset，立即数，表示寻址的偏移值
    - 源地址计算公式：$rs + offset * [27]
    - 目的地址计算公式：$rd + offset * [26]
  */
  int rs = getReg(regmap, op.getOperand());
  Inst inst = {
    {"class", -1},
    {"rs", rs}
  };
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::TransOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list){
  /*
  - [31, 29]，3bit：class，指令类别码，值为110
  - [28, 28]，1bit：type，指令类型码，值为0
  - [27, 26]，1bit：offset mask，偏移值掩码，0表示该地址不使用偏移值，1表示使用偏移值
    - [27]，1bit：source offset mask，源地址偏移值掩码
    - [26]，1bit：destination offset mask，目的地址偏移值掩码
  - [25, 21]，5bit：rs，通用寄存器1，表示传输源地址的基址
  - [20, 16]，5bit：rd，通用寄存器2，表示传输目的地址的基址
  - [15, 0]，16bit：offset，立即数，表示寻址的偏移值
    - 源地址计算公式：$rs + offset * [27]
    - 目的地址计算公式：$rd + offset * [26]
  */
  int rs = getReg(regmap, op.getOperand(0));
  int rd = getReg(regmap, op.getOperand(1));
  int size = getReg(regmap, op.getOperand(2));
  Inst inst = {
    {"class", 0b110},
    {"type", 0b0},
    {"offset mask", 0b0},
    {"rs", rs},
    {"rd", rd},
    {"offset", 0b0},
    {"size", size},
  };
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::CIMComputeOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list){
  /*
    - [31, 30]，2bit：class，指令类别码，值为00
    - [29, 29]，1bit：type，指令类型码，值为0
    - [28, 25]，4bit：reserve，保留字段
    - [24, 20]，5bit：flag，功能扩展字段
      - [24]，1bit：value sparse，表示是否使用值稀疏，稀疏掩码Mask的起始地址由专用寄存器给出
      - [23]，1bit：bit sparse，表示是否使用bit级稀疏，稀疏Meta数据的起始地址由专用寄存器给出
      - [22]，1bit：group，表示是否进行分组，组大小及激活的组数量由专用寄存器给出
      - [21]，1bit：group input mode，表示多组输入的模式
        - 0：每一组输入向量的起始地址相对于上一组的增量（步长，step）是一个定值，由专用寄存器给出
        - 1：每一组输入向量的起始地址相对于上一组的增量不是定值，其相对于rs1的偏移量（offset）在存储器中给出，地址（offset addr）由专用寄存器给出
      - [20]，1bit：accumulate，表示是否进行累加
    - [19, 15]，5bit：rs1，通用寄存器1，表示input向量起始地址
    - [14, 10]，5bit：rs2，通用寄存器2，表示input向量长度
    - [9, 5]，5bit：rs3，通用寄存器3，表示激活的row的index
    - [4, 0]，5bit：rd，通用寄存器4，表示output写入的起始地址

    - input bit width：输入的bit长度
    - output bit width：输出的bit长度
    - weight bit width：权重的bit长度
    - activation element col num：每个group内激活的element列的数量
  */
  int input_addr_reg = getReg(regmap, op.getOperand(0));
  int output_addr_reg = getReg(regmap, op.getOperand(1));
  int activate_row_reg = getReg(regmap, op.getOperand(2));
  Inst inst = {
    {"class", 0b00},
    {"type", 0b0},
    {"value sparse", static_cast<int>(op.getValueSparseFlag())},
    {"bit sparse", static_cast<int>(op.getBitSparseFlag())},
    {"group", 0b0},
    {"group input mode", 0b0},
    {"accumulate", static_cast<int>(op.getAccFlag())},
    {"rs1", input_addr_reg},
    {"rs2", 16},
    {"rs3", activate_row_reg},
    {"rd", output_addr_reg},
  };
  Inst inst_input_bw = {
    {"class", 0b10},{"type", 0b11},{"opcode", 0b01},
    {"rd", SPECIAL_REG_INPUT_BIT_WIDTH},
    {"imm", op.getInputBw()},
  };
  Inst inst_output_bw = {
    {"class", 0b10},{"type", 0b11},{"opcode", 0b01},
    {"rd", SPECIAL_REG_OUTPUT_BIT_WIDTH},
    {"imm", op.getOutputBw()},
  };
  Inst inst_weight_bw = {
    {"class", 0b10},{"type", 0b11},{"opcode", 0b01},
    {"rd", SPECIAL_REG_WEIGHT_BIT_WIDTH},
    {"imm", op.getWeightBw()},
  };
  instr_list.push_back(inst_input_bw);
  instr_list.push_back(inst_output_bw);
  instr_list.push_back(inst_weight_bw);
  instr_list.push_back(inst);
}

static void codeGen(mlir::cf::BranchOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list){
  /*
- [31, 29]，3bit：class，指令类别码，值为111
- [28, 26]，3bit：type，指令类型码，值为100
- [25, 0]，26bit：offset，立即数，表示跳转指令地址相对于该指令的偏移值
  */
  Inst inst = {
    {"class", 0b111},
    {"type", 0b100},
    {"offset", -1},
  };
  instr_list.push_back(inst);
}

static void codeGen(mlir::cf::CondBranchOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list){
  /*
    - [31, 29]，3bit：class，指令类别码，值为111
    - [28, 26]，3bit：type，指令类型码
      - 000：beq，相等跳转
      - 001：bne，不等跳转
      - 010：bgt，大于跳转
      - 011：blt，小于跳转
    - [25, 21]，5bit：rs1，通用寄存器1，表示进行比较的操作数1
    - [20, 16]，5bit：rs2，通用寄存器2，表示进行比较的操作数2
    - [15, 0]，16bit：offset，立即数，表示跳转指令地址相对于该指令的偏移值
  */
  arith::CmpIOp cmpi_op = op.getOperand(0).getDefiningOp<arith::CmpIOp>();
  if(!cmpi_op){
    std::cerr << "cmpi_op is null!" << std::endl;
  }
  string predicate = cmpi_op.getPredicateAttrName().str();
  mlir::Value lhs = cmpi_op.getLhs();
  mlir::Value rhs = cmpi_op.getRhs();
  std::cout << "predicate=" << predicate << std::endl;
  Inst inst = {
    {"class", 0b111},
    {"type", 0},
    {"rs1", getReg(regmap, cmpi_op.getLhs())},
    {"rs2", getReg(regmap, cmpi_op.getRhs())},
    {"offset", -1},
  };
  instr_list.push_back(inst);
}


static void codeGen(mlir::func::FuncOp func, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list, std::map<Operation*, int>& op2line){
  func.walk([&](mlir::Operation *op) {
    op2line[op] = instr_list.size();
    if(auto _op = dyn_cast<mlir::arith::ConstantOp>(op) ){
      codeGen(_op, regmap, instr_list);
    }else if(auto _op = dyn_cast<mlir::arith::AddIOp>(op)){
      codeGenArith<mlir::arith::AddIOp>(_op, regmap, instr_list);
    }else if(auto _op = dyn_cast<mlir::arith::SubIOp>(op)){
      codeGenArith<mlir::arith::SubIOp>(_op, regmap, instr_list);
    }else if(auto _op = dyn_cast<mlir::arith::MulIOp>(op)){
      codeGenArith<mlir::arith::MulIOp>(_op, regmap, instr_list);
    }else if(auto _op = dyn_cast<mlir::arith::DivSIOp>(op)){
      codeGenArith<mlir::arith::DivSIOp>(_op, regmap, instr_list);
    }else if(auto _op = dyn_cast<mlir::cimisa::CIMComputeOp>(op)){
      codeGen(_op, regmap, instr_list);
    }else if(auto _op = dyn_cast<mlir::cimisa::TransOp>(op)){
      codeGen(_op, regmap, instr_list);
    }else if(auto _op = dyn_cast<mlir::cf::CondBranchOp>(op)){
      codeGen(_op, regmap, instr_list);
    }else if(auto _op = dyn_cast<mlir::cf::BranchOp>(op)){
      codeGen(_op, regmap, instr_list);
    }else{
      std::cerr << "error: unsupport operator: " << op->getName().getStringRef().str() << std::endl;
    }
  });
}

static void mapValueAsRegister(mlir::Value& value, std::unordered_map<llvm::hash_code, int>& mapping, int &reg_cnt){
  llvm::hash_code hash_code = mlir::hash_value(value);
  if(!mapping.count(hash_code)){
    mapping[hash_code] = reg_cnt++; 
  }
}

template <typename Ty>
static void mapResultAsRegister(Ty op, std::unordered_map<llvm::hash_code, int>& mapping, int &reg_cnt){
  // std::cout << "mapResultAsRegister 1" << std::endl;
  mlir::Value result = op.getResult();
  mapValueAsRegister(result, mapping, reg_cnt);
  // std::cout << "mapResultAsRegister 4" << std::endl;
}

static void _getRegisterMappingAliasBetweenBasicBlock(
  mlir::func::FuncOp func,
  std::unordered_map<llvm::hash_code, int >& mapping,
  int& reg_cnt
){
  auto regions = func->getRegions();
  std::cout << "regions.size()" << regions.size() << std::endl;
  for (Region &region : regions){
    // for each block
    for (Block &block : region.getBlocks()){
      
      auto block_arguments = block.getArguments();
      for (int arg_i = 0; arg_i < block_arguments.size(); arg_i++){
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

        // map all alias to same
        for(mlir::Value& alias : alias_values){
          mapValueAsRegister(alias, mapping, reg_cnt);
        }
        reg_cnt++;
      }
    }
  }
    
}

static void _getRegisterMappingGeneral(
  mlir::func::FuncOp func,
  std::unordered_map<llvm::hash_code, int >& mapping,
  int& reg_cnt){

  func.walk([&](mlir::Operation *op) {
    if(auto _op = dyn_cast<mlir::arith::ConstantOp>(op) ){
      mapResultAsRegister<mlir::arith::ConstantOp>(_op, mapping, reg_cnt);
    }else if(auto _op = dyn_cast<mlir::arith::AddIOp>(op)){
      mapResultAsRegister<mlir::arith::AddIOp>(_op, mapping, reg_cnt);
    }else if(auto _op = dyn_cast<mlir::arith::SubIOp>(op)){
      mapResultAsRegister<mlir::arith::SubIOp>(_op, mapping, reg_cnt);
    }else if(auto _op = dyn_cast<mlir::arith::MulIOp>(op)){
      mapResultAsRegister<mlir::arith::MulIOp>(_op, mapping, reg_cnt);
    }else if(auto _op = dyn_cast<mlir::arith::DivSIOp>(op)){
      mapResultAsRegister<mlir::arith::DivSIOp>(_op, mapping, reg_cnt);
    }else if(auto _op = dyn_cast<mlir::arith::RemSIOp>(op)){
      mapResultAsRegister<mlir::arith::RemSIOp>(_op, mapping, reg_cnt);
    }
  });
  std::cout << "_getRegisterMappingGeneral finish" << std::endl;
}
static std::unordered_map<llvm::hash_code, int > getRegisterMapping(mlir::func::FuncOp func){
  /*
    Step 1: get special register mapping
    Step 2: get alias rf between basic block
    Step 3: get other rf
  */
  std::unordered_map<llvm::hash_code, int > mapping;
  int reg_cnt = 0;


  _getRegisterMappingAliasBetweenBasicBlock(func, mapping, reg_cnt);
  _getRegisterMappingGeneral(func, mapping, reg_cnt);
  std::cout << "getRegisterMapping finish" << std::endl;
  return mapping;
}

static string instToStr(Inst& inst){
    std::string json = "{";
    for (auto it = inst.begin(); it != inst.end();) {
        json += "\"" + it->first + "\": " + std::to_string(it->second);
        if ((++it) != inst.end()) {
            json += ", ";
        }
    }
    json += "}";
    return json;
}

static void fillJumpBranchOffset(mlir::func::FuncOp func, std::vector<Inst>& instr_list, std::map<Operation*, int>& op2line){
  func.walk([&](mlir::Operation *op) {
    if(auto _op = dyn_cast<mlir::cf::BranchOp>(op)){
      Block* dest_block = _op.getDest();
      Operation &dest_op = dest_block->getOperations().front();
      if(!op2line.count(&dest_op)){
        std::cerr << "error: can't find branch target" << std::endl;
        std::exit(1);
      }
      int target_line = op2line[&dest_op];
      int current_line = op2line[op];
      int offset = target_line - current_line;
      instr_list[current_line]["offset"] = offset;
    }else if(auto _op = dyn_cast<mlir::cf::CondBranchOp>(op)){
      Block* dest_block = _op.getTrueDest();
      Operation &dest_op = dest_block->getOperations().front();
      if(!op2line.count(&dest_op)){
        std::cerr << "error: can't find branch target" << std::endl;
        std::exit(1);
      }
      int target_line = op2line[&dest_op];
      int current_line = op2line[op];
      int offset = target_line - current_line;
      instr_list[current_line]["offset"] = offset;
    }
  });
}
struct CodeGenerationPass
    : public mlir::PassWrapper<CodeGenerationPass, OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CodeGenerationPass)

  std::string outputFilePath;

  void runOnOperation() override {
    std::cout << "run on operation" << std::endl;
    auto f = getOperation();
    std::cout << "code generation pass!" << std::endl;

    std::unordered_map<llvm::hash_code, int > regmap = getRegisterMapping(f);
    std::vector<Inst> instr_list;
    std::map<Operation*, int> op2line;
    std::cout << "getRegisterMapping finish!" << std::endl;
    codeGen(f, regmap, instr_list, op2line);
    std::cout << "codegen finish!" << std::endl;
    fillJumpBranchOffset(f, instr_list, op2line);

    // std::string filename = "result.json";
    std::ofstream file(outputFilePath);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << outputFilePath << std::endl;
    } else {
        file << "[\n";
        for(auto it = instr_list.begin(); it != instr_list.end();){
          file << instToStr(*it);
          if ((++it) != instr_list.end()) {
              file << ",";
          }
          file << "\n";
        }
        file << "]";
        // 关闭文件
        file.close();
    }
    std::cout << "Generated code was saved to " << outputFilePath << std::endl;
  }

};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::cim::createCodeGenerationPass(std::string outputFilePath) {
  auto pass = std::make_unique<CodeGenerationPass>();
  pass->outputFilePath = outputFilePath;
  return pass;
}
