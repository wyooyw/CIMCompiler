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
#include <algorithm>

#define DEBUG_TYPE "shape-inference"

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

static void codeGen(mlir::arith::ConstantOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
  std::set<int> &def, std::set<int> &use){
  /*
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为11
    - [27, 26]，2bit：opcode，指令操作码，值为00
    - [25, 21]，5bit：rd，通用寄存器编号，即要赋值的通用寄存器
    - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
  */
  int value = cast<IntegerAttr>(op.getValueAttr()).getInt();
  int reg = getReg(regmap, op.getResult());
  def.insert(reg);
  Inst inst = {
    {"class", 0b10},
    {"type", 0b11},
    {"opcode", 0b00},
    {"rd", reg},
    {"imm", value}
  };
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::GeneralRegLiOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
  std::set<int> &def, std::set<int> &use){
  /*
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为11
    - [27, 26]，2bit：opcode，指令操作码，值为00
    - [25, 21]，5bit：rd，通用寄存器编号，即要赋值的通用寄存器
    - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
  */
  int64_t value = op.getValue().getSExtValue();
  int reg = getReg(regmap, op.getResult());
  def.insert(reg);
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
static void codeGenArith(Ty op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
  std::set<int> &def, std::set<int> &use){
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
  def.insert(rd);
  use.insert(rs1);
  use.insert(rs2);

  int opcode = 0b000; // 默认值
  if constexpr (std::is_same<Ty, mlir::arith::AddIOp>::value) {
      opcode = 0b000; // Ty1 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::arith::SubIOp>::value) {
      opcode = 0b001; // Ty2 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::arith::MulIOp>::value) {
      opcode = 0b010; // Ty2 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::arith::DivSIOp>::value) {
      opcode = 0b011; // Ty2 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::arith::RemSIOp>::value) {
      opcode = 0b111; // Ty2 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::arith::MinSIOp>::value) {
      opcode = 0b1000; // Ty2 的 opcode
  } else {
    std::cerr << "Unsupport arith op!" << std::endl;
    std::exit(1);
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

template <typename Ty>
static void codeGenRI(Ty op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
  std::set<int> &def, std::set<int> &use){
  /*
    R-I型整数运算指令：scalar-RI
    指令字段划分：
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为01
    - [27, 26]，2bit：opcode，操作类别码，表示具体计算的类型
      - 00：addi，整型立即数加法
      - 01：muli，整型立即数乘法，结果寄存器仅保留低32位
      - 10：lui，高16位立即数赋值
    - [25, 21]，5bit：rs，通用寄存器1，表示运算数1的值
    - [20, 16]，5bit：rd，通用寄存器2，即运算结果写回的寄存器
    - [15, 0]，16bit：imm，立即数，表示运算数2的值
  */
  int rs = getReg(regmap, op.getOperand());
  int rd = getReg(regmap, op.getResult());
  int64_t imm = op.getConstant().getSExtValue();
  
  def.insert(rd);
  use.insert(rs);

  int opcode = 0b000; // 默认值
  if constexpr (std::is_same<Ty, mlir::cimisa::RIAddIOp>::value) {
      opcode = 0b000; // Ty1 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::cimisa::RISubIOp>::value) {
      opcode = 0b001; // Ty2 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::cimisa::RIMulIOp>::value) {
      opcode = 0b010; // Ty2 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::cimisa::RIDivSIOp>::value) {
      opcode = 0b011; // Ty2 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::cimisa::RIRemSIOp>::value) {
      opcode = 0b111; // Ty2 的 opcode
  } else if constexpr (std::is_same<Ty, mlir::cimisa::RIMinSIOp>::value) {
      opcode = 0b1000; // Ty2 的 opcode
  } else {
    std::cerr << "Unsupport arith ri op!" << std::endl;
    std::exit(1);
  }
  
  Inst inst = {
    {"class", 0b10},
    {"type", 0b01},
    {"opcode", opcode},
    {"rs", rs},
    {"rd", rd},
    {"imm", imm}
  };
  instr_list.push_back(inst);
}

/*
  SIMD
  VVAdd, VVMul, VSAdd, Quantify, QuantifyResAdd, QuantifyMultiply
*/

static void codeGen(mlir::cimisa::VVAddOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
  std::set<int> &def, std::set<int> &use){
  /*
    SIMD计算：SIMD-compute
    指令字段划分：
    - [31, 30]，2bit：class，指令类别码，值为01
    - [29, 28]，2bit：input num，input向量的个数，范围是1到4
      - 00：1个输入向量，地址由rs1给出
      - 01：2个输入向量，地址由rs1和rs2给出
      - 10：3个输入向量，地址由rs1，rs1+1，rs2给出
      - 11：4个输入向量，地址由rs1，rs1+1，rs2，rs2+1给出
    - [27, 20]，8bit：opcode，操作类别码，表示具体计算的类型
      - 0x00：add，向量加法
      - 0x01：add-scalar，向量和标量加法
      - 0x02：multiply，向量逐元素乘法
      - 0x03：quantify，量化
      - 0x04：quantify-resadd，resadd量化
      - 0x05：quantify-multiply，乘法量化
    - [19, 15]，5bit：rs1，通用寄存器1，表示input向量起始地址1
    - [14, 10]，5bit：rs2，通用寄存器2，表示input向量起始地址2
    - [9, 5]，5bit：rs3，通用寄存器3，表示input向量长度
    - [4, 0]，5bit：rd，通用寄存器4，表示output写入的起始地址
    使用的专用寄存器：
    - input 1 bit width：输入向量1每个元素的bit长度
    - input 2 bit width：输入向量2每个元素的bit长度
    - input 3 bit width：输入向量3每个元素的bit长度
    - input 4 bit width：输入向量4每个元素的bit长度
    - output bit width：输出向量每个元素的bit长度
  */
  int lhs = getReg(regmap, op.getOperand(0));
  int rhs = getReg(regmap, op.getOperand(1));
  int rd = getReg(regmap, op.getOperand(2));
  int size = getReg(regmap, op.getOperand(3));
  use.insert(lhs);
  use.insert(rhs);
  use.insert(rd);
  use.insert(size);

  Inst inst = {
    {"class", 0b01},
    {"input_num", 0b01},
    {"opcode", 0b00},
    {"rs1", lhs},
    {"rs2", rhs},
    {"rs3", size},
    {"rd", rd}
  };
  instr_list.push_back(inst);
}


/*
  PrintOp
*/

static void codeGen(mlir::cim::PrintOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
  std::set<int> &def, std::set<int> &use){

  int rs = getReg(regmap, op.getOperand());
  use.insert(rs);
  Inst inst = {
    {"class", -1},
    {"type", 0},
    {"rs", rs}
  };
  instr_list.push_back(inst);
}

static void codeGen(mlir::cim::DebugOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
  std::set<int> &def, std::set<int> &use){

  Inst inst = {
    {"class", -1},
    {"type", 1},
  };
  instr_list.push_back(inst);
}

/*
  Memory
  TransOp, LoadOp, StoreOp
*/

static void codeGen(mlir::cimisa::TransOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
  std::set<int> &def, std::set<int> &use){
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
  use.insert(rs);
  use.insert(rd);
  use.insert(size);
  Inst inst = {
    {"class", 0b110},
    {"type", 0b0},
    {"source_offset_mask", 0b0},
    {"destination_offset_mask", 0b0},
    {"rs1", rs},
    {"rd", rd},
    {"offset", 0b0},
    {"rs2", size},
  };
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::LoadOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
  std::set<int> &def, std::set<int> &use){
  /*
    Load/Store指令：scalar-SL
    指令字段划分：
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为10
    - [27, 26]，2bit：opcode，操作类别码，表示具体操作的类型
      - 00：本地存储load至寄存器
      - 01：寄存器值store至本地存储
      - 10：全局存储load至寄存器
      - 11：寄存器值store至全局存储
    - [25, 21]，5bit：rs1，通用寄存器1，即寻址的基址寄存器base
    - [20, 16]，5bit：rs2，通用寄存器2，即存储load/store值的寄存器
    - [15, 0]，16bit：offset，立即数，表示寻址的偏移值
      - 地址计算公式：$rs + offset
  */
  int rs1 = getReg(regmap, op.getOperand());
  int rs2 = getReg(regmap, op.getResult());
  int64_t imm = op.getConstant().getSExtValue();
  use.insert(rs1);
  def.insert(rs2);
  Inst inst = {
    {"class", 0b10},
    {"type", 0b10},
    {"opcode", 0b00},
    {"rs1", rs1},
    {"rs2", rs2},
    {"offset", imm},
  };
  instr_list.push_back(inst);
}


static void codeGen(mlir::cimisa::StoreOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
 std::set<int> &def, std::set<int> &use){
  /*
    Load/Store指令：scalar-SL
    指令字段划分：
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为10
    - [27, 26]，2bit：opcode，操作类别码，表示具体操作的类型
      - 00：本地存储load至寄存器
      - 01：寄存器值store至本地存储
      - 10：全局存储load至寄存器
      - 11：寄存器值store至全局存储
    - [25, 21]，5bit：rs1，通用寄存器1，即寻址的基址寄存器base
    - [20, 16]，5bit：rs2，通用寄存器2，即存储load/store值的寄存器
    - [15, 0]，16bit：offset，立即数，表示寻址的偏移值
      - 地址计算公式：$rs + offset
  */
  int rs1 = getReg(regmap, op.getOperand(0));
  int rs2 = getReg(regmap, op.getOperand(1));
  int64_t imm = op.getConstant().getSExtValue();
  use.insert(rs1);
  use.insert(rs2);
  Inst inst = {
    {"class", 0b10},
    {"type", 0b10},
    {"opcode", 0b01},
    {"rs1", rs1},
    {"rs2", rs2},
    {"offset", imm},
  };
  instr_list.push_back(inst);
}

/*
  CIMCompute
*/

static void codeGen(mlir::cimisa::CIMComputeOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
    std::set<int> &def, std::set<int> &use){
  // TODO: 这里没加上input_size寄存器
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
  int input_size_reg = getReg(regmap, op.getOperand(3));
  use.insert(input_addr_reg);
  use.insert(output_addr_reg);
  use.insert(activate_row_reg);
  use.insert(input_size_reg);
  Inst inst = {
    {"class", 0b00},
    {"type", 0b0},
    {"value_sparse", static_cast<int>(op.getValueSparseFlag())},
    {"bit_sparse", static_cast<int>(op.getBitSparseFlag())},
    {"group", 0b1},
    {"group_input_mode", 0b0},
    {"accumulate", static_cast<int>(op.getAccFlag())},
    {"rs1", input_addr_reg},
    {"rs2", input_size_reg},
    {"rs3", activate_row_reg},
    {"rd", output_addr_reg},
  };
  instr_list.push_back(inst);
}


static void codeGen(mlir::cimisa::CIMOutputOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
    std::set<int> &def, std::set<int> &use){
  /*

  */
  int output_addr_reg = getReg(regmap, op.getOperand());
  use.insert(output_addr_reg);
  Inst inst = {
    {"class", 0b00},
    {"type", 0b10},
    {"outsum_move", 0},
    {"outsum", 0},
    // {"rs1", input_addr_reg},
    // {"rs2", input_size_reg},
    {"rd", output_addr_reg},
  };
  instr_list.push_back(inst);
}

/*
  ControlFlow
  BranchOp, CondBranchOp
*/

static void codeGen(mlir::cf::BranchOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list,
    std::set<int> &def, std::set<int> &use ){
  /*
- [31, 29]，3bit：class，指令类别码，值为111
- [28, 26]，3bit：type，指令类型码，值为100
- [25, 0]，26bit：offset，立即数，表示跳转指令地址相对于该指令的偏移值
  */
  Block* dest_block = op.getDest();
  auto dest_args = dest_block->getArguments();
  auto dest_operands = op.getDestOperands();
  int args_size = dest_args.size();
  int operands_size = dest_operands.size();
  if (args_size != operands_size){
    std::cerr << "error: args_size != operands_size" << args_size << " vs "<< operands_size << std::endl;
    std::exit(1);
  }

  for (int i = 0; i < args_size; i++){
    mlir::Value arg = llvm::cast<mlir::Value>(dest_args[i]);
    mlir::Value operand = dest_operands[i];
    int arg_reg = getReg(regmap, arg);
    int operand_reg = getReg(regmap, operand);
    // we don't have a move instruct between general register, so we use addi zero instead
    Inst add_zero_inst = {
      {"class", 0b10},
      {"type", 0b01},
      {"opcode", 0b00},
      {"rs", operand_reg},
      {"rd", arg_reg},
      {"imm", 0}
    };
    instr_list.push_back(add_zero_inst);
    use.insert(operand_reg);
    // def.insert(arg_reg);
    use.insert(arg_reg);
  }

  Inst inst = {
    {"class", 0b111},
    {"type", 0b100},
    {"offset", -1},
  };
  instr_list.push_back(inst);
}

static void codeGen(mlir::cf::CondBranchOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list, 
    std::set<int> &def, std::set<int> &use ){
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
  auto predicate = cmpi_op.getPredicate();
  int compare = 0;
  if (predicate == arith::CmpIPredicate::eq){
    compare = 0;
  }else if(predicate == arith::CmpIPredicate::ne){
    compare = 1;
  }else if(predicate == arith::CmpIPredicate::sgt){
    compare = 2;
  }else if(predicate == arith::CmpIPredicate::slt){
    compare = 3;
  }else{
    std::cerr << "error: unsupport predicate" << std::endl;
    std::exit(1);
  }
  mlir::Value lhs = cmpi_op.getLhs();
  mlir::Value rhs = cmpi_op.getRhs();

  int lhs_reg = getReg(regmap, cmpi_op.getLhs());
  int rhs_reg = getReg(regmap, cmpi_op.getRhs());
  Inst inst = {
    {"class", 0b111},
    {"type", compare},
    {"rs1", lhs_reg},
    {"rs2", rhs_reg},
    {"offset", -1},
  };
  instr_list.push_back(inst);
  use.insert(lhs_reg);
  use.insert(rhs_reg);
}

static void codeGen(mlir::cimisa::SpecialRegLiOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list, 
    std::set<int> &def, std::set<int> &use ){
  /*
    专用寄存器立即数赋值指令：special-li
    指令字段划分：
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为11
    - [27, 26]，2bit：opcode，指令操作码，值为01
    - [25, 21]，5bit：rd，专用寄存器编号，即要赋值的通用寄存器
    - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
  */

  int special_reg = static_cast<int>(op.getSpecialReg());
  int set_value = static_cast<int>(op.getSetValue());
  Inst inst = {
    {"class", 0b10},
    {"type", 0b11},
    {"opcode", 0b01},
    {"rd", special_reg},
    {"imm", set_value},
  };
  instr_list.push_back(inst);
}

static void codeGen(mlir::cimisa::SpecialRegAssignOp op, std::unordered_map<llvm::hash_code, int > &regmap, std::vector<Inst>& instr_list, 
    std::set<int> &def, std::set<int> &use ){
  /*
    专用/通用寄存器赋值指令：special-general-assign
    指令字段划分：
    - [31, 30]，2bit：class，指令类别码，值为10
    - [29, 28]，2bit：type，指令类型码，值为11
    - [27, 26]，2bit：opcode，指令操作码
      - 10：表示将通用寄存器的值赋给专用寄存器
      - 11：表示将专用寄存器的值赋给通用寄存器
    - [25, 21]，5bit：rs1，通用寄存器编号，即涉及赋值的通用寄存器
    - [20, 16]，5bit：rs2，专用寄存器编号，即涉及赋值的专用寄存器
    - [15, 0]，16bit：reserve，保留字段
  */

  int special_reg = static_cast<int>(op.getSpecialReg());
  int from_general_reg = getReg(regmap, op.getOperand());
  use.insert(from_general_reg);
  Inst inst = {
    {"class", 0b10},
    {"type", 0b11},
    {"opcode", 0b10},
    {"rs1", from_general_reg},
    {"rs2", special_reg}
  };
  instr_list.push_back(inst);
}
/*
  CodeGen For Operator Finish!
*/

static std::vector<Block*> getBlockList(mlir::func::FuncOp func){
  std::cout << "getBlockList begin" << std::endl;
  auto regions = func->getRegions();
  if (regions.size()>1){
    std::cout << "regions.size()" << regions.size() << std::endl;
    std::exit(1);
  }
  Region &region = regions.front();
  std::vector<Block*> blocks;
  std::unordered_map<Block*, bool> blocks_completed;
  for (Block &block : region.getBlocks()){
    blocks_completed[&block] = false;
  }

  int block_cnt = 0;
  int total_block_cnt = region.getBlocks().size();
  while(block_cnt < total_block_cnt){
    if (block_cnt==0){

      // find the block with no predeccessor
      for (Block &block : region.getBlocks()){
        int num_predecessors = 0;
        for (auto *b : block.getPredecessors()) num_predecessors++;
        if (num_predecessors==0){
          blocks.push_back(&block);
          blocks_completed[&block] = true;
          break;
        }
      }
      block_cnt = 1;

    }else{

      // find the block with no false-dest predecessor
      int find = 0;
      for (Block &block : region.getBlocks()){
        if (blocks_completed[&block]) continue;
        int flag = 1;
        for (auto *b : block.getPredecessors()){
          auto terminator = b->getTerminator();
          if (auto _op = dyn_cast<mlir::cf::CondBranchOp>(terminator)){
            if (_op.getFalseDest()==&block){
              flag = 0;
              break;
            }
          }
        }
        if (flag){
          find = 1;
          blocks.push_back(&block);
          blocks_completed[&block] = true;
          break;
        }
      }
      if (!find){
        std::cout << "can't find block with no false-dest predecessor" << std::endl;
        std::exit(1);
      }
      block_cnt++;

    } // end if block_cnt==0

    Block *selected_block = blocks.back();

    // False-dest chain
    while(true){
      auto terminator = selected_block->getTerminator();
      if (auto _op = dyn_cast<mlir::cf::CondBranchOp>(terminator)){
        if (blocks_completed[_op.getFalseDest()]){
          std::cerr << "Error: false-dest block already completed" << std::endl;
          std::exit(1);
        }
        blocks.push_back(_op.getFalseDest());
        blocks_completed[_op.getFalseDest()] = true;
        selected_block = _op.getFalseDest();
        block_cnt++;
      }else{
        break;
      }
    }
  }
  std::cout << "getBlockList end" << std::endl;
  return blocks;
}

static void codeGen(std::vector<Block*> &blocks, std::unordered_map<llvm::hash_code, int > &regmap, 
          std::vector<Inst>& instr_list, 
          std::map<Block*, int>& block2line,
          std::map<Block*, int>& block2line_end,
          std::map<Operation*, int>& jump2line,
          std::map<Block*, std::set<int> >& def,
          std::map<Block*, std::set<int> >& use
          ){
  
  for (Block *block : blocks){
    // iter all Operation in this block
    block2line[block] = instr_list.size();

    std::set<int> _write;
    std::set<int> _read;
    for (Operation &op_obj : block->getOperations()){
      Operation *op = &op_obj;
      
      if(auto _op = dyn_cast<mlir::arith::ConstantOp>(op) ){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::GeneralRegLiOp>(op) ){
        codeGen(_op, regmap, instr_list, _write, _read);

      // RR
      }else if(auto _op = dyn_cast<mlir::arith::AddIOp>(op)){
        codeGenArith<mlir::arith::AddIOp>(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::arith::SubIOp>(op)){
        codeGenArith<mlir::arith::SubIOp>(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::arith::MulIOp>(op)){
        codeGenArith<mlir::arith::MulIOp>(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::arith::DivSIOp>(op)){
        codeGenArith<mlir::arith::DivSIOp>(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::arith::RemSIOp>(op)){
        codeGenArith<mlir::arith::RemSIOp>(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::arith::MinSIOp>(op)){
        codeGenArith<mlir::arith::MinSIOp>(_op, regmap, instr_list, _write, _read);

      // RI
      }else if(auto _op = dyn_cast<mlir::cimisa::RIAddIOp>(op)){
        codeGenRI<mlir::cimisa::RIAddIOp>(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::RISubIOp>(op)){
        codeGenRI<mlir::cimisa::RISubIOp>(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::RIMulIOp>(op)){
        codeGenRI<mlir::cimisa::RIMulIOp>(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::RIDivSIOp>(op)){
        codeGenRI<mlir::cimisa::RIDivSIOp>(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::RIRemSIOp>(op)){
        codeGenRI<mlir::cimisa::RIRemSIOp>(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::RIMinSIOp>(op)){
        codeGenRI<mlir::cimisa::RIMinSIOp>(_op, regmap, instr_list, _write, _read);

      }else if(auto _op = dyn_cast<mlir::cimisa::VVAddOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::CIMComputeOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::CIMOutputOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::TransOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::LoadOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::StoreOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cim::PrintOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cim::DebugOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cf::CondBranchOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
        jump2line[op] = instr_list.size() - 1;
      }else if(auto _op = dyn_cast<mlir::cf::BranchOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
        jump2line[op] = instr_list.size() - 1;
      }else if(auto _op = dyn_cast<mlir::cimisa::SpecialRegLiOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else if(auto _op = dyn_cast<mlir::cimisa::SpecialRegAssignOp>(op)){
        codeGen(_op, regmap, instr_list, _write, _read);
      }else{
        std::cerr << "error: unsupport operator: " << op->getName().getStringRef().str() << std::endl;
      }
    }
    block2line_end[block] = instr_list.size() - 1;

    std::set<int> _def;
    std::set<int> _use;
    _def.insert(_write.begin(), _write.end());
    std::set_difference(_read.begin(), _read.end(), _write.begin(), _write.end(), std::inserter(_use, _use.begin()));
    
    def[block] = _def;
    use[block] = _use;
  }
}

static void mapValueAsRegister(mlir::Value& value, std::unordered_map<llvm::hash_code, int>& mapping, int &reg_cnt){
  llvm::hash_code hash_code = mlir::hash_value(value);
  if(!mapping.count(hash_code)){
    mapping[hash_code] = reg_cnt++; 
  }else{
    std::cout << "register already allocted! " << mapping[hash_code] << std::endl;
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
        std::cout << "reg = " << reg_cnt << ", alias_values.size() = " << alias_values.size() << std::endl;
        // map all alias to same
        for(mlir::Value& alias : alias_values){
          int _reg_cnt = reg_cnt;
          mapValueAsRegister(alias, mapping, _reg_cnt);
        }
        reg_cnt++;
      }
    }
  }
}

static void _getRegisterMappingForBlockArgs(
  mlir::func::FuncOp func,
  std::unordered_map<llvm::hash_code, int >& mapping,
  int& reg_cnt){
    

  auto regions = func->getRegions();
  for (Region &region : regions){
    // for each block
    for (Block &block : region.getBlocks()){
      auto block_arguments = block.getArguments();
      for (int arg_i = 0; arg_i < block_arguments.size(); arg_i++){
        BlockArgument block_arg = block_arguments[arg_i];
        mlir::Value block_arg_val = llvm::cast<mlir::Value>(block_arg);
        mapValueAsRegister(block_arg_val, mapping, reg_cnt);
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
    }else if(auto _op = dyn_cast<mlir::cimisa::GeneralRegLiOp>(op) ){
      mapResultAsRegister<mlir::cimisa::GeneralRegLiOp>(_op, mapping, reg_cnt);
    
    // RR
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
    }else if(auto _op = dyn_cast<mlir::arith::MinSIOp>(op)){
      mapResultAsRegister<mlir::arith::MinSIOp>(_op, mapping, reg_cnt);
    
    // RI
    }else if(auto _op = dyn_cast<mlir::cimisa::RIAddIOp>(op)){
      mapResultAsRegister<mlir::cimisa::RIAddIOp>(_op, mapping, reg_cnt);
    }else if(auto _op = dyn_cast<mlir::cimisa::RISubIOp>(op)){
      mapResultAsRegister<mlir::cimisa::RISubIOp>(_op, mapping, reg_cnt);
    }else if(auto _op = dyn_cast<mlir::cimisa::RIMulIOp>(op)){
      mapResultAsRegister<mlir::cimisa::RIMulIOp>(_op, mapping, reg_cnt);
    }else if(auto _op = dyn_cast<mlir::cimisa::RIDivSIOp>(op)){
      mapResultAsRegister<mlir::cimisa::RIDivSIOp>(_op, mapping, reg_cnt);
    }else if(auto _op = dyn_cast<mlir::cimisa::RIRemSIOp>(op)){
      mapResultAsRegister<mlir::cimisa::RIRemSIOp>(_op, mapping, reg_cnt);
    }else if(auto _op = dyn_cast<mlir::cimisa::RIMinSIOp>(op)){
      mapResultAsRegister<mlir::cimisa::RIMinSIOp>(_op, mapping, reg_cnt);

    }else if(auto _op = dyn_cast<mlir::cimisa::LoadOp>(op)){
      mapResultAsRegister<mlir::cimisa::LoadOp>(_op, mapping, reg_cnt);
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


  // _getRegisterMappingAliasBetweenBasicBlock(func, mapping, reg_cnt);
  _getRegisterMappingForBlockArgs(func, mapping, reg_cnt);
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

static void fillJumpBranchOffset(mlir::func::FuncOp func, std::vector<Inst>& instr_list, 
    std::map<Block*, int>& block2line, 
    std::map<Operation*, int>& jump2line){
  func.walk([&](mlir::Operation *op) {
    if(auto _op = dyn_cast<mlir::cf::BranchOp>(op)){
      Block* dest_block = _op.getDest();
      if(!block2line.count(dest_block)){
        std::cerr << "error: can't find branch target" << std::endl;
        std::exit(1);
      }
      if(!jump2line.count(op)){
        std::cerr << "error: can't find op in jump2line" << std::endl;
        std::exit(1);
      }
      int target_line = block2line[dest_block];
      int current_line = jump2line[op];
      int offset = target_line - current_line;
      instr_list[current_line]["offset"] = offset;
      std::cout << "[jump]set offset in line "<< current_line << " to " << offset << std::endl;
    }else if(auto _op = dyn_cast<mlir::cf::CondBranchOp>(op)){
      Block* dest_block = _op.getTrueDest();
      if(!block2line.count(dest_block)){
        std::cerr << "error: can't find branch target" << std::endl;
        std::exit(1);
      }
      if(!jump2line.count(op)){
        std::cerr << "error: can't find op in jump2line" << std::endl;
        std::exit(1);
      }
      int target_line = block2line[dest_block];
      int current_line = jump2line[op];
      int offset = target_line - current_line;
      instr_list[current_line]["offset"] = offset;
      std::cout << "[condbranch]set offset in line "<< current_line << " to " << offset << std::endl;
    }
  });
}

static void liveVariableAnalysis(
    std::vector<Block*> blocks,
    std::map<Block*, std::set<int> > &def, 
    std::map<Block*, std::set<int> > &use,
    std::map<Block*, std::set<int> > &in,
    std::map<Block*, std::set<int> > &out ){
    Block* exit_block = blocks.back();
    if(exit_block->getSuccessors().size()!=0){
      std::cerr << "error: exit block should have no successor" << std::endl;
      std::exit(1);
    }
    for(Block* block : blocks){
      in[block] = {};
    }
    int change;
    do{
      change = 0;
      for(int i = 0; i<blocks.size()-1; i++){ // TODO: 必须保证blocks有一个单独的exit block,且位于最后一个位置
        Block* block = blocks[i];
        std::set<int> _in = in[block];
        std::set<int> _out = out[block];
        std::set<int> _def = def[block];
        std::set<int> _use = use[block];
        std::set<int> _new_in;
        std::set<int> _new_out;

        // out[B] = \union_{S:successor of B} in[B]
        for(Block *successor : block->getSuccessors()){
          std::set<int> _succ_in = in[successor];
          _new_out.insert(_succ_in.begin(), _succ_in.end());
        }

        // in[B] = use[B] \union (out[B] - def[B])
        std::set<int> difference;
        std::set_difference(_new_out.begin(), _new_out.end(), _def.begin(), _def.end(), std::inserter(difference, difference.begin()));
        std::set_union(difference.begin(), difference.end(), _use.begin(), _use.end(), std::inserter(_new_in, _new_in.begin()));

        bool is_equal_in = (_in==_new_in);
        bool is_equal_out = (_out==_new_out);
        bool is_equal = is_equal_in && is_equal_out;
        if(!is_equal) change = 1;

        in[block] = _new_in;
        out[block] = _new_out;
        // std::cout << "def: ";
        // for(int i : _def) std::cout << i << " ";
        // std::cout << " | use: ";
        // for(int i : _use) std::cout << i << " ";
        // std::cout << " | _new_in: ";
        // for(int i : _new_in) std::cout << i << " ";
        // std::cout << " | _new_out: ";
        // for(int i : _new_out) std::cout << i << " ";
        // std::cout << std::endl;
        

      }
      // std::cout << "------" << change << std::endl;
    }while(change);
    
}

static bool isPrefix(const std::string& str, const std::string& prefix) {
    // 检查前缀长度是否大于字符串长度
    if (prefix.length() > str.length()) {
        return false;
    }

    // 获取字符串的子串，长度等于前缀的长度，从字符串的开始位置
    std::string strPrefix = str.substr(0, prefix.length());

    // 比较子串和前缀是否相等
    return strPrefix == prefix;
}

static bool isSpecialLi(Inst& inst){
  if((inst.count("class") && inst["class"]==0b10) && 
     (inst.count("type") && inst["type"]==0b11 ) && 
     (inst.count("opcode") && inst["opcode"]==0b01)){
    return true;
  }
  return false;
}

static bool isSpecialAssign(Inst& inst){
  if((inst.count("class") && inst["class"]==0b10) && 
     (inst.count("type") && inst["type"]==0b11 ) && 
     (inst.count("opcode") && inst["opcode"]==0b10)){
    return true;
  }
  return false;
}

static void mappingRegisterLogicalToPhysical(
      std::vector<Inst>& instr_list,
      std::map<Block*, std::set<int> > &in,
      std::map<Block*, std::set<int> > &out,
      std::map<Block*, int> &block2line,
      std::map<Block*, int> &block2line_end){

  // Step 1: get life cycle of each logical register
  unordered_map<int, int> logic_reg_life_begin;
  unordered_map<int, int> logic_reg_life_end;
  for(int inst_id = 0;inst_id < instr_list.size(); inst_id++){
    Inst inst = instr_list[inst_id];
    // skip special register instruction
    if(isSpecialLi(inst)) continue;

    for (const auto& [key, value] : inst) {
      bool is_special_assign = isSpecialAssign(inst);
      bool is_reg_general = (is_special_assign && key=="rs1") || ((!is_special_assign) && (isPrefix(key, "rs") || isPrefix(key, "rd")));
      if(is_reg_general){
        int reg_id = value;
        if (!logic_reg_life_begin.count(reg_id)){
          logic_reg_life_begin[reg_id] = inst_id;
          logic_reg_life_end[reg_id] = inst_id ;
        }else{
          logic_reg_life_end[reg_id] = inst_id;
        }
      }
    }
  }
  for (const auto& [block, regs] : in) {
    for(auto reg_id : regs){
      logic_reg_life_begin[reg_id] = min(logic_reg_life_begin[reg_id], block2line[block]);
      logic_reg_life_end[reg_id] = max(logic_reg_life_end[reg_id], block2line_end[block]);
    }
  }
  for (const auto& [block, regs] : out) {
    for(auto reg_id : regs){
      logic_reg_life_begin[reg_id] = min(logic_reg_life_begin[reg_id], block2line[block]);
      logic_reg_life_end[reg_id] = max(logic_reg_life_end[reg_id], block2line_end[block]);
    }
  }
  
  // Step 2: Construct a mapping from logical register to physical register
  int num_logical_regs = logic_reg_life_begin.size();
  int num_physical_regs = 32;
  std::priority_queue<int, std::vector<int>, std::greater<int>> physical_regs;
  std::unordered_map<int, int> logical_to_physical_mapping;
  for(int i = 0; i < num_physical_regs; i++) physical_regs.push(i);
  for(int inst_id = 0;inst_id < instr_list.size(); inst_id++){
    for(int logical_reg_id = 0; logical_reg_id < num_logical_regs ;logical_reg_id++){
      if(logic_reg_life_begin[logical_reg_id]==inst_id){
        if (physical_regs.empty()){
          std::cerr << "No more physical_regs can use!" << std::endl;
          std::exit(1);
        }
        int physical_reg = physical_regs.top();
        physical_regs.pop();
        logical_to_physical_mapping[logical_reg_id] = physical_reg;
      }else if(logic_reg_life_end[logical_reg_id]==inst_id){
        int physical_reg = logical_to_physical_mapping[logical_reg_id];
        physical_regs.push(physical_reg);
      }
    }
  }
  for(int logical_reg_id = 0; logical_reg_id < num_logical_regs ;logical_reg_id++){
    std::cout << "logical_reg: " << logical_reg_id << " -> physical_reg: " << logical_to_physical_mapping[logical_reg_id] << std::endl;
  }
  for(int logical_reg_id = 0; logical_reg_id < num_logical_regs ;logical_reg_id++){
    std::cout << "logical_reg:" << logical_reg_id << " begin: " << logic_reg_life_begin[logical_reg_id] << " end: " << logic_reg_life_end[logical_reg_id] << std::endl;
  }
  // return;
  // Step 3: replace logical register to physical register
  for(int inst_id = 0;inst_id < instr_list.size(); inst_id++){
    Inst &inst = instr_list[inst_id];
     // skip special register instruction
    if(isSpecialLi(inst)) continue;
    
    std::unordered_map<string, int> replace;
    for (const auto& [key, value] : inst) {
      bool is_special_assign = isSpecialAssign(inst);
      bool is_reg_general = (is_special_assign && key=="rs1") || ((!is_special_assign) && (isPrefix(key, "rs") || isPrefix(key, "rd")));

      if(is_reg_general){
        replace[key] = logical_to_physical_mapping[value];
      }
    }
    for (const auto& [key, value] : replace) {
      inst[key] = value;
    }
  }
}

struct CodeGenerationPass
    : public mlir::PassWrapper<CodeGenerationPass, OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CodeGenerationPass)

  std::string outputFilePath;

  void runOnOperation() override {
    std::cout << "run on operation" << std::endl;
    std::cout << "code generation pass!" << std::endl;
    auto f = getOperation();
    if (f.getName()!="main"){
      return;
    }
    std::cout << "code generation pass, run on main!" << std::endl;
    

    std::unordered_map<llvm::hash_code, int > regmap = getRegisterMapping(f);
    std::cout << "getRegisterMapping finish!" << std::endl;

    std::vector<Inst> instr_list;
    std::map<Block*, int> block2line;
    std::map<Block*, int> block2line_end;
    std::map<Operation*, int> jump2line;
    std::map<Block*, std::set<int> > def;
    std::map<Block*, std::set<int> > use;
    std::vector<Block*> blocks = getBlockList(f);
    codeGen(blocks, regmap, instr_list, block2line, block2line_end, jump2line, def, use);
    std::cout << "codegen finish!" << std::endl;

    fillJumpBranchOffset(f, instr_list, block2line, jump2line);
    std::cout << "fill jump offset finish!" << std::endl;

    std::map<Block*, std::set<int> > in;
    std::map<Block*, std::set<int> > out;
    liveVariableAnalysis(blocks, def, use, in, out);
    std::cout << "live variable analysis finish!" << std::endl;

    mappingRegisterLogicalToPhysical(instr_list, in, out, block2line, block2line_end);

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
