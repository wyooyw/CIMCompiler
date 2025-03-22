#include "cim/Dialect.h"
#include "cim/Passes.h"
#include "cimisa/Dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>

#include "common/macros.h"

using namespace mlir;

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
    // tell user
    std::cerr << "[safe_get_] Key error: " << key << std::endl;
    std::exit(1);
    // return nullptr;
  }
}
const boost::property_tree::ptree &
safe_get_child(const boost::property_tree::ptree &ast, const std::string &key) {
  if (ast.count(key)) {
    return ast.get_child(key);
  } else {
    // tell user
    std::cerr << "[safe_get_child] Key error: " << key << std::endl;
    std::exit(1);
    return ast;
  }
}
static std::map<std::string, int> memory_addr_list;
static void getMemoryAddrList(std::string config_path) {
  boost::property_tree::ptree ast;
  boost::property_tree::read_json(config_path, ast);

  // std::map<string, int> memory_addr_list;
  LOG_DEBUG << "getMemoryAddrList";
  auto json_memory_list = safe_get_child(ast, "memory_list");
  for (const auto &pair : json_memory_list) {
    auto json_memory = pair.second;
    std::string name = safe_get_as<std::string>(json_memory, "name");
    auto json_address = safe_get_child(json_memory, "addressing");
    int offset = safe_get_as<int>(json_address, "offset_byte");
    int size = safe_get_as<int>(json_address, "size_byte");

    memory_addr_list[name] = offset;
    LOG_DEBUG << "name: " << name << " offset: " << offset << " size: " << size;
  }

  // return memory_addr_list;
}

static int getMemoryBaseAddr(Value buffer) {
  mlir::MemRefType type = llvm::cast<mlir::MemRefType>(buffer.getType());
  mlir::DictionaryAttr memory_space =
      llvm::cast<mlir::DictionaryAttr>(type.getMemorySpace());
  std::string memory =
      llvm::cast<mlir::StringAttr>(memory_space.get("memory")).getValue().str();
  if (!memory_addr_list.count(memory)) {
    LOG_ERROR << "can't find memory: " << memory;
    std::exit(1);
  }
  int memory_addr = memory_addr_list[memory];
  return memory_addr;
}

static Value getValue(OpFoldResult offset, PatternRewriter &rewriter) {
  if (Attribute attr = llvm::dyn_cast_if_present<Attribute>(offset)) {
    Value value = rewriter.create<arith::ConstantIndexOp>(
        rewriter.getUnknownLoc(), cast<IntegerAttr>(attr).getInt());
    return value;
  } else if (Value value = llvm::dyn_cast_if_present<Value>(offset)) {
    return value;
  } else {
    return nullptr;
  }
}

static int getBitWidth(mlir::Type type) {
  if (type.isa<mlir::IntegerType>()) {
    return type.getIntOrFloatBitWidth();
  } else if (type.isa<mlir::FloatType>()) {
    return type.getIntOrFloatBitWidth();
  } else if (type.isa<mlir::IndexType>()) {
    return 32;
  } else {
    LOG_ERROR << "getBitWidth fail";
    std::exit(1);
    return 0;
  }
}

static int getBitWidthMemRefOperand(mlir::Value operand) {
  mlir::MemRefType type = llvm::cast<mlir::MemRefType>(operand.getType());
  return getBitWidth(type.getElementType());
}

static Value getBufferBaseAddr(Value buffer, PatternRewriter &rewriter) {
  mlir::MemRefType type = llvm::cast<mlir::MemRefType>(buffer.getType());
  mlir::DictionaryAttr memory_space =
      llvm::cast<mlir::DictionaryAttr>(type.getMemorySpace());
  int _buffer_offset =
      llvm::cast<mlir::IntegerAttr>(memory_space.get("address")).getInt();
  int _memory_offset = getMemoryBaseAddr(buffer);
  int _offset = _buffer_offset + _memory_offset;
  mlir::Value offset = rewriter.create<arith::ConstantIndexOp>(
      rewriter.getUnknownLoc(), _offset);
  return offset;
}

static Value getAddrValue(Value operand, PatternRewriter &rewriter) {
  if (auto alloc_op = operand.getDefiningOp<memref::AllocOp>()) {
    mlir::Value addr = getBufferBaseAddr(alloc_op.getResult(), rewriter);
    return addr;
  } else if (auto subViewOp = operand.getDefiningOp<memref::SubViewOp>()) {
    auto allocOp = subViewOp.getOperand(0).getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
      LOG_ERROR << "getAddrValue allocOp==nullptr";
      std::exit(1);
      return nullptr;
    }
    llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
    SmallVector<OpFoldResult> offsets = subViewOp.getMixedOffsets();

    Value addr_offset = getValue(offsets[0], rewriter);
    for (int i = 1; i < offsets.size(); i++) {
      if (Value offset_i = getValue(offsets[i], rewriter)) {
        Value shape_i = rewriter.create<arith::ConstantIndexOp>(
            rewriter.getUnknownLoc(), allocShapes[i]);
        Value mul = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(),
                                                   addr_offset, shape_i);
        Value add = rewriter.create<arith::AddIOp>(rewriter.getUnknownLoc(),
                                                   mul, offset_i);
        addr_offset = add;
      } else {
        return nullptr;
      }
    }
    int64_t bitwidth = getBitWidthMemRefOperand(operand);
    Value byte_addr_offset;
    if (bitwidth == 1) {
      Value bytewidth_reciprocal_value =
          rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 8);
      // we assume that addr_offset is multiple of 8
      byte_addr_offset = rewriter.create<arith::DivSIOp>(
          rewriter.getUnknownLoc(), addr_offset, bytewidth_reciprocal_value);
    } else if (bitwidth >= 8 && bitwidth % 8 == 0) {
      int64_t bytewidth = bitwidth / 8;
      Value bytewidth_value = rewriter.create<arith::ConstantIndexOp>(
          rewriter.getUnknownLoc(), bytewidth);
      byte_addr_offset = rewriter.create<arith::MulIOp>(
          rewriter.getUnknownLoc(), addr_offset, bytewidth_value);
    } else {
      LOG_ERROR << "Wrong bitwidth: " << bitwidth;
      std::exit(1);
    }
    // int64_t bytewidth = bitwidth / 8;
    // Value bytewidth_value =
    // rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(),
    // bytewidth); Value byte_addr_offset =
    // rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(), addr_offset,
    // bytewidth_value);

    mlir::Value addr_base = getBufferBaseAddr(allocOp.getResult(), rewriter);
    mlir::Value real_address = rewriter.create<arith::AddIOp>(
        rewriter.getUnknownLoc(), addr_base, byte_addr_offset);
    return real_address;
  } else {
    LOG_ERROR << "getAddrValue fail";
    std::exit(1);
    return nullptr;
  }
}

static std::pair<Value, Value>
getAddrBaseAndOffsetValue(Value operand, PatternRewriter &rewriter) {
  if (auto alloc_op = operand.getDefiningOp<memref::AllocOp>()) {
    mlir::Value base = getBufferBaseAddr(alloc_op.getResult(), rewriter);
    mlir::Value offset =
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
    return std::make_pair(base, offset);
  } else if (auto subViewOp = operand.getDefiningOp<memref::SubViewOp>()) {
    auto allocOp = subViewOp.getOperand(0).getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
      LOG_ERROR << "getAddrValue allocOp==nullptr";
      std::exit(1);
    }
    llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
    SmallVector<OpFoldResult> offsets = subViewOp.getMixedOffsets();

    Value addr_offset = getValue(offsets[0], rewriter);
    for (int i = 1; i < offsets.size(); i++) {
      if (Value offset_i = getValue(offsets[i], rewriter)) {
        Value shape_i = rewriter.create<arith::ConstantIndexOp>(
            rewriter.getUnknownLoc(), allocShapes[i]);
        Value mul = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(),
                                                   addr_offset, shape_i);
        Value add = rewriter.create<arith::AddIOp>(rewriter.getUnknownLoc(),
                                                   mul, offset_i);
        addr_offset = add;
      } else {
        LOG_ERROR << "Can't get value!";
        std::exit(1);
      }
    }
    int64_t bitwidth = getBitWidthMemRefOperand(operand);
    Value byte_addr_offset;
    if (bitwidth == 1) {
      Value bytewidth_reciprocal_value =
          rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 8);
      // we assume that addr_offset is multiple of 8
      byte_addr_offset = rewriter.create<arith::DivSIOp>(
          rewriter.getUnknownLoc(), addr_offset, bytewidth_reciprocal_value);
    } else if (bitwidth >= 8 && bitwidth % 8 == 0) {
      int64_t bytewidth = bitwidth / 8;
      Value bytewidth_value = rewriter.create<arith::ConstantIndexOp>(
          rewriter.getUnknownLoc(), bytewidth);
      byte_addr_offset = rewriter.create<arith::MulIOp>(
          rewriter.getUnknownLoc(), addr_offset, bytewidth_value);
    } else {
      LOG_ERROR << "Wrong bitwidth: " << bitwidth;
      std::exit(1);
    }
    // int64_t bytewidth = bitwidth / 8;
    // Value bytewidth_value =
    // rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(),
    // bytewidth); Value byte_addr_offset =
    // rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(), addr_offset,
    // bytewidth_value);

    mlir::Value addr_base = getBufferBaseAddr(allocOp.getResult(), rewriter);
    // mlir::Value real_address =
    // rewriter.create<arith::AddIOp>(rewriter.getUnknownLoc(), addr_base,
    // byte_addr_offset);
    return std::make_pair(addr_base, byte_addr_offset);
  } else {
    LOG_ERROR << "getAddrValue fail";
    std::exit(1);
  }
}

static Value getLengthValue(Value operand, PatternRewriter &rewriter) {
  if (auto allocOp = operand.getDefiningOp<memref::AllocOp>()) {

    llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
    int64_t size = 1;
    for (int i = 0; i < allocShapes.size(); i++) {
      size *= allocShapes[i];
    }
    mlir::Value zero =
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), size);
    return zero;
  } else if (auto subViewOp = operand.getDefiningOp<memref::SubViewOp>()) {
    SmallVector<OpFoldResult> shapes = subViewOp.getMixedSizes();

    Value size =
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 1);
    for (int i = 0; i < shapes.size(); i++) {
      if (Value shape_i = getValue(shapes[i], rewriter)) {
        size = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(), size,
                                              shape_i);
      } else {
        return nullptr;
      }
    }
    return size;
  } else {
    LOG_ERROR << "getSizeValue fail";
    std::exit(1);
    return nullptr;
  }
}

static Value getShapeValue(Value operand, int index, PatternRewriter &rewriter) {
  if (auto allocOp = operand.getDefiningOp<memref::AllocOp>()) {

    llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
    mlir::Value shape_i =
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), allocShapes[index]);
    return shape_i;
  } else if (auto subViewOp = operand.getDefiningOp<memref::SubViewOp>()) {
    SmallVector<OpFoldResult> shapes = subViewOp.getMixedSizes();
    if (Value shape_i = getValue(shapes[index], rewriter)) {
      return shape_i;
    } else {
      return nullptr;
    }
  } else {
    LOG_ERROR << "getSizeValue fail";
    std::exit(1);
    return nullptr;
  }
}

static Value getSizeValue(Value operand, PatternRewriter &rewriter) {
  if (auto allocOp = operand.getDefiningOp<memref::AllocOp>()) {

    llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
    int64_t size = 1;
    for (int i = 0; i < allocShapes.size(); i++) {
      size *= allocShapes[i];
    }

    int bitwidth = getBitWidthMemRefOperand(operand);
    int bytewidth = bitwidth / 8;
    if (bitwidth == 1) {
      size = size / 8;
    } else if (bitwidth >= 8 && bitwidth % 8 == 0) {
      size = size * (bitwidth / 8);
    } else {
      LOG_ERROR << "Wrong bitwidth: " << bitwidth;
      std::exit(1);
    }

    mlir::Value zero =
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), size);
    return zero;
  } else if (auto subViewOp = operand.getDefiningOp<memref::SubViewOp>()) {
    SmallVector<OpFoldResult> shapes = subViewOp.getMixedSizes();

    Value size =
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 1);
    for (int i = 0; i < shapes.size(); i++) {
      if (Value shape_i = getValue(shapes[i], rewriter)) {
        size = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(), size,
                                              shape_i);
      } else {
        return nullptr;
      }
    }

    int bitwidth = getBitWidthMemRefOperand(operand);
    int bytewidth = bitwidth / 8;

    if (bitwidth == 1) {
      Value bytewidth_reciprocal_value =
          rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 8);
      size = rewriter.create<arith::DivSIOp>(rewriter.getUnknownLoc(), size,
                                             bytewidth_reciprocal_value);
    } else if (bitwidth >= 8 && bitwidth % 8 == 0) {
      int64_t bytewidth = bitwidth / 8;
      Value bytewidth_value = rewriter.create<arith::ConstantIndexOp>(
          rewriter.getUnknownLoc(), bytewidth);
      size = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(), size,
                                            bytewidth_value);
    } else {
      LOG_ERROR << "Wrong bitwidth: " << bitwidth;
      std::exit(1);
    }

    return size;
  } else {
    LOG_ERROR << "getSizeValue fail";
    std::exit(1);
    return nullptr;
  }
}

static std::vector<Value> _getMacroActivatePositionBySubview(
    cim::CIMComputeOp op, PatternRewriter &rewriter, int operand_index) {
  // <N_ROW, N_COMP, N_GROUP, N_MACRO * N_VCOL>
  auto subViewOp =
      op.getOperand(operand_index).getDefiningOp<memref::SubViewOp>();
  auto allocOp = subViewOp.getOperand(0).getDefiningOp<memref::AllocOp>();
  llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();
  SmallVector<OpFoldResult> offsets = subViewOp.getMixedOffsets();
  SmallVector<OpFoldResult> shapes = subViewOp.getMixedSizes();

  Value activate_row_begin ;
  Value activate_group_num ;
  Value activate_macro_length ;
  Value activate_element_col_num ;
  // get activate row
  if (allocShapes.size() == 5) {
      activate_row_begin = getValue(offsets[0], rewriter);
      Value activate_outer_group_num = getValue(shapes[2], rewriter);
      Value activate_inner_group_num = getValue(shapes[3], rewriter);
      activate_group_num = rewriter.create<arith::MulIOp>(rewriter.getUnknownLoc(),
                                                   activate_outer_group_num, activate_inner_group_num);
      // activate_macro_length = getValue(shapes[2], rewriter);
      activate_element_col_num = getValue(shapes[4], rewriter);
  }else if (allocShapes.size() == 4) {
      activate_row_begin = getValue(offsets[0], rewriter);
      activate_group_num = getValue(shapes[2], rewriter);
      activate_macro_length = getValue(shapes[2], rewriter);
      activate_element_col_num = getValue(shapes[3], rewriter);
  }else if (allocShapes.size() == 2) {
      activate_row_begin = getValue(offsets[0], rewriter);
      activate_group_num = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 1);
      activate_macro_length = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 1);
      activate_element_col_num = getValue(shapes[1], rewriter);
  }else{
      // error
      LOG_ERROR << "_getMacroActivatePositionBySubview fail";
      std::exit(1);
      return {};
  }
  // Value activate_row_begin = getValue(offsets[0], rewriter);
  // Value activate_group_num = getValue(shapes[2], rewriter);
  // Value activate_macro_length = getValue(shapes[2], rewriter);
  // Value activate_element_col_num = getValue(shapes[3], rewriter);

  return {activate_row_begin, activate_element_col_num, activate_group_num};
}

static std::vector<Value> _getMacroActivatePositionByAlloc(
    cim::CIMComputeOp op, PatternRewriter &rewriter, int operand_index) {
  // <N_ROW, N_COMP, N_GROUP, N_MACRO * N_VCOL>
  auto allocOp =
      op.getOperand(operand_index).getDefiningOp<memref::AllocOp>();

  llvm::ArrayRef<int64_t> allocShapes = allocOp.getType().getShape();

  Value activate_row_begin ;
  Value activate_group_num ;
  Value activate_macro_length ;
  Value activate_element_col_num ;
  // get activate row
  if (allocShapes.size() == 5) {
      activate_row_begin = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
      activate_group_num = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), allocShapes[2] * allocShapes[3]);
      // activate_macro_length = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), allocShapes[3]);
      activate_element_col_num = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), allocShapes[4]);
  }else if (allocShapes.size() == 4) {
      activate_row_begin = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
      activate_group_num = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), allocShapes[2]);
      activate_element_col_num = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), allocShapes[3]);
  }else if (allocShapes.size() == 2) {
      activate_row_begin = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
      activate_group_num = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 1);
      activate_macro_length = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 1);
      activate_element_col_num = rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), allocShapes[1]);
  }else{
      // error
      LOG_ERROR << "_getMacroActivatePositionByAlloc fail";
      std::exit(1);
      return {};
  }
  // Value activate_row_begin = getValue(offsets[0], rewriter);
  // Value activate_group_num = getValue(shapes[2], rewriter);
  // Value activate_macro_length = getValue(shapes[2], rewriter);
  // Value activate_element_col_num = getValue(shapes[3], rewriter);

  return {activate_row_begin, activate_element_col_num, activate_group_num};
}

static std::vector<Value> getMacroActivatePosition(cim::CIMComputeOp op,
                                                   PatternRewriter &rewriter,
                                                   int operand_index) {
  if (op.getOperand(operand_index).getDefiningOp<memref::SubViewOp>()) {
    return _getMacroActivatePositionBySubview(op, rewriter, operand_index);
  } else if (op.getOperand(operand_index).getDefiningOp<memref::AllocOp>()){
    return _getMacroActivatePositionByAlloc(op, rewriter, operand_index);
  } else {
    // fail
    LOG_ERROR << "getMacroActivatePosition fail";
    return {};
  }
}

static IntegerAttr getI1IntegerAttr(int32_t value, PatternRewriter &rewriter) {
  return IntegerAttr::get(rewriter.getIntegerType(1), APInt(1, value));
}

static bool isConstant(Value operand) {
  return operand.getDefiningOp<arith::ConstantOp>();
}

static IntegerAttr getConstantInt(Value operand) {
  if (auto constantOp = operand.getDefiningOp<arith::ConstantOp>()) {
    return constantOp.getValue().cast<IntegerAttr>();
  } else {
    LOG_ERROR << "getConstantInt fail";
    std::exit(1);
    return 0;
  }
}

// why need this namespace ?
namespace {

struct LoadOpLowering : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const final {
    /*
      Now, all index of load is 0.
    */
    LOG_DEBUG << "LoadOpLowering::matchAndRewrite 1";
    // Value addr_src = getAddrValue(op.getOperand(0), rewriter);
    std::pair<Value, Value> base_and_offset =
        getAddrBaseAndOffsetValue(op.getOperand(0), rewriter);
    Value base = base_and_offset.first;
    Value offset = base_and_offset.second;
    LOG_DEBUG << "LoadOpLowering::matchAndRewrite 4";
    if (!base | !offset) {
      LOG_ERROR << "LoadOpLowering::matchAndRewrite fail";
      return failure();
    }
    LOG_DEBUG << "LoadOpLowering::matchAndRewrite success";

    MemRefType memtype =
        llvm::cast<mlir::MemRefType>(op.getOperand(0).getType());
    Type type = memtype.getElementType();
    auto new_op = rewriter.create<mlir::cimisa::LoadBaseAndOffsetOp>(
        op.getLoc(), type, base, offset);
    rewriter.replaceOp(op, {new_op.getResult()});
    // rewriter.replaceOpWithNewOp<mlir::cimisa::LoadOp>(op, type, addr_src);
    return success();
  }
};

struct StoreOpLowering : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const final {
    /*
      Now, all index of load is 0.
    */
    LOG_DEBUG << "StoreOpLowering::matchAndRewrite 1";
    Value value = op.getOperand(0);
    // Value addr_dst = getAddrValue(op.getOperand(1), rewriter);
    std::pair<Value, Value> base_and_offset =
        getAddrBaseAndOffsetValue(op.getOperand(1), rewriter);
    Value base = base_and_offset.first;
    Value offset = base_and_offset.second;

    // if (isConstant(offset))

    LOG_DEBUG << "StoreOpLowering::matchAndRewrite 4";
    if (!base || !offset) {
      LOG_ERROR << "StoreOpLowering::matchAndRewrite fail";
      return failure();
    }
    LOG_DEBUG << "StoreOpLowering::matchAndRewrite success";

    rewriter.replaceOpWithNewOp<cimisa::StoreBaseAndOffsetOp>(op, base, offset,
                                                              value);
    return success();
  }
};

struct TransOpLowering : public OpRewritePattern<cim::CopyOp> {
  using OpRewritePattern<cim::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::CopyOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "TransOpLowering::matchAndRewrite begin";
    Value addr_src = getAddrValue(op.getOperand(0), rewriter);
    Value addr_dst = getAddrValue(op.getOperand(1), rewriter);
    Value size = getSizeValue(op.getOperand(0), rewriter);
    LOG_DEBUG << "TransOpLowering::matchAndRewrite";
    if (!addr_src || !addr_dst || !size) {
      LOG_ERROR << "TransOpLowering::matchAndRewrite fail";
      return failure();
    }
    LOG_DEBUG << "TransOpLowering::matchAndRewrite success";

    rewriter.replaceOpWithNewOp<cimisa::TransOp>(op, addr_src, addr_dst, size, 0, false, false);

    return success();
  }
};

struct CIMComputeOpLowering : public OpRewritePattern<cim::CIMComputeOp> {
  using OpRewritePattern<cim::CIMComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::CIMComputeOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "CIMComputeOpLowering::matchAndRewrite 1";
    Value addr_input = getAddrValue(op.getOperand(0), rewriter);
    LOG_DEBUG << "CIMComputeOpLowering::matchAndRewrite 2";
    // Value addr_output = getAddrValue(op.getOperand(2), rewriter);
    LOG_DEBUG << "CIMComputeOpLowering::matchAndRewrite 3";
    std::vector<Value> macro_activate =
        getMacroActivatePosition(op, rewriter, 1);
    LOG_DEBUG << "CIMComputeOpLowering::matchAndRewrite 4";
    if (macro_activate.size() < 2) {
      LOG_ERROR << "CIMComputeOpLowering::matchAndRewrite fail 1";
      return failure();
    }
    Value row_index = macro_activate[0];
    Value num_group = macro_activate[2];
    Value input_size_all = getLengthValue(op.getOperand(0), rewriter);
    Value input_size =
        rewriter.create<arith::DivSIOp>(op.getLoc(), input_size_all, num_group);

    if (!addr_input || !row_index || !input_size) {
      LOG_ERROR << "CIMComputeOpLowering::matchAndRewrite fail 2";
      return failure();
    }
    LOG_DEBUG << "CIMComputeOpLowering::matchAndRewrite success";

    IntegerAttr input_bw = rewriter.getI8IntegerAttr(8);
    IntegerAttr output_bw = rewriter.getI8IntegerAttr(32);
    IntegerAttr weight_bw = rewriter.getI8IntegerAttr(8);

    IntegerAttr acc_flag = getI1IntegerAttr(1, rewriter);
    IntegerAttr value_sparse_flag =
        getI1IntegerAttr(op.getValueSparseFlag(), rewriter);
    IntegerAttr bit_sparse_flag =
        getI1IntegerAttr(op.getBitSparseFlag(), rewriter);

    rewriter.replaceOpWithNewOp<cimisa::CIMComputeOp>(
        op,
        addr_input, // AnyTypeOf<[AnyInteger, Index]>:$input_addr,
        // addr_output,          // AnyTypeOf<[AnyInteger, Index]>:$output_addr,
        row_index,         // AnyTypeOf<[AnyInteger, Index]>:$row_index,
        input_size,        // AnyTypeOf<[AnyInteger, Index]>:$input_size,
        acc_flag,          // I1:$acc_flag,
        value_sparse_flag, // I1:$value_sparse_flag,
        bit_sparse_flag    // I1:$bit_sparse_flag
    );

    return success();
  }
};

struct SIMDOpLowering : public OpRewritePattern<cim::SIMDOp> {
  using OpRewritePattern<cim::SIMDOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::SIMDOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "SIMDOpLowering::matchAndRewrite begin";
    int SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_1 = 21;
    int SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_2 = 22;

    int num_operands = op.getNumOperands();
    int num_inputs = num_operands - 2;
    if (num_inputs > 4){
      LOG_ERROR << "SIMDOpLowering::matchAndRewrite fail";
      return failure();
    }

    IntegerAttr op_id_ = getConstantInt(op.getOperand(0));
    IntegerAttr op_id = rewriter.getI32IntegerAttr(op_id_.getInt());
    if (!op_id) {
      LOG_ERROR << "SIMDOpLowering::matchAndRewrite fail";
      return failure();
    }

    Value size = getLengthValue(op.getOperand(1), rewriter);
    if (!size) {
      LOG_ERROR << "SIMDOpLowering::matchAndRewrite fail";
      return failure();
    }

    llvm::SmallVector<Value, 2> inputs_addr;
    for (int i = 0; i < num_inputs; i++) {
      int operand_id = i + 1;
      Value addr = getAddrValue(op.getOperand(operand_id), rewriter);
      if (!addr) {
        LOG_ERROR << "SIMDOpLowering::matchAndRewrite fail";
        return failure();
      }
      if (i <= 1){
        inputs_addr.push_back(addr);
      }else if(i == 2){
        IntegerAttr special_reg = rewriter.getI32IntegerAttr(SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_1);
        rewriter.create<cimisa::SpecialRegAssignOp>(rewriter.getUnknownLoc(), special_reg, addr);
      }else if(i == 3){
        IntegerAttr special_reg = rewriter.getI32IntegerAttr(SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_2);
        rewriter.create<cimisa::SpecialRegAssignOp>(rewriter.getUnknownLoc(), special_reg, addr);
      }else{
        LOG_ERROR << "SIMDOpLowering::matchAndRewrite fail";
        return failure();
      }
    }

    Value output_addr = getAddrValue(op.getOperand(num_operands - 1), rewriter);
    if (!output_addr) {
      LOG_ERROR << "SIMDOpLowering::matchAndRewrite fail";
      return failure();
    }


    LOG_DEBUG << "SIMDOpLowering::matchAndRewrite";

    IntegerAttr num_inputs_attr = rewriter.getI32IntegerAttr(num_inputs);
    rewriter.replaceOpWithNewOp<cimisa::SIMDOp>(op, op_id, num_inputs_attr, inputs_addr, output_addr, size);

    return success();
  }
};

struct SpecialRegSetOpLowering : public OpRewritePattern<cim::SpecialRegSetOp> {
  using OpRewritePattern<cim::SpecialRegSetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::SpecialRegSetOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "SpecialRegSetOpLowering begin";
    Value special_reg = op.getOperand(0);
    LOG_DEBUG << "SpecialRegSetOpLowering 1";
    Value set_value = op.getOperand(1);
    LOG_DEBUG << "SpecialRegSetOpLowering 2";

    if (!special_reg || !set_value) {
      LOG_ERROR << "SpecialRegSetOpLowering::matchAndRewrite fail";
      return failure();
    }

    if (auto special_reg_const_op =
            special_reg.getDefiningOp<arith::ConstantIndexOp>()) {
      int64_t special_reg_const = special_reg_const_op.value();
      if (auto set_value_const_op =
              set_value.getDefiningOp<arith::ConstantIndexOp>()) {
        int64_t set_value_const = set_value_const_op.value();
        rewriter.replaceOpWithNewOp<cimisa::SpecialRegLiOp>(
            op, special_reg_const, set_value_const);
      } else {
        rewriter.replaceOpWithNewOp<cimisa::SpecialRegAssignOp>(
            op, special_reg_const, set_value);
      }
    } else {
      LOG_ERROR << "SpecialRegSetOpLowering special_reg is not constant";
      return failure(); 
    }
    LOG_DEBUG << "SpecialRegSetOpLowering::matchAndRewrite success";
    return success();
  }
};

struct CIMOutputOpLowering : public OpRewritePattern<cim::CIMOutputOp> {
  using OpRewritePattern<cim::CIMOutputOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::CIMOutputOp op,
                                PatternRewriter &rewriter) const final {
    /*
      Now, all index of load is 0.
    */
    LOG_DEBUG << "CIMOutputOpLowering::matchAndRewrite begin";
    Value out_n = op.getOperand(0);
    Value mask_addr =
        op.getOperand(1); // this is true, no need to call getAddrValue
    Value addr_dst = getAddrValue(op.getOperand(2), rewriter);
    if (!out_n || !mask_addr || !addr_dst) {
      std::cerr << "CIMOutputOpLowering::matchAndRewrite fail" << std::endl;
      std::exit(1);
    }
    rewriter.replaceOpWithNewOp<cimisa::CIMOutputOp>(op, out_n, mask_addr,
                                                     addr_dst);
    LOG_DEBUG << "CIMOutputOpLowering::matchAndRewrite finish";
    return success();
  }
};

struct CIMOutputSumOpLowering : public OpRewritePattern<cim::CIMOutputSumOp> {
  using OpRewritePattern<cim::CIMOutputSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::CIMOutputSumOp op,
                                PatternRewriter &rewriter) const final {
    /*
      Now, all index of load is 0.
    */
    LOG_DEBUG << "CIMOutputSumOpLowering::matchAndRewrite begin";
    Value out_n = op.getOperand(0);
    Value out_mask_addr = getAddrValue(op.getOperand(1), rewriter);
    Value output_addr = getAddrValue(op.getOperand(2), rewriter);
    if (!out_mask_addr || !output_addr) {
      std::cerr << "CIMOutputSumOpLowering::matchAndRewrite fail" << std::endl;
      std::exit(1);
    }
    rewriter.replaceOpWithNewOp<cimisa::CIMOutputSumOp>(
        op, out_n, out_mask_addr, output_addr);
    LOG_DEBUG << "CIMOutputSumOpLowering::matchAndRewrite finish";
    return success();
  }
};

struct CIMTransferOpLowering : public OpRewritePattern<cim::CIMTransferOp> {
  using OpRewritePattern<cim::CIMTransferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::CIMTransferOp op,
                                PatternRewriter &rewriter) const final {
    /*
      Now, all index of load is 0.
    */
    LOG_DEBUG << "CIMTransferOpLowering::matchAndRewrite begin";
    Value src_addr = getAddrValue(op.getOperand(0), rewriter);
    Value output_number = op.getOperand(1);
    Value output_mask_addr = getAddrValue(op.getOperand(2), rewriter);
    Value buffer_addr = getAddrValue(op.getOperand(3), rewriter);
    Value dst_addr = getAddrValue(op.getOperand(4), rewriter);
    if (!src_addr || !output_number || !output_mask_addr || !buffer_addr ||
        !dst_addr) {
      std::cerr << "CIMTransferOpLowering::matchAndRewrite fail" << std::endl;
      std::exit(1);
    }
    rewriter.replaceOpWithNewOp<cimisa::CIMTransferOp>(
        op, src_addr, output_number, output_mask_addr, buffer_addr, dst_addr);
    LOG_DEBUG << "CIMTransferOpLowering::matchAndRewrite finish";
    return success();
  }
};

struct CIMSetOpLowering : public OpRewritePattern<cim::CIMSetOp> {
  using OpRewritePattern<cim::CIMSetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::CIMSetOp op,
                                PatternRewriter &rewriter) const final {
    /*
      Now, all index of load is 0.
    */
    LOG_DEBUG << "CIMSetOpLowering::matchAndRewrite begin";
    Value mask_addr = getAddrValue(op.getOperand(), rewriter);
    if (!mask_addr) {
      std::cerr << "CIMSetOpLowering::matchAndRewrite fail" << std::endl;
      std::exit(1);
    }
    rewriter.replaceOpWithNewOp<cimisa::CIMSetOp>(op, mask_addr);
    LOG_DEBUG << "CIMSetOpLowering::matchAndRewrite finish";
    return success();
  }
};

struct SendOpLowering : public OpRewritePattern<cim::SendOp> {
  using OpRewritePattern<cim::SendOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::SendOp op,
                                PatternRewriter &rewriter) const final {
    /*
      Now, all index of load is 0.
    */
    LOG_DEBUG << "SendOpLowering::matchAndRewrite begin";
    Value src_addr = getAddrValue(op.getOperand(0), rewriter);
    Value dst_addr = getAddrValue(op.getOperand(1), rewriter);
    Value size = getSizeValue(op.getOperand(0), rewriter);
    Value core_id = op.getOperand(2);
    Value transfer_id = op.getOperand(3);
    if ((!src_addr) || (!dst_addr) || (!core_id) || (!transfer_id) ) {
      std::cerr << "SendOpLowering::matchAndRewrite fail" << std::endl;
      std::exit(1);
    }
    rewriter.replaceOpWithNewOp<cimisa::SendOp>(op, src_addr, dst_addr, size, core_id, transfer_id);
    LOG_DEBUG << "SendOpLowering::matchAndRewrite finish";
    return success();
  }
};

struct RecvOpLowering : public OpRewritePattern<cim::RecvOp> {
  using OpRewritePattern<cim::RecvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::RecvOp op,
                                PatternRewriter &rewriter) const final {
    /*
      Now, all index of load is 0.
    */
    LOG_DEBUG << "RecvOpLowering::matchAndRewrite begin";
    Value src_addr = getAddrValue(op.getOperand(0), rewriter);
    LOG_DEBUG << "RecvOpLowering::matchAndRewrite 1";
    Value dst_addr = getAddrValue(op.getOperand(1), rewriter);
    LOG_DEBUG << "RecvOpLowering::matchAndRewrite 2";
    Value size = getSizeValue(op.getOperand(0), rewriter);
    LOG_DEBUG << "RecvOpLowering::matchAndRewrite 3";
    Value core_id = op.getOperand(2);
    Value transfer_id = op.getOperand(3);
    if ((!src_addr) || (!dst_addr) || (!core_id) || (!transfer_id) ) {
      std::cerr << "RecvOpLowering::matchAndRewrite fail" << std::endl;
      std::exit(1);
    }
    rewriter.replaceOpWithNewOp<cimisa::RecvOp>(op, src_addr, dst_addr, size, core_id, transfer_id);
    LOG_DEBUG << "RecvOpLowering::matchAndRewrite finish";
    return success();
  }
};

struct AddrOpLowering : public OpRewritePattern<cim::AddrOp> {
  using OpRewritePattern<cim::AddrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::AddrOp op,
                                PatternRewriter &rewriter) const final {
    /*
      Now, all index of load is 0.
    */
    LOG_DEBUG << "AddrOpLowering::matchAndRewrite begin";
    Value src_addr = getAddrValue(op.getOperand(), rewriter);
    if (!src_addr) {
      LOG_ERROR << "AddrOpLowering::matchAndRewrite fail";
      return failure();
    }
    rewriter.replaceOp(op, {src_addr});
    LOG_DEBUG << "AddrOpLowering::matchAndRewrite finish";
    return success();
  }
};

struct ShapeOpLowering : public OpRewritePattern<cim::ShapeOp> {
  using OpRewritePattern<cim::ShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cim::ShapeOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "ShapeOpLowering::matchAndRewrite begin";
    // Value input = getAddrValue(op.getOperand(0), rewriter);
    IntegerAttr indexAttr = getConstantInt(op.getOperand(1));
    int64_t index = indexAttr.getInt();
    Value shape_i = getShapeValue(op.getOperand(0), index, rewriter);

    LOG_DEBUG << "ShapeOpLowering::matchAndRewrite";
    if (!shape_i) {
      LOG_ERROR << "ShapeOpLowering::matchAndRewrite fail";
      return failure();
    }
    LOG_DEBUG << "ShapeOpLowering::matchAndRewrite success";


    // replace ShapeOp with shape_i
    rewriter.replaceOp(op, {shape_i});

    return success();
  }
};
} // namespace

namespace {
struct CIMLoweringPass
    : public PassWrapper<CIMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CIMLoweringPass)
  std::string config_path;
  void getDependentDialects(DialectRegistry &registry) const override {
    // registry.insert<affine::AffineDialect, func::FuncDialect,
    //                 memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void CIMLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  LOG_DEBUG << "CIMLoweringPass::runOnOperation";
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  // target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
  //                        arith::ArithDialect, func::FuncDialect,
  //                        cimisa::CIMISADialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  // target.addIllegalDialect<cim::CIMDialect>();
  //   target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
  //     return llvm::none_of(op->getOperandTypes(),
  //                          [](Type type) { return
  //                          llvm::isa<TensorType>(type); });
  //   });

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns
      .add<TransOpLowering, CIMComputeOpLowering, LoadOpLowering,
           StoreOpLowering, SpecialRegSetOpLowering,
           CIMOutputOpLowering, CIMOutputSumOpLowering, CIMTransferOpLowering,
           AddrOpLowering, CIMSetOpLowering, ShapeOpLowering,
           SIMDOpLowering, SendOpLowering, RecvOpLowering>(&getContext());
  
  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  // if (failed(
  //         applyPartialConversion(getOperation(), target,
  //         std::move(patterns))))
  //   signalPassFailure();
  getMemoryAddrList(config_path);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
  LOG_DEBUG << "CIMLoweringPass::runOnOperation finish!";
}

std::unique_ptr<Pass>
mlir::cim::createCIMLoweringPass(std::string config_path) {
  auto pass = std::make_unique<CIMLoweringPass>();
  pass->config_path = config_path;
  return pass;
}