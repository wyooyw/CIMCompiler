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

template <typename Ty, typename Tz>
static LogicalResult rewrite_rr_to_ri(Ty &op, PatternRewriter &rewriter) {
  IntegerAttr constant;
  Value value;

  bool operand0_is_const = isConstant(op.getOperand(0));
  bool operand1_is_const = isConstant(op.getOperand(1));

  if (operand0_is_const && operand1_is_const) {
    LOG_ERROR << "This should not happend!";
    std::exit(1);
  } else if (operand0_is_const) {
    constant = getConstantInt(op.getOperand(0));
    value = op.getOperand(1);
  } else if (operand1_is_const) {
    constant = getConstantInt(op.getOperand(1));
    value = op.getOperand(0);
  } else {
    return failure();
  }

  rewriter.replaceOpWithNewOp<Tz>(op, op.getResult().getType(), value,
                                  constant);
  return success();
}

template <typename Ty, typename Tz>
static LogicalResult
rewrite_rr_to_ri_non_commutative(Ty &op, PatternRewriter &rewriter) {
  IntegerAttr constant;
  Value value;

  bool operand0_is_const = isConstant(op.getOperand(0));
  bool operand1_is_const = isConstant(op.getOperand(1));

  if (operand0_is_const && operand1_is_const) {
    LOG_ERROR << "This should not happend!";
    std::exit(1);
  } else if (operand1_is_const) {
    constant = getConstantInt(op.getOperand(1));
    value = op.getOperand(0);
  } else {
    return failure();
  }

  rewriter.replaceOpWithNewOp<Tz>(op, op.getResult().getType(), value,
                                  constant);
  return success();
}

namespace {

struct AddIOpConvert : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "AddIOpConvert::matchAndRewrite begin";
    LogicalResult result =
        rewrite_rr_to_ri<arith::AddIOp, cimisa::RIAddIOp>(op, rewriter);
    LOG_DEBUG << "AddIOpConvert::matchAndRewrite finish";
    return result;
  }
};
struct SubIOpConvert : public OpRewritePattern<arith::SubIOp> {
  using OpRewritePattern<arith::SubIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SubIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "SubIOpConvert::matchAndRewrite begin";
    LogicalResult result =
        rewrite_rr_to_ri_non_commutative<arith::SubIOp, cimisa::RISubIOp>(
            op, rewriter);
    LOG_DEBUG << "SubIOpConvert::matchAndRewrite finish";
    return result;
  }
};
struct MulIOpConvert : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "MulIOpConvert::matchAndRewrite begin";
    LogicalResult result =
        rewrite_rr_to_ri<arith::MulIOp, cimisa::RIMulIOp>(op, rewriter);
    LOG_DEBUG << "MulIOpConvert::matchAndRewrite finish";
    return result;
  }
};
struct DivSIOpConvert : public OpRewritePattern<arith::DivSIOp> {
  using OpRewritePattern<arith::DivSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "DivSIOpConvert::matchAndRewrite begin";
    LogicalResult result =
        rewrite_rr_to_ri_non_commutative<arith::DivSIOp, cimisa::RIDivSIOp>(
            op, rewriter);
    LOG_DEBUG << "DivSIOpConvert::matchAndRewrite finish";
    return result;
  }
};
struct RemSIOpConvert : public OpRewritePattern<arith::RemSIOp> {
  using OpRewritePattern<arith::RemSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::RemSIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "RemSIOpConvert::matchAndRewrite begin";
    LogicalResult result =
        rewrite_rr_to_ri_non_commutative<arith::RemSIOp, cimisa::RIRemSIOp>(
            op, rewriter);
    LOG_DEBUG << "RemSIOpConvert::matchAndRewrite finish";
    return result;
  }
};
struct MinSIOpConvert : public OpRewritePattern<arith::MinSIOp> {
  using OpRewritePattern<arith::MinSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MinSIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "MinSIOpConvert::matchAndRewrite begin";
    LogicalResult result =
        rewrite_rr_to_ri<arith::MinSIOp, cimisa::RIMinSIOp>(op, rewriter);
    LOG_DEBUG << "MinSIOpConvert::matchAndRewrite finish";
    return result;
  }
};

struct TransOpConvert : public OpRewritePattern<cimisa::TransOp> {
  using OpRewritePattern<cimisa::TransOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cimisa::TransOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "TransOpConvert::matchAndRewrite begin";
    IntegerAttr constant;
    Value value;

    Value src_addr = op.getOperand(0);
    Value dst_addr = op.getOperand(1);
    Value size = op.getOperand(2);

    bool change = false;
    if (isConstant(src_addr)) {
      IntegerAttr constant = getConstantInt(src_addr);
      src_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), src_addr.getType(), constant);
      change = true;
    }
    if (isConstant(dst_addr)) {
      IntegerAttr constant = getConstantInt(dst_addr);
      dst_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), src_addr.getType(), constant);
      change = true;
    }
    if (isConstant(size)) {
      IntegerAttr constant = getConstantInt(size);
      size = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), src_addr.getType(), constant);
      change = true;
    }

    if (!change) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<cimisa::TransOp>(op, src_addr, dst_addr, size, 0, false, false);
    LOG_DEBUG << "TransOpConvert::matchAndRewrite finish";
    return success();
  }
};

struct StoreBaseAndOffsetOpConvert
    : public OpRewritePattern<cimisa::StoreBaseAndOffsetOp> {
  using OpRewritePattern<cimisa::StoreBaseAndOffsetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cimisa::StoreBaseAndOffsetOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "StoreBaseAndOffsetOpConvert::matchAndRewrite begin";

    Value base = op.getOperand(0);
    Value offset = op.getOperand(1);
    Value value = op.getOperand(2);
    IntegerAttr constant;
    if (isConstant(base)) {
      IntegerAttr base_constant = getConstantInt(base);
      base = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), base.getType(), base_constant);
    }
    if (isConstant(offset)) {
      constant = getConstantInt(offset);
    } else {
      constant = rewriter.getIndexAttr(0);
      base = rewriter.create<arith::AddIOp>(op.getLoc(), base, offset);
    }
    if (isConstant(value)) {
      IntegerAttr value_constant = getConstantInt(value);
      value = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), value.getType(), value_constant);
    }

    rewriter.replaceOpWithNewOp<cimisa::StoreOp>(op, base, value, constant);
    LOG_DEBUG << "StoreBaseAndOffsetOpConvert::matchAndRewrite finish";
    return success();
  }
};

struct LoadBaseAndOffsetOpConvert
    : public OpRewritePattern<cimisa::LoadBaseAndOffsetOp> {
  using OpRewritePattern<cimisa::LoadBaseAndOffsetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cimisa::LoadBaseAndOffsetOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "LoadBaseAndOffsetOpConvert::matchAndRewrite begin";

    Value base = op.getOperand(0);
    Value offset = op.getOperand(1);
    IntegerAttr constant;
    LOG_DEBUG << "LoadBaseAndOffsetOpConvert::matchAndRewrite 1";
    if (isConstant(base)) {
      IntegerAttr base_constant = getConstantInt(base);
      base = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), base.getType(), base_constant);
    }

    if (isConstant(offset)) {
      constant = getConstantInt(offset);
    } else {
      constant = rewriter.getIndexAttr(0);
      base = rewriter.create<arith::AddIOp>(op.getLoc(), base, offset);
    }

    LOG_DEBUG << "LoadBaseAndOffsetOpConvert::matchAndRewrite 2";
    // MemRefType memtype =
    // llvm::cast<mlir::MemRefType>(op.getOperand(0).getType()); Type type =
    // memtype.getElementType();
    rewriter.replaceOpWithNewOp<cimisa::LoadOp>(op, op.getResult().getType(),
                                                base, constant);
    LOG_DEBUG << "LoadBaseAndOffsetOpConvert::matchAndRewrite finish";
    return success();
  }
};

struct CIMTransferOpConvert : public OpRewritePattern<cimisa::CIMTransferOp> {
  using OpRewritePattern<cimisa::CIMTransferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cimisa::CIMTransferOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "CIMTransferOpConvert::matchAndRewrite begin";

    Value src_addr = op.getOperand(0);
    Value output_number = op.getOperand(1);
    Value output_mask_addr = op.getOperand(2);
    Value buffer_addr = op.getOperand(3);
    Value dst_addr = op.getOperand(4);
    LOG_DEBUG << "CIMTransferOpConvert::matchAndRewrite 1";

    bool change = false;
    if (isConstant(src_addr)) {
      IntegerAttr constant = getConstantInt(src_addr);
      src_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), src_addr.getType(), constant);
      change = true;
    }
    if (isConstant(output_number)) {
      IntegerAttr constant = getConstantInt(output_number);
      output_number = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), output_number.getType(), constant);
      change = true;
    }
    if (isConstant(output_mask_addr)) {
      IntegerAttr constant = getConstantInt(output_mask_addr);
      output_mask_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), output_mask_addr.getType(), constant);
      change = true;
    }
    if (isConstant(buffer_addr)) {
      IntegerAttr constant = getConstantInt(buffer_addr);
      buffer_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), buffer_addr.getType(), constant);
      change = true;
    }
    if (isConstant(dst_addr)) {
      IntegerAttr constant = getConstantInt(dst_addr);
      dst_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), dst_addr.getType(), constant);
      change = true;
    }

    if (!change) {
      return failure();
    }

    LOG_DEBUG << "CIMTransferOpConvert::matchAndRewrite 2";
    // MemRefType memtype =
    // llvm::cast<mlir::MemRefType>(op.getOperand(0).getType()); Type type =
    // memtype.getElementType();
    rewriter.replaceOpWithNewOp<cimisa::CIMTransferOp>(
        op, src_addr, output_number, output_mask_addr, buffer_addr, dst_addr);
    LOG_DEBUG << "CIMTransferOpConvert::matchAndRewrite finish";
    return success();
  }
};

struct CIMComputeOpConvert : public OpRewritePattern<cimisa::CIMComputeOp> {
  using OpRewritePattern<cimisa::CIMComputeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cimisa::CIMComputeOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "CIMComputeOpConvert::matchAndRewrite begin";

    Value input_addr = op.getOperand(0);
    // Value output_addr = op.getOperand(1);
    Value row_index = op.getOperand(1);
    Value input_size = op.getOperand(2);
    LOG_DEBUG << "CIMComputeOpConvert::matchAndRewrite 1";

    bool change = false;
    if (isConstant(input_addr)) {
      IntegerAttr constant = getConstantInt(input_addr);
      input_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), input_addr.getType(), constant);
      change = true;
    }
    // if (isConstant(output_addr)) {
    //     IntegerAttr constant = getConstantInt(output_addr);
    //     output_addr = rewriter.create<cimisa::GeneralRegLiOp>(op.getLoc(),
    //     output_addr.getType(), constant); change = true;
    // }
    if (isConstant(row_index)) {
      IntegerAttr constant = getConstantInt(row_index);
      row_index = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), row_index.getType(), constant);
      change = true;
    }
    if (isConstant(input_size)) {
      IntegerAttr constant = getConstantInt(input_size);
      input_size = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), input_size.getType(), constant);
      change = true;
    }

    if (!change) {
      return failure();
    }

    LOG_DEBUG << "CIMComputeOpConvert::matchAndRewrite 2";
    // MemRefType memtype =
    // llvm::cast<mlir::MemRefType>(op.getOperand(0).getType()); Type type =
    // memtype.getElementType();
    rewriter.replaceOpWithNewOp<cimisa::CIMComputeOp>(
        op, input_addr, row_index, input_size, op.getAccFlag(),
        op.getValueSparseFlag(), op.getBitSparseFlag());
    LOG_DEBUG << "CIMComputeOpConvert::matchAndRewrite finish";
    return success();
  }
};

struct SpecialRegAssignOpConvert : public OpRewritePattern<cimisa::SpecialRegAssignOp> {
  using OpRewritePattern<cimisa::SpecialRegAssignOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cimisa::SpecialRegAssignOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "SpecialRegAssignOpConvert::matchAndRewrite begin";

    Value value = op.getOperand();
    LOG_DEBUG << "SpecialRegAssignOpConvert::matchAndRewrite 1";

    if (isConstant(value)) {
      IntegerAttr constant = rewriter.getI32IntegerAttr(getConstantInt(value).getInt());
      IntegerAttr special_reg = rewriter.getI32IntegerAttr(op.getSpecialReg());
      rewriter.replaceOpWithNewOp<cimisa::SpecialRegLiOp>(op, special_reg, constant);
      return success();
    }
    return failure();
  }
};

struct CIMOutputOpConvert : public OpRewritePattern<cimisa::CIMOutputOp> {
  using OpRewritePattern<cimisa::CIMOutputOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cimisa::CIMOutputOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "CIMOutputOpConvert::matchAndRewrite begin";

    Value out_n = op.getOperand(0);
    Value mask_addr = op.getOperand(1);
    Value out_addr = op.getOperand(2);
      LOG_DEBUG << "CIMOutputOpConvert::matchAndRewrite 1";

    bool change = false;
    if (isConstant(out_n)) {
      IntegerAttr constant = getConstantInt(out_n);
      out_n = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), out_n.getType(), constant);
      change = true;
    }
    if (isConstant(mask_addr)) {
      IntegerAttr constant = getConstantInt(mask_addr);
      mask_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), mask_addr.getType(), constant);
      change = true;
    }
    if (isConstant(out_addr)) {
      IntegerAttr constant = getConstantInt(out_addr);
      out_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), out_addr.getType(), constant);
      change = true;
    }

    if (!change) {
      return failure();
    }

    LOG_DEBUG << "CIMOutputOpConvert::matchAndRewrite 2";
    // MemRefType memtype =
    // llvm::cast<mlir::MemRefType>(op.getOperand(0).getType()); Type type =
    // memtype.getElementType();
    rewriter.replaceOpWithNewOp<cimisa::CIMOutputOp>(op, out_n, mask_addr,
                                                     out_addr);
    LOG_DEBUG << "CIMOutputOpConvert::matchAndRewrite finish";
    return success();
  }
};

struct CIMOutputSumOpConvert : public OpRewritePattern<cimisa::CIMOutputSumOp> {
  using OpRewritePattern<cimisa::CIMOutputSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cimisa::CIMOutputSumOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "CIMOutputSumOpConvert::matchAndRewrite begin";

    Value out_n = op.getOperand(0);
    Value out_mask_addr = op.getOperand(1);
    Value output_addr = op.getOperand(2);
    LOG_DEBUG << "CIMOutputSumOpConvert::matchAndRewrite 1";

    bool change = false;
    if (isConstant(out_n)) {
      IntegerAttr constant = getConstantInt(out_n);
      out_n = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), out_n.getType(), constant);
      change = true;
    }
    if (isConstant(out_mask_addr)) {
      IntegerAttr constant = getConstantInt(out_mask_addr);
      out_mask_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), out_mask_addr.getType(), constant);
      change = true;
    }
    if (isConstant(output_addr)) {
      IntegerAttr constant = getConstantInt(output_addr);
      output_addr = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), output_addr.getType(), constant);
      change = true;
    }

    if (!change) {
      return failure();
    }

      LOG_DEBUG << "CIMOutputSumOpConvert::matchAndRewrite 2";
    // MemRefType memtype =
    // llvm::cast<mlir::MemRefType>(op.getOperand(0).getType()); Type type =
    // memtype.getElementType();
    rewriter.replaceOpWithNewOp<cimisa::CIMOutputSumOp>(
        op, out_n, out_mask_addr, output_addr);
    LOG_DEBUG << "CIMOutputSumOpConvert::matchAndRewrite finish";
    return success();
  }
};

struct CmpIOpConvert : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "CmpIOpConvert::matchAndRewrite begin";

    Value operand0 = op.getOperand(0);
    Value operand1 = op.getOperand(1);

    LOG_DEBUG << "CmpIOpConvert::matchAndRewrite 1";

    bool change = false;
    if (isConstant(operand0)) {
      IntegerAttr constant = getConstantInt(operand0);
      operand0 = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), operand0.getType(), constant);
      change = true;
    }
    if (isConstant(operand1)) {
      IntegerAttr constant = getConstantInt(operand1);
      operand1 = rewriter.create<cimisa::GeneralRegLiOp>(
          op.getLoc(), operand1.getType(), constant);
      change = true;
    }

    if (!change) {
      return failure();
    }

    LOG_DEBUG << "CmpIOpConvert::matchAndRewrite 2";
    // MemRefType memtype =
    // llvm::cast<mlir::MemRefType>(op.getOperand(0).getType()); Type type =
    // memtype.getElementType();
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, op.getPredicate(), operand0,
                                               operand1);
    LOG_DEBUG << "CmpIOpConvert::matchAndRewrite finish";
    return success();
  }
};
} // namespace

namespace {
struct RR2RIPass : public PassWrapper<RR2RIPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RR2RIPass)
  std::string config_path;
  void getDependentDialects(DialectRegistry &registry) const override {
    // registry.insert<affine::AffineDialect, func::FuncDialect,
    //                 memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void RR2RIPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  LOG_DEBUG << "RR2RIPass::runOnOperation";
  ConversionTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  patterns.add<AddIOpConvert, SubIOpConvert, MulIOpConvert, DivSIOpConvert,
               RemSIOpConvert, MinSIOpConvert, StoreBaseAndOffsetOpConvert,
               LoadBaseAndOffsetOpConvert, 
              //  TransOpConvert, 
               CIMTransferOpConvert,
               // CIMComputeOpConvert, 
               CIMOutputSumOpConvert,
               CmpIOpConvert, CIMOutputOpConvert, 
               SpecialRegAssignOpConvert>(
      &getContext());
  // ForOpConvert

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
  LOG_DEBUG << "RR2RIPass::runOnOperation finish!";
}

std::unique_ptr<Pass> mlir::cim::createRR2RIPass() {
  return std::make_unique<RR2RIPass>();
}