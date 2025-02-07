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
static int64_t getConstantIntValue(Value operand) {
  if (auto constantOp = operand.getDefiningOp<arith::ConstantIndexOp>()) {
    return constantOp.value();
  } else {
    LOG_ERROR << "getConstantInt fail";
    std::exit(1);
    return 0;
  }
}

static bool isAddInteger(Value operand) {
  auto add_op = operand.getDefiningOp<arith::AddIOp>();
  if (!add_op)
    return false;
  mlir::Value lhs = add_op.getOperand(0);
  mlir::Value rhs = add_op.getOperand(1);
  bool is_lhs_const = isConstant(lhs);
  bool is_rhs_const = isConstant(rhs);
  return (is_lhs_const && !is_rhs_const) || (!is_lhs_const && is_rhs_const);
}

static std::pair<mlir::Value, mlir::Value>
getAddIntegerOperators(Value operand) {
  auto add_op = operand.getDefiningOp<arith::AddIOp>();
  if (!add_op) {
    LOG_ERROR << "getAddInteger fail";
    std::exit(1);
  }
  mlir::Value lhs = add_op.getOperand(0);
  mlir::Value rhs = add_op.getOperand(1);
  bool is_lhs_const = isConstant(lhs);
  bool is_rhs_const = isConstant(rhs);

  mlir::Value val_const, val_var;
  if (is_lhs_const && !is_rhs_const) {
    val_const = lhs;
    val_var = rhs;
  } else if (!is_lhs_const && is_rhs_const) {
    val_const = rhs;
    val_var = lhs;
  } else {
    LOG_ERROR << "getAddInteger fail";
    std::exit(1);
  }
  return std::make_pair(val_var, val_const);
}

static bool isMulInteger(Value operand) {
  auto mul_op = operand.getDefiningOp<arith::MulIOp>();
  if (!mul_op)
    return false;
  mlir::Value lhs = mul_op.getOperand(0);
  mlir::Value rhs = mul_op.getOperand(1);
  bool is_lhs_const = isConstant(lhs);
  bool is_rhs_const = isConstant(rhs);
  return (is_lhs_const && !is_rhs_const) || (!is_lhs_const && is_rhs_const);
}

static std::pair<mlir::Value, mlir::Value>
getMulIntegerOperators(Value operand) {
  auto mul_op = operand.getDefiningOp<arith::MulIOp>();
  if (!mul_op) {
    LOG_ERROR << "getAddInteger fail";
    std::exit(1);
  }
  mlir::Value lhs = mul_op.getOperand(0);
  mlir::Value rhs = mul_op.getOperand(1);
  bool is_lhs_const = isConstant(lhs);
  bool is_rhs_const = isConstant(rhs);

  mlir::Value val_const, val_var;
  if (is_lhs_const && !is_rhs_const) {
    val_const = lhs;
    val_var = rhs;
  } else if (!is_lhs_const && is_rhs_const) {
    val_const = rhs;
    val_var = lhs;
  } else {
    LOG_ERROR << "getAddInteger fail";
    std::exit(1);
  }
  return std::make_pair(val_var, val_const);
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

struct AddiAddPattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "AddiAddPattern::matchAndRewrite begin";
    mlir::Value lhs = op.getOperand(0);
    mlir::Value rhs = op.getOperand(1);
    if (isConstant(lhs) || isConstant(rhs))
      return failure();

    mlir::Value v1, c1, v2;
    if (isAddInteger(lhs)) {
      auto value_and_constant = getAddIntegerOperators(lhs);
      v1 = value_and_constant.first;
      c1 = value_and_constant.second;
      v2 = rhs;
    } else if (isAddInteger(rhs)) {
      auto value_and_constant = getAddIntegerOperators(rhs);
      v1 = value_and_constant.first;
      c1 = value_and_constant.second;
      v2 = lhs;
    } else {
      return failure();
    }
    mlir::Value o1 = rewriter.create<arith::AddIOp>(op.getLoc(), v1, v2);
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, o1, c1);
    LOG_DEBUG << "AddiAddPattern::matchAndRewrite finish";
    return success();
  }
};

struct AddiMuliPattern : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "AddiMuliPattern::matchAndRewrite begin";
    mlir::Value lhs = op.getOperand(0);
    mlir::Value rhs = op.getOperand(1);
    mlir::Value v2, c2, v1, c1;
    if (isConstant(lhs) && !isConstant(rhs)) {
      c2 = lhs;
      v2 = rhs;
    } else if (isConstant(rhs) && !isConstant(lhs)) {
      c2 = rhs;
      v2 = lhs;
    } else {
      return failure();
    }

    if (isAddInteger(v2)) {
      auto value_and_constant = getAddIntegerOperators(v2);
      v1 = value_and_constant.first;
      c1 = value_and_constant.second;
    } else {
      return failure();
    }
    mlir::Value o1 = rewriter.create<arith::MulIOp>(op.getLoc(), v1, c2);
    mlir::Value o2 = rewriter.create<arith::MulIOp>(op.getLoc(), c1, c2);
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, o1, o2);
    LOG_DEBUG << "AddiMuliPattern::matchAndRewrite finish";
    return success();
  }
};

struct AddiAddiPattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "AddiAddiPattern::matchAndRewrite begin";
    mlir::Value lhs = op.getOperand(0);
    mlir::Value rhs = op.getOperand(1);
    mlir::Value v2, c2, v1, c1;
    if (isConstant(lhs) && !isConstant(rhs)) {
      c2 = lhs;
      v2 = rhs;
    } else if (isConstant(rhs) && !isConstant(lhs)) {
      c2 = rhs;
      v2 = lhs;
    } else {
      return failure();
    }

    if (isAddInteger(v2)) {
      auto value_and_constant = getAddIntegerOperators(v2);
      v1 = value_and_constant.first;
      c1 = value_and_constant.second;
    } else {
      return failure();
    }
    mlir::Value o1 = rewriter.create<arith::AddIOp>(op.getLoc(), c1, c2);
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, v1, o1);
    LOG_DEBUG << "AddiAddiPattern::matchAndRewrite finish";
    return success();
  }
};

struct MuliDiviPattern : public OpRewritePattern<arith::DivSIOp> {
  using OpRewritePattern<arith::DivSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const final {
    LOG_DEBUG << "MuliDiviPattern::matchAndRewrite begin";
    mlir::Value lhs = op.getOperand(0);
    mlir::Value rhs = op.getOperand(1);
    mlir::Value v2, c2, v1, c1;
    if (isConstant(lhs) && !isConstant(rhs)) {
      c2 = lhs;
      v2 = rhs;
    } else if (isConstant(rhs) && !isConstant(lhs)) {
      c2 = rhs;
      v2 = lhs;
    } else {
      return failure();
    }

    if (isMulInteger(v2)) {
      auto value_and_constant = getMulIntegerOperators(v2);
      v1 = value_and_constant.first;
      c1 = value_and_constant.second;
    } else {
      return failure();
    }

    int64_t c1_value = getConstantIntValue(c1);
    int64_t c2_value = getConstantIntValue(c2);
    if (c1_value % c2_value != 0) {
      return failure();
    }
    int64_t mul_factor = c1_value / c2_value;
    mlir::Value c3 =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), mul_factor);
    rewriter.replaceOpWithNewOp<arith::MulIOp>(op, v1, c3);
    LOG_DEBUG << "MuliDiviPattern::matchAndRewrite finish";
    return success();
  }
};
} // namespace

namespace {
struct CommonSubexpressionExposePass
    : public PassWrapper<CommonSubexpressionExposePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CommonSubexpressionExposePass)
  std::string config_path;
  void getDependentDialects(DialectRegistry &registry) const override {
    // registry.insert<affine::AffineDialect, func::FuncDialect,
    //                 memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void CommonSubexpressionExposePass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  LOG_DEBUG << "CommonSubexpressionExposePass::runOnOperation";
  ConversionTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  patterns
      .add<AddiAddPattern, AddiMuliPattern, AddiAddiPattern, MuliDiviPattern>(
          &getContext());
  // ForOpConvert

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
  LOG_DEBUG << "CommonSubexpressionExposePass::runOnOperation finish!";
}

std::unique_ptr<Pass> mlir::cim::createCommonSubexpressionExposePass() {
  LOG_DEBUG << "createCommonSubexpressionExposePass";
  return std::make_unique<CommonSubexpressionExposePass>();
}
