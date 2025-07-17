#include "cim/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LogicalResult.h"
#include <iostream>
#include "common/macros.h"
using namespace mlir;
using namespace cim;


struct ShapeToConstant : public mlir::OpRewritePattern<mlir::cim::ShapeOp> {
 
  ShapeToConstant(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::cim::ShapeOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::cim::ShapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    LOG_DEBUG << "ShapeToConstant";

    auto operands = op.getOperands();

    mlir::Value source = operands[0];
    mlir::MemRefType source_type =
        llvm::cast<mlir::MemRefType>(source.getType());
    ArrayRef<int64_t> shape = source_type.getShape();

    mlir::Value index = operands[1];
    mlir::arith::ConstantIndexOp const_index_op =
        index.getDefiningOp<mlir::arith::ConstantIndexOp>();
    int64_t index_value = const_index_op.value();

    int64_t size = shape[index_value];
    if (size == mlir::ShapedType::kDynamic) {
      return failure();
    }

    mlir::Value new_constant =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), size);
    rewriter.replaceOp(op, {new_constant});
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void ShapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  LOG_DEBUG << "ShapeOp::getCanonicalizationPatterns";
  results.add<ShapeToConstant>(context);
}
