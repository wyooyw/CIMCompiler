//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

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
// using namespace arith;

// namespace {
// /// Include the patterns defined in the Declarative Rewrite framework.
// #include "ToyCombine.inc"
// } // namespace

// struct VVAddDuplicateOperand_to_VSMul : public mlir::OpRewritePattern<VVAddOp> {
//   /// We register this pattern to match every cim.vv_add in the IR.
//   /// The "benefit" is used by the framework to order the patterns and process
//   /// them in order of profitability.
//   VVAddDuplicateOperand_to_VSMul(mlir::MLIRContext *context)
//       : OpRewritePattern<VVAddOp>(context, /*benefit=*/1) {}

//   /// This method attempts to match a pattern and rewrite it.
//   mlir::LogicalResult
//   matchAndRewrite(VVAddOp op, mlir::PatternRewriter &rewriter) const override {
//     // Look through the input of the current transpose.
//     auto operands = op.getOperands();
//     LOG_DEBUG << "VVAddDuplicateOperand_to_VSMul";
//     // std::cout << "operands[0]: " << operands[0] << std::endl;
//     // std::cout << "operands[1]: " << operands[1] << std::endl;

//     if (operands[0] != operands[1]) {
//       return failure();
//     }

//     // Otherwise, we replace VVAddOp to VSMulOp. Use the rewriter.
//     mlir::Value factor =
//         rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 2, 32);
//     auto vs_mul = rewriter.create<VSMulOp>(op.getLoc(), operands[0], factor);
//     rewriter.replaceOp(op, {vs_mul.getResult()});
//     return success();
//   }
// };

// /// Register our patterns as "canonicalization" patterns on the TransposeOp so
// /// that they can be picked up by the Canonicalization framework.
// void VVAddOp::getCanonicalizationPatterns(RewritePatternSet &results,
//                                           MLIRContext *context) {
//   LOG_DEBUG << "VVAddOp::getCanonicalizationPatterns";
//   results.add<VVAddDuplicateOperand_to_VSMul>(context);
// }

struct ShapeToConstant : public mlir::OpRewritePattern<mlir::cim::ShapeOp> {
  /// We register this pattern to match every cim.vv_add in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  ShapeToConstant(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::cim::ShapeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it.
  mlir::LogicalResult
  matchAndRewrite(mlir::cim::ShapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    LOG_DEBUG << "ShapeToConstant";
    // Look through the input of the current transpose.
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
