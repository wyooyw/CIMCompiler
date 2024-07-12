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

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "cim/Dialect.h"
#include "mlir/InitAllDialects.h"
#include <iostream>
using namespace mlir;
using namespace cim;
// using namespace arith;

// namespace {
// /// Include the patterns defined in the Declarative Rewrite framework.
// #include "ToyCombine.inc"
// } // namespace

struct VVAddDuplicateOperand_to_VSMul : public mlir::OpRewritePattern<VVAddOp> {
  /// We register this pattern to match every cim.vv_add in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  VVAddDuplicateOperand_to_VSMul(mlir::MLIRContext *context)
      : OpRewritePattern<VVAddOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it.
  mlir::LogicalResult
  matchAndRewrite(VVAddOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    auto operands = op.getOperands();
    std::cout << "VVAddDuplicateOperand_to_VSMul" << std::endl;
    // std::cout << "operands[0]: " << operands[0] << std::endl;
    // std::cout << "operands[1]: " << operands[1] << std::endl;

    if (operands[0]!=operands[1]) {
      return failure();
    }

    // Otherwise, we replace VVAddOp to VSMulOp. Use the rewriter.
    mlir::Value factor = rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 2, 32);
    auto vs_mul = rewriter.create<VSMulOp>(op.getLoc(), operands[0], factor);
    rewriter.replaceOp(op, {vs_mul.getResult()});
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void VVAddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  std::cout << "VVAddOp::getCanonicalizationPatterns" << std::endl;                      
  results.add<VVAddDuplicateOperand_to_VSMul>(context);
}



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
    std::cout << "ShapeToConstant" << std::endl;
    // Look through the input of the current transpose.
    auto operands = op.getOperands();

    mlir::Value source = operands[0];
    mlir::MemRefType source_type = llvm::cast<mlir::MemRefType>(source.getType());
    ArrayRef<int64_t> shape = source_type.getShape();

    mlir::Value index = operands[1];
    mlir::arith::ConstantIntOp const_index_op = index.getDefiningOp<mlir::arith::ConstantIntOp>();
    int64_t index_value = const_index_op.value();

    int64_t size = shape[index_value];
    if(size==mlir::ShapedType::kDynamic){
      return failure();
    }
    
    mlir::Value new_constant = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(size));
    rewriter.replaceOp(op, {new_constant});
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void ShapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  std::cout << "ShapeOp::getCanonicalizationPatterns" << std::endl;                      
  results.add<ShapeToConstant>(context);
}

/*
 Cast Op
*/


struct CastToNoOp : public mlir::OpRewritePattern<mlir::cim::CastOp> {
  /// We register this pattern to match every cim.vv_add in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  CastToNoOp(mlir::MLIRContext *context)
      : OpRewritePattern<mlir::cim::CastOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it.
  mlir::LogicalResult
  matchAndRewrite(mlir::cim::CastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    std::cout << "CastToNoOp" << std::endl;
    // Look through the input of the current transpose.
    auto operand = op.getOperand();

    mlir::Value source = operand;
    mlir::MemRefType source_type = llvm::cast<mlir::MemRefType>(source.getType());
    ArrayRef<int64_t> source_shape = source_type.getShape();
    // ArrayRef<int64_t> source_shape = source_type.getStride();

    mlir::Value target = op.getResult();
    mlir::MemRefType target_type = llvm::cast<mlir::MemRefType>(target.getType());
    ArrayRef<int64_t> target_shape = target_type.getShape();

    // int64_t size = shape[index_value];
    // if(size==mlir::ShapedType::kDynamic){
    //   return failure();
    // }
    
    // mlir::Value new_constant = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(size));
    rewriter.replaceOp(op, {source});
    return success();
  }
};

void CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  std::cout << "CastOp::getCanonicalizationPatterns" << std::endl;                      
  results.add<CastToNoOp>(context);
}