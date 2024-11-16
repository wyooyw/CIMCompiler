#include "cim/Dialect.h"
#include "cim/Passes.h"
#include "cimisa/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>
using namespace mlir;

namespace {

struct CondBranchOpConvert : public OpRewritePattern<cf::CondBranchOp> {
  using OpRewritePattern<cf::CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cf::CondBranchOp op,
                                PatternRewriter &rewriter) const final {
    /*
    cf.cond_br %0, ^bb1, ^bb2
    ->
    cf.cond_br %0, ^bb1, ^bb3
    ^bb3:
    cf.br %^bb2
    */

    cf::CondBranchOp ter_op =
        llvm::cast<cf::CondBranchOp>(op->getBlock()->getTerminator());
    if ((!ter_op) or (ter_op != op)) {
      std::cerr << "this condbranch op is not the terminator of block."
                << std::endl;
      std::exit(1);
    }

    Block *false_block = op.getFalseDest();
    // if the false_block is a block with only one jump, then return
    int i = 0;
    for (Operation &op_obj : false_block->getOperations()) {
      Operation *op = &op_obj;
      if (auto _op = dyn_cast<mlir::cf::BranchOp>(op)) {
        return failure();
      }
      break;
    }
    // int num_predecessors = 0;
    // int num_successors = 0;
    // for (auto *b : false_block->getPredecessors()) num_predecessors++;
    // for (auto *b : false_block->getSuccessors()) num_successors++;
    // std::cout << "num_predecessors=" << num_predecessors << "num_successors="
    // << num_successors << std::endl; if (num_predecessors==1 && num_successors
    // > 0){
    //   return failure();
    // }
    SmallVector<Type, 8> argTypes;
    SmallVector<Location, 8> argLocs;
    auto block_arguments = false_block->getArguments();
    for (int arg_i = 0; arg_i < block_arguments.size(); arg_i++) {
      mlir::Value block_arg_val =
          llvm::cast<mlir::Value>(block_arguments[arg_i]);
      argTypes.push_back(block_arg_val.getType());
      argLocs.push_back(block_arg_val.getLoc());
    }

    Block *jump_block =
        rewriter.createBlock(false_block->getParent(), {}, argTypes, argLocs);
    rewriter.setInsertionPointToStart(jump_block);
    rewriter.create<cf::BranchOp>(rewriter.getUnknownLoc(), false_block,
                                  jump_block->getArguments());

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, op.getCondition(), op.getTrueDest(), op.getTrueOperands(),
        jump_block, op.getFalseOperands());
    std::cout << "CondBranchOpConvert::matchAndRewrite finish!" << std::endl;
    return success();
  }
};

} // namespace

namespace {
struct CIMBranchConvertPass
    : public PassWrapper<CIMBranchConvertPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CIMBranchConvertPass)

  void getDependentDialects(DialectRegistry &registry) const override {}
  void runOnOperation() final;
};
} // namespace

void CIMBranchConvertPass::runOnOperation() {
  std::cout << "CIMBranchConvertPass::runOnOperation" << std::endl;
  RewritePatternSet patterns(&getContext());
  patterns.add<CondBranchOpConvert>(&getContext());

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
  std::cout << "CIMBranchConvertPass::runOnOperation finish!" << std::endl;
}

std::unique_ptr<Pass> mlir::cim::createCIMBranchConvertPass() {
  return std::make_unique<CIMBranchConvertPass>();
}