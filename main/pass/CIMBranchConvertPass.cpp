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
#include "cim/Dialect.h"
#include "cimisa/Dialect.h"
#include "cim/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <iostream>
using namespace mlir;


// why need this namespace ?
namespace {


    struct CondBranchOpConvert : public OpRewritePattern<cf::CondBranchOp> {
      using OpRewritePattern<cf::CondBranchOp>::OpRewritePattern;

      LogicalResult
      matchAndRewrite(cf::CondBranchOp op, PatternRewriter &rewriter) const final {
        /*
        cf.cond_br %0, ^bb1, ^bb2
        ->
        cf.cond_br %0, ^bb1, ^bb3
        ^bb3:
        cf.br %^bb2
        */
      //  return failure();
      
        Block* block = op->getBlock();
        cf::CondBranchOp ter_op = llvm::cast<cf::CondBranchOp>(block->getTerminator());
        if(!ter_op){
          std::cout << "CondBranchOpConvert::matchAndRewrite ter_op==nullptr" << std::endl;
          return failure();
        }else if (ter_op != op){
          std::cout << "CondBranchOpConvert::matchAndRewrite ter_op != op" << std::endl;
          return failure();
        }else{
          std::cout << "CondBranchOpConvert::matchAndRewrite ter_op == op" << std::endl;
        }


        // Block* dest_block = _op.getTrueDest();
        // Block* dest_block = _op.getFalseDest();
        
          arith::CmpIOp cmpi_op = op.getOperand(0).getDefiningOp<arith::CmpIOp>();
          if(!cmpi_op){
            std::cerr << "cmpi_op is null!" << std::endl;
          }
          // string predicate = cmpi_op.getPredicateAttrName().str();
          mlir::Value lhs = cmpi_op.getLhs();
          mlir::Value rhs = cmpi_op.getRhs();
        
       std::cout << "CondBranchOpConvert::matchAndRewrite begin!" << std::endl;
        Block* true_block = op.getTrueDest();
        Block* false_block = op.getFalseDest();

        SmallVector<Type, 8> argTypes;
        SmallVector<Location, 8> argLocs;
        auto block_arguments = false_block->getArguments();
        for (int arg_i = 0; arg_i < block_arguments.size(); arg_i++){
          BlockArgument block_arg = block_arguments[arg_i];
          mlir::Value block_arg_val = llvm::cast<mlir::Value>(block_arg);
          argTypes.push_back(block_arg_val.getType());
          argLocs.push_back(block_arg_val.getLoc());
        }
        
        // Block* jump_block = rewriter.createBlock(false_block->getParent(), {}, argTypes, argLocs);
        // rewriter.setInsertionPointToStart(jump_block);
        // rewriter.create<cf::BranchOp>(rewriter.getUnknownLoc(), false_block, false_block->getArguments());
        // mlir::Value value = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1, rewriter.getI32Type());
        // rewriter.create<cim::PrintOp>(rewriter.getUnknownLoc(), rhs);

        // rewriter.create<cimisa::BranchOp>(rewriter.getUnknownLoc(), compare, lhs, rhs, true_block);
        // rewriter.replaceOpWithNewOp<cf::BranchOp>(op, false_block);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<cf::BranchOp>(op, 
                // op.getCondition(), 
                 op.getFalseDest(),
                op.getFalseOperands()
                );
        // op.getBlock()->dump();
        std::cout << "CondBranchOpConvert::matchAndRewrite finish!" << std::endl;
        return success();
      }
    };

}



namespace {
struct CIMBranchConvertPass
    : public PassWrapper<CIMBranchConvertPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CIMBranchConvertPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // registry.insert<affine::AffineDialect, func::FuncDialect,
    //                 memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void CIMBranchConvertPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  std::cout << "CIMBranchConvertPass::runOnOperation" << std::endl;
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  patterns.add<CondBranchOpConvert>(
      &getContext());
  target.addLegalOp<cf::BranchOp>();
  // target.addIllegalOp<cf::CondBranchOp>();
  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns)))){
            signalPassFailure();
            std::cout << "CIMBranchConvertPass::runOnOperation failed!" << std::endl;
          }
    

  // cf::CondBranchOp op = getOperation();

  std::cout << "CIMBranchConvertPass::runOnOperation finish!" << std::endl;
}

std::unique_ptr<Pass> mlir::cim::createCIMBranchConvertPass() {
  return std::make_unique<CIMBranchConvertPass>();
}