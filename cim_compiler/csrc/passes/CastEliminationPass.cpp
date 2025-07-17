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

namespace {

struct MemRefCastEliminate : public OpRewritePattern<memref::CastOp> {
  using OpRewritePattern<memref::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp op,
                                PatternRewriter &rewriter) const final {
    /*
      Now, all index of load is 0.
    */
    std::cout << "MemRefCastEliminate::matchAndRewrite" << std::endl;
    rewriter.replaceOp(op, {op.getOperand()});
    std::cout << "MemRefCastEliminate::matchAndRewrite finish" << std::endl;
    return success();
  }
};

} // namespace

namespace {
struct CastEliminationPass
    : public PassWrapper<CastEliminationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CastEliminationPass)
  std::string config_path;
  void getDependentDialects(DialectRegistry &registry) const override {
  }
  void runOnOperation() final;
};
} // namespace

void CastEliminationPass::runOnOperation() {
  LOG_DEBUG << "CastEliminationPass::runOnOperation";
  ConversionTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  patterns.add<MemRefCastEliminate>(&getContext());

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
    LOG_DEBUG << "CIMLoweringPass::runOnOperation finish!";
}

std::unique_ptr<Pass> mlir::cim::createCastEliminationPass() {
  auto pass = std::make_unique<CastEliminationPass>();
  return pass;
}