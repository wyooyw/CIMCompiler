#include "cim/Dialect.h"
#include "cim/Passes.h"
#include "cim/ShapeInferenceInterface.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>

#include "common/macros.h"

using namespace mlir;

namespace {
struct ExtractAddressComputationPass
    : public PassWrapper<ExtractAddressComputationPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExtractAddressComputationPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    // registry.insert<affine::AffineDialect, func::FuncDialect,
    //                 memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ExtractAddressComputationPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  LOG_DEBUG << "ExtractAddressComputationPass::runOnOperation";
  RewritePatternSet patterns(&getContext());
  memref::populateExtractAddressComputationsPatterns(patterns);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
  LOG_DEBUG << "ExtractAddressComputationPass::runOnOperation finish!";
}

std::unique_ptr<Pass> mlir::cim::createExtractAddressComputationPass() {
  return std::make_unique<ExtractAddressComputationPass>();
}