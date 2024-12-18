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

// why need this namespace ?
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
    // registry.insert<affine::AffineDialect, func::FuncDialect,
    //                 memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void CastEliminationPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  std::cout << "CastEliminationPass::runOnOperation" << std::endl;
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
  patterns.add<MemRefCastEliminate>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  // if (failed(
  //         applyPartialConversion(getOperation(), target,
  //         std::move(patterns))))
  //   signalPassFailure();

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
    LOG_DEBUG << "CIMLoweringPass::runOnOperation finish!";
}

std::unique_ptr<Pass> mlir::cim::createCastEliminationPass() {
  auto pass = std::make_unique<CastEliminationPass>();
  return pass;
}