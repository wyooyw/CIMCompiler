
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

#define DEBUG_TYPE "shape-inference"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_FOLDMEMREFALIASOPS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;
using namespace cim;

namespace {

struct FoldMemRefAliasOpsPass final
    : public mlir::memref::impl::FoldMemRefAliasOpsBase<
          FoldMemRefAliasOpsPass> {
  void runOnOperation() override;
};

} // namespace

// void cim::populateFoldMemRefAliasOpPatterns(RewritePatternSet &patterns) {
//   patterns.add<CIMComuteOpSubViewOpFolder>(
//       patterns.getContext());
// }

void FoldMemRefAliasOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<mlir::Pass> cim::createFoldMemRefAliasOpsPass() {
  return std::make_unique<FoldMemRefAliasOpsPass>();
}