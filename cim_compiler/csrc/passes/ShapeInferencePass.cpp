#include "cim/Dialect.h"
#include "cim/Passes.h"
#include "cim/ShapeInferenceInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <memory>
#include "common/macros.h"

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace cim;

#include "cim/ShapeInferenceOpInterfaces.cpp.inc"

namespace {
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass,
                               OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  void runOnOperation() override {
    LOG_DEBUG << "ShapeInferencePass::runOnOperation";
    auto f = getOperation();

    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
    });

  }

};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::cim::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
