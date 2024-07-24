
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "cim/Dialect.h"
#include "cim/Passes.h"
#include "cim/ShapeInferenceInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <iostream>

#define PASS_NAME "test-decompose-affine-ops"


using namespace mlir;
using namespace cim;
using namespace mlir::affine;

namespace {

struct TestDecomposeAffineOps
    : public PassWrapper<TestDecomposeAffineOps, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDecomposeAffineOps)

  StringRef getArgument() const final { return PASS_NAME; }
  StringRef getDescription() const final {
    return "Tests affine ops decomposition utility functions.";
  }
  TestDecomposeAffineOps() = default;
  TestDecomposeAffineOps(const TestDecomposeAffineOps &pass) = default;

  void runOnOperation() override;
};

} // namespace

void TestDecomposeAffineOps::runOnOperation() {
  IRRewriter rewriter(&getContext());
  this->getOperation().walk([&](AffineApplyOp op) {
    std::cout << "TestDecomposeAffineOps::runOnOperation.walk" << std::endl;
    rewriter.setInsertionPoint(op);
    mlir::affine::reorderOperandsByHoistability(rewriter, op);
    (void)mlir::affine::decompose(rewriter, op);
  });
}

std::unique_ptr<mlir::Pass> mlir::cim::createTestDecomposeAffineOpPass() {
  return std::make_unique<TestDecomposeAffineOps>();
}
