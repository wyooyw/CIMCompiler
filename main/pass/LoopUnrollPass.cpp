//===- ShapeInferencePass.cpp - Shape Inference ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a Function level pass performing interprocedural
// propagation of array shapes through function specialization.
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect.h"
#include "cim/Passes.h"
#include "cim/ShapeInferenceInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
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

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace cim;

/// Include the auto-generated definitions for the shape inference interfaces.

namespace {

struct LoopUnrollPass
    : public mlir::PassWrapper<LoopUnrollPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopUnrollPass)

  std::vector<mlir::scf::ForOp> unrollForOps;

  void runOnOperation() override {
    std::cout << "LoopUnrollPass begin." << std::endl;
    int fail_cnt = unrollForOps.size();
    for (auto it = unrollForOps.rbegin(); it != unrollForOps.rend(); ++it) {
      auto forOp = *it;
      // for (auto &forOp : unrollForOps) {
      std::optional<int64_t> lbCstOp =
          getConstantIntValue(forOp.getLowerBound());
      std::optional<int64_t> ubCstOp =
          getConstantIntValue(forOp.getUpperBound());
      std::optional<int64_t> stepCstOp = getConstantIntValue(forOp.getStep());
      if (lbCstOp && ubCstOp && stepCstOp) {
        // Constant loop bounds computation.
        int64_t lbCst = lbCstOp.value();
        int64_t ubCst = ubCstOp.value();
        int64_t stepCst = stepCstOp.value();
        if (stepCst == 1) {
          int64_t unrollFactor = (ubCst - lbCst) / stepCst;
          if (!failed(mlir::loopUnrollByFactor(forOp, unrollFactor))) {
            fail_cnt -= 1;
          }
        }
      }
    }
    std::cout << "LoopUnrollPass end. fail: " << fail_cnt
              << " total: " << unrollForOps.size() << std::endl;
  }
};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass>
mlir::cim::createLoopUnrollPass(std::vector<mlir::scf::ForOp> &unrollForOps) {
  std::cout << "createLoopUnrollPass" << std::endl;
  auto pass = std::make_unique<LoopUnrollPass>();
  pass->unrollForOps = unrollForOps;
  return pass;
}
