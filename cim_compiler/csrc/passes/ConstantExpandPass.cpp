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

using namespace mlir;
using namespace cim;

namespace {

struct ConstantExpandPass
    : public mlir::PassWrapper<ConstantExpandPass,
                               OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantExpandPass)

  void runOnOperation() override {
    auto fn = getOperation();
    // Collect all ConstantOp instances first
    SmallVector<mlir::arith::ConstantOp, 8> constantOps;
    fn.walk([&](mlir::Operation *op) {
        if (auto constantOp = llvm::dyn_cast<mlir::arith::ConstantOp>(op)) {
            constantOps.push_back(constantOp);
        }
    });

    // Iterate over the collected ConstantOps
    for (auto constantOp : constantOps) {
        // Store users in a separate container
        SmallVector<mlir::Operation*, 8> users(constantOp->getUsers().begin(), constantOp->getUsers().end());
        
        // block to users
        DenseMap<mlir::Block*, SmallVector<mlir::Operation*, 8>> blockToUsersMap;
        for (auto *user : users) {
            auto *block = user->getBlock();
            blockToUsersMap[block].push_back(user);
        }


        // Map to track if a block has a shared newConstantOp
        DenseMap<mlir::Block*, mlir::arith::ConstantOp> blockToConstantOpMap;

        for (auto *user : users) {
            auto *block = user->getBlock();

            mlir::arith::ConstantOp newConstantOp;

            if (blockToUsersMap[block].size() > 2) {
              if (blockToConstantOpMap.count(block) == 0) {
                  std::cout << "miss: " << user->getName().getStringRef().str() << std::endl;
                  OpBuilder builder(block, block->begin());
                  newConstantOp = llvm::cast<mlir::arith::ConstantOp>(builder.clone(*constantOp));
                  blockToConstantOpMap[block] = newConstantOp;
              } else {
                  std::cout << "hit: " << user->getName().getStringRef().str() << std::endl;
                  newConstantOp = blockToConstantOpMap[block];
              }
            } else {
              std::cout << "single: " << user->getName().getStringRef().str() << std::endl;
              OpBuilder builder(user);
              newConstantOp = llvm::cast<mlir::arith::ConstantOp>(builder.clone(*constantOp));
            }


            user->replaceUsesOfWith(constantOp->getResult(0), newConstantOp->getResult(0));
        }
        // Erase the original constantOp
        constantOp->erase();
    }
  }
};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::cim::createConstantExpandPass() {
  return std::make_unique<ConstantExpandPass>();
}
