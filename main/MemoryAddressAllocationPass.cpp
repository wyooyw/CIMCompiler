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

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
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

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace cim;

/// Include the auto-generated definitions for the shape inference interfaces.
#include "cim/MemoryAddressAllocationPass.cpp.inc"

namespace {
/// The ShapeInferencePass is a pass that performs intra-procedural
/// shape inference.
///
///    Algorithm:
///
///   1) Build a worklist containing all the operations that return a
///      dynamically shaped tensor: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the argument types.
///   3) If the worklist is empty, the algorithm succeeded.
///

struct MemoryAddressAllocationPass
    : public mlir::PassWrapper<MemoryAddressAllocationPass, OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryAddressAllocationPass)

  void runOnOperation() override {
    std::cout << "run on operation" << std::endl;
    auto f = getOperation();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::memref::AllocOp *, 1024> alloc_op_list;
    f.walk([&](mlir::Operation *op) {
      // std::cout << "Inferring shape for: " c<< *op << std::endl;
      if (mlir::memref::AllocOp* alloc_op = dyn_cast<mlir::memref::AllocOp>(op)) {
        alloc_op_list.insert(alloc_op);
      }
    });
    std::cout << "alloc_op_list.size()=" << alloc_op_list.size() << std::endl;

    std::unordered_map<std::string, int > address_table;
    for(auto iter = alloc_op_list.begin(); iter!=alloc_op_list.end();iter++){
      mlir::MemRefType type = iter->getType();

      mlir::DictionaryAttr memory_space = type->getMemorySpace();
      std::string memory = memory_space.get("memory").getValue(); 

      auto shape = type.getShape(); // TODO: how to get memref's size?
      int size = 1;
      for(auto s in shape){
        size *= s
      }
      
      if (!address_table.count(memory)){
        address_table[memory] = 0;
      }
      int address = address_table[memory];
      memory_space.set("address", address); // TODO: how to set value in DictionaryAttr?

      address_table[memory] += size;
    }
  }

};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::cim::createMemoryAddressAllocationPass() {
  return std::make_unique<MemoryAddressAllocationPass>();
}
