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

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace cim;

/// Include the auto-generated definitions for the shape inference interfaces.

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

static int getBitWidth(mlir::Type type) {
  if (type.isa<mlir::IntegerType>()) {
    return type.getIntOrFloatBitWidth();
  } else if (type.isa<mlir::FloatType>()) {
    return type.getIntOrFloatBitWidth();
  } else if (type.isa<mlir::IndexType>()) {
    return 32;
  } else {
    LOG_ERROR << "getBitWidth fail";
    std::exit(1);
    return 0;
  }
}

struct MemoryAddressAllocationPass
    : public mlir::PassWrapper<MemoryAddressAllocationPass,
                               OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryAddressAllocationPass)

  void runOnOperation() override {
    LOG_DEBUG << "run on operation";
    auto f = getOperation();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    std::vector<mlir::memref::AllocOp> alloc_op_list;
    f.walk([&](mlir::Operation *op) {
      // std::cout << "Inferring shape for: " c<< *op << std::endl;
      if (mlir::memref::AllocOp alloc_op =
              dyn_cast<mlir::memref::AllocOp>(op)) {
        alloc_op_list.push_back(alloc_op);
      }
    });
    LOG_DEBUG << "alloc_op_list.size()=" << alloc_op_list.size();

    std::unordered_map<std::string, int> address_table;
    for (auto iter = alloc_op_list.begin(); iter != alloc_op_list.end();
         iter++) {
      mlir::memref::AllocOp op = *iter;
      auto context = op.getContext();
      mlir::MemRefType type = op.getResult().getType();

      mlir::DictionaryAttr memory_space =
          llvm::cast<mlir::DictionaryAttr>(type.getMemorySpace());
      llvm::StringRef _memory =
          llvm::cast<mlir::StringAttr>(memory_space.get("memory")).getValue();
      std::string memory = _memory.str();

      auto shape = type.getShape(); // TODO: how to get memref's size?

      int size = 1;
      for (auto s = shape.begin(); s != shape.end(); s++) {
        size *= (*s);
      }

      int bitwidth = getBitWidth(type.getElementType());
      if (bitwidth == 1) {
        size = size / 8;
      } else if (bitwidth >= 8 && bitwidth % 8 == 0) {
        size = size * bitwidth / 8;
      } else {
        LOG_ERROR << "Unsupported bit width: " << bitwidth;
        std::exit(1);
      }

      if (!address_table.count(memory)) {
        address_table[memory] = 0;
      }
      int address = address_table[memory];

      mlir::SmallVector<mlir::NamedAttribute, 2> nameAttrs;
      nameAttrs.push_back(
          mlir::NamedAttribute(mlir::StringAttr::get(context, "memory"),
                               mlir::StringAttr::get(context, memory)));
      nameAttrs.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(context, "address"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                 address)));

      mlir::DictionaryAttr new_memory_space =
          mlir::DictionaryAttr::get(op.getContext(), nameAttrs);

      // type.setMemorySpace(new_memory_space);
      mlir::MemRefType new_type =
          mlir::MemRefType::get(type.getShape(), type.getElementType(),
                                type.getLayout(), new_memory_space);
      op.getResult().setType(new_type);


      // get SubviewOp that use this alloc op
      // set result of SubviewOp to new_type
      for (mlir::OpOperand &use : op.getResult().getUses()) {
        if (auto subview = llvm::dyn_cast<mlir::memref::SubViewOp>(use.getOwner())) {
          // Set the result type of the SubviewOp to use the same memory space
          mlir::MemRefType subview_type = subview.getType();
          mlir::MemRefType new_subview_type = mlir::MemRefType::get(
              subview_type.getShape(),
              subview_type.getElementType(),
              subview_type.getLayout(),
              new_memory_space);
          subview.getResult().setType(new_subview_type);
        }
      }


      address_table[memory] += size;
    }
  }
};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::cim::createMemoryAddressAllocationPass() {
  return std::make_unique<MemoryAddressAllocationPass>();
}
