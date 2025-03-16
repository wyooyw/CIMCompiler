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
#include "cimisa/Dialect.h"
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
#include <unordered_map>

using namespace mlir;
using namespace cim;

namespace {

struct TransOffsetOptimizePass
    : public mlir::PassWrapper<TransOffsetOptimizePass,
                               OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransOffsetOptimizePass)

  void runOnOperation() override {
    auto fn = getOperation();
    for (auto &block : fn.getBlocks()) {
        processBlockLI(block);
        processBlockRI(block);
    }
  }

  

  void processBlockLI(mlir::Block &block) {
    std::vector<Operation *> transOps;
    std::vector<int64_t> src_addr_list; // List to store src addresses
    for (auto &op : block) {
        if (auto transOp = dyn_cast<mlir::cimisa::TransOp>(&op)) {
            if (
              transOp.getSrcOffsetFlag() == false
              && transOp.getDstOffsetFlag() == false
              && isa<mlir::arith::ConstantOp>(transOp.getSrcAddr().getDefiningOp()) 
              && isa<mlir::arith::ConstantOp>(transOp.getDstAddr().getDefiningOp())) {
                transOps.push_back(&op);
            }
        }
    }
    // Sort transOps based on their order of appearance in the block
    std::sort(transOps.begin(), transOps.end(), [&](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
    });

    // Process each TransOp and extract src address value
    for (auto *op : transOps) {
        if (auto transOp = dyn_cast<mlir::cimisa::TransOp>(op)) {
            auto srcConstOp = dyn_cast<mlir::arith::ConstantOp>(transOp.getSrcAddr().getDefiningOp());
            if (auto intAttr = srcConstOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
                src_addr_list.push_back(intAttr.getInt());
            }
        }
    }

    // Function to find shared base regions
    std::vector<std::tuple<int, int>> share_base_region_list = findSharedBaseRegions(src_addr_list);

    // Process each share_base_region
    for (auto [start, end] : share_base_region_list) {
        // Get the base value from the first element in the region
        auto baseOp = dyn_cast<mlir::cimisa::TransOp>(transOps[start]);
        auto baseValue = baseOp.getSrcAddr(); // Assuming getSrcAddr() returns the Value

        for (size_t k = start + 1; k <= end; ++k) {
            auto transOp = dyn_cast<mlir::cimisa::TransOp>(transOps[k]);
            int64_t offset = src_addr_list[k] - src_addr_list[start];

            // Create a new TransOp with updated src_addr and imm
            OpBuilder builder(transOp);
            builder.setInsertionPoint(transOp);
            auto newTransOp = builder.create<mlir::cimisa::TransOp>(
                transOp.getLoc(),
                baseValue, // New src_addr
                transOp.getDstAddr(), // Keep the same dst_addr
                transOp.getSize(), // Keep the same size
                builder.getI32IntegerAttr(offset), // New imm
                builder.getBoolAttr(true), // Keep the same src_offset_flag
                builder.getBoolAttr(false)  // Keep the same dst_offset_flag
            );

            // Replace the old operation with the new one
            transOp->replaceAllUsesWith(newTransOp);
            transOp.erase();
        }
    }
  }

  // Define a hash function for mlir::Value
  struct ValueHash {
    std::size_t operator()(const mlir::Value &val) const {
        return std::hash<void*>()(val.getAsOpaquePointer());
    }
  };

  // Define an equality function for mlir::Value
  struct ValueEqual {
    bool operator()(const mlir::Value &lhs, const mlir::Value &rhs) const {
        return lhs == rhs;
    }
  };

  void processBlockRI(mlir::Block &block) {
    std::unordered_map<Value, std::vector<Operation *>, ValueHash, ValueEqual> transOps; // Change to unordered_map
    std::vector<int64_t> src_addr_list; // List to store src addresses
    for (auto &op : block) {
        if (auto transOp = dyn_cast<mlir::cimisa::TransOp>(&op)) {
            if (
              transOp.getSrcOffsetFlag() == false
              && transOp.getDstOffsetFlag() == false
              && isa<mlir::cimisa::RIAddIOp>(transOp.getSrcAddr().getDefiningOp())) {
                // Get the AddIOp operand value
                auto addIOp = cast<mlir::cimisa::RIAddIOp>(transOp.getSrcAddr().getDefiningOp());
                Value operandValue = addIOp.getOperand(); // Assuming you want the first operand
                transOps[operandValue].push_back(&op); // Store in map
            }
        }
    }
    for(auto [value, op_list] : transOps) {
      if (op_list.size() < 2) {
        continue;
      }
      // std::cout << "op_list size: " << op_list.size() << std::endl;

      // sort op_list
      std::sort(op_list.begin(), op_list.end(), [&](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });

      std::vector<int64_t> src_addr_list;
      for (auto op : op_list) {
        auto transOp = dyn_cast<mlir::cimisa::TransOp>(op);
        auto addIOp = cast<mlir::cimisa::RIAddIOp>(transOp.getSrcAddr().getDefiningOp());
        src_addr_list.push_back(addIOp.getConstant().getSExtValue());
      }
      std::vector<std::tuple<int, int>> share_base_region_list = findSharedBaseRegions(src_addr_list);
      for (auto [start, end] : share_base_region_list) {
        // std::cout << "start: " << start << " end: " << end << std::endl;
        auto baseOp = dyn_cast<mlir::cimisa::TransOp>(op_list[start]);
        auto baseValue = baseOp.getSrcAddr(); // Assuming getSrcAddr() returns the Value
        for (size_t k = start + 1; k <= end; ++k) {
          auto transOp = dyn_cast<mlir::cimisa::TransOp>(op_list[k]);
          int64_t offset = src_addr_list[k] - src_addr_list[start];

          // Create a new TransOp with updated src_addr and imm
          OpBuilder builder(transOp);
          builder.setInsertionPoint(transOp);
            auto newTransOp = builder.create<mlir::cimisa::TransOp>(
                transOp.getLoc(),
                baseValue, // New src_addr
                transOp.getDstAddr(), // Keep the same dst_addr
                transOp.getSize(), // Keep the same size
                builder.getI32IntegerAttr(offset), // New imm
                builder.getBoolAttr(true), // Keep the same src_offset_flag
                builder.getBoolAttr(false)  // Keep the same dst_offset_flag
            );

          // Replace the old operation with the new one
          transOp->replaceAllUsesWith(newTransOp);
          auto addIOp = cast<mlir::cimisa::RIAddIOp>(transOp.getSrcAddr().getDefiningOp());
          transOp.erase();
          // std::cout << "addIOp user count: " << std::distance(addIOp.getResult().use_begin(), addIOp.getResult().use_end()) << std::endl;

          if (addIOp.getResult().use_empty()) {
            addIOp.erase();
          }
        }
      }

      // 检查:若此时value没有user了,则删除value
      // std::cout << "value user count: " << std::distance(value.use_begin(), value.use_end()) << std::endl;
      // if (value.use_empty()) {
      //   value.getDefiningOp()->erase();
      // }
    }    
  }

  // Function to find shared base regions
  std::vector<std::tuple<int, int>> findSharedBaseRegions(const std::vector<int64_t>& src_addr_list) {
    std::vector<std::tuple<int, int>> share_base_region_list;
    for (size_t i = 0; i < src_addr_list.size(); ++i) {
        int64_t base_addr = src_addr_list[i];
        size_t j = i + 1;
        while (j < src_addr_list.size() && src_addr_list[j] >= base_addr && src_addr_list[j] - base_addr <= 1024) {
            ++j;
        }
        if (j - i > 1) { // Ensure at least two trans ops are included
            share_base_region_list.emplace_back(i, j - 1);
            i = j - 1; // Update i to the end of the current region to avoid overlap
        }
    }
    return share_base_region_list;
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::cim::createTransOffsetOptimizePass() {
  return std::make_unique<TransOffsetOptimizePass>();
}
