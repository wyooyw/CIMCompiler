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
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>
#include <memory>
#include "common/macros.h"

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace cim;


namespace {

static const boost::property_tree::ptree &
get_item(const boost::property_tree::ptree &ast, int index) {
  auto it = ast.begin();
  std::advance(it, index);
  return it->second;
}

template <typename Ty>
Ty safe_get_as(const boost::property_tree::ptree &ast, const std::string &key) {
  if (ast.count(key)) {
    return ast.get<Ty>(key);
  } else {
    // tell user
    std::cerr << "[safe_get_] Key error: " << key << std::endl;
    std::exit(1);
    // return nullptr;
  }
}
const boost::property_tree::ptree &
safe_get_child(const boost::property_tree::ptree &ast, const std::string &key) {
  if (ast.count(key)) {
    return ast.get_child(key);
  } else {
    // tell user
    std::cerr << "[safe_get_child] Key error: " << key << std::endl;
    std::exit(1);
    return ast;
  }
}
static std::map<std::string, int> memory_addr_list;
static std::map<std::string, int> memory_size_list;
static void getMemoryAddrList(std::string config_path) {
  boost::property_tree::ptree ast;
  boost::property_tree::read_json(config_path, ast);

  // std::map<string, int> memory_addr_list;
  LOG_DEBUG << "getMemoryAddrList";
  auto json_memory_list = safe_get_child(ast, "memory_list");
  for (const auto &pair : json_memory_list) {
    auto json_memory = pair.second;
    std::string name = safe_get_as<std::string>(json_memory, "name");
    auto json_address = safe_get_child(json_memory, "addressing");
    int offset = safe_get_as<int>(json_address, "offset_byte");
    int size = safe_get_as<int>(json_address, "size_byte");

    memory_addr_list[name] = offset;
    memory_size_list[name] = size;
    LOG_DEBUG << "name: " << name << " offset: " << offset << " size: " << size;
  }

  // return memory_addr_list;
}

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

// Update the BufferLifetime structure to include first/last operations
struct BufferLifetime {
  mlir::Operation *allocOp;
  mlir::Operation *startOp; // first (earliest) operation where buffer is live (usually alloc)
  mlir::Operation *endOp;   // last (latest) operation that uses the buffer
};

// Helper that tries to decide if opA happens strictly before opB.
static bool opComesBefore(mlir::Operation *opA, mlir::Operation *opB, mlir::DominanceInfo &dom) {
  if (opA == opB)
    return false; // same op – treat as not strictly before
  // Same block – use IR order.
  if (opA->getBlock() == opB->getBlock())
    return opA->isBeforeInBlock(opB);
  // Different blocks – fall back to dominance.
  if (dom.dominates(opA, opB) && !dom.dominates(opB, opA))
    return true;
  return false; // conservatively return false otherwise.
}

// Determine if two lifetimes overlap.
static bool lifetimesOverlap(const BufferLifetime &a, const BufferLifetime &b, mlir::DominanceInfo &dom) {
  // If a ends strictly before b starts, or b ends strictly before a starts, there is NO overlap.
  if (opComesBefore(a.endOp, b.startOp, dom) || opComesBefore(b.endOp, a.startOp, dom))
    return false;
  // Otherwise, conservatively assume they overlap.
  return true;
}

struct MemoryAddressAllocationPass
    : public mlir::PassWrapper<MemoryAddressAllocationPass,
                               OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryAddressAllocationPass)
  std::string config_path;
  std::map<mlir::Operation *, std::string> buffer_type;
  
  void runOnOperation() override {
    LOG_DEBUG << "run on operation";
    getMemoryAddrList(config_path);

    auto f = getOperation();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    std::vector<mlir::memref::AllocOp> alloc_op_list;
    f.walk([&](mlir::Operation *op) {
      if (mlir::memref::AllocOp alloc_op =
              dyn_cast<mlir::memref::AllocOp>(op)) {
        alloc_op_list.push_back(alloc_op);
      }
    });
    LOG_DEBUG << "alloc_op_list.size()=" << alloc_op_list.size();

    // Initialize dominance info (needed both for ordering across blocks and dominance queries)
    mlir::DominanceInfo dominanceInfo(f);

    // Track the lifetime of each buffer
    std::vector<BufferLifetime> bufferLifetimes;
    for (auto &allocOp : alloc_op_list) {
      BufferLifetime lifetime;
      lifetime.allocOp = allocOp;
      lifetime.startOp = allocOp;
      lifetime.endOp   = allocOp;

      for (mlir::OpOperand &use : allocOp.getResult().getUses()) {
        mlir::Operation *user = use.getOwner();

        // Update earliest op.
        if (opComesBefore(user, lifetime.startOp, dominanceInfo))
          lifetime.startOp = user;

        // Update latest op.
        if (opComesBefore(lifetime.endOp, user, dominanceInfo))
          lifetime.endOp = user;
      }
      bufferLifetimes.push_back(lifetime);
    }

    // Find conflicting buffer pairs
    std::vector<std::pair<mlir::Operation *, mlir::Operation *>> conflictingPairs;
    for (size_t i = 0; i < bufferLifetimes.size(); ++i) {
      for (size_t j = i + 1; j < bufferLifetimes.size(); ++j) {
        if (lifetimesOverlap(bufferLifetimes[i], bufferLifetimes[j], dominanceInfo)) {
          conflictingPairs.emplace_back(bufferLifetimes[i].allocOp, bufferLifetimes[j].allocOp);
        }
      }
    }

    // Log the conflicting pairs
    for (const auto &pair : conflictingPairs) {
      LOG_DEBUG << "Conflicting pair: " << pair.first << " and " << pair.second;
    }

    std::unordered_map<std::string, int> address_table;
    for (auto iter = alloc_op_list.begin(); iter != alloc_op_list.end();
         iter++) {
      mlir::memref::AllocOp op = *iter;
      auto context = op.getContext();
      mlir::MemRefType type = op.getResult().getType();

      // mlir::DictionaryAttr memory_space =
      //     llvm::cast<mlir::DictionaryAttr>(type.getMemorySpace());
      // llvm::StringRef _memory =
      //     llvm::cast<mlir::StringAttr>(memory_space.get("memory")).getValue();
      // std::string memory = _memory.str();
      std::string memory = buffer_type[op];
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
      if (address_table[memory] > memory_size_list[memory]) {
        LOG_ERROR << "Memory address overflow: " << memory << " total size: " << memory_size_list[memory] << " use size: " << address_table[memory] << " buffer size: " << size;
        std::exit(1);
      }
    }
  }
};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::cim::createMemoryAddressAllocationPass(std::string config_path, std::map<mlir::Operation *, std::string> buffer_type) {
  auto pass = std::make_unique<MemoryAddressAllocationPass>();
  pass->config_path = config_path;
  pass->buffer_type = buffer_type;
  return pass;
}
