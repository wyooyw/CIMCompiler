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
#include <unordered_map>
#include "common/macros.h"
#include <glpk.h>

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
  std::string bufType;       // group key (e.g., memory name) for this buffer
  mlir::Operation *startOp;  // earliest op where buffer is live (usually alloc)
  mlir::Operation *endOp;    // latest op that uses the buffer
};

// Helper that tries to decide if opA happens strictly before opB.
static bool opComesBefore(mlir::Operation *opA, mlir::Operation *opB, mlir::DominanceInfo &dom) {
  if (opA == opB)
    return false; // same op – not strictly before

  // Fast-path: same block → use insertion order.
  if (opA->getBlock() == opB->getBlock())
    return opA->isBeforeInBlock(opB);

  // If one op is an (proper) ancestor of the other, the ancestor lexically comes first.
  if (opA->isProperAncestor(opB))
    return true;
  if (opB->isProperAncestor(opA))
    return false;

  // Build ancestor chains up to the root (inclusive).
  llvm::SmallVector<mlir::Operation *, 8> chainA, chainB;
  for (auto *cur = opA; cur; cur = cur->getParentOp())
    chainA.push_back(cur);
  for (auto *cur = opB; cur; cur = cur->getParentOp())
    chainB.push_back(cur);

  // Reverse to have root → leaf order.
  std::reverse(chainA.begin(), chainA.end());
  std::reverse(chainB.begin(), chainB.end());

  // Find first differing node.
  size_t minLen = std::min(chainA.size(), chainB.size());
  size_t idx = 0;
  while (idx < minLen && chainA[idx] == chainB[idx])
    ++idx;

  // If one chain is a prefix of the other, earlier prefix op is ancestor – already handled.
  if (idx == chainA.size() || idx == chainB.size())
    return false; // should not happen due to ancestor test above

  mlir::Operation *childA = chainA[idx];
  mlir::Operation *childB = chainB[idx];

  // Both children reside in the same block (their parent's region). Use order in that block.
  if (childA->getBlock() == childB->getBlock())
    return childA->isBeforeInBlock(childB);

  // Fallback to dominance when blocks differ and order still unknown.
  if (dom.dominates(opA, opB) && !dom.dominates(opB, opA))
    return true;

  return false; // Unable to decide → treat as not strictly before.
}

// Determine if two lifetimes overlap.
static bool lifetimesOverlap(const BufferLifetime &a, const BufferLifetime &b, mlir::DominanceInfo &dom) {
  // If a ends strictly before b starts, or b ends strictly before a starts, there is NO overlap.
  if (opComesBefore(a.endOp, b.startOp, dom) || opComesBefore(b.endOp, a.startOp, dom))
    return false;
  // Otherwise, conservatively assume they overlap.
  return true;
}

struct BufferEntry {
  mlir::memref::AllocOp alloc;
  int64_t size; // in bytes
};

// Solve ILP for a set of buffers with given conflict pairs and capacity using GLPK.
static std::unordered_map<mlir::Operation *, int64_t>
solveILP(const std::vector<BufferEntry> &buffers,
         const std::vector<std::pair<int, int>> &conflictIdxPairs,
         int64_t capacity) {

  glp_prob *prob = glp_create_prob();
  glp_set_prob_name(prob, "mem_addr_alloc");
  glp_set_obj_dir(prob, GLP_MIN);

  int nBuf = buffers.size();
  int nConflict = conflictIdxPairs.size();
  int nBin = nConflict; // one binary per conflict

  int nCols = nBuf + nBin + 1; // addresses, binaries, maxAddr
  int nRows = 2 * nBuf           // capacity & maxAddr constraints for each buffer
            + 2 * nConflict;     // conflict constraints

  glp_add_cols(prob, nCols);

  // column index mapping
  std::vector<int> addrCol(nBuf);
  int colIdx = 1;
  for (int i = 0; i < nBuf; ++i) {
    addrCol[i] = colIdx;
    glp_set_col_name(prob, colIdx, ("addr_" + std::to_string(i)).c_str());
    glp_set_col_kind(prob, colIdx, GLP_IV);
    glp_set_col_bnds(prob, colIdx, GLP_DB, 0.0, capacity - buffers[i].size);
    ++colIdx;
  }

  // binary vars a_ij
  std::vector<int> binCol(nBin);
  for (int k = 0; k < nBin; ++k) {
    binCol[k] = colIdx;
    glp_set_col_name(prob, colIdx, ("a_" + std::to_string(k)).c_str());
    glp_set_col_kind(prob, colIdx, GLP_BV);
    glp_set_col_bnds(prob, colIdx, GLP_DB, 0.0, 1.0);
    ++colIdx;
  }

  int maxAddrCol = colIdx;
  glp_set_col_name(prob, maxAddrCol, "max_addr");
  glp_set_col_kind(prob, maxAddrCol, GLP_IV);
  glp_set_col_bnds(prob, maxAddrCol, GLP_DB, 0.0, capacity);
  glp_set_obj_coef(prob, maxAddrCol, 1.0); // objective

  // Add rows (constraints)
  glp_add_rows(prob, nRows);

  // Use CSR arrays for coefficients
  int estNnz = 5 * nBuf + 6 * nConflict; // rough estimation
  std::vector<int> ia(1 + estNnz);
  std::vector<int> ja(1 + estNnz);
  std::vector<double> ar(1 + estNnz);
  int idx = 1; // 1-based
  int row = 1;

  // capacity and maxAddr constraints
  for (int i = 0; i < nBuf; ++i) {
    // addr_i + size_i <= capacity  (row)
    glp_set_row_bnds(prob, row, GLP_UP, 0.0, capacity - buffers[i].size);
    ia[idx] = row; ja[idx] = addrCol[i]; ar[idx++] = 1.0;
    row++;

    // addr_i + size_i - maxAddr <= 0
    glp_set_row_bnds(prob, row, GLP_UP, 0.0, 0.0);
    ia[idx] = row; ja[idx] = addrCol[i]; ar[idx++] = 1.0;
    ia[idx] = row; ja[idx] = maxAddrCol; ar[idx++] = -1.0;
    row++;
  }

  double M = static_cast<double>(capacity);
  for (int k = 0; k < nConflict; ++k) {
    int i = conflictIdxPairs[k].first;
    int j = conflictIdxPairs[k].second;
    int aCol = binCol[k];

    // addr_i - addr_j + M*a <= M - size_i
    glp_set_row_bnds(prob, row, GLP_UP, 0.0, M - buffers[i].size);
    ia[idx] = row; ja[idx] = addrCol[i]; ar[idx++] = 1.0;
    ia[idx] = row; ja[idx] = addrCol[j]; ar[idx++] = -1.0;
    ia[idx] = row; ja[idx] = aCol; ar[idx++] = M;
    row++;

    // addr_j - addr_i - M*a <= -size_j
    glp_set_row_bnds(prob, row, GLP_UP, 0.0, -buffers[j].size);
    ia[idx] = row; ja[idx] = addrCol[j]; ar[idx++] = 1.0;
    ia[idx] = row; ja[idx] = addrCol[i]; ar[idx++] = -1.0;
    ia[idx] = row; ja[idx] = aCol; ar[idx++] = -M;
    row++;
  }

  glp_load_matrix(prob, idx - 1, ia.data(), ja.data(), ar.data());

  // Solve
  glp_iocp parm;
  glp_init_iocp(&parm);
  parm.presolve = GLP_ON;
  int status = glp_intopt(prob, &parm);

  std::unordered_map<mlir::Operation *, int64_t> result;
  if (status == 0) {
    for (int i = 0; i < nBuf; ++i) {
      double val = glp_mip_col_val(prob, addrCol[i]);
      result[buffers[i].alloc.getOperation()] = static_cast<int64_t>(val);
    }
  } else {
    LOG_ERROR << "GLPK failed to solve ILP (status=" << status << "). Falling back to sequential allocation.";
    int64_t next = 0;
    for (auto &b : buffers) {
      result[b.alloc.getOperation()] = next;
      next += b.size;
    }
  }

  glp_delete_prob(prob);
  return result;
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

    // Track the lifetime of each buffer and group by buffer_type
    std::unordered_map<std::string, std::vector<BufferLifetime>> lifetimesByType;
    for (auto &allocOp : alloc_op_list) {
      BufferLifetime lifetime;
      lifetime.allocOp = allocOp;
      lifetime.bufType = buffer_type[allocOp];
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
      lifetimesByType[lifetime.bufType].push_back(lifetime);
    }

    // Map from allocOp* to computed address across all groups.
    std::unordered_map<mlir::Operation *, int64_t> assignedAddr;

    for (auto &kv : lifetimesByType) {
      const std::string &typeName = kv.first;
      auto &vec = kv.second;

      // Generate conflict index pairs and buffer entries.
      std::vector<std::pair<int, int>> conflictIdxPairs;
      std::vector<BufferEntry> bufferEntries;
      bufferEntries.reserve(vec.size());

      for (size_t i = 0; i < vec.size(); ++i) {
        // compute size in bytes for this alloc
        auto type = vec[i].allocOp.getResult().getType().cast<mlir::MemRefType>();
        int64_t size = 1;
        for (auto dim : type.getShape())
          size *= dim;
        int bwidth = getBitWidth(type.getElementType());
        if (bwidth == 1)
          size = size / 8;
        else if (bwidth >= 8 && bwidth % 8 == 0)
          size = size * bwidth / 8;
        else {
          LOG_ERROR << "Unsupported bit width in ILP allocation: " << bwidth;
          std::exit(1);
        }

        bufferEntries.push_back(BufferEntry{vec[i].allocOp, size});
      }

      // Build conflicts by index
      for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = i + 1; j < vec.size(); ++j) {
          if (lifetimesOverlap(vec[i], vec[j], dominanceInfo)) {
            conflictIdxPairs.emplace_back(i, j);
          }
        }
      }

      // Debug log of conflicts
      for (auto &p : conflictIdxPairs) {
        LOG_DEBUG << "[bufType=" << typeName << "] Conflict idx: " << p.first << "," << p.second;
      }

      int64_t capacity = memory_size_list[typeName];

      auto addrMap = solveILP(bufferEntries, conflictIdxPairs, capacity);

      // Store assigned addresses
      assignedAddr.insert(addrMap.begin(), addrMap.end());
    }

    // --------------------------------------------
    // Address assignment update (uses assignedAddr)

    std::unordered_map<std::string, int> address_table; // not used anymore for assignment but kept for overflow check
    for (auto iter = alloc_op_list.begin(); iter != alloc_op_list.end(); ++iter) {
      mlir::memref::AllocOp op = *iter;

      std::string memory = buffer_type[op];
      int64_t address = assignedAddr[op.getOperation()];

      auto context = op.getContext();
      mlir::MemRefType type = op.getResult().getType();

      // Recompute size for overflow check
      int64_t size = 1;
      for (auto dim : type.getShape())
        size *= dim;
      int bitwidth = getBitWidth(type.getElementType());
      if (bitwidth == 1)
        size = size / 8;
      else if (bitwidth >= 8 && bitwidth % 8 == 0)
        size = size * bitwidth / 8;

      // overflow check and accumulate
      if (!address_table.count(memory))
        address_table[memory] = 0;
      address_table[memory] = std::max(address_table[memory], static_cast<int>(address + size));
      if (address + size > memory_size_list[memory]) {
        LOG_ERROR << "Memory address overflow after ILP: " << memory;
        std::exit(1);
      }

      // create memory space attr
      mlir::SmallVector<mlir::NamedAttribute, 2> nameAttrs;
      nameAttrs.push_back(mlir::NamedAttribute(mlir::StringAttr::get(context, "memory"), mlir::StringAttr::get(context, memory)));
      nameAttrs.push_back(mlir::NamedAttribute(mlir::StringAttr::get(context, "address"), mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), address)));

      mlir::DictionaryAttr new_memory_space = mlir::DictionaryAttr::get(op.getContext(), nameAttrs);
      mlir::MemRefType new_type = mlir::MemRefType::get(type.getShape(), type.getElementType(), type.getLayout(), new_memory_space);
      op.getResult().setType(new_type);

      // propagate to subview results
      for (mlir::OpOperand &use : op.getResult().getUses()) {
        if (auto subview = llvm::dyn_cast<mlir::memref::SubViewOp>(use.getOwner())) {
          mlir::MemRefType subview_type = subview.getType();
          mlir::MemRefType new_subview_type = mlir::MemRefType::get(subview_type.getShape(), subview_type.getElementType(), subview_type.getLayout(), new_memory_space);
          subview.getResult().setType(new_subview_type);
        }
      }
    }

    // return early since rest of old address allocation is removed/handled
    return;
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
