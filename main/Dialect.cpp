//===- Dialect.cpp - Toy IR Dialect registration in MLIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the Toy IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::cim;

#include "cim/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CIMDialect
//===----------------------------------------------------------------------===//

struct CIMInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within toy can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within toy can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // All functions within toy can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(toy.return) by replacing it with a new
  /// operation as necessary.
  // void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
  //   // Only "toy.return" needs to be handled here.
  //   auto returnOp = cast<ReturnOp>(op);

  //   // Replace the values directly with the return operands.
  //   assert(returnOp.getNumOperands() == valuesToRepl.size());
  //   for (const auto &it : llvm::enumerate(returnOp.getOperands()))
  //     valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  // }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void CIMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cim/Ops.cpp.inc"
      >();
  addInterfaces<CIMInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// CIM Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VecAddOp
//===----------------------------------------------------------------------===//

void VVAddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  // same shape
  auto type = lhs.getType().cast<RankedTensorType>();
  if(type){
    auto shape = type.getShape();
    auto element_type = builder.getI32Type();
    auto encoding = type.getEncoding();
    RankedTensorType::Builder _builder =
        RankedTensorType::Builder(shape, element_type, encoding);
    RankedTensorType newTensorType = RankedTensorType(_builder);
    state.addTypes(newTensorType);
  }else{
    state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
  }
  
  state.addOperands({lhs, rhs});
}

void VSMulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value vec, mlir::Value scalar) {
  auto type = vec.getType().cast<RankedTensorType>();
  if(type){
    auto shape = type.getShape();
    auto element_type = builder.getI32Type();
    auto encoding = type.getEncoding();
    RankedTensorType::Builder _builder =
        RankedTensorType::Builder(shape, element_type, encoding);
    RankedTensorType newTensorType = RankedTensorType(_builder);
    state.addTypes(newTensorType);
  }else{
    state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
  }
  // state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
  state.addOperands({vec, scalar});
  
}

//===----------------------------------------------------------------------===//
// CIMComputeOp
//===----------------------------------------------------------------------===//


void CIMComputeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value vec, mlir::Value mat) {
  state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
  state.addOperands({vec, mat});
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = inputs.front().dyn_cast<TensorType>();
  TensorType output = outputs.front().dyn_cast<TensorType>();
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cim/Ops.cpp.inc"
