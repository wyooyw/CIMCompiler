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

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void CIMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cim/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CIM Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VecAddOp
//===----------------------------------------------------------------------===//

void VVAddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
  state.addOperands({lhs, rhs});
}

void VSMulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value vec, mlir::Value scalar) {
  state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
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
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cim/Ops.cpp.inc"
