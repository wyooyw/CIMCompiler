//===- Dialect.h - Dialect definition for the Toy IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Toy language.
// See docs/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_CIM_DIALECT_H_
#define MLIR_TUTORIAL_CIM_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "cim/Dialect.h.inc"
#include "cim/ShapeInferenceInterface.h"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "cim/Ops.h.inc"

namespace mlir {
class DialectRegistry;
namespace cim {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace cim

void registerCIMInlinerInterface(mlir::DialectRegistry &registry);

} // namespace mlir

#endif // MLIR_TUTORIAL_CIM_DIALECT_H_
