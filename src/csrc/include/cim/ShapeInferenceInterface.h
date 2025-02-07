//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_
#define MLIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/InitAllDialects.h"

namespace mlir {
namespace cim {

/// Include the auto-generated declarations.
#include "cim/ShapeInferenceOpInterfaces.h.inc"

} // namespace cim
} // namespace mlir

#endif // MLIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_
