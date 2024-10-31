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

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
#include <iostream>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::cim;

#include "cim/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CIMDialect
//===----------------------------------------------------------------------===//

// modify from mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp
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

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, Block *newDest) const final {
    // Only return needs to be handled here.
    auto returnOp = dyn_cast<mlir::func::ReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    OpBuilder builder(op);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest,
                                       returnOp.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // Only return needs to be handled here.
    auto returnOp = cast<mlir::func::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    std::cout << "materializeCallConversion" << std::endl;
    return builder.create<mlir::cim::CastOp>(conversionLoc, resultType, input);
  }
};

struct CIM_InlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

struct IndexInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void CIMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cim/Ops.cpp.inc"
      >();
  addInterfaces<CIM_InlinerInterface>();
}

void mlir::registerCIMInlinerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    dialect->addInterfaces<CIMInlinerInterface>();
  });
  registry.addExtension(+[](MLIRContext *ctx, index::IndexDialect *dialect) {
    dialect->addInterfaces<IndexInlinerInterface>();
  });
}

//===----------------------------------------------------------------------===//
// CIM Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VecAddOp
//===----------------------------------------------------------------------===//

// void VVAddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   // same shape
//   auto type = lhs.getType().cast<RankedTensorType>();
//   if(type){
//     auto shape = type.getShape();
//     auto element_type = builder.getI32Type();
//     auto encoding = type.getEncoding();
//     RankedTensorType::Builder _builder =
//         RankedTensorType::Builder(shape, element_type, encoding);
//     RankedTensorType newTensorType = RankedTensorType(_builder);
//     state.addTypes(newTensorType);
//   }else{
//     state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
//   }

//   state.addOperands({lhs, rhs});
// }

// void BufVVAddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs, mlir::Value result) {
// same shape
// auto type = lhs.getType().cast<RankedTensorType>();
// if(type){
//   auto shape = type.getShape();
//   auto element_type = builder.getI32Type();
//   auto encoding = type.getEncoding();
//   RankedTensorType::Builder _builder =
//       RankedTensorType::Builder(shape, element_type, encoding);
//   RankedTensorType newTensorType = RankedTensorType(_builder);
//   state.addTypes(newTensorType);
// }else{
//   state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
// }
// auto output_type = llvm::cast<MemRefType>(lhs.getType());
// state.addTypes(output_type);
//   state.addOperands({lhs, rhs, result});
// }

void VSMulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value vec, mlir::Value scalar) {
  auto type = vec.getType().cast<RankedTensorType>();
  if (type) {
    auto shape = type.getShape();
    auto element_type = builder.getI32Type();
    auto encoding = type.getEncoding();
    RankedTensorType::Builder _builder =
        RankedTensorType::Builder(shape, element_type, encoding);
    RankedTensorType newTensorType = RankedTensorType(_builder);
    state.addTypes(newTensorType);
  } else {
    state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
  }
  // state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
  state.addOperands({vec, scalar});
}

//===----------------------------------------------------------------------===//
// CIMComputeOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return true;
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

/*
  Bulitin Functions
*/

void ShapeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value input, mlir::Value index) {
  state.addTypes(builder.getIndexType());
  state.addOperands({input, index});
}

OpFoldResult ShapeOp::fold(FoldAdaptor adaptor) {
  // prefetch(memrefcast) -> prefetch
  return succeeded(memref::foldMemRefCast(*this)) ? getResult() : Value();
}

LogicalResult
CopyOp::fold(FoldAdaptor adaptor,
             llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return memref::foldMemRefCast(*this);
}

LogicalResult
CIMComputeOp::fold(FoldAdaptor adaptor,
                   llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return memref::foldMemRefCast(*this);
}

LogicalResult
VVAddOp::fold(FoldAdaptor adaptor,
              llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return memref::foldMemRefCast(*this);
}

LogicalResult
QuantifyOp::fold(FoldAdaptor adaptor,
                 llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return memref::foldMemRefCast(*this);
}

LogicalResult
CIMOutputOp::fold(FoldAdaptor adaptor,
                  llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return memref::foldMemRefCast(*this);
}

LogicalResult
CIMOutputSumOp::fold(FoldAdaptor adaptor,
                     llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return memref::foldMemRefCast(*this);
}

LogicalResult
CIMTransferOp::fold(FoldAdaptor adaptor,
                    llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return memref::foldMemRefCast(*this);
}

LogicalResult
CIMSetOp::fold(FoldAdaptor adaptor,
               llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return memref::foldMemRefCast(*this);
}

void AddrOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value src) {
  state.addTypes(builder.getIndexType());
  state.addOperands({src});
}

OpFoldResult AddrOp::fold(FoldAdaptor adaptor) {
  // prefetch(memrefcast) -> prefetch
  return succeeded(memref::foldMemRefCast(*this)) ? getResult() : Value();
}
// Bufferize

static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Bufferization of cim.vv_add. Replace with cim.b_vv_add
// struct VVAddOpInterface
//     : public
//     bufferization::BufferizableOpInterface::ExternalModel<VVAddOpInterface,
//                                                     cim::VVAddOp> {
//   bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
//                               const bufferization::AnalysisState &state)
//                               const {
//     return false;
//   }

//   bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
//                                const bufferization::AnalysisState &state)
//                                const {
//     return false;
//   }

//   bufferization::AliasingValueList getAliasingValues(Operation *op, OpOperand
//   &opOperand,
//                                       const bufferization::AnalysisState
//                                       &state) const {
//     return {{op->getOpResult(0), bufferization::BufferRelation::Unknown}};
//   }

//   LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
//                           const bufferization::BufferizationOptions &options)
//                           const {
//     auto vv_add_op = cast<cim::VVAddOp>(op);
//     Location loc = vv_add_op.getLoc();

//     // Get source buffer.
//     FailureOr<Value> src0Memref =
//         getBuffer(rewriter, vv_add_op.getOperand(0), options);
//     FailureOr<Value> src1Memref =
//         getBuffer(rewriter, vv_add_op.getOperand(1), options);
//     if (failed(src0Memref) || failed(src1Memref))
//       return failure();

//     // Take a subview of the source buffer.
//     auto resultMemrefType =
//         convertTensorToMemRef(vv_add_op.getResult().getType().cast<RankedTensorType>());
//     auto alloc = rewriter.create<memref::AllocOp>(loc, resultMemrefType);
//     // if (failed(resultMemrefType))
//     //   return failure();
//     rewriter.create<cim::BufVVAddOp>(
//         loc, *src0Memref, *src1Memref, alloc);

//     bufferization::replaceOpWithBufferizedValues(rewriter, vv_add_op,
//     ValueRange({alloc})); return success();
//   }

// FailureOr<BaseMemRefType>
// getBufferType(Operation *op, Value value, const BufferizationOptions
// &options,
//               SmallVector<Value> &invocationStack) const {
//   auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
//   assert(value == extractSliceOp.getResult() && "invalid value");
//   auto srcMemrefType = bufferization::getBufferType(
//       extractSliceOp.getSource(), options, invocationStack);
//   if (failed(srcMemrefType))
//     return failure();
//   SmallVector<OpFoldResult> mixedOffsets = extractSliceOp.getMixedOffsets();
//   SmallVector<OpFoldResult> mixedSizes = extractSliceOp.getMixedSizes();
//   SmallVector<OpFoldResult> mixedStrides = extractSliceOp.getMixedStrides();
//   return cast<BaseMemRefType>(memref::SubViewOp::inferRankReducedResultType(
//       extractSliceOp.getType().getShape(),
//       llvm::cast<MemRefType>(*srcMemrefType), mixedOffsets, mixedSizes,
//       mixedStrides));
// }
// };

void mlir::cim::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  // registry.addExtension(+[](MLIRContext *ctx, cim::CIMDialect *dialect) {
  //   VVAddOp::attachInterface<VVAddOpInterface>(*ctx);
  // });
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cim/Ops.cpp.inc"