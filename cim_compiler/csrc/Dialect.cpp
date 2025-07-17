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
#include "common/macros.h"

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
    LOG_DEBUG << "materializeCallConversion";
    return builder.create<mlir::memref::CastOp>(conversionLoc, resultType, input);
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

/*
  Built-in Functions
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
SIMDOp::fold(FoldAdaptor adaptor,
             llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return memref::foldMemRefCast(*this);
}

LogicalResult
ReduceOp::fold(FoldAdaptor adaptor,
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

LogicalResult
SendOp::fold(FoldAdaptor adaptor,
               llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
}

LogicalResult
RecvOp::fold(FoldAdaptor adaptor,
               llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
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

void mlir::cim::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {

}

#define GET_OP_CLASSES
#include "cim/Ops.cpp.inc"