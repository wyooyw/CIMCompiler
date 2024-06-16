//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/ValueRange.h"
#include <iostream>

using namespace mlir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::cim::CIMDialect>();
  

  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(theModule.getBody());

  llvm::ArrayRef<int64_t> shape = {10, 10};
  mlir::Value a = builder.create<tensor::EmptyOp>(loc, shape, builder.getI8Type());
  mlir::Value b1 = builder.create<tensor::EmptyOp>(loc, shape, builder.getI8Type());
  mlir::Value b2 = builder.create<tensor::EmptyOp>(loc, shape, builder.getI8Type());
  // mlir::Value c = builder.create<mlir::arith::AddFOp>(loc, a, b);
  mlir::Value c1 = builder.create<mlir::cim::CIMComputeOp>(loc, a, b1);
  mlir::Value c2 = builder.create<mlir::cim::CIMComputeOp>(loc, a, b2);
  mlir::Value d = builder.create<mlir::cim::VecAddOp>(loc, c1, c2);

  // ValueRange offsetsArray = {builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(0)),
  //                            builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(4))};
  // ValueRange sizesArray = {builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(4)),
  //                            builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(8))};
  // ValueRange stridesArray = {builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(1)),
  //                            builder.create<arith::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(1))};
  // Value d = builder.create<tensor::ExtractSliceOp>(loc,  c, offsetsArray, sizesArray, stridesArray);

  // bufferization::runOneShotBufferize(theModule,bufferization::OneShotBufferizationOptions());
  theModule.dump();
  std::cout << "Hello World" << std::endl;
  return 0;
}
