//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::cim::CIMDialect>();
  

  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(theModule.getBody());

  auto func_type = builder.getFunctionType(std::nullopt, std::nullopt);
  std::string name = "main";
  auto function = builder.create<func::FuncOp>(loc, name, func_type);
  Block *funcBody = function.addEntryBlock();
  builder.setInsertionPointToStart(funcBody);

  llvm::ArrayRef<int64_t> shape = {10, 10};
  mlir::Value a = builder.create<tensor::EmptyOp>(loc, shape, builder.getI32Type());
  mlir::Value b = builder.create<tensor::EmptyOp>(loc, shape, builder.getI32Type());
  mlir::Value c = builder.create<mlir::cim::VVAddOp>(loc, a, a);
  mlir::Value d = builder.create<mlir::cim::VVAddOp>(loc, c, b);
  builder.create<func::ReturnOp>(loc);
  theModule.dump();

  mlir::PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(theModule))) {
    std::cout << "Pass fail." << std::endl;
  }else{
    std::cout << "Pass success." << std::endl;
  }
  theModule.dump();
  std::cout << "Hello World" << std::endl;
  return 0;
}
