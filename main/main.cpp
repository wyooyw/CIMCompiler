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
// #include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
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
  mlir::func::registerAllExtensions(registry);
  // mlir::func::registerInlinerExtension(registry);

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
  

  // Create Sub Function
  builder.setInsertionPointToEnd(theModule.getBody());
  RankedTensorType::Builder _sub_builder = 
        RankedTensorType::Builder({10,10}, builder.getI32Type(), Attribute());
  RankedTensorType sub_func_ret_type = RankedTensorType(_sub_builder);
  RankedTensorType sub_func_arg0_type = RankedTensorType(_sub_builder);
  RankedTensorType sub_func_arg1_type = RankedTensorType(_sub_builder);
  auto sub_func_type = builder.getFunctionType({sub_func_arg0_type, sub_func_arg1_type}, {sub_func_ret_type});
  std::string sub_func_name = "sub";
  auto sub_function = builder.create<func::FuncOp>(loc, sub_func_name, sub_func_type);
  sub_function.setPrivate();
  Block *subfuncBody = sub_function.addEntryBlock();
  builder.setInsertionPointToStart(subfuncBody);
  mlir::Value func_arg0 = subfuncBody->getArgument(0);
  mlir::Value func_arg1 = subfuncBody->getArgument(1);

  mlir::Value c = builder.create<mlir::arith::AddIOp>(loc, func_arg0, func_arg0);
  mlir::Value d = builder.create<mlir::arith::AddIOp>(loc, c, func_arg1);
  builder.create<func::ReturnOp>(loc, d);

  // Create Main Function
  builder.setInsertionPointToEnd(theModule.getBody());
  RankedTensorType::Builder _builder = 
        RankedTensorType::Builder({10,10}, builder.getI32Type(), Attribute());
  RankedTensorType func_ret_type = RankedTensorType(_builder);
  auto func_type = builder.getFunctionType({}, {func_ret_type});
  std::string func_name = "main";
  auto function = builder.create<func::FuncOp>(loc, func_name, func_type);
  Block *funcBody = function.addEntryBlock();
  builder.setInsertionPointToStart(funcBody);

  llvm::ArrayRef<int64_t> shape = {10, 10};
  mlir::Value a = builder.create<tensor::EmptyOp>(loc, shape, builder.getI32Type());
  mlir::Value b = builder.create<tensor::EmptyOp>(loc, shape, builder.getI32Type());
  func::CallOp call = builder.create<func::CallOp>(loc, sub_function, ValueRange({a, b}));
  builder.create<func::ReturnOp>(loc, call.getResults());
  theModule.dump();

  mlir::PassManager pm(&context);
  // pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createInlinerPass());
  if (mlir::failed(pm.run(theModule))) {
    std::cout << "Pass fail." << std::endl;
  }else{
    std::cout << "Pass success." << std::endl;
  }
  theModule.dump();
  std::cout << "Hello World" << std::endl;
  return 0;
}
