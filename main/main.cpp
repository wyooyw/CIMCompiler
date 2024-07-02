//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect.h"
#include "cim/Parser.h"

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
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include <iostream>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#define BOOST_NO_EXCEPTIONS
#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const & e){
//do nothing
}
using namespace mlir;


int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cim::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::registerAllDialects(registry);
  // mlir::func::registerInlinerExtension(registry);

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::cim::CIMDialect>();
  

  MLIRGenImpl gen_impl(context);
  mlir::ModuleOp module = gen_impl.parseJson("/home/wangyiou/project/cim_compiler_frontend/playground/out.json");
  
  module.dump();
  return 0;

  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  

  // Create Sub Function
  // builder.setInsertionPointToEnd(theModule.getBody());
  // RankedTensorType::Builder _sub_builder = 
  //       RankedTensorType::Builder({10,10}, builder.getI32Type(), Attribute()); // builder.getStringAttr("global")
  // RankedTensorType sub_func_ret_type = RankedTensorType::get({10,10}, builder.getI32Type(), Attribute());
  // RankedTensorType sub_func_arg0_type = RankedTensorType::get({10, 10}, builder.getI32Type(), Attribute());
  // RankedTensorType sub_func_arg1_type = RankedTensorType::get({10, 10}, builder.getI32Type(), Attribute());
  // auto sub_func_type = builder.getFunctionType({sub_func_arg0_type, sub_func_arg1_type}, {sub_func_ret_type});
  // std::string sub_func_name = "sub";
  // auto sub_function = builder.create<func::FuncOp>(loc, sub_func_name, sub_func_type);
  // sub_function.setPrivate();
  // Block *subfuncBody = sub_function.addEntryBlock();
  // builder.setInsertionPointToStart(subfuncBody);
  // mlir::Value func_arg0 = subfuncBody->getArgument(0);
  // mlir::Value func_arg1 = subfuncBody->getArgument(1);

  // mlir::Value c = builder.create<mlir::cim::VVAddOp>(loc, func_arg0, func_arg1);
  // mlir::Value d = builder.create<mlir::cim::VVAddOp>(loc, c, func_arg1);
  // builder.create<func::ReturnOp>(loc, d);

  // theModule.dump();
  // return 0;

  // Create Main Function
  builder.setInsertionPointToEnd(theModule.getBody());
  RankedTensorType::Builder _builder = 
        RankedTensorType::Builder({10,10}, builder.getI32Type(), Attribute());//builder.getStringAttr("global")
  RankedTensorType func_ret_type = RankedTensorType(_builder);
  mlir::MemRefType return_type =  mlir::MemRefType::get({10, 10}, builder.getI32Type());
  mlir::MemRefType param_type =  mlir::MemRefType::get({10, 10}, builder.getI32Type());
  auto func_type = builder.getFunctionType({param_type}, {func_ret_type});
  std::string func_name = "main";
  auto function = builder.create<func::FuncOp>(loc, func_name, func_type);
  Block *funcBody = function.addEntryBlock();
  builder.setInsertionPointToStart(funcBody);

  // llvm::ArrayRef<int64_t> shape = {10, 10};
  // mlir::Value _a = builder.create<tensor::EmptyOp>(loc, shape, builder.getI32Type());
  // mlir::Value _b = builder.create<tensor::EmptyOp>(loc, shape, builder.getI32Type());
  // mlir::Value a = builder.create<tensor::CastOp>(loc, sub_func_ret_type, _a);
  // mlir::Value b = builder.create<tensor::CastOp>(loc, sub_func_ret_type, _b);
  // func::CallOp call = builder.create<func::CallOp>(loc, sub_function, ValueRange({a, b}));
  // builder.create<func::ReturnOp>(loc, call.getResults());


  mlir::MemRefType type =  mlir::MemRefType::get({10, 10}, builder.getI32Type());
  mlir::memref::AllocOp alloc1 = builder.create<memref::AllocOp>(loc, type);
  mlir::memref::AllocOp alloc2 = builder.create<memref::AllocOp>(loc, type);
  mlir::memref::AllocOp alloc3 = builder.create<memref::AllocOp>(loc, type);
  mlir::Value buf1 = alloc1.getResult();
  mlir::Value buf2 = alloc2.getResult();
  mlir::Value buf3 = alloc3.getResult();
  builder.create<mlir::cim::VVAddOp>(loc, buf1, buf2, buf3);

  mlir::Value a1 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  mlir::Value a2 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  mlir::Value b1 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(4));
  mlir::Value b2 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(4));
  mlir::Value c1 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(4));
  mlir::Value c2 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(4));
  mlir::ValueRange offsets = {a1,a2};
  mlir::ValueRange sizes = {b1,b2};
  mlir::ValueRange strides = {c1,c2};
  mlir::Value result = builder.create<mlir::memref::SubViewOp>(loc, buf3, offsets, sizes, strides);
  // builder.create<func::ReturnOp>(loc, buf3);

  theModule.dump();

  // mlir::PassManager pm(&context);
  // pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::createInlinerPass());
  // pm.addPass(mlir::func::createFuncBufferizePass());
  // pm.addPass(mlir::bufferization::createOneShotBufferizePass());
  // pm.addPass(mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  // if (mlir::failed(pm.run(theModule))) {
  //   std::cout << "Pass fail." << std::endl;
  // }else{
  //   std::cout << "Pass success." << std::endl;
  // }
  // theModule.dump();
  // bufferization::runOneShotBufferize(b.getOperator(),bufferization::OneShotBufferizationOptions());
  // theModule.dump();
  std::cout << "Hello World" << std::endl;
  return 0;
}
