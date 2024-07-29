//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect.h"
#include "cim/Parser.h"
// #include "cim/ShapeInferenceInterface.h"
#include "cim/Passes.h"
#include "cimisa/Dialect.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
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
  if (argc < 4) {
    std::cerr << "Error: Not enough arguments provided." << std::endl;
    return 1;
  }
  std::string inputFilePath(argv[1]);
  std::string outputFilePath(argv[2]);
  std::string configPath(argv[3]);

  mlir::DialectRegistry registry;
  mlir::registerCIMInlinerInterface(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cim::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::registerAllDialects(registry);
  // mlir::func::registerInlinerExtension(registry);

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::cim::CIMDialect>();
  context.getOrLoadDialect<mlir::cimisa::CIMISADialect>();
  

  MLIRGenImpl gen_impl(context);
  mlir::ModuleOp module = gen_impl.parseJson(inputFilePath);
  
  module.dump();
  std::cout << "\n\n\n\n" << std::endl;
  // return 0;
  mlir::PassManager pm(&context);
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  // optPM.addPass(mlir::createCSEPass());
  optPM.addPass(cim::createFoldMemRefAliasOpsPass());
  optPM.addPass(cim::createExtractAddressComputationPass());
  optPM.addPass(mlir::createLowerAffinePass());
  // optPM.addPass(mlir::cim::createTestDecomposeAffineOpPass());
  optPM.addPass(mlir::cim::createMemoryAddressAllocationPass());
  
  if (mlir::failed(pm.run(module))) {
    std::cout << "Pass fail." << std::endl;
  }else{
    std::cout << "Pass success." << std::endl;
  }
  module.dump();
  std::cout << "\n\n\n\n" << std::endl;

  // step3: lower
  mlir::PassManager lower_passes(&context);
  lower_passes.addPass(mlir::cim::createCIMLoweringPass(configPath));
  lower_passes.addPass(mlir::createCanonicalizerPass());
  lower_passes.addPass(mlir::cim::createRR2RIPass());
  lower_passes.addPass(mlir::createCanonicalizerPass());
  lower_passes.addPass(mlir::createLoopInvariantCodeMotionPass());
  lower_passes.addPass(mlir::createConvertSCFToCFPass());
  if (mlir::failed(lower_passes.run(module))) {
    std::cout << "Lower Passes fail." << std::endl;
    module.dump();
    return 1;
  }else{
    std::cout << "Lower Passes success." << std::endl;
    module.dump();
  }
  // return 0;

  // step4: convert control flow
  std::cout << "\n\n\n\n" << std::endl;
  mlir::PassManager cf_convert_passes(&context);
  cf_convert_passes.addPass(mlir::cim::createCIMBranchConvertPass());
  if (mlir::failed(cf_convert_passes.run(module))) {
    std::cout << "CF Convert Passes fail." << std::endl;
    module.dump();
    return 1;
  }else{
    std::cout << "CF Convert Passes success." << std::endl;
    module.dump();
  }

  // step5: codegen
  std::cout << "\n\n\n\n" << std::endl;
  mlir::PassManager codegen_passes(&context);
  mlir::OpPassManager &codegen_op_passes = codegen_passes.nest<mlir::func::FuncOp>();
  std::cout << outputFilePath << std::endl;
  codegen_op_passes.addPass(cim::createCodeGenerationPass(outputFilePath));
  if (mlir::failed(codegen_passes.run(module))) {
    std::cout << "CodeGen Passes fail." << std::endl;
    return 1;
  }else{
    std::cout << "CodeGen Passes success." << std::endl;
  }
  
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


  mlir::MemRefType type1 =  mlir::MemRefType::get({10, 10}, builder.getI32Type());
  mlir::memref::AllocOp alloc1 = builder.create<memref::AllocOp>(loc, type1);
  mlir::Value buf1 = alloc1.getResult();

  mlir::MemRefType type2 =  mlir::MemRefType::get({2,2}, builder.getI32Type());
  mlir::memref::AllocOp alloc2 = builder.create<memref::AllocOp>(loc, type2);
  mlir::Value buf2 = alloc2.getResult();
  // mlir::Value buf2 = alloc2.getResult();
  // mlir::Value buf3 = alloc3.getResult();
  // builder.create<mlir::cim::VVAddOp>(loc, buf1, buf2, buf3);
  mlir::Value _a1 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  mlir::Value _a2 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  OpFoldResult a1 = OpFoldResult(_a1);
  OpFoldResult a2 = OpFoldResult(_a2);
  OpFoldResult b1 = OpFoldResult(builder.getI64IntegerAttr(1));
  OpFoldResult b2 = OpFoldResult(builder.getI64IntegerAttr(4));
  OpFoldResult c1 = OpFoldResult(builder.getI64IntegerAttr(1));
  OpFoldResult c2 = OpFoldResult(builder.getI64IntegerAttr(1));
  llvm::SmallVector<OpFoldResult> offsets1 = {a1,a2};
  llvm::SmallVector<OpFoldResult> sizes1 = {b1,b2};
  llvm::SmallVector<OpFoldResult> strides1 = {c1,c2};
  // mlir::ValueRange sizes1 = {b1,b2};
  // mlir::ValueRange strides1 = {c1,c2};
  // std::vector<int64_t> offsets1 = {0,0};
  // std::vector<int64_t> sizes1 = {1,4};
  // std::vector<int64_t> strides1 = {1,1};
  mlir::Value result = builder.create<mlir::memref::SubViewOp>(loc, buf1, offsets1, sizes1, strides1);
  // llvm::SmallVector<ReassociationIndices> reassociation1;
  // ReassociationIndices collapsedIndices1 = {0,1};
  // reassociation1.push_back(collapsedIndices1);
  // mlir::Value flatten_buf1 = builder.create<mlir::memref::CollapseShapeOp>(loc, result, reassociation1);
  // mlir::Value _tmp = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(4));
  // FailureOr<Value> _reduce_rank_buf1 = mlir::memref::SubViewOp::rankReduceIfNeeded(builder, loc,result,{mlir::ShapedType::kDynamic});
  // if (failed(_reduce_rank_buf1)){
  //   std::cout << "fail" << std::endl;
  //   std::exit(1);
  // }
  // mlir::Value reduce_rank_buf1 = *_reduce_rank_buf1;
  // std::cout << "A" << std::endl;
  // llvm::SmallVector<ReassociationIndices> reassociation;
  // ReassociationIndices collapsedIndices = {0,1};
  // reassociation.push_back(collapsedIndices);
  // mlir::Value flatten_buf2 = builder.create<mlir::memref::CollapseShapeOp>(loc, buf2, reassociation);
  // std::cout << "B" << std::endl;
  builder.create<mlir::cim::CopyOp>(loc, buf2, result);
  // builder.create<func::ReturnOp>(loc, buf3);

  // mlir::Value a3 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  // mlir::Value a4 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  // mlir::Value b3 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(2));
  // mlir::Value b4 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(2));
  // mlir::Value c3 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));
  // mlir::Value c4 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));
  // mlir::ValueRange offsets2 = {a3,a4};
  // mlir::ValueRange sizes2 = {b3,b4};
  // mlir::ValueRange strides2 = {c3,c4};
  // std::vector<int64_t> offsets2 = {2,2};
  // std::vector<int64_t> sizes2 = {2,2};
  // std::vector<int64_t> strides2 = {1,1};
  // mlir::Value result2 = builder.create<mlir::memref::SubViewOp>(loc, result, offsets2, sizes2, strides2);
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
