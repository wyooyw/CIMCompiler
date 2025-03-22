#include "cim/Dialect.h"
#include "cim/Parser.h"
#include "cim/Passes.h"
#include "cimisa/Dialect.h"
#include "common/macros.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <glog/logging.h>
#include "common/macros.h"

#define BOOST_NO_EXCEPTIONS
#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const &e) {
  // do nothing
}
using namespace mlir;

void debugLogIR(mlir::ModuleOp &module) {
  std::string irString;
  llvm::raw_string_ostream os(irString);
  module.print(os);
  os.flush();

  LOG_DEBUG << "IR: " << irString << "\n\n";
}

void errorLogIR(mlir::ModuleOp &module) {
  std::string irString;
  llvm::raw_string_ostream os(irString);
  module.print(os);
  LOG_ERROR << "IR: " << irString << "\n\n";
}

void MyPrefixFormatter(std::ostream& s, const google::LogMessage& m, void* /*data*/) {
   s << std::setw(4) << 1900 + m.time().year() << "-"
   << std::setw(2) << 1 + m.time().month() << "-"
   << std::setw(2) << m.time().day()
   << " "
   << std::setw(2) << m.time().hour() << ":"
   << std::setw(2) << m.time().min()  << ":"
   << std::setw(2) << m.time().sec()
   << " - "
   << m.basename() << ':' << m.line()
   << " - "
   << google::GetLogSeverityName(m.severity())
   << " - ";
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallPrefixFormatter(&MyPrefixFormatter);


  if (argc < 4) {
    LOG_ERROR << "Error: Not enough arguments provided.";
    return 1;
  }
  std::string inputFilePath(argv[1]);
  std::string outputDirPath(argv[2]);
  std::string configPath(argv[3]);

  std::string outputFileName = "final_code.json";
  std::string outputFilePath = outputDirPath + "/" + outputFileName;

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

  debugLogIR(module);  

  if (failed(verify(module))) {
    LOG_ERROR << "Module verification failed";
    errorLogIR(module);
    return 1;
  }  
  


  mlir::PassManager init_passes(&context);
  init_passes.addPass(mlir::createInlinerPass());
  init_passes.addPass(mlir::createCanonicalizerPass());
  mlir::OpPassManager &init_op_passes = init_passes.nest<mlir::func::FuncOp>();
  init_op_passes.addPass(mlir::createCSEPass());
  init_op_passes.addPass(cim::createFoldMemRefAliasOpsPass());
  init_op_passes.addPass(cim::createExtractAddressComputationPass());
  init_op_passes.addPass(mlir::createLowerAffinePass());
  init_op_passes.addPass(mlir::cim::createMemoryAddressAllocationPass());

  mlir::PassManager unroll_passes(&context);
  unroll_passes.addPass(cim::createLoopUnrollPass());

  mlir::PassManager lower_passes(&context);
  lower_passes.addPass(mlir::cim::createCIMLoweringPass(configPath));
  lower_passes.addPass(mlir::createCanonicalizerPass());
  lower_passes.addPass(mlir::createLoopInvariantCodeMotionPass());
  lower_passes.addPass(mlir::createCanonicalizerPass());
  lower_passes.addPass(mlir::cim::createCommonSubexpressionExposePass());
  mlir::OpPassManager &cse_passes = lower_passes.nest<mlir::func::FuncOp>();
  cse_passes.addPass(mlir::createCSEPass());

  mlir::PassManager rr2ri_passes(&context);
  rr2ri_passes.addPass(mlir::cim::createRR2RIPass());
  rr2ri_passes.addPass(mlir::createCanonicalizerPass());

  mlir::PassManager cf_passes(&context);
  cf_passes.addPass(mlir::createConvertSCFToCFPass());
  // cf_passes.addPass(mlir::cim::createRR2RIPass());
  cf_passes.addPass(mlir::cim::createCIMBranchConvertPass());
  mlir::OpPassManager &cf_op_passes = cf_passes.nest<mlir::func::FuncOp>();
  cf_op_passes.addPass(mlir::cim::createTransOffsetOptimizePass());
  cf_op_passes.addPass(mlir::cim::createConstantExpandPass());
  
  

  mlir::PassManager codegen_passes(&context);
  mlir::OpPassManager &codegen_op_passes =
      codegen_passes.nest<mlir::func::FuncOp>();
  codegen_op_passes.addPass(cim::createCodeGenerationPass(outputFilePath));


  if (mlir::failed(init_passes.run(module))) {
    LOG_ERROR << "Init passes fail.";
    errorLogIR(module);
    return 1;
  }else{
    LOG_INFO << "Init Passes success.";
    debugLogIR(module);
  }

  if (mlir::failed(unroll_passes.run(module))) {
    LOG_ERROR << "Unroll passes fail.";
    errorLogIR(module);
    return 1;
  }else{
    LOG_INFO << "Unroll Passes success.";
    debugLogIR(module);
  }

  if (mlir::failed(lower_passes.run(module))) {
    LOG_ERROR << "Lower passes fail.";
    errorLogIR(module);
    return 1;
  }else{
    LOG_INFO << "Lower Passes success.";
    debugLogIR(module);
  }

  if (mlir::failed(rr2ri_passes.run(module))) {
    LOG_ERROR << "RR2RI passes fail.";
    errorLogIR(module);
    return 1;
  }else{
    LOG_INFO << "RR2RI Passes success.";
    debugLogIR(module);
  }

  if (mlir::failed(cf_passes.run(module))) {
    LOG_ERROR << "CF passes fail.";
    errorLogIR(module);
    return 1;
  }else{
    LOG_INFO << "CF Passes success.";
    debugLogIR(module);
  }

  // return 0;

  if (mlir::failed(codegen_passes.run(module))) {
    LOG_ERROR << "CodeGen Passes fail.";
    errorLogIR(module);
    return 1;
  }else{
    LOG_INFO << "CodeGen Passes success.";
    debugLogIR(module);
  }
  return 0;
}