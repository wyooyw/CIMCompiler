#ifndef CIM_PASSES_H
#define CIM_PASSES_H

#include <memory>
#include <string>
#include <map>
#include "mlir/IR/Operation.h"

namespace mlir {
class Pass;

namespace cim {
std::unique_ptr<Pass> createShapeInferencePass();
std::unique_ptr<Pass> createMemoryAddressAllocationPass(std::string configPath, std::map<mlir::Operation *, std::string> buffer_type);
std::unique_ptr<Pass> createTestDecomposeAffineOpPass();
std::unique_ptr<Pass> createFoldMemRefAliasOpsPass();
std::unique_ptr<Pass> createExtractAddressComputationPass();
std::unique_ptr<Pass> createCIMLoweringPass(std::string configPath);
std::unique_ptr<Pass> createCIMBranchConvertPass();
std::unique_ptr<Pass> createCodeGenerationPass(std::string outputFilePath);
std::unique_ptr<Pass> createRR2RIPass();
std::unique_ptr<Pass> createConstantExpandPass();
std::unique_ptr<Pass> createLoopUnrollPass();
std::unique_ptr<Pass> createCastEliminationPass();
std::unique_ptr<Pass> createCommonSubexpressionExposePass();
std::unique_ptr<Pass> createTransOffsetOptimizePass();
} // namespace cim
} // namespace mlir

#endif // CIM_PASSES_H
