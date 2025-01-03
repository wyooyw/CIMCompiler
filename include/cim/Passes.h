#ifndef CIM_PASSES_H
#define CIM_PASSES_H

#include <memory>
#include <string>
namespace mlir {
class Pass;

namespace cim {
std::unique_ptr<Pass> createShapeInferencePass();
std::unique_ptr<Pass> createMemoryAddressAllocationPass();
std::unique_ptr<Pass> createTestDecomposeAffineOpPass();
std::unique_ptr<Pass> createFoldMemRefAliasOpsPass();
std::unique_ptr<Pass> createExtractAddressComputationPass();
std::unique_ptr<Pass> createCIMLoweringPass(std::string configPath);
std::unique_ptr<Pass> createCIMBranchConvertPass();
std::unique_ptr<Pass> createCodeGenerationPass(std::string outputFilePath);
std::unique_ptr<Pass> createRR2RIPass();
std::unique_ptr<Pass>
createLoopUnrollPass(std::vector<mlir::scf::ForOp> &unrollForOps);
std::unique_ptr<Pass> createCastEliminationPass();
std::unique_ptr<Pass> createCommonSubexpressionExposePass();
} // namespace cim
} // namespace mlir

#endif // CIM_PASSES_H
