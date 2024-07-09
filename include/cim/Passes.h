#ifndef CIM_PASSES_H
#define CIM_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace cim {
std::unique_ptr<Pass> createShapeInferencePass();
std::unique_ptr<Pass> createMemoryAddressAllocationPass();
} // namespace cim
} // namespace mlir

#endif // CIM_PASSES_H
