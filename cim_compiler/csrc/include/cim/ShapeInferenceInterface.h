#ifndef SHAPEINFERENCEINTERFACE_H_
#define SHAPEINFERENCEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/InitAllDialects.h"

namespace mlir {
namespace cim {

#include "cim/ShapeInferenceOpInterfaces.h.inc"

} // namespace cim
} // namespace mlir

#endif // SHAPEINFERENCEINTERFACE_H_
