#ifndef CIM_DIALECT_H_
#define CIM_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cim/Dialect.h.inc"
#include "cim/ShapeInferenceInterface.h"

#define GET_OP_CLASSES
#include "cim/Ops.h.inc"

namespace mlir {
class DialectRegistry;
namespace cim {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace cim

void registerCIMInlinerInterface(mlir::DialectRegistry &registry);

} // namespace mlir

#endif // CIM_DIALECT_H_
