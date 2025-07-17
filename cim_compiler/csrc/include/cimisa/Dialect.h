#ifndef CIMISA_DIALECT_H_
#define CIMISA_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "cimisa/Dialect.h.inc"

#define GET_OP_CLASSES
#include "cimisa/Ops.h.inc"

namespace mlir {
class DialectRegistry;
} // namespace mlir

#endif // CIMISA_DIALECT_H_
