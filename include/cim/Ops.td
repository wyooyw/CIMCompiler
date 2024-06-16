
#ifndef CIM_OPS
#define CIM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def CIM_Dialect : Dialect {
  let name = "cim";
  let cppNamespace = "::mlir::cim";
}

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class CIM_Op<string mnemonic, list<Trait> traits = []> :
    Op<CIM_Dialect, mnemonic, traits>;

def VVAddOp : CIM_Op<"vv_add"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I32Tensor:$lhs, I32Tensor:$rhs);
  let results = (outs I32Tensor);

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def VSMulOp : CIM_Op<"vs_mul"> {
  let summary = "Multiplying scalar and vector";
  let description = [{
    Multiplying scalars and vectors.
  }];

  let arguments = (ins I32Tensor:$vec, I32Attr:$scalar);
  let results = (outs I32Tensor);

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def CIMComputeOp : CIM_Op<"cim_compute"> {
  let summary = "cim compute";
  let description = [{
    cim compute use macro
  }];

  let arguments = (ins I8Tensor:$vec, I8Tensor:$mat);
  let results = (outs I32Tensor);

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$vec, "Value":$mat)>
  ];
}
#endif // CIM_OPS