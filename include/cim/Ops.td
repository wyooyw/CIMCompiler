
#ifndef CIM_OPS
#define CIM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

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

def VVAddOp : CIM_Op<"vv_add", [Pure]> {
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

  let hasCanonicalizer = 1;
}

def BufVVAddOp: CIM_Op<"buf_vv_add", [Pure]>{
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs);
  let results = (outs AnyMemRef);
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

def VSMulOp : CIM_Op<"vs_mul", [Pure]> {
  let summary = "Multiplying scalar and vector";
  let description = [{
    Multiplying scalars and vectors.
  }];

  let arguments = (ins I32Tensor:$vec, I32:$scalar);
  let results = (outs I32Tensor);

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$vec, "Value":$scalar)>
  ];
}

def CIMComputeOp : CIM_Op<"cim_compute", [Pure]> {
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

def CastOp : CIM_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    Pure,
    SameOperandsAndResultShape]
  > {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked,
    then shape is required to match. The operation is invalid if converting
    to a mismatching constant dimension.
  }];

  let arguments = (ins I32Tensor:$input);
  let results = (outs I32Tensor:$output);
}
#endif // CIM_OPS