
#ifndef CIM_OPS
#define CIM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "cim/ShapeInferenceInterface.td"


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


/////////////////////////////////
// Vector Compute //
/////////////////////////////////

def VVAddOp : CIM_Op<"vv_add"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$out);

  // Allow building an AddOp with from the two input operands.
  // let builders = [
  //  OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$out)>
  // ];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
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


/////////////////////////////////
// CIM Compute //
/////////////////////////////////

def CIMComputeOp : CIM_Op<"cim_compute"> {
  let summary = "cim compute";
  let description = [{
    cim compute use macro
  }];

  let arguments = (ins 
    AnyMemRef:$input, 
    AnyMemRef:$macro, 
    AnyMemRef:$output,
    I1Attr:$value_sparse_flag,
    I1Attr:$bit_sparse_flag
  );

  let hasFolder = 1;
}


/////////////////////////////////
// Other Operator //
/////////////////////////////////

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

  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$output);

  let hasCanonicalizer = 1;
}

def ShapeOp : CIM_Op<"shape", [Pure]> {
  let summary = "get shape of a tensor";
  let description = [{
    get shape of a tensor
  }];

  let arguments = (ins AnyMemRef:$input, Index:$index); // TODO: which type can represent any Tensor?
  let results = (outs Index:$output);

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$index)>
  ];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def PrintOp : CIM_Op<"print"> {
  let summary = "print a integer";
  let description = [{
    print an integer
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$value); // TODO: which type can represent any Tensor?

  //let builders = [
  //  OpBuilder<(ins "Value":$value)>
  //];

}

def DebugOp : CIM_Op<"debug"> {
  let summary = "break for debug";
  let description = [{
    break for debug
  }];

  // let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$value); // TODO: which type can represent any Tensor?

  //let builders = [
  //  OpBuilder<(ins "Value":$value)>
  //];

}

def CopyOp : CIM_Op<"copy"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  // Allow building an AddOp with from the two input operands.
  // let builders = [
  //  OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$out)>
  // ];

  // let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def SpecialRegSetOp : CIM_Op<"special_reg_set"> {
  let summary = "set a special reg";
  let description = [{
    set a special reg
  }];
  let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$special_reg, AnyTypeOf<[AnyInteger, Index]>:$set_value); // TODO: which type can represent any Tensor?
}

def CIMOutputOp : CIM_Op<"cim_output"> {
  let summary = "cim output and clear";
  let description = [{
    cim output and clear
  }];

  let arguments = (ins AnyMemRef:$dst);

  let hasFolder = 1;
}

def CIMOutputSumOp : CIM_Op<"cim_output_sum"> {
  let summary = "cim output and clear";
  let description = [{
    cim output and clear
  }];

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$out_n, 
    AnyMemRef:$out_mask, 
    AnyMemRef:$dst
  );

  let hasFolder = 1;
}

def CIMTransferOp : CIM_Op<"cim_transfer"> {
  let summary = "cim output and clear";
  let description = [{
    cim output and clear
  }];

  let arguments = (ins 
    AnyMemRef:$src,
    AnyTypeOf<[AnyInteger, Index]>:$output_num, 
    AnyMemRef:$output_mask,
    AnyMemRef:$buffer,
    AnyMemRef:$dst
  );

  let hasFolder = 1;
}

#endif // CIM_OPS