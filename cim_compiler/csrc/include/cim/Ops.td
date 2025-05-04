
#ifndef CIM_OPS
#define CIM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
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

def SIMDOp : CIM_Op<"simd"> {
  let summary = "simd";
  let description = [{
    simd
  }];

  let arguments = (ins 
    Index:$op_id,
    Variadic<AnyMemRef>:$inputs,
    AnyMemRef:$output
  );

  let hasFolder = 1;
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
    I1Attr:$value_sparse_flag,
    I1Attr:$bit_sparse_flag,
    I1Attr:$batch_flag
  );

  let hasFolder = 1;
}


/////////////////////////////////
// Other Operator //
/////////////////////////////////

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

def AddrOp : CIM_Op<"addr", [Pure]> {
  let summary = "get addr of a buffer";
  let description = [{
    get addr of a buffer
  }];

  let arguments = (ins AnyMemRef:$src);
  let results = (outs Index:$output);

  let builders = [
    OpBuilder<(ins "Value":$src)>
  ];

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

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$out_n,
    AnyTypeOf<[AnyInteger, Index]>:$mask_addr,
    AnyMemRef:$dst
  );

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

def CIMSetOp : CIM_Op<"cim_set"> {
  let summary = "cim set";
  let description = [{
    cim set
  }];

  let arguments = (ins 
    AnyMemRef:$mask
  );

  let hasFolder = 1;
}

def SendOp : CIM_Op<"send"> {
  let summary = "send";
  let description = [{
    send
  }];

  let arguments = (ins 
    AnyMemRef:$src_buffer,
    AnyMemRef:$dst_buffer,
    AnyTypeOf<[AnyInteger, Index]>:$core_id,
    AnyTypeOf<[AnyInteger, Index]>:$transfer_id
  );

  let hasFolder = 1;
}

def RecvOp : CIM_Op<"recv"> {
  let summary = "recv";
  let description = [{
    recv
  }];

  let arguments = (ins 
    AnyMemRef:$src_buffer,
    AnyMemRef:$dst_buffer,
    AnyTypeOf<[AnyInteger, Index]>:$core_id,
    AnyTypeOf<[AnyInteger, Index]>:$transfer_id
  );

  let hasFolder = 1;
}


#endif // CIM_OPS