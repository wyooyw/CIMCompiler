
#ifndef CIMISA_OPS
#define CIMISA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def CIMISA_Dialect : Dialect {
  let name = "cimisa";
  let cppNamespace = "::mlir::cimisa";
}

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class CIMISA_Op<string mnemonic, list<Trait> traits = []> :
    Op<CIMISA_Dialect, mnemonic, traits>;


/////////////////////////////////
// Vector Compute //
/////////////////////////////////

def VVAddOp : CIMISA_Op<"vvadd"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$lhs_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$rhs_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$out_addr,
    AnyTypeOf<[AnyInteger, Index]>:$size,
    I8Attr:$lhs_bw, // bw: bit width
    I8Attr:$rhs_bw, 
    I8Attr:$out_bw
  );
}
/////////////////////////////////
// CIM Compute //
/////////////////////////////////

def CIMComputeOp : CIMISA_Op<"cim_compute"> {
  let summary = "cim compute";
  let description = [{
    cim compute use macro
  }];

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$input_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$output_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$row_index,
    AnyTypeOf<[AnyInteger, Index]>:$activate_element_num,
    I8Attr:$input_bw, // bw: bit width
    I8Attr:$output_bw, 
    I8Attr:$weight_bw,
    I1Attr:$acc_flag,
    I1Attr:$value_sparse_flag,
    I1Attr:$bit_sparse_flag
  );

}


/////////////////////////////////
// Other Operator //
/////////////////////////////////

def TransOp : CIMISA_Op<"trans"> {
  let summary = "trans";
  let description = [{
    trans
  }];

  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$src_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$dst_addr,
    AnyTypeOf<[AnyInteger, Index]>:$size
  );
}

def BranchOp : CIMISA_Op<"branch"> {
  let summary = "branch";
  let description = [{
    if condition satisfied, jump to 'trueDestOperands'.
    otherwise, run next op.
  }];

  let arguments = (ins
    I8:$compare, 
    AnyTypeOf<[AnyInteger, Index]>:$lhs, 
    AnyTypeOf<[AnyInteger, Index]>:$rhs,
    Variadic<AnyType>:$trueDestOperands
  );
}

def JumpOp: CIMISA_Op<"jump"> {
  let summary = "jump";
  let description = [{
    if condition satisfied, jump to 'trueDestOperands'.
    otherwise, run next op.
  }];

  let arguments = (ins Variadic<AnyType>:$trueDestOperands);
}

def StoreOp : CIMISA_Op<"store"> {
  let summary = "trans";
  let description = [{
    store
  }];

  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$dst_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$value
  );
}
// TODO: mark source addr is global or local.
def LoadOp : CIMISA_Op<"load"> {
  let summary = "load";
  let description = [{
    load
  }];

  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$src_addr
  );
  let results = (outs AnyTypeOf<[AnyInteger, Index]>);
}

#endif // CIMISA_OPS