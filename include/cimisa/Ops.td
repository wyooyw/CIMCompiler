
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

def VVMulOp : CIMISA_Op<"vvmul"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "mul" operation performs element-wise addition between two tensors.
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

def QuantifyOp : CIMISA_Op<"quantify"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$input_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$out_zp_addr,
    AnyTypeOf<[AnyInteger, Index]>:$output_addr,
    AnyTypeOf<[AnyInteger, Index]>:$size,
    I1Attr:$relu
  );
}

def ResAddQuantifyOp : CIMISA_Op<"res_add_quantify"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$input_1_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$input_2_addr,
    AnyTypeOf<[AnyInteger, Index]>:$output_addr,
    AnyTypeOf<[AnyInteger, Index]>:$size
  );
}

def ResMulQuantifyOp : CIMISA_Op<"res_mul_quantify"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$input_1_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$input_2_addr,
    AnyTypeOf<[AnyInteger, Index]>:$output_addr,
    AnyTypeOf<[AnyInteger, Index]>:$size
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
    AnyTypeOf<[AnyInteger, Index]>:$row_index,
    AnyTypeOf<[AnyInteger, Index]>:$input_size,
    I1Attr:$acc_flag,
    I1Attr:$value_sparse_flag,
    I1Attr:$bit_sparse_flag
  );

}

def CIMOutputOp : CIMISA_Op<"cim_output"> {
  let summary = "cim output and clear";
  let description = [{
    cim output and clear
  }];

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$out_n,
    AnyTypeOf<[AnyInteger, Index]>:$out_mask_addr,
    AnyTypeOf<[AnyInteger, Index]>:$output_addr
  );

}

def CIMOutputSumOp : CIMISA_Op<"cim_output_sum"> {
  let summary = "cim output and clear";
  let description = [{
    cim output and clear
  }];

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$out_n,
    AnyTypeOf<[AnyInteger, Index]>:$out_mask_addr,
    AnyTypeOf<[AnyInteger, Index]>:$output_addr
  );

}

          

def CIMTransferOp : CIMISA_Op<"cim_transfer"> {
  let summary = "cim output and clear";
  let description = [{
    cim output and clear
  }];

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$src_addr,
    AnyTypeOf<[AnyInteger, Index]>:$output_number, 
    AnyTypeOf<[AnyInteger, Index]>:$output_mask_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$buffer_addr, 
    AnyTypeOf<[AnyInteger, Index]>:$dst_addr
  );

}

def CIMSetOp : CIMISA_Op<"cim_set"> {
  let summary = "cim set";
  let description = [{
    cim set
  }];

  let arguments = (ins 
    AnyTypeOf<[AnyInteger, Index]>:$mask_addr
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
    AnyTypeOf<[AnyInteger, Index]>:$value,
    IndexAttr:$constant
  );
}
def StoreBaseAndOffsetOp : CIMISA_Op<"store_base_and_offset"> {
  let summary = "trans";
  let description = [{
    store
  }];

  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$base, 
    AnyTypeOf<[AnyInteger, Index]>:$offset, 
    AnyTypeOf<[AnyInteger, Index]>:$value
    //IndexAttr:$imm
  );
}
// TODO: mark source addr is global or local.
def LoadOp : CIMISA_Op<"load"> {
  let summary = "load";
  let description = [{
    load
  }];

  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$src_addr,
    IndexAttr:$constant
  );
  let results = (outs AnyTypeOf<[AnyInteger, Index]>);
}
def LoadBaseAndOffsetOp : CIMISA_Op<"load_base_and_offset"> {
  let summary = "trans";
  let description = [{
    load
  }];

  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$base, 
    AnyTypeOf<[AnyInteger, Index]>:$offset
  );
  let results = (outs AnyTypeOf<[AnyInteger, Index]>);
}
def GeneralRegLiOp : CIMISA_Op<"general_reg_li", [Pure]> {
  let summary = "load immediate into general reg";
  let description = [{
    load immediate into general reg
  }];

  let arguments = (ins
    IndexAttr:$value
  );
  let results = (outs AnyTypeOf<[AnyInteger, Index]>);
}
def SpecialRegLiOp : CIMISA_Op<"special_reg_li"> {
  let summary = "load immediate into special reg";
  let description = [{
    load immediate into special reg
  }];

  let arguments = (ins
    I32Attr:$special_reg,
    I32Attr:$set_value
  );
}
def SpecialRegAssignOp : CIMISA_Op<"special_reg_assign"> {
  let summary = "load immediate into special reg";
  let description = [{
    load immediate into special reg
  }];

  let arguments = (ins
    I32Attr:$special_reg,
    AnyTypeOf<[AnyInteger, Index]>:$set_value
  );
}

// RI Ops

def RIAddIOp : CIMISA_Op<"ri_addi",[Pure]> {
  let summary = "ri_addi";
  let description = [{
    ri_addi
  }];
  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$value,
    IndexAttr:$constant
  );
  let results = (outs AnyTypeOf<[AnyInteger, Index]>);
}

def RISubIOp : CIMISA_Op<"ri_subi",[Pure]> {
  let summary = "ri_subi";
  let description = [{
    ri_subi
  }];
  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$value,
    IndexAttr:$constant
  );
  let results = (outs AnyTypeOf<[AnyInteger, Index]>);
}

def RIMulIOp : CIMISA_Op<"ri_muli",[Pure]> {
  let summary = "ri_muli";
  let description = [{
    ri_muli
  }];
  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$value,
    IndexAttr:$constant
  );
  let results = (outs AnyTypeOf<[AnyInteger, Index]>);
}

def RIDivSIOp : CIMISA_Op<"ri_divsi",[Pure]> {
  let summary = "ri_divsi";
  let description = [{
    ri_divsi
  }];
  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$value,
    IndexAttr:$constant
  );
  let results = (outs AnyTypeOf<[AnyInteger, Index]>);
}

def RIRemSIOp : CIMISA_Op<"ri_remsi",[Pure]> {
  let summary = "ri_remsi";
  let description = [{
    ri_remsi
  }];
  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$value,
    IndexAttr:$constant
  );
  let results = (outs AnyTypeOf<[AnyInteger, Index]>);
}

def RIMinSIOp : CIMISA_Op<"ri_minsi",[Pure]> {
  let summary = "ri_minsi";
  let description = [{
    ri_minsi
  }];
  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]>:$value,
    IndexAttr:$constant
  );
  let results = (outs AnyTypeOf<[AnyInteger, Index]>);
}

#endif // CIMISA_OPS