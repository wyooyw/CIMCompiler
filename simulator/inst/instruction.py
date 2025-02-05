from dataclasses import dataclass

@dataclass
class GeneralLiInst:
    reg: int
    value: int

@dataclass
class SpecialLiInst:
    reg: int
    value: int

@dataclass
class SpecialToGeneralAssignInst:
    reg_general: int
    reg_special: int

@dataclass
class GeneralToSpecialAssignInst:
    reg_general: int
    reg_special: int

@dataclass
class ArithInst:
    opcode: int
    reg_lhs: int
    reg_rhs: int
    reg_out: int

@dataclass
class RIInst:
    opcode: int
    reg_in: int
    reg_out: int
    imm: int

@dataclass
class SIMDInst:
    opcode: int
    input_num: int
    reg_in1: int
    reg_in2: int
    reg_size: int
    reg_out: int

@dataclass
class TransInst:
    reg_in: int
    reg_out: int
    reg_size: int
    flag_src_offset: int
    flag_dst_offset: int
    offset: int

@dataclass
class LoadInst:
    reg_addr: int
    reg_value: int
    offset: int

@dataclass
class StoreInst:
    reg_addr: int
    reg_value: int
    offset: int

@dataclass
class PrintInst:
    reg: int

@dataclass
class DebugInst:
    pass

@dataclass
class BranchInst:
    compare: int
    reg_lhs: int
    reg_rhs: int
    offset: int

@dataclass
class JumpInst:
    offset: int

@dataclass
class CIMComputeInst:
    reg_input_addr: int
    reg_input_size: int
    reg_activate_row: int
    flag_accumulate: int
    flag_value_sparse: int
    flag_bit_sparse: int
    flag_group: int
    flag_group_input_mode: int

@dataclass
class CIMConfigInst:
    reg_single_group_id: int
    reg_mask_addr: int
    flag_group_broadcast: int

@dataclass
class CIMOutputInst:
    reg_out_n: int
    reg_out_mask_addr: int
    reg_out_addr: int
    flag_outsum: int
    flag_outsum_move: int

@dataclass
class CIMTransferInst:
    reg_src_addr: int
    reg_out_n: int
    reg_out_mask_addr: int
    reg_buffer_addr: int
    reg_dst_addr: int
