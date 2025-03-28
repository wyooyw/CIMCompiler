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
class RRInst:
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
    flag_src_offset: bool
    flag_dst_offset: bool
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
    flag_accumulate: bool
    flag_value_sparse: bool
    flag_bit_sparse: bool
    flag_group: bool
    flag_group_input_mode: bool

@dataclass
class CIMConfigInst:
    reg_single_group_id: int
    reg_mask_addr: int
    flag_group_broadcast: bool

@dataclass
class CIMOutputInst:
    reg_out_n: int
    reg_out_mask_addr: int
    reg_out_addr: int
    flag_outsum: bool
    flag_outsum_move: bool

@dataclass
class CIMTransferInst:
    reg_src_addr: int
    reg_out_n: int
    reg_out_mask_addr: int
    reg_buffer_addr: int
    reg_dst_addr: int

@dataclass
class SendInst:
    reg_src_addr: int
    reg_dst_addr: int
    reg_size: int
    reg_dst_core: int
    reg_transfer_id: int

@dataclass
class RecvInst:
    reg_src_addr: int
    reg_dst_addr: int
    reg_size: int
    reg_src_core: int
    reg_transfer_id: int
