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

