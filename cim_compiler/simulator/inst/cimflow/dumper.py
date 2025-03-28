import json
from cim_compiler.simulator.inst.instruction import *
from cim_compiler.utils.json_utils import (
    dumps_list_of_dict,
    dumps_dict_of_list_of_dict
)

class CIMFlowDumper:
    def __init__(self):
        pass

    def dump(self, instructions):
        data = []

        if isinstance(instructions[-1], (JumpInst, BranchInst)):
            no_op_inst = RIInst(opcode=0, reg_in=0, reg_out=0, imm=0 )
            instructions.append(no_op_inst)
        
        for inst in instructions:
            inst_dict = self._convert_to_dict(inst)
            data.append(inst_dict)

        return data

    def dump_str(self, instructions, core_id=None):
        data = self.dump(instructions)
        if core_id is None:
            return dumps_list_of_dict(data)
        else:
            return dumps_dict_of_list_of_dict({core_id: data})

    def dump_to_file(self, instructions, path, core_id=None):
        with open(path, 'w') as file:
            data_str = self.dump_str(instructions, core_id)
            file.write(data_str)

    def _convert_to_dict(self, inst):
        if isinstance(inst, GeneralLiInst):
            return {
                "opcode": 0b101100,
                "rd": inst.reg,
                "imm": inst.value
            }
        elif isinstance(inst, SpecialLiInst):
            return {
                "opcode": 0b101101,
                "rd": inst.reg,
                "imm": inst.value
            }
        elif isinstance(inst, RRInst):
            return {
                "opcode": 0b100000,
                "rs": inst.reg_lhs,
                "rt": inst.reg_rhs,
                "rd": inst.reg_out,
                "funct": inst.opcode
            }
        elif isinstance(inst, RIInst):
            return {
                "opcode": 0b100100,
                "rs": inst.reg_in,
                "rd": inst.reg_out,
                "funct": inst.opcode,
                "imm": inst.imm
            }
        elif isinstance(inst, SIMDInst):
            assert inst.input_num >=1 and inst.input_num <= 4
            return {
                "opcode": 0b010000 + (inst.input_num - 1 << 2),
                "rs": inst.reg_in1,
                "rt": inst.reg_in2,
                "rd": inst.reg_out,
                "re": inst.reg_size,
                "funct": inst.opcode
            }
        elif isinstance(inst, TransInst):
            assert inst.flag_src_offset in [0, 1]
            assert inst.flag_dst_offset in [0, 1]
            return {
                "opcode": 0b110000 + (inst.flag_src_offset << 1) + inst.flag_dst_offset,
                "rs": inst.reg_in,
                "rt": inst.reg_size,
                "rd": inst.reg_out,
                "imm": inst.offset
            }
        elif isinstance(inst, LoadInst):
            return {
                "opcode": 0b101000,
                "rs": inst.reg_addr,
                "rt": inst.reg_value,
                "imm": inst.offset
            }
        elif isinstance(inst, StoreInst):
            return {
                "opcode": 0b101001,
                "rs": inst.reg_addr,
                "rt": inst.reg_value,
                "imm": inst.offset
            }
        elif isinstance(inst, PrintInst):
            return {
                "opcode": -1,
                "rs": inst.reg
            }
        elif isinstance(inst, DebugInst):
            return {
                "opcode": -2
            }
        elif isinstance(inst, BranchInst):
            assert inst.compare >=0  and inst.compare <= 3
            return {
                "opcode": 0b111000 + inst.compare,
                "rs": inst.reg_lhs,
                "rt": inst.reg_rhs,
                "imm": inst.offset
            }
        elif isinstance(inst, JumpInst):
            return {
                "opcode": 0b111100,
                "imm": inst.offset
            }
        elif isinstance(inst, GeneralToSpecialAssignInst):
            return {
                "opcode": 0b101110,
                "rs": inst.reg_general,
                "rd": inst.reg_special
            }
        elif isinstance(inst, SpecialToGeneralAssignInst):
            return {
                "opcode": 0b101111,
                "rs": inst.reg_special,
                "rd": inst.reg_general
            }
        elif isinstance(inst, CIMComputeInst):
            return {
                "opcode": 0b000000,
                "rs": inst.reg_input_addr,
                "rt": inst.reg_input_size,
                "re": inst.reg_activate_row,
                "SP_V": inst.flag_value_sparse,
                "SP_B": inst.flag_bit_sparse,
                "GRP": inst.flag_group,
                "GRP_I": inst.flag_group_input_mode
            }
        elif isinstance(inst, CIMConfigInst):
            return {
                "opcode": 0b000100,
                "rs": inst.reg_single_group_id,
                "rt": inst.reg_mask_addr,
                "GRP_B": inst.flag_group_broadcast
            }
        elif isinstance(inst, CIMOutputInst):
            return {
                "opcode": 0b001000,
                "rs": inst.reg_out_n,
                "rt": inst.reg_out_mask_addr,
                "rd": inst.reg_out_addr,
                "OSUM": inst.flag_outsum,
                "OSUM_MOV": inst.flag_outsum_move
            }
        elif isinstance(inst, SendInst):
            return {
                "opcode": 0b110100,
                "rs": inst.reg_src_addr,
                "rt": inst.reg_dst_core,
                "rd": inst.reg_dst_addr,
                "re": inst.reg_size,
                "rf": inst.reg_transfer_id
            }
        elif isinstance(inst, RecvInst):
            return {
                "opcode": 0b110110,
                "rs": inst.reg_src_core,
                "rt": inst.reg_src_addr,
                "rd": inst.reg_dst_addr,
                "re": inst.reg_size,
                "rf": inst.reg_transfer_id
            }
        else:
            raise ValueError(f"Unknown instruction type: {type(inst)}")
        