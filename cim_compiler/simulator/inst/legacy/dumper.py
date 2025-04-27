import json
from cim_compiler.simulator.inst.instruction import *
from cim_compiler.utils.json_utils import (
    dumps_list_of_dict,
    dumps_dict_of_list_of_dict
)

class LegacyDumper:
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

    def dump_str(self, instructions, core_id=None, curly=True):
        data = self.dump(instructions)
        if core_id is None:
            return dumps_list_of_dict(data)
        else:
            return dumps_dict_of_list_of_dict({core_id: data}, curly=curly)

    def dump_to_file(self, instructions, path, core_id=None):
        with open(path, 'a') as file:
            data_str = self.dump_str(instructions, core_id)
            file.write(data_str)

    def _convert_to_dict(self, inst):
        if isinstance(inst, GeneralLiInst):
            return {
                "class": 0b10,
                "type": 0b11,
                "opcode": 0b00,
                "rd": inst.reg,
                "imm": inst.value
            }
        elif isinstance(inst, SpecialLiInst):
            return {
                "class": 0b10,
                "type": 0b11,
                "opcode": 0b01,
                "rd": inst.reg,
                "imm": inst.value
            }
        elif isinstance(inst, GeneralToSpecialAssignInst):
            return {
                "class": 0b10,
                "type": 0b11,
                "opcode": 0b10,
                "rs1": inst.reg_general,
                "rs2": inst.reg_special
            }
        elif isinstance(inst, SpecialToGeneralAssignInst):
            return {
                "class": 0b10,
                "type": 0b11,
                "opcode": 0b11,
                "rs1": inst.reg_general,
                "rs2": inst.reg_special
            }
        elif isinstance(inst, RRInst):
            return {
                "class": 0b10,
                "type": 0b00,
                "opcode": inst.opcode,
                "rs1": inst.reg_lhs,
                "rs2": inst.reg_rhs,
                "rd": inst.reg_out
            }
        elif isinstance(inst, RIInst):
            return {
                "class": 0b10,
                "type": 0b01,
                "opcode": inst.opcode,
                "rs": inst.reg_in,
                "rd": inst.reg_out,
                "imm": inst.imm
            }
        elif isinstance(inst, SIMDInst):
            return {
                "class": 0b01,
                "input_num": inst.input_num - 1,
                "opcode": inst.opcode,
                "rs1": inst.reg_in1,
                "rs2": inst.reg_in2,
                "rs3": inst.reg_size,
                "rd": inst.reg_out
            }
        elif isinstance(inst, TransInst):
            return {
                "class": 0b110,
                "type": 0b0,
                "source_offset_mask": int(inst.flag_src_offset),
                "destination_offset_mask": int(inst.flag_dst_offset),
                "rs1": inst.reg_in,
                "rd": inst.reg_out,
                "offset": inst.offset,
                "rs2": inst.reg_size
            }
        elif isinstance(inst, LoadInst):
            return {
                "class": 0b10,
                "type": 0b10,
                "opcode": 0b00,
                "rs1": inst.reg_addr,
                "rs2": inst.reg_value,
                "offset": inst.offset
            }
        elif isinstance(inst, StoreInst):
            return {
                "class": 0b10,
                "type": 0b10,
                "opcode": 0b01,
                "rs1": inst.reg_addr,
                "rs2": inst.reg_value,
                "offset": inst.offset
            }
        elif isinstance(inst, PrintInst):
            return {
                "class": -1,
                "type": 0,
                "rs": inst.reg
            }
        elif isinstance(inst, DebugInst):
            return {
                "class": -1,
                "type": 1
            }
        elif isinstance(inst, BranchInst):
            return {
                "class": 0b111,
                "type": inst.compare,
                "rs1": inst.reg_lhs,
                "rs2": inst.reg_rhs,
                "offset": inst.offset
            }
        elif isinstance(inst, JumpInst):
            return {
                "class": 0b111,
                "type": 0b100,
                "offset": inst.offset
            }
        elif isinstance(inst, CIMComputeInst):
            return {
                "class": 0b00,
                "type": 0b0,
                "value_sparse": int(inst.flag_value_sparse),
                "bit_sparse": int(inst.flag_bit_sparse),
                "group": int(inst.flag_group),
                "group_input_mode": int(inst.flag_group_input_mode),
                "accumulate": int(inst.flag_accumulate),
                "rs1": inst.reg_input_addr,
                "rs2": inst.reg_input_size,
                "rs3": inst.reg_activate_row
            }
        elif isinstance(inst, CIMConfigInst):
            return {
                "class": 0b00,
                "type": 0b01,
                "group_broadcast": int(inst.flag_group_broadcast),
                "rs1": inst.reg_single_group_id,
                "rs2": inst.reg_mask_addr
            }
        elif isinstance(inst, CIMOutputInst):
            return {
                "class": 0b00,
                "type": 0b10,
                "outsum_move": int(inst.flag_outsum_move),
                "outsum": int(inst.flag_outsum),
                "rs1": inst.reg_out_n,
                "rs2": inst.reg_out_mask_addr,
                "rd": inst.reg_out_addr
            }
        elif isinstance(inst, SendInst):
            return {
                "class": 0b110,
                "type": 0b10,
                "sync": 0,
                "rs": inst.reg_src_addr,
                "rd1": inst.reg_dst_core,
                "rd2": inst.reg_dst_addr,
                "reg_id": inst.reg_transfer_id,
                "reg_len": inst.reg_size
            }
        elif isinstance(inst, RecvInst):
            return {
                "class": 0b110,
                "type": 0b11,
                "sync": 0,
                "rs1": inst.reg_src_core,
                "rs2": inst.reg_src_addr,
                "rd": inst.reg_dst_addr,
                "reg_id": inst.reg_transfer_id,
                "reg_len": inst.reg_size
            }
        else:
            raise ValueError(f"Unknown instruction type: {type(inst)}")
        
