import json
from cim_compiler.simulator.inst.instruction import *
from cim_compiler.simulator.inst.legacy.dumper import LegacyDumper
from cim_compiler.simulator.inst.asm.dumper import AsmDumper
from cim_compiler.simulator.inst.asm.parser import AsmParser
class CIMFlowParser:
    def __init__(self):
        pass

    def parse_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data, self.parse(data)

    def parse(self, data):

        if isinstance(data, dict):
            assert len(data.keys()) == 1
            data = data.values()[0]
        
        instructions = []
        for item in data:
            inst = self._parse_inst(item)
            instructions.append(inst)
        return instructions
    
    def _parse_inst(self, inst):
        opcode = inst.get("opcode")

        if opcode == 0b101100:
            return GeneralLiInst(
                reg=inst["rd"],
                value=inst["imm"]
            )
        elif opcode == 0b101101:
            return SpecialLiInst(
                reg=inst["rd"],
                value=inst["imm"]
            )
        elif opcode == 0b100000:
            return RRInst(
                opcode=inst["funct"],
                reg_lhs=inst["rs"],
                reg_rhs=inst["rt"],
                reg_out=inst["rd"]
            )
        elif opcode == 0b100100:
            return RIInst(
                opcode=inst["funct"],
                reg_in=inst["rs"],
                reg_out=inst["rd"],
                imm=inst["imm"]
            )
        elif opcode in (
            0b010000,
            0b010100,
            0b011000,
            0b011100,
        ):
            input_num = ((opcode & 0b1100) >> 2) + 1
            return SIMDInst(
                opcode=inst["funct"],
                reg_in1=inst["rs"],
                reg_in2=inst["rt"],
                reg_size=inst["re"],
                reg_out=inst["rd"],
                input_num=input_num
            )
        elif (opcode & 0b111100) == 0b110000:
            flag_src_offset = (opcode & 0b10) >> 1
            flag_dst_offset = (opcode & 0b1)
            # print(inst)
            return TransInst(
                reg_in=inst["rs"],
                reg_size=inst["rt"],
                reg_out=inst["rd"],
                flag_src_offset=flag_src_offset,
                flag_dst_offset=flag_dst_offset,
                offset=inst["imm"]
            )
        elif opcode == 0b101000:
            return LoadInst(
                reg_addr=inst["rs"],
                reg_value=inst["rt"],
                offset=inst["imm"]
            )
        elif opcode == 0b101001:
            return StoreInst(
                reg_addr=inst["rs"],
                reg_value=inst["rt"],
                offset=inst["imm"]
            )
        elif opcode == -1:
            return PrintInst(
                reg=inst["rs"]
            )
        elif opcode == -2:
            return DebugInst()
        elif opcode >= 0b111000 and opcode <= 0b111011:
            compare = opcode - 0b111000
            return BranchInst(
                compare=compare,
                reg_lhs=inst["rs"],
                reg_rhs=inst["rt"],
                offset=inst["imm"]
            )
        elif opcode == 0b111100:
            return JumpInst(
                offset=inst["imm"]
            )
        elif opcode == 0b101110:
            return GeneralToSpecialAssignInst(
                reg_general=inst["rs"],
                reg_special=inst["rd"]
            )
        elif opcode == 0b101111:
            return SpecialToGeneralAssignInst(
                reg_special=inst["rs"],
                reg_general=inst["rd"]
            )
        elif opcode == 0b000000:
            return CIMComputeInst(
                reg_input_addr=inst["rs"],
                reg_input_size=inst["rt"],
                reg_activate_row=inst["re"],
                flag_value_sparse=inst["SP_V"],
                flag_bit_sparse=inst["SP_B"],
                flag_group=inst["GRP"],
                flag_group_input_mode=inst["GRP_I"],
                flag_accumulate=inst["ACC"]
            )
        elif opcode == 0b000100:
            return CIMConfigInst(
                reg_single_group_id=inst["rs"],
                reg_mask_addr=inst["rt"],
                flag_group_broadcast=inst["GRP_B"]
            )
        elif opcode == 0b001000:
            return CIMOutputInst(
                reg_out_n=inst["rs"],
                reg_out_mask_addr=inst["rt"],
                reg_out_addr=inst["rd"],
                flag_outsum=inst["OSUM"],
                flag_outsum_move=inst["OSUM_MOV"]
            )
        elif opcode == 0b001100:
            return CIMTransferInst(
                reg_src_addr=inst["rs"],
                reg_out_n=inst["rt"],
                reg_out_mask_addr=inst["re"],
                reg_buffer_addr=inst["rf"],
                reg_dst_addr=inst["rd"]
            )
        elif opcode == 0b110100:
            return SendInst(
                reg_src_addr=inst["rs"],
                reg_dst_addr=inst["rd"],
                reg_size=inst["re"],
                reg_dst_core=inst["rt"],
                reg_transfer_id=inst["rf"]
            )
        elif opcode == 0b110110:
            return RecvInst(
                reg_src_addr=inst["rt"],
                reg_dst_addr=inst["rd"],
                reg_size=inst["re"],
                reg_src_core=inst["rs"],
                reg_transfer_id=inst["rf"]
            )
        else:
            raise ValueError(f"Unknown opcode: {opcode}")
        