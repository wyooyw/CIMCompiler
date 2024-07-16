class InstUtil:
    def __init__(self):
        pass
    
    def general_li(self,rd,imm):
        return {
            "class": 0b10,
            "type": 0b11,
            "opcode": 0b00,
            "rd": rd,
            "imm": imm
        }
    def special_li(self,rd,imm):
        return {
            "class": 0b10,
            "type": 0b11,
            "opcode": 0b01,
            "rd": rd,
            "imm": imm
        }
    def special_to_general(self,general_reg,special_reg):
        return {
            "class": 0b10,
            "type": 0b11,
            "opcode": 0b11,
            "rs1": general_reg,
            "rs2": special_reg
        }
    def general_to_special(self,general_reg,special_reg):
        return {
            "class": 0b10,
            "type": 0b11,
            "opcode": 0b10,
            "rs1": general_reg,
            "rs2": special_reg
        }
    def scalar_rr(self,rs1,rs2,rd,opcode):
        if type(opcode)==str:
            opcode = {
                "add": 0b000,
                "sub": 0b001,
                "mul": 0b010,
                "div": 0b011,
                "sll": 0b100,
                "srl": 0b101,
                "sra": 0b110,
            }[opcode]
        return {
            "class": 0b10,
            "type": 0b00,
            "opcode": opcode,
            "rs1": rs1,
            "rs2": rs2,
            "rd": rd
        }

    def jump(self, offset):
        return {
            "class": 0b111,
            "type": 0b100,
            "offset": offset
        }

    def branch(self, cmp, rs1, rs2, offset):
        if type(cmp)==str:
            cmp = {
                "beq": 0b000,
                "bne": 0b001,
                "bgt": 0b010,
                "blt": 0b011
            }[cmp]
        return {
            "class": 0b111,
            "type": cmp,
            "rs1": rs1,
            "rs2": rs2,
            "offset": offset
        }

    def trans(self, rs, rd, size, src_offset_mask=0, dst_offset_mask=0, offset=0):
        return {
            "class": 0b110,
            "type": 0b0,
            "source_offset_mask": src_offset_mask,
            "destination_offset_mask": dst_offset_mask,
            "rs": rs,
            "rd": rd,
            "offset": offset,
            "size": size
        }
    
    def debug_print(self, rd):
        return {
            "class": -1,
            "rd": rd
        }

