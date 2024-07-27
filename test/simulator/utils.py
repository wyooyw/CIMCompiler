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
                "mod": 0b111,
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
            "rs1": rs,
            "rd": rd,
            "offset": offset,
            "rs2": size
        }

    def pimcompute_value_sparse(self, accumulate, rs1, rs2, rs3, rd):
        return self.pimcompute(1, 0, group=1, group_input_mode=0, accumulate=accumulate, rs1=rs1, rs2=rs2, rs3=rs3, rd=rd)

    def pimcompute_dense_single_group(self, accumulate, rs1, rs2, rs3, rd):
        return self.pimcompute_dense(1, 0, accumulate, rs1, rs2, rs3, rd)

    def pimcompute_dense_multi_group(self, accumulate, rs1, rs2, rs3, rd):
        return self.pimcompute_dense(1, 0, accumulate, rs1, rs2, rs3, rd)

    def pimcompute_dense(self, group, group_input_mode, accumulate, rs1, rs2, rs3, rd):
        return self.pimcompute(0, 0, group, group_input_mode, accumulate, rs1, rs2, rs3, rd)
    
    def pimcompute(self, value_sparse, bit_sparse, group, group_input_mode, accumulate, rs1, rs2, rs3, rd):
        """
        pim计算：pim-compute
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为00
        - [29, 29]，1bit：type，指令类型码，值为0
        - [28, 25]，4bit：reserve，保留字段
        - [24, 20]，5bit：flag，功能扩展字段
            - [24]，1bit：value sparse，表示是否使用值稀疏，稀疏掩码Mask的起始地址由专用寄存器给出
            - [23]，1bit：bit sparse，表示是否使用bit级稀疏，稀疏Meta数据的起始地址由专用寄存器给出
            - [22]，1bit：group，表示是否进行分组，组大小及激活的组数量由专用寄存器给出
            - [21]，1bit：group input mode，表示多组输入的模式
                - 0：每一组输入向量的起始地址相对于上一组的增量（步长，step）是一个定值，由专用寄存器给出
                - 1：每一组输入向量的起始地址相对于上一组的增量不是定值，其相对于rs1的偏移量（offset）在存储器中给出，地址（offset addr）由专用寄存器给出
        - [20]，1bit：accumulate，表示是否进行累加
        - [19, 15]，5bit：rs1，通用寄存器1，表示input向量起始地址
        - [14, 10]，5bit：rs2，通用寄存器2，表示input向量长度
        - [9, 5]，5bit：rs3，通用寄存器3，表示激活的row的index
        - [4, 0]，5bit：rd，通用寄存器4，表示output写入的起始地址
        """
        # value_sparse, bit_sparse, group, group_input_mode, accumulate, rs1, rs2, rs3, rd
        return {
            "class": 0b00,
            "type": 0b0,
            "value_sparse": value_sparse,
            "bit_sparse": bit_sparse,
            "group":group,
            "group_input_mode": group_input_mode,
            "accumulate": accumulate,
            "rs1": rs1,
            "rs2": rs2,
            "rs3": rs3,
            "rd": rd
        }

    def simd_vvadd(self, rs1, rs2, rs3, rd):
        return self.simd(0x00, 0b01, rs1, rs2, rs3, rd)

    def simd_vsadd(self, rs1, rs2, rs3, rd):
        return self.simd(0x01, 0b00, rs1, rs2, rs3, rd)

    def simd(self, opcode, input_num, rs1, rs2, rs3, rd):
        """
        SIMD计算：SIMD-compute
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为01
        - [29, 28]，2bit：input num，input向量的个数，范围是1到4
            - 00：1个输入向量，地址由rs1给出
            - 01：2个输入向量，地址由rs1和rs2给出
            - 10：3个输入向量，地址由rs1，rs1+1，rs2给出
            - 11：4个输入向量，地址由rs1，rs1+1，rs2，rs2+1给出
        - [27, 20]，8bit：opcode，操作类别码，表示具体计算的类型
            - 0x00：add，向量加法
            - 0x01：add-scalar，向量和标量加法
            - 0x02：multiply，向量逐元素乘法
            - 0x03：quantify，量化
            - 0x04：quantify-resadd，resadd量化
            - 0x05：quantify-multiply，乘法量化
        - [19, 15]，5bit：rs1，通用寄存器1，表示input向量起始地址1
        - [14, 10]，5bit：rs2，通用寄存器2，表示input向量起始地址2
        - [9, 5]，5bit：rs3，通用寄存器3，表示input向量长度
        - [4, 0]，5bit：rd，通用寄存器4，表示output写入的起始地址
        使用的专用寄存器：
        - input 1 bit width：输入向量1每个元素的bit长度
        - input 2 bit width：输入向量2每个元素的bit长度
        - input 3 bit width：输入向量3每个元素的bit长度
        - input 4 bit width：输入向量4每个元素的bit长度
        - output bit width：输出向量每个元素的bit长度
        """
        return {
            "class": 0b01,
            "input_num": input_num,
            "opcode": opcode,
            "rs1": rs1,
            "rs2": rs2,
            "rs3": rs3,
            "rd": rd
        }

    def debug_print(self, rd):
        return {
            "class": -1,
            "rd": rd
        }

