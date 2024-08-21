from enum import Enum
import numpy as np
import os
import json
import copy

class SpecialReg(Enum):

    # pim special reg
    INPUT_BIT_WIDTH = 0
    OUTPUT_BIT_WIDTH = 1
    WEIGHT_BIT_WIDTH = 2
    GROUP_SIZE = 3
    ACTIVATION_GROUP_NUM = 4
    ACTIVATION_ELEMENT_COL_NUM = 5
    GROUP_INPUT_STEP = 6
    GROUP_INPUT_OFFSET_ADDR = 6
    VALUE_SPARSE_MASK_ADDR = 7
    BIT_SPARSE_META_ADDR = 8

    # simd special reg
    SIMD_INPUT_1_BIT_WIDTH = 16
    SIMD_INPUT_2_BIT_WIDTH = 17
    SIMD_INPUT_3_BIT_WIDTH = 18
    SIMD_INPUT_4_BIT_WIDTH = 19
    SIMD_OUTPUT_BIT_WIDTH = 20
    SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_1 = 21
    SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_2 = 22

class FlatInstUtil:
    def __init__(self, general_rf, special_rf):
        self.general_rf = general_rf
        self.special_rf = special_rf
        
        self.flat_general_rf = np.zeros([32], dtype=np.int32)
        self.flat_special_rf = np.zeros([32], dtype=np.int32)

        self.flat_inst_list = []

    def get_flat_code(self):
        return copy.deepcopy(self.flat_inst_list)

    def _li_general_inst(self, reg, imm):
        return {
            "class": 2,
            "type": 3,
            "opcode": 0,
            "rd": reg,
            "imm": imm.item()
        }

    def _li_special_inst(self, reg, imm):
        return {
            "class": 2,
            "type": 3,
            "opcode": 1,
            "rd": reg,
            "imm": imm.item()
        }
    
    def _load_general_regs(self, regs):
        """
        通用寄存器立即数赋值指令：general-li
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为10
        - [29, 28]，2bit：type，指令类型码，值为11
        - [27, 26]，2bit：opcode，指令操作码，值为00
        - [25, 21]，5bit：rd，通用寄存器编号，即要赋值的通用寄存器
        - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
        """
        assert type(regs)==list
        for reg in regs:
            if not self.flat_general_rf[reg] == self.general_rf[reg]:
                self.flat_general_rf[reg] = self.general_rf[reg]
                self.flat_inst_list.append(self._li_general_inst(reg, self.general_rf[reg]))

    def _load_special_regs(self, regs):
        """
        专用寄存器立即数赋值指令：special-li
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为10
        - [29, 28]，2bit：type，指令类型码，值为11
        - [27, 26]，2bit：opcode，指令操作码，值为01
        - [25, 21]，5bit：rd，专用寄存器编号，即要赋值的通用寄存器
        - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
        """
        assert type(regs)==list
        for reg in regs:
            reg = reg.value
            if not self.flat_special_rf[reg] == self.special_rf[reg]:
                self.flat_special_rf[reg] = self.special_rf[reg]
                self.flat_inst_list.append(self._li_special_inst(reg, self.special_rf[reg]))

    def flat_inst(self, inst):
        if inst["class"]==0b110 and inst["type"]==0b0: # trans
            self._flat_trans(inst)
        elif inst["class"]==0b00 and inst["type"]==0b00: # pim_compute
            self._flat_pim_compute(inst)
        elif inst["class"]==0b00 and inst["type"]==0b10: # pim_output
            # self._flat_pim_output(inst)
            pass
        elif inst["class"]==0b00 and inst["type"]==0b11: # pim_transfer
            # self._flat_pim_transfer(inst)
            pass
        elif inst["class"]==0b01: # simd
            self._flat_simd(inst)
        else:
            assert inst["class"] in [
                0b10,  # scalar
                0b110, # trans
                0b111  # branch
            ], f"Unsupported instruction class: {inst['class']}"

    def _flat_trans(self, inst):
        self._load_general_regs([inst["rs1"], inst["rs2"], inst["rd"]])
        self.flat_inst_list.append(inst)
    
    def _flat_pim_compute(self, inst):
        self._load_general_regs([inst["rs1"], inst["rs2"], inst["rs3"]])
        self._load_special_regs([
            SpecialReg.INPUT_BIT_WIDTH,
            SpecialReg.OUTPUT_BIT_WIDTH,
            SpecialReg.WEIGHT_BIT_WIDTH,
            SpecialReg.ACTIVATION_ELEMENT_COL_NUM,
            SpecialReg.ACTIVATION_GROUP_NUM,
            SpecialReg.GROUP_SIZE,
            SpecialReg.GROUP_INPUT_STEP,
            SpecialReg.VALUE_SPARSE_MASK_ADDR,
            SpecialReg.BIT_SPARSE_META_ADDR
        ])
        self.flat_inst_list.append(inst)

    def _flat_simd(self, inst):
        self._load_general_regs([inst["rs1"], inst["rs2"], inst["rs3"], inst["rd"]])
        self._load_special_regs([
            SpecialReg.SIMD_INPUT_1_BIT_WIDTH,
            SpecialReg.SIMD_INPUT_2_BIT_WIDTH,
            SpecialReg.SIMD_INPUT_3_BIT_WIDTH,
            SpecialReg.SIMD_INPUT_4_BIT_WIDTH,
            SpecialReg.SIMD_OUTPUT_BIT_WIDTH,
            SpecialReg.SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_1,
            SpecialReg.SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_2
        ])
        self.flat_inst_list.append(inst)

    def _flat_pim_output(self, inst):
        self._load_general_regs([inst["rs1"], inst["rs2"], inst["rd"]])
        self._load_special_regs([
            SpecialReg.WEIGHT_BIT_WIDTH,
            SpecialReg.OUTPUT_BIT_WIDTH,
            SpecialReg.GROUP_SIZE,
            SpecialReg.ACTIVATION_GROUP_NUM,
        ])
        self.flat_inst_list.append(inst)

    def _flat_pim_transfer(self, inst):
        self._load_general_regs([inst["rs1"], inst["rs2"], inst["rs3"], inst["rs4"], inst["rd"]])
        self._load_special_regs([
            SpecialReg.OUTPUT_BIT_WIDTH
        ])
        self.flat_inst_list.append(inst)

    def dump(self, out_dir):
        file_path = os.path.join(out_dir, "flat_code.json")
        with open(file_path, "w") as f:
            f.write("[\n")
            for i,inst in enumerate(self.flat_inst_list):
                str_inst = json.dumps(inst)
                f.write(str_inst)
                if i < len(self.flat_inst_list)-1:
                    f.write(",")
                f.write("\n")
            f.write("]\n")
        print("Flatten code saved to", file_path)