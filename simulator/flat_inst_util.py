from enum import Enum
import numpy as np
import os
import json
import copy
import os

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
        self.last_access_time = np.zeros([32], dtype=np.int32)

        self.flat_inst_list = []

        self.max_int32 = np.iinfo(np.int32).max


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
    
    def _load_general_regs(self, inst, regs_name, idx):
        """
        通用寄存器立即数赋值指令：general-li
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为10
        - [29, 28]，2bit：type，指令类型码，值为11
        - [27, 26]，2bit：opcode，指令操作码，值为00
        - [25, 21]，5bit：rd，通用寄存器编号，即要赋值的通用寄存器
        - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
        """
        regs = [inst[reg] for reg in regs_name]

        assert type(regs)==list
        for reg in regs:
            if not self.flat_general_rf[reg] == self.general_rf[reg]:
                self.flat_general_rf[reg] = self.general_rf[reg]
                self.flat_inst_list.append(self._li_general_inst(reg, self.general_rf[reg]))


    # def _load_general_regs(self, inst, regs_name, idx):
    #     """
    #     通用寄存器立即数赋值指令：general-li
    #     指令字段划分：
    #     - [31, 30]，2bit：class，指令类别码，值为10
    #     - [29, 28]，2bit：type，指令类型码，值为11
    #     - [27, 26]，2bit：opcode，指令操作码，值为00
    #     - [25, 21]，5bit：rd，通用寄存器编号，即要赋值的通用寄存器
    #     - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
    #     """
        
    #     regs = [inst[reg] for reg in regs_name]
        
    #     assert type(regs)==list
    #     new_regs = copy.deepcopy(regs)
    #     need_alloc_regs = []
    #     use_old_regs = []
    #     for i_reg, reg in enumerate(regs):
    #         need_value = self.general_rf[reg]
    #         # if need_value==660480:
    #         #     import pdb; pdb.set_trace()
    #         if (self.flat_general_rf==need_value).any():
    #             # get index
    #             new_reg = np.where(self.flat_general_rf==need_value)[0][0]
    #             new_regs[i_reg] = new_reg.item()
    #             use_old_regs.append(i_reg)
    #         else:
    #             need_alloc_regs.append(i_reg)
    #     # if self.general_rf[regs[0]]==883840:
    #     #     import pdb; pdb.set_trace()
    #     last_access_time = self.last_access_time.copy()
    #     for lock_reg in use_old_regs:
    #         last_access_time[new_regs[lock_reg]] = self.max_int32
    #     for i_reg in need_alloc_regs:
    #         need_value = self.general_rf[regs[i_reg]]
    #         use_reg = last_access_time.argmin().item()
    #         last_access_time[use_reg] = self.max_int32
    #         self.flat_general_rf[use_reg] = need_value
    #         self.flat_inst_list.append(self._li_general_inst(use_reg, need_value))
    #         new_regs[i_reg] = use_reg
            
    #     for i_reg, reg in enumerate(new_regs):
    #         self.last_access_time[reg] = idx
        
    #     origin_values = [self.general_rf[reg] for reg in regs]
    #     new_values = [self.flat_general_rf[reg] for reg in new_regs]
    #     assert origin_values==new_values, f"{origin_values=}, {new_values=}"
    #     # print([self.flat_general_rf[reg] for reg in new_regs])
    #     # print(f"{inst=}")
        
    #     for i_reg, reg in enumerate(regs_name):
    #         inst[reg] = new_regs[i_reg]
    #         # print(type(inst[reg]))
    #     # print(f"{inst=}")
    #     # import pdb; pdb.set_trace()
        

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

    def flat_inst(self, inst, idx):
        inst = copy.deepcopy(inst)
        if inst["class"]==0b110 and inst["type"]==0b0: # trans
            self._flat_trans(inst, idx)
        elif inst["class"]==0b00 and inst["type"]==0b00: # pim_compute
            self._flat_pim_compute(inst, idx)
        elif inst["class"]==0b00 and inst["type"]==0b01: # pim_set
            self._flat_pim_set(inst, idx)
        elif inst["class"]==0b00 and inst["type"]==0b10: # pim_output
            if int(os.environ.get("FAST_MODE"))==0:
                self._flat_pim_output(inst, idx)
        elif inst["class"]==0b00 and inst["type"]==0b11: # pim_transfer
            if int(os.environ.get("FAST_MODE"))==0:
                self._flat_pim_transfer(inst, idx)
        elif inst["class"]==0b01: # simd
            self._flat_simd(inst, idx)
        else:
            assert inst["class"] in [
                0b10,  # scalar
                0b110, # trans
                0b111,  # branch
                -1,     #bebug
            ], f"Unsupported instruction class: {inst['class']}"

    def _flat_trans(self, inst, idx):
        self._load_general_regs(inst, ["rs1", "rs2", "rd"], idx)
        self.flat_inst_list.append(inst)
    
    def _flat_pim_compute(self, inst, idx):
        self._load_general_regs(inst, ["rs1", "rs2", "rs3"], idx)
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

    def _flat_pim_set(self, inst, idx):
        self._load_general_regs(inst, ["rs2"], idx)
        self._load_special_regs([
            SpecialReg.WEIGHT_BIT_WIDTH,
            SpecialReg.GROUP_SIZE
        ])
        self.flat_inst_list.append(inst)

    def _flat_simd(self, inst, idx):
        self._load_general_regs(inst, ["rs1", "rs2", "rs3", "rd"], idx)
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

    def _flat_pim_output(self, inst, idx):
        self._load_general_regs(inst, ["rs1", "rs2", "rd"], idx)
        self._load_special_regs([
            SpecialReg.WEIGHT_BIT_WIDTH,
            SpecialReg.OUTPUT_BIT_WIDTH,
            SpecialReg.GROUP_SIZE,
            SpecialReg.ACTIVATION_GROUP_NUM,
        ])
        self.flat_inst_list.append(inst)

    def _flat_pim_transfer(self, inst, idx):
        self._load_general_regs(inst, ["rs1", "rs2", "rs3", "rs4", "rd"], idx)
        self._load_special_regs([
            SpecialReg.OUTPUT_BIT_WIDTH
        ])
        self.flat_inst_list.append(inst)

    def dump(self, out_dir):
        file_path = os.path.join(out_dir, "flat_code.json")
        with open(file_path, "w") as f:
            f.write("[\n")
            for i,inst in enumerate(self.flat_inst_list):
                # print(i, inst)
                # for key,valye in inst.items():
                #     print(key, valye, type(valye))
                str_inst = json.dumps(inst)
                f.write(str_inst)
                if i < len(self.flat_inst_list)-1:
                    f.write(",")
                f.write("\n")
            f.write("]\n")
        print("Flatten code saved to", file_path)