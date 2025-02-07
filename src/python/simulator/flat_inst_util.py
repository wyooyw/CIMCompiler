import copy
import json
import os
from enum import Enum

import numpy as np
from simulator.inst.instruction import *

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

        self.flat_general_rf = np.zeros([64], dtype=np.int32)
        self.flat_special_rf = np.zeros([32], dtype=np.int32)
        self.last_access_time = np.zeros([64], dtype=np.int32)

        self.flat_inst_list = []

        self.max_int32 = np.iinfo(np.int32).max

    def get_flat_code(self):
        return copy.deepcopy(self.flat_inst_list)

    def _li_general_inst(self, reg, imm):
        return GeneralLiInst(reg, int(imm))
        # return {"class": 2, "type": 3, "opcode": 0, "rd": reg, "imm": imm.item()}

    def _li_special_inst(self, reg, imm):
        return SpecialLiInst(reg, int(imm))
        # return {"class": 2, "type": 3, "opcode": 1, "rd": reg, "imm": imm.item()}

    def _load_general_regs(self, inst, regs, idx):
        """
        通用寄存器立即数赋值指令：general-li
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为10
        - [29, 28]，2bit：type，指令类型码，值为11
        - [27, 26]，2bit：opcode，指令操作码，值为00
        - [25, 21]，5bit：rd，通用寄存器编号，即要赋值的通用寄存器
        - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
        """
        assert type(regs) == list, str(regs)
        assert all(isinstance(reg, int) for reg in regs), str(regs)

        assert type(regs) == list
        for reg in regs:
            if not self.flat_general_rf[reg] == self.general_rf[reg]:
                self.flat_general_rf[reg] = self.general_rf[reg]
                self.flat_inst_list.append(
                    self._li_general_inst(reg, self.general_rf[reg])
                )

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
        assert type(regs) == list
        for reg in regs:
            reg = reg.value
            if not self.flat_special_rf[reg] == self.special_rf[reg]:
                self.flat_special_rf[reg] = self.special_rf[reg]
                self.flat_inst_list.append(
                    self._li_special_inst(reg, self.special_rf[reg])
                )

    def flat_inst(self, inst, idx):
        if isinstance(inst, TransInst):
            self._flat_trans(inst, idx)
        elif isinstance(inst, CIMComputeInst):
            self._flat_pim_compute(inst, idx)
        elif isinstance(inst, CIMConfigInst):
            self._flat_pim_set(inst, idx)
        elif isinstance(inst, CIMOutputInst):
            if int(os.environ.get("FAST_MODE")) == 0:
                self._flat_pim_output(inst, idx)
        elif isinstance(inst, CIMTransferInst):
            if int(os.environ.get("FAST_MODE")) == 0:
                self._flat_pim_transfer(inst, idx)
        elif isinstance(inst, SIMDInst):
            self._flat_simd(inst, idx)
        else:
            assert (
                isinstance(inst, RRInst) or
                isinstance(inst, RIInst) or
                isinstance(inst, GeneralLiInst) or
                isinstance(inst, SpecialLiInst) or
                isinstance(inst, StoreInst) or
                isinstance(inst, LoadInst) or
                isinstance(inst, SpecialToGeneralAssignInst) or
                isinstance(inst, GeneralToSpecialAssignInst) or
                isinstance(inst, BranchInst) or
                isinstance(inst, JumpInst) or
                isinstance(inst, PrintInst) or
                isinstance(inst, DebugInst)
            ), f"Unsupported instruction type: {type(inst)}"

    def _flat_trans(self, inst, idx):
        self._load_general_regs(inst, [inst.reg_in, inst.reg_size, inst.reg_out], idx)
        self.flat_inst_list.append(inst)

    def _flat_pim_compute(self, inst, idx):
        self._load_general_regs(inst, [inst.reg_input_addr, inst.reg_input_size, inst.reg_activate_row], idx)
        self._load_special_regs(
            [
                SpecialReg.INPUT_BIT_WIDTH,
                SpecialReg.OUTPUT_BIT_WIDTH,
                SpecialReg.WEIGHT_BIT_WIDTH,
                SpecialReg.ACTIVATION_ELEMENT_COL_NUM,
                SpecialReg.ACTIVATION_GROUP_NUM,
                SpecialReg.GROUP_SIZE,
                SpecialReg.GROUP_INPUT_STEP,
                SpecialReg.VALUE_SPARSE_MASK_ADDR,
                SpecialReg.BIT_SPARSE_META_ADDR,
            ]
        )
        self.flat_inst_list.append(inst)

    def _flat_pim_set(self, inst, idx):
        self._load_general_regs(inst, [inst.reg_single_group_id, inst.reg_mask_addr], idx)
        self._load_special_regs([SpecialReg.WEIGHT_BIT_WIDTH, SpecialReg.GROUP_SIZE])
        self.flat_inst_list.append(inst)

    def _flat_simd(self, inst, idx):
        self._load_general_regs(inst, [inst.reg_in1, inst.reg_in2, inst.reg_size, inst.reg_out], idx)
        self._load_special_regs(
            [
                SpecialReg.SIMD_INPUT_1_BIT_WIDTH,
                SpecialReg.SIMD_INPUT_2_BIT_WIDTH,
                SpecialReg.SIMD_INPUT_3_BIT_WIDTH,
                SpecialReg.SIMD_INPUT_4_BIT_WIDTH,
                SpecialReg.SIMD_OUTPUT_BIT_WIDTH,
                SpecialReg.SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_1,
                SpecialReg.SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_2,
            ]
        )
        self.flat_inst_list.append(inst)

    def _flat_pim_output(self, inst, idx):
        self._load_general_regs(inst, [inst.reg_out_n, inst.reg_out_mask_addr, inst.reg_out_addr], idx)
        self._load_special_regs(
            [
                SpecialReg.WEIGHT_BIT_WIDTH,
                SpecialReg.OUTPUT_BIT_WIDTH,
                SpecialReg.GROUP_SIZE,
                SpecialReg.ACTIVATION_GROUP_NUM,
            ]
        )
        self.flat_inst_list.append(inst)

    def _flat_pim_transfer(self, inst, idx):
        self._load_general_regs(inst, [inst.reg_src_addr, inst.reg_out_n, inst.reg_out_mask_addr, inst.reg_buffer_addr, inst.reg_dst_addr], idx)
        self._load_special_regs([SpecialReg.OUTPUT_BIT_WIDTH])
        self.flat_inst_list.append(inst)

    def dump(self, out_dir):
        file_path = os.path.join(out_dir, "flat_code.json")
        with open(file_path, "w") as f:
            f.write("[\n")
            for i, inst in enumerate(self.flat_inst_list):
                # print(i, inst)
                # for key,valye in inst.items():
                #     print(key, valye, type(valye))
                # str_inst = json.dumps(inst)
                str_inst = str(inst)
                f.write(str_inst)
                if i < len(self.flat_inst_list) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]\n")
        print("Flatten code saved to", file_path)
