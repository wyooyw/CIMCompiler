import copy
import cProfile
import json
import logging
from enum import Enum

import numpy as np
from tqdm import tqdm
import math
import os
import torch
import torch.nn.functional as F

from cim_compiler.simulator.data_type import get_bitwidth_from_dtype, get_dtype_from_bitwidth
from cim_compiler.simulator.special_regs import SpecialReg
from cim_compiler.simulator.flat_inst_util import FlatInstUtil
from cim_compiler.simulator.macro_utils import MacroConfig, MacroUtil
from cim_compiler.simulator.mask_utils import MaskConfig, MaskUtil
from cim_compiler.simulator.meta_utils import MetaUtil
from cim_compiler.simulator.stats_util import StatsUtil
from cim_compiler.simulator.simd_utils import SIMDConfig, SIMDUtil
from cim_compiler.utils.df_layout import tensor_int8_to_bits
from cim_compiler.utils.round import banker_round
from cim_compiler.simulator.inst.instruction import *
from cim_compiler.simulator.inst import LegacyParser, CIMFlowParser
from cim_compiler.utils.logger import get_logger
from cim_compiler.simulator.reduce_util import ReduceSumUtil, ReduceSumConfig, ReduceMaxUtil, ReduceMaxConfig

logger = get_logger(__name__)

class Memory:
    def __init__(self, name, memtype, offset, size):
        self.name = name
        self.memtype = memtype
        self.offset = offset
        self.size = size
        self.end = self.offset + self.size
        self._data = bytearray(np.zeros((size,), dtype=np.int8))
        self.write_hook_list = []

    def _check_range(self, offset, size):
        return (offset >= self.offset) and (offset + size <= self.offset + self.size)

    def read(self, offset, size):
        assert self._check_range(offset, size), f"offset={offset}, size={size}"
        offset = offset - self.offset
        return copy.copy(self._data[offset : offset + size])

    def read_all(self):
        return copy.copy(self._data)

    def write(self, data, offset, size):
        assert self._check_range(offset, size), f"{offset=}, {size=}"
        assert type(data) in [np.array, np.ndarray, bytearray], f"{type(data)=}"
        if type(data) in [np.array, np.ndarray]:
            data = bytearray(data)
        assert len(data) == size, f"{len(data)=}, {size=}"

        offset = offset - self.offset
        self._data[offset : offset + size] = data
        assert (
            len(self._data) == self.size
        ), f"{len(self._data)=}, {self.size=}, {offset=}, "

        for hook in self.write_hook_list:
            hook()

    def clear(self):
        self._data[:] = bytearray(np.zeros((self.size,), dtype=np.int8))

    def register_write_hook(self, hook):
        self.write_hook_list.append(hook)

class TransposeMemory(Memory):
    def __init__(self, name, memtype, offset, size):
        super().__init__(name, memtype, offset, size)
        self._transpose_buffer = None
    
    def get_transpose_buffer(self):
        if self._transpose_buffer is not None:
            return self._transpose_buffer
    
        src_np = np.frombuffer(self._data, dtype=np.float16).reshape(-1)
        assert src_np.shape[0] % (16 * 128) == 0
        src_np = src_np.reshape(-1, 16, 128)
        src_np = np.transpose(src_np, (0, 2, 1)).reshape(-1)
        transpose_data = bytearray(src_np.tobytes())
        self._transpose_buffer = transpose_data

        return transpose_data
    
    def write(self, data, offset, size):
        super().write(data, offset, size)
        self._transpose_buffer = None

    def clear(self):
        super().clear()
        self._transpose_buffer = None

    def read(self, offset, size):
        transpose_data = self.get_transpose_buffer()
        
        assert self._check_range(offset, size), f"offset={offset}, size={size}"
        offset = offset - self.offset
        return copy.copy(transpose_data[offset : offset + size])

    def read_all(self):
        transpose_data = self.get_transpose_buffer()
        return copy.copy(transpose_data)
        

class MemorySpace:
    def __init__(self):
        self.memory_space = []

    def _sort_by_offset(self):
        self.memory_space.sort(key=lambda m: m.offset)

    def _check_no_overlap(self):
        self._sort_by_offset()
        offset = 0
        flag = True
        for memory in self.memory_space:
            if memory.offset < offset:
                flag = False
                break
            offset = memory.offset + memory.size
        return flag

    def _check_no_hole(self):
        self._sort_by_offset()
        offset = 0
        flag = True
        for memory in self.memory_space:
            if offset < memory.offset:
                flag = False
                break
            offset = memory.offset + memory.size
        return flag

    def check(self):
        assert self._check_no_overlap()
        assert self._check_no_hole()

    def add_memory(self, memory: Memory):
        self.memory_space.append(memory)
        self._sort_by_offset()

    def get_memory_by_address(self, address):
        for memory in self.memory_space:
            if memory.offset <= address and address < memory.offset + memory.size:
                return memory
        return None

    def get_memory_and_check_range(self, offset, size):
        """
        check [offset, offset + size) is in one memory
        """
        for memory in self.memory_space:
            if offset >= memory.offset and (
                offset + size <= memory.offset + memory.size
            ):
                return memory
        assert False, f"can not find memory! {offset=}, {size=}"

    def check_memory_type(self, offset, size, memtype):
        """
        check [offset, offset + size) is in one memory
        """
        if type(memtype) == str:
            memtype = [memtype]
        assert type(memtype) == list
        assert type(size) in (int, np.int32, np.int64), f"{type(size)=}"
        assert size >= 0, f"{size=}"
        for memory in self.memory_space:
            if ( offset >= memory.offset 
                and offset < memory.offset + memory.size 
                and offset + size <= memory.offset + memory.size
            ):
                assert (
                    memory.memtype in memtype
                ), f"require {memtype=}, but get {memory.memtype=}. {offset=}, {size=}"
                return
        assert False, f"can not find memory! {offset=}, {size=}, {memtype=}"

    def check_memory_name(self, offset, size, memory_name):
        """
        check [offset, offset + size) is in one memory
        """
        if type(memory_name) == str:
            memory_name = [memory_name]
        assert type(memory_name) == list
        for memory in self.memory_space:
            if offset >= memory.offset and (
                offset + size <= memory.offset + memory.size
            ):
                assert (
                    memory.name in memory_name
                ), f"require {memory_name=}, but get {memory.name=}. {offset=}, {size=}"
                return
        assert False, f"can not find memory! {offset=}, {size=}, {memory_name=}"

    def read(self, offset, size):
        memory = self.get_memory_and_check_range(offset, size)
        return memory.read(offset, size)

    def write(self, data, offset, size):
        memory = self.get_memory_and_check_range(offset, size)
        memory.write(data, offset, size)

    def read_as(self, offset, size, dtype):
        data_bytes = self.read(offset, size)
        np_array = np.frombuffer(data_bytes, dtype=dtype)
        return np_array

    def get_macro_memory(self):
        for memory in self.memory_space:
            if memory.memtype == "macro":
                return memory
        return None

    def get_mask_memory(self):
        for memory in self.memory_space:
            if memory.name == "mask":
                return memory
        return None

    def get_memory_by_name(self, name):
        if type(name) == str:
            name = [name]
        assert type(name) == list
        for memory in self.memory_space:
            if memory.name in name:
                return memory
        return None

    def get_base_of(self, name):
        for memory in self.memory_space:
            if memory.name == name:
                return memory.offset
        assert False, f"Can not find {name=}"
        return None

    def clear(self):
        for memory in self.memory_space:
            memory.clear()

    def load_memory_image(self, memory_image_path):
        with open(memory_image_path, "rb") as file:
            content = file.read()
        byte_array = bytearray(content)
        total_size = sum([memory.size for memory in self.memory_space])
        assert len(byte_array) == total_size, f"{len(byte_array)=}, {total_size=}"

        offset = 0
        for memory in self.memory_space:
            size = memory.size
            memory.write(byte_array[offset : offset + size], memory.offset, size)
            offset += size

    def save_memory_image(self, memory_image_path):
        byte_array = bytearray()
        for memory in self.memory_space:
            byte_array += memory.read_all()
        with open(memory_image_path, "wb") as file:
            file.write(byte_array)

    @classmethod
    def from_memory_config(
        cls,
        memory_config_path,
    ):
        with open(memory_config_path, "r") as f:
            memory_config = json.load(f)
        memory_space = cls()
        for memory in memory_config["memory_list"]:
            name = memory["name"]
            memtype = memory["type"]
            offset = memory["addressing"]["offset_byte"]
            size = memory["addressing"]["size_byte"]
            logger.debug(f"Add memory: {name=}, {memtype=}, {offset=}, {size=}")
            if name == "transpose_memory":
                memory_space.add_memory(TransposeMemory(name, memtype, offset, size))
            else:
                memory_space.add_memory(Memory(name, memtype, offset, size))
        return memory_space


class Simulator:
    FINISH = 0
    TIMEOUT = 1
    ERROR = 2

    def __init__(
        self,
        memory_space,
        macro_config,
        mask_config,
        reduce_sum_config = None,
        reduce_max_config = None,
        simd_config = None,
        safe_time=999999999,
        mask_memory_name="mask",
    ):
        super().__init__()
        self.general_rf = np.zeros([64], dtype=np.int32)
        self.special_rf = np.zeros([32], dtype=np.int32)
        self.memory_space = memory_space
        self.macro_config = macro_config
        self.mask_config = mask_config
        self.reduce_sum_config = reduce_sum_config
        self.reduce_max_config = reduce_max_config
        self.simd_config = simd_config
        self.macro_util = MacroUtil(self.memory_space.get_macro_memory(), macro_config)
        self.mask_util = MaskUtil(
            self.memory_space.get_memory_by_name(mask_memory_name),
            macro_config,
            mask_config,
        )
        self.meta_util = MetaUtil(
            self.memory_space.get_memory_by_name("pim_meta_data_reg_buffer"),
            macro_config,
        )
        self.reduce_sum_util = ReduceSumUtil(
            self.reduce_sum_config
        )
        self.reduce_max_util = ReduceMaxUtil(
            self.reduce_max_config
        )
        self.simd_util = SIMDUtil(
            self.simd_config,
            self,
        )
        self.jump_offset = None
        self.safe_time = safe_time

        self.debug_hook = None

        self._int_data_type = {8: np.int8, 16: np.int16, 32: np.int32}

        self.print_record = list()

        self._add_internel_macro_output_buffer()

        self._read_reg_value_directly = False

        self.pimset_mask = None

        self.pipes = None

        self.core_id = -1
    
    def get_pimset_mask(self):
        if self.pimset_mask is not None:
            return self.pimset_mask

        group_size = self.read_special_reg(SpecialReg.GROUP_SIZE)
        vcol = self.read_special_reg(SpecialReg.WEIGHT_BIT_WIDTH)
        n_vcol_per_group = self.macro_config.n_vcol(vcol) * group_size
        return np.zeros([n_vcol_per_group], dtype=bool)
        
    
    def set_pimset_mask(self, mask):
        self.pimset_mask = mask

    def _add_internel_macro_output_buffer(self):
        """
        This is an internal memory for doing accumulate for macro's output
        """
        # if self.memory_space.get_memory_by_name(["pim_output_reg_buffer", "cim_output_reg_buffer"]) is None:
        #     logger.debug(
        #         "[Warning] Can't find pim_output_reg_buffer or cim_output_reg_buffer. Make sure the code has no macro-related instruction."
        #     )
        #     return
        end_memory = self.memory_space.memory_space[-1]
        end_offset = end_memory.offset + end_memory.size
        # output_buffer_size = self.memory_space.get_memory_by_name(
        #     ["pim_output_reg_buffer", "cim_output_reg_buffer"]
        # ).size
        output_buffer_size = self.macro_config.get_n_group_vcol(8) * self.macro_config.n_group
        internel_macro_output_buffer = Memory(
            "internel_macro_output_reg_buffer",
            "reg_buffer",
            offset=end_offset,
            size=output_buffer_size,
        )
        logger.debug(f"{end_offset=}")
        self.memory_space.add_memory(internel_macro_output_buffer)

    @classmethod
    def from_config(
        cls,
        config_path,
    ):
        with open(config_path, "r") as f:
            config = json.load(f)
        memory_space = MemorySpace.from_memory_config(config_path)
        macro_config = MacroConfig.from_config(config_path)
        mask_config = MaskConfig.from_config(config_path)
        reduce_sum_config = ReduceSumConfig.from_config(config_path)
        reduce_max_config = ReduceMaxConfig.from_config(config_path)
        simd_config = SIMDConfig.from_config(config_path)
        if "mask_memory_name" in config:
            return cls(
                memory_space,
                macro_config,
                mask_config,
                reduce_sum_config,
                reduce_max_config,
                simd_config,
                mask_memory_name=config["mask_memory_name"],
            )
        else:
            return cls(
                memory_space, 
                macro_config, 
                mask_config, 
                reduce_sum_config, 
                reduce_max_config, 
                simd_config
            )

    def clear(self):
        self.memory_space.clear()
        self.general_rf = np.zeros([64], dtype=np.int32)
        self.special_rf = np.zeros([32], dtype=np.int32)
        self.print_record = list()
        self.jump_offset = None
        self.meta_util._clear_buffer()
        self._read_reg_value_directly = False

    def get_dtype(self, bitwidth):
        assert bitwidth in self._int_data_type
        return self._int_data_type[bitwidth]

    def run_code_with_profile(self, code: list[dict], total_pim_compute_count=0):
        profiler = cProfile.Profile()
        profiler.enable()
        status = self.run_code(code, total_pim_compute_count)
        profiler.disable()
        profiler.print_stats(sort="cumtime")
        return status

    def run_code(self, code: list[dict], total_pim_compute_count=0, record_flat=True):
        pc = 0
        cnt = 0
        self.pbar = tqdm(total=total_pim_compute_count)
        self.pimcompute_cnt = 0
        self.stats_util = StatsUtil()
        self.flat_inst_util = FlatInstUtil(self.general_rf, self.special_rf)
        
        while pc < len(code) and cnt < self.safe_time:
            inst = code[pc]
            self.stats_util.record(inst)
            # self.stats_util.record_reg_status(pc + 1, cnt, self.general_rf)
            if record_flat:
                self.flat_inst_util.flat_inst(inst, cnt)

            self._run_inst(inst)

            if self.jump_offset is not None:
                pc += self.jump_offset
                self.jump_offset = None
            else:
                pc += 1

            cnt += 1

        # print(f"{self.pimcompute_cnt=}, {total_pim_compute_count=}")
        self.pbar.close()
        
        # if self.pipes is not None:
        #     for pipe in self.pipes:
        #         if pipe:
        #             pipe.close()

        if pc == len(code):
            logger.debug("Run finish!")
            return self.FINISH, self.stats_util, self.flat_inst_util
        elif pc < len(code) and cnt == self.safe_time:
            logger.debug("Meet safe time!")
            return self.TIMEOUT, self.stats_util, self.flat_inst_util
        else:
            print(
                f"Strange exit situation! {pc=}, {len(code)=}, {cnt=}, {self.safe_time=}"
            )
            return self.ERROR, self.stats_util, self.flat_inst_util

    def read_general_reg(self, regid):
        if self._read_reg_value_directly:
            return regid
        return self.read_reg(self.general_rf, regid)

    def write_general_reg(self, regid, value):
        self.write_reg(self.general_rf, regid, value)

    def read_special_reg(self, regid):
        if type(regid) == SpecialReg:
            regid = regid.value
        assert type(regid) == int, f"{regid=}"
        return self.read_reg(self.special_rf, regid)

    def write_special_reg(self, regid, value):
        if type(regid) == SpecialReg:
            regid = regid.value
        assert type(regid) == int, f"{regid=}"
        self.write_reg(self.special_rf, regid, value)

    def read_reg(self, rf, regid):
        assert type(regid) == int, f"{type(regid)=}"
        assert 0 <= regid and regid < rf.shape[0], f"{regid=}"
        return rf[regid]

    def write_reg(self, rf, regid, value):
        assert type(regid) == int, f"{regid=}"
        assert 0 <= regid and regid < rf.shape[0], f"{regid=}, {rf.shape[0]=}"
        # TODO: check value is in range of int32
        rf[regid] = value

    """
    Classes
    """

    def _run_inst(self, inst):
        # CIM
        if isinstance(inst, CIMComputeInst):
            self._run_pim_class_pim_compute_type_inst(inst)
        elif isinstance(inst, CIMConfigInst):
            self._run_pim_class_pim_set_type_inst(inst)
        elif isinstance(inst, CIMOutputInst):
            self._run_pim_class_pim_output_type_inst(inst)
        elif isinstance(inst, CIMTransferInst):
            self._run_pim_class_pim_transfer_type_inst(inst)

        # SIMD
        elif isinstance(inst, SIMDInst):
            self.simd_util.run(inst)
            # self._run_simd_class_inst(inst)

        # Scalar
        elif isinstance(inst, RRInst):
            self._run_scalar_class_rr_type_inst(inst)
        elif isinstance(inst, RIInst):
            self._run_scalar_class_ri_type_inst(inst)
        elif isinstance(inst, LoadInst) or isinstance(inst, StoreInst):
            self._run_scalar_class_load_store_type_inst(inst)
        elif isinstance(inst, GeneralLiInst) or isinstance(inst, SpecialLiInst):
            self._run_scalar_class_other_type_li_inst(inst)
        elif isinstance(inst, GeneralToSpecialAssignInst) or isinstance(inst, SpecialToGeneralAssignInst):
            self._run_scalar_class_other_type_special_general_assign_inst(inst)

        # Trans
        elif isinstance(inst, TransInst):
            self._run_trans_class_trans_type_inst(inst)

        # Control
        elif isinstance(inst, BranchInst):
            self._run_control_class_br_type_inst(inst)
        elif isinstance(inst, JumpInst):
            self._run_control_class_jump_type_inst(inst)

        elif isinstance(inst, PrintInst):
            self._run_debug_class_inst(inst)

        # Communication
        elif isinstance(inst, SendInst):
            self._run_send_inst(inst)
        elif isinstance(inst, RecvInst):
            self._run_recv_inst(inst)
        else:
            assert False, f"Not support {inst=}"

    def _run_simd_class_inst(self, inst):
        opcode = inst.opcode
        if opcode in [0x00, 0x02, 6]:  # vec add / mul
            self._run_simd_class_vector_vector_inst(inst)
        elif opcode in [
            0b01, 
            7, 
            9, 
            14, # VS_DIV
            15  # VS_SUB
        ]: 
            self._run_simd_class_scalar_vector_inst(inst)
        elif opcode == 3:
            self._run_simd_class_quantify_inst(inst)
        elif opcode == 4:
            self._run_simd_class_resadd_quantify_inst(inst)
        elif opcode == 5:
            self._run_simd_class_resmul_quantify_inst(inst)
        elif opcode == 8:
            self._run_simd_class_floor_inst(inst)
        # elif opcode == 9:
        #     self._run_simd_class_set_inst(inst)
        elif opcode == 10:
            self._run_simd_class_softmax_inst(inst)
        elif opcode in [
            11, # reduce max
            13 # reduce sum
        ]:
            self._run_simd_class_reduce_inst(inst)
        elif opcode in [
            12 # vector exp
        ]:
            self._run_simd_class_vector_inst(inst)
        elif opcode == 16:
            self._run_simd_class_sqrt_inst(inst)
        elif opcode == 17:
            self._run_simd_class_gelu_inst(inst)
        else:
            assert False, f"Not support {opcode=} yet."

    """
    Types
    """

    def _run_scalar_class_rr_type_inst(self, inst):
        value1 = self.read_general_reg(inst.reg_lhs)
        value2 = self.read_general_reg(inst.reg_rhs)
        opcode = inst.opcode
        if opcode == 0b000:  # add
            result = value1 + value2
        elif opcode == 0b001:  # sub
            result = value1 - value2
        elif opcode == 0b010:  # mul
            result = value1 * value2
        elif opcode == 0b011:  # div
            result = value1 // value2
        elif opcode == 0b100:  # sll
            assert False, "Not support sll yet"
        elif opcode == 0b101:  # srl
            assert False, "Not support srl yet"
        elif opcode == 0b110:  # sra
            assert False, "Not support sra yet"
        elif opcode == 0b111:  # mod
            result = value1 % value2
        elif opcode == 0b1000:  # min
            result = min(value1, value2)
        elif opcode == 0b1001:  # max
            result = max(value1, value2)
        elif opcode == 0b1010:  # and
            result = value1 & value2
        elif opcode == 0b1011:  # or
            result = value1 | value2
        elif opcode == 0b1100:  # comp
            result = value1 == value2
        elif opcode == 0b1101:  # comp
            result = value1 != value2
        elif opcode == 0b1110:  # comp
            result = value1 > value2
        elif opcode == 0b1111:  # comp
            result = value1 < value2
        else:
            assert False, f"Not support {opcode=}."
        self.write_general_reg(inst.reg_out, result)

    def _run_scalar_class_ri_type_inst(self, inst):
        value = self.read_general_reg(inst.reg_in)
        imm = inst.imm
        opcode = inst.opcode
        if opcode == 0b000:  # add
            result = value + imm
        elif opcode == 0b001:  # sub
            result = value - imm
        elif opcode == 0b010:  # mul
            result = value * imm
        elif opcode == 0b011:  # div
            result = value / imm
        elif opcode == 0b111:  # mod
            result = value % imm
        elif opcode == 0b1000:  # min
            result = min(value, imm)
        else:
            assert False, f"Not support {opcode=}."
        self.write_general_reg(inst.reg_out, result)

    def _run_scalar_class_load_store_type_inst(self, inst):

        if isinstance(inst, LoadInst):  # load

            addr = self.read_general_reg(inst.reg_addr)
            offset = inst.offset
            addr += offset
            self.memory_space.check_memory_type(
                addr, 4, ["rf", "reg_buffer", "sram"]
            )
            value = self.memory_space.read_as(addr, 4, np.int32).item()
            self.write_general_reg(inst.reg_value, value)

        elif isinstance(inst, StoreInst):  # store

            addr = self.read_general_reg(inst.reg_addr)
            value = self.read_general_reg(inst.reg_value)
            offset = inst.offset
            addr += offset
            self.memory_space.check_memory_type(
                addr, 4, ["rf", "reg_buffer", "sram"]
            )
            self.memory_space.write(np.array([value], dtype=np.int32), addr, 4)

        else:
            assert False, f"Not support {inst=}."

    def _run_scalar_class_other_type_li_inst(self, inst):
        """
        通用寄存器立即数赋值指令：general-li
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为10
        - [29, 28]，2bit：type，指令类型码，值为11
        - [27, 26]，2bit：opcode，指令操作码，值为00
        - [25, 21]，5bit：rd，通用寄存器编号，即要赋值的通用寄存器
        - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值

        专用寄存器立即数赋值指令：special-li
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为10
        - [29, 28]，2bit：type，指令类型码，值为11
        - [27, 26]，2bit：opcode，指令操作码，值为01
        - [25, 21]，5bit：rd，专用寄存器编号，即要赋值的通用寄存器
        - [20, 0]，21bit：imm，立即数，表示将要赋给寄存器的值
        """
        rd = inst.reg
        imm = inst.value
        if isinstance(inst, GeneralLiInst):  # 通用寄存器
            rf = self.general_rf
        elif isinstance(inst, SpecialLiInst):  # 专用寄存器
            rf = self.special_rf
        self.write_reg(rf, rd, imm)

    def _run_scalar_class_other_type_special_general_assign_inst(self, inst):
        """
        专用/通用寄存器赋值指令：special-general-assign
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为10
        - [29, 28]，2bit：type，指令类型码，值为11
        - [27, 26]，2bit：opcode，指令操作码
        - 10：表示将通用寄存器的值赋给专用寄存器
        - 11：表示将专用寄存器的值赋给通用寄存器
        - [25, 21]，5bit：rs1，通用寄存器编号，即涉及赋值的通用寄存器
        - [20, 16]，5bit：rs2，专用寄存器编号，即涉及赋值的专用寄存器
        - [15, 0]，16bit：reserve，保留字段
        """
        if isinstance(inst, GeneralToSpecialAssignInst):
            value = self.read_general_reg(inst.reg_general)
            self.write_special_reg(inst.reg_special, value)
        elif isinstance(inst, SpecialToGeneralAssignInst):
            value = self.read_special_reg(inst.reg_special)
            self.write_general_reg(inst.reg_general, value)
        else:
            assert False, f"Not support {opcode=}"

    def _run_trans_class_trans_type_inst(self, inst):
        """
        核内数据传输指令：trans
        指令字段划分：
        - [31, 29]，3bit：class，指令类别码，值为110
        - [28, 28]，1bit：type，指令类型码，值为0
        - [27, 26]，1bit：offset mask，偏移值掩码，0表示该地址不使用偏移值，1表示使用偏移值
        - [27]，1bit：source offset mask，源地址偏移值掩码
        - [26]，1bit：destination offset mask，目的地址偏移值掩码
        - [25, 21]，5bit：rs1，通用寄存器1，表示传输源地址的基址
        - [20, 16]，5bit：rs2，通用寄存器2，表示传输数据的字节大小
        - [15, 11]，5bit：rd，通用寄存器3，表示传输目的地址的基址
        - [10, 0]，11bit：offset，立即数，表示寻址的偏移值
        - 源地址计算公式：$rs + offset * [27]
        - 目的地址计算公式：$rd + offset * [26]
        """
        src_base = self.read_general_reg(inst.reg_in)
        dst_base = self.read_general_reg(inst.reg_out)
        offset = inst.offset
        src_offset_mask = inst.flag_src_offset
        dst_offset_mask = inst.flag_dst_offset
        size = self.read_general_reg(inst.reg_size)

        src_addr = src_base + src_offset_mask * offset
        dst_addr = dst_base + dst_offset_mask * offset

        src_data = self.memory_space.read(src_addr, size)
        self.memory_space.write(src_data, dst_addr, size)

        logger.debug(
            "Trans: from {}({}) to {}({}), {} bytes. Data (first 16 bytes): {}".format(
                str(src_addr),
                self.memory_space.get_memory_by_address(src_addr).name,
                str(dst_addr),
                self.memory_space.get_memory_by_address(dst_addr).name,
                str(size),
                src_data[0:min(size, 16)]
            )
        )

        self.stats_util.record_trans_addr(src_addr, dst_addr, size)

    def _run_control_class_br_type_inst(self, inst):
        """
        指令字段划分：
        - [31, 29]，3bit：class，指令类别码，值为111
        - [28, 26]，3bit：type，指令类型码
            - 000：beq，相等跳转
            - 001：bne，不等跳转
            - 010：bgt，大于跳转
            - 011：blt，小于跳转
        - [25, 21]，5bit：rs1，通用寄存器1，表示进行比较的操作数1
        - [20, 16]，5bit：rs2，通用寄存器2，表示进行比较的操作数2
        - [15, 0]，16bit：offset，立即数，表示跳转指令地址相对于该指令的偏移值
        """
        val1 = self.read_general_reg(inst.reg_lhs)
        val2 = self.read_general_reg(inst.reg_rhs)
        inst_type = inst.compare
        if inst_type == 0b000:  # equals
            cond = val1 == val2
        elif inst_type == 0b001:  # not equals
            cond = not (val1 == val2)
        elif inst_type == 0b010:  # greater than
            cond = val1 > val2
        elif inst_type == 0b011:  # less than
            cond = val1 < val2
        else:
            assert False, f"Unsupported {inst_type=} in control instruction!"

        if cond:
            self.jump_offset = inst.offset

    def _run_control_class_jump_type_inst(self, inst):
        self.jump_offset = inst.offset

    def _run_pim_class_pim_compute_type_inst(self, inst):
        if inst.flag_bit_sparse == 1:
            self._run_pim_class_pim_compute_type_inst_bit_sparse(inst)
        else:
            self._run_pim_class_pim_compute_type_inst_dense(inst)
        self.pbar.update(1)
        self.pimcompute_cnt += 1

    """
    Diffenet Pim Compute
    """

    def _stats_macro_util(self, macro_id, group_size, n_use_comp, width_bw):
        pimset_mask = self.get_pimset_mask().reshape(group_size, -1)
        n_comp = self.macro_config.n_comp
        n_col = self.macro_config.n_bcol

        macro_pimset_mask = ~pimset_mask[macro_id]
        n_use_col = (macro_pimset_mask.sum() * width_bw).item()
        if n_use_col > 0 and n_use_comp > 0:
            self.stats_util.record_macro_ultilize(n_use_comp, n_use_col, n_comp * n_col)

    def _value_sparsity_compute(self, inst, input_data, weight_data):
        if inst.flag_value_sparse == 0:
            return input_data
        output_bw = self.read_special_reg(SpecialReg.OUTPUT_BIT_WIDTH)
        width_bw = self.read_special_reg(SpecialReg.WEIGHT_BIT_WIDTH)
        group_size = self.read_special_reg(SpecialReg.GROUP_SIZE)
        logger.debug(f"{group_size=}, {self.macro_config.n_macro=}")
        logger.debug(f"old {weight_data.shape=}")
        weight_data = np.pad(
            weight_data,
            (
                (0, 0),
                (
                    0,
                    group_size * self.macro_config.n_vcol(width_bw)
                    - weight_data.shape[1],
                ),
            ),
            mode="constant",
            constant_values=0,
        )
        logger.debug(f"new {weight_data.shape=}")
        weight_data = weight_data.reshape(
            self.macro_config.n_comp, group_size, self.macro_config.n_vcol(width_bw)
        )

        assert weight_data.ndim == 3
        assert weight_data.shape[0] == self.macro_config.n_comp
        assert weight_data.shape[1] == group_size
        assert weight_data.shape[2] == self.macro_config.n_vcol(width_bw)

        assert (
            input_data.size == self.mask_config.n_from
        ), f"{input_data.size=}, {self.mask_config.n_from=}"
        mask_addr = self.read_special_reg(SpecialReg.VALUE_SPARSE_MASK_ADDR)
        # logger.debug(f"{mask_addr=}")
        mask_data = self.mask_util.get_mask(mask_addr, input_data.size, group_size)
        assert mask_data.ndim == 2, f"{mask_data.ndim=}"
        assert (
            mask_data.shape[0] == group_size
            and mask_data.shape[1] == self.mask_config.n_from
        ), f"{mask_data.shape=}, {group_size=}, {self.mask_config.n_from=}"
        assert mask_data.dtype == bool, f"{mask_data.dtype}"
        assert (
            mask_data.sum(axis=1) <= self.mask_config.n_to
        ).all(), f"{mask_data.sum(axis=1)=}, {self.mask_config.n_to=}"
        # np.set_printoptions(threshold=65536)
        # logger.debug(f"{mask_data.shape=}")
        # logger.debug(mask_data.astype(np.int8))
        # logger.debug(f"{weight_data.shape=}")
        # logger.debug(weight_data.reshape(self.macro_config.n_comp, -1))
        # import pdb; pdb.set_trace()
        pimset_mask = self.get_pimset_mask().reshape(group_size, -1)
        n_comp = self.macro_config.n_comp
        n_col = self.macro_config.n_bcol

        output_list = []
        out_dtype = get_dtype_from_bitwidth(output_bw)
        for macro_id in range(group_size):
            # get macro input data
            macro_mask = mask_data[macro_id]
            # import pdb; pdb.set_trace()
            macro_input_data = input_data[macro_mask]
            # logger.debug(f"{input_data=}, {macro_mask=}, {macro_input_data=}")
            assert macro_input_data.ndim == 1
            assert macro_input_data.size <= self.mask_config.n_to
            macro_input_data = np.pad(
                macro_input_data,
                (0, self.mask_config.n_to - macro_input_data.size),
                mode="constant",
                constant_values=0,
            )
            assert macro_input_data.size == self.mask_config.n_to

            macro_weight = weight_data[:, macro_id, :]
            # import pdb; pdb.set_trace();
            macro_output = np.dot(
                macro_input_data.astype(out_dtype), macro_weight.astype(out_dtype)
            )
            # logger.debug(f"{macro_input_data=}, {macro_weight=}, {macro_output=}")
            output_list.append(macro_output)

            n_use_comp = macro_mask.sum().item()
            # macro_pimset_mask = ~pimset_mask[macro_id]
            # n_use_col = (macro_pimset_mask.sum() * width_bw).item()
            # if n_use_col > 0:
            #     self.stats_util.record_macro_ultilize(n_use_comp, n_use_col, n_comp * n_col)
            self._stats_macro_util(macro_id, group_size, n_use_comp, width_bw)
        # import pdb; pdb.set_trace()
        output_data = np.concatenate(output_list)
        # logger.debug(f"{output_data=}")
        return output_data

    def _run_pim_class_pim_compute_type_inst_dense(self, inst):
        assert isinstance(inst, CIMComputeInst)
        input_bw = self.read_special_reg(SpecialReg.INPUT_BIT_WIDTH)
        input_offset = self.read_general_reg(inst.reg_input_addr)
        input_size = self.read_general_reg(inst.reg_input_size)
        activate_row = self.read_general_reg(inst.reg_activate_row)
        batch_size = self.read_general_reg(inst.reg_batch_size)
        flag_batch = inst.flag_batch
        if not flag_batch:
            batch_size = 1
        # import pdb; pdb.set_trace()
        for batch_id in range(batch_size):
            self._run_pim_class_pim_compute_type_inst_impl(
                input_offset=input_offset + (batch_id * input_size * input_bw) // 8,
                input_size=input_size,
                activate_row=activate_row + batch_id,
                inst=inst,
            )

    def _run_pim_class_pim_compute_type_inst_impl(self, 
        input_offset,
        input_size,
        activate_row,
        inst
        ):
        # input_offset = self.read_general_reg(inst.reg_input_addr)
        # input_size = self.read_general_reg(inst.reg_input_size)
        # activate_row = self.read_general_reg(inst.reg_activate_row)

        input_bw = self.read_special_reg(SpecialReg.INPUT_BIT_WIDTH)
        output_bw = self.read_special_reg(SpecialReg.OUTPUT_BIT_WIDTH)
        width_bw = self.read_special_reg(SpecialReg.WEIGHT_BIT_WIDTH)
        activation_element_col_num = self.read_special_reg(
            SpecialReg.ACTIVATION_ELEMENT_COL_NUM
        )

        activation_group_num = self.read_special_reg(SpecialReg.ACTIVATION_GROUP_NUM)
        group_size = self.read_special_reg(SpecialReg.GROUP_SIZE)
        assert self.macro_config.n_macro % group_size == 0
        group_num = self.macro_config.n_macro // group_size
        assert group_num == self.macro_config.n_group
        
        group_input_step = self.read_special_reg(SpecialReg.GROUP_INPUT_STEP)
        assert inst.flag_group == 1
        assert inst.flag_group_input_mode == 0
        logger.debug(f"{group_num=}")

        value_sparsity = inst.flag_value_sparse
        # Get input vector
        input_byte_size = input_size * input_bw // 8

        # self.memory_space.check_memory_name(input_offset, input_byte_size, ["pim_input_reg_buffer", "cim_input_reg_buffer"])
        group_input_data = []
        for group_id in range(activation_group_num):
            group_input_offset = input_offset + group_id * group_input_step
            input_data = self.memory_space.read_as(
                group_input_offset, 
                input_byte_size, 
                get_dtype_from_bitwidth(input_bw, is_float=self.read_special_reg(SpecialReg.DTYPE_MACRO_IS_FLOAT))
            )
            # self.memory_space.check_memory_name(group_input_offset, input_byte_size, ["pim_input_reg_buffer", "cim_input_reg_buffer"])
            group_input_data.append(input_data)

        # Get weight matrix
        activate_element_row_num = input_size
        weight_data = self.macro_util.get_macro_data(
            activate_row,
            get_dtype_from_bitwidth(width_bw, is_float=self.read_special_reg(SpecialReg.DTYPE_MACRO_IS_FLOAT)),
            activate_element_row_num,
            activation_element_col_num,
            activation_group_num,
        )  # shape: [compartment, group, vcolumn]
        
        logger.debug(f"{weight_data.shape=}")
        group_weight_data = []
        for group_id in range(activation_group_num):
            group_weight_data.append(weight_data[:, group_id, :])
        
        # compute
        group_output_data = []
        for group_id in range(activation_group_num):
            input_data = group_input_data[group_id]
            weight_data = group_weight_data[group_id]
            # logger.debug(f"{input_data=}, {weight_data=}")

            # use pimset to mask weight
            pimset_mask = self.get_pimset_mask()
            assert pimset_mask is not None
            assert (
                len(pimset_mask) == weight_data.shape[1]
            ), f"{len(pimset_mask)=}, {weight_data.shape[1]=}"
            assert pimset_mask.dtype == bool, f"{pimset_mask.dtype=}"
            weight_data[:, pimset_mask] = 0

            assert input_data.ndim == 1
            assert weight_data.ndim == 2, f"{weight_data.shape=}"
            out_dtype = get_dtype_from_bitwidth(output_bw, is_float=self.read_special_reg(SpecialReg.DTYPE_MACRO_IS_FLOAT))
            if value_sparsity:
                output_data = self._value_sparsity_compute(
                    inst, input_data, weight_data
                )
            else:
                for macro_id in range(group_size):
                    n_use_comp = input_data.size
                    self._stats_macro_util(macro_id, group_size, n_use_comp, width_bw)

                assert (
                    0 < input_data.size and input_data.size <= self.macro_config.n_comp
                ), f"{input_data.size=}, {self.macro_config.n_comp=}"
                input_data = np.pad(
                    input_data,
                    (0, self.macro_config.n_comp - input_data.size),
                    mode="constant",
                    constant_values=0,
                )
                assert (
                    input_data.shape[0] == weight_data.shape[0]
                ), f"{input_data.shape=}, {weight_data.shape=}"
                output_data = np.dot(
                    input_data.astype(out_dtype), weight_data.astype(out_dtype)
                )
                # import pdb; pdb.set_trace()
                pass
            
            group_output_data.append(output_data)

        # Save output
        n_macro_per_group = group_size
        group_output_step = (
            self.macro_config.n_vcol(width_bw) * n_macro_per_group * output_bw // 8
        )
        output_offset = self.memory_space.get_base_of(
            "internel_macro_output_reg_buffer"
        )
        for group_id in range(activation_group_num):
            output_data = group_output_data[group_id]
            output_byte_size = output_data.size * output_bw // 8
            group_output_offset = output_offset + group_id * group_output_step
            self.memory_space.check_memory_type(
                group_output_offset, output_byte_size, ["rf", "reg_buffer"]
            )

            # Accumulate
            if inst.flag_accumulate == 1:
                output_data_ori = self.memory_space.read_as(
                    group_output_offset, output_byte_size, out_dtype
                )
                output_data = output_data + output_data_ori
            # else:
            #     assert False
            self.memory_space.write(output_data, group_output_offset, output_byte_size)

    def _run_pim_class_pim_compute_type_inst_bit_sparse(self, inst):
        input_offset = self.read_general_reg(inst.reg_input_addr)
        input_size = self.read_general_reg(inst.reg_input_size)
        activate_row = self.read_general_reg(inst.reg_activate_row)

        input_bw = self.read_special_reg(SpecialReg.INPUT_BIT_WIDTH)
        output_bw = self.read_special_reg(SpecialReg.OUTPUT_BIT_WIDTH)
        width_bw = self.read_special_reg(SpecialReg.WEIGHT_BIT_WIDTH)
        assert input_bw == 8, f"{input_bw=}"
        assert output_bw == 32, f"{output_bw=}"
        assert width_bw == 1, f"{width_bw=}"
        activation_element_col_num = self.read_special_reg(
            SpecialReg.ACTIVATION_ELEMENT_COL_NUM
        )

        activation_group_num = self.read_special_reg(SpecialReg.ACTIVATION_GROUP_NUM)
        group_size = self.read_special_reg(SpecialReg.GROUP_SIZE)
        assert self.macro_config.n_macro % group_size == 0
        group_num = self.macro_config.n_macro // group_size
        assert group_num == self.macro_config.n_group
        group_input_step = self.read_special_reg(SpecialReg.GROUP_INPUT_STEP)
        assert inst.flag_group == 1
        assert inst.flag_group_input_mode == 0
        logger.debug(f"{group_num=}")
        # logger.debug(f"{self.macro_config.n_macro=}")
        # logger.debug(f"{self.macro_config.n_macro=}")
        value_sparsity = inst.flag_value_sparse
        assert inst.flag_bit_sparse == 1, str(inst)
        meta_addr = self.read_special_reg(SpecialReg.BIT_SPARSE_META_ADDR)

        # Get input vector
        input_byte_size = input_size * input_bw // 8
        # self.memory_space.check_memory_type(input_offset, input_byte_size, "reg_buffer")
        group_input_data = []
        for group_id in range(activation_group_num):
            group_input_offset = input_offset + group_id * group_input_step
            input_data = self.memory_space.read_as(
                group_input_offset, input_byte_size, self.get_dtype(input_bw)
            )
            group_input_data.append(input_data)

        # Get weight matrix
        activate_element_row_num = input_size
        weight_data = self.macro_util.get_macro_data(
            activate_row,
            8,  # width_bw,
            activate_element_row_num,
            self.macro_config.n_vcol(8) * group_size,  # activation_element_col_num,
            activation_group_num,
        )  # shape: [compartment, group, vcolumn]
        logger.debug(f"{weight_data.shape=}")
        group_weight_data = []
        for group_id in range(activation_group_num):
            _weight = weight_data[:, group_id, :]
            _weight = self.meta_util.recover_weight(meta_addr, _weight)
            group_weight_data.append(_weight)

        # compute
        group_output_data = []
        for group_id in range(activation_group_num):
            input_data = group_input_data[group_id]
            weight_data = group_weight_data[group_id]
            # logger.debug(f"{input_data=}, {weight_data=}")

            # use pimset to mask weight
            pimset_mask = self.get_pimset_mask()
            assert pimset_mask is not None
            assert (
                len(pimset_mask) == weight_data.shape[1]
            ), f"{len(pimset_mask)=}, {weight_data.shape[1]=}"
            assert pimset_mask.dtype == bool, f"{pimset_mask.dtype=}"
            weight_data[:, pimset_mask] = 0

            assert input_data.ndim == 1
            assert weight_data.ndim == 2, f"{weight_data.shape=}"
            out_dtype = get_dtype_from_bitwidth(output_bw)
            if value_sparsity:
                output_data = self._value_sparsity_compute(
                    inst, input_data, weight_data
                )
            else:
                for macro_id in range(group_size):
                    n_use_comp = input_data.size
                    self._stats_macro_util(macro_id, group_size, n_use_comp, width_bw)

                assert (
                    0 < input_data.size and input_data.size <= self.macro_config.n_comp
                ), f"{input_data.size=}, {self.macro_config.n_comp=}"
                input_data = np.pad(
                    input_data,
                    (0, self.macro_config.n_comp - input_data.size),
                    mode="constant",
                    constant_values=0,
                )
                assert (
                    input_data.shape[0] == weight_data.shape[0]
                ), f"{input_data.shape=}, {weight_data.shape=}"
                output_data = np.dot(
                    input_data.astype(out_dtype), weight_data.astype(out_dtype)
                )

            group_output_data.append(output_data)
        # import pdb; pdb.set_trace()
        # Save output
        n_macro_per_group = group_size
        group_output_step = (
            self.macro_config.n_vcol(width_bw) * n_macro_per_group * output_bw // 8
        )
        output_offset = self.memory_space.get_base_of(
            "internel_macro_output_reg_buffer"
        )
        for group_id in range(activation_group_num):
            output_data = group_output_data[group_id]
            output_byte_size = output_data.size * output_bw // 8
            group_output_offset = output_offset + group_id * group_output_step
            self.memory_space.check_memory_type(
                group_output_offset, output_byte_size, ["rf", "reg_buffer"]
            )

            # Accumulate
            if inst.flag_accumulate == 1:
                output_data_ori = self.memory_space.read_as(
                    group_output_offset, output_byte_size, out_dtype
                )
                output_data = output_data + output_data_ori
            # else:
            #     assert False
            self.memory_space.write(output_data, group_output_offset, output_byte_size)

    def _run_pim_class_pim_output_type_inst(self, inst):
        outsum_move = inst.flag_outsum_move
        outsum = inst.flag_outsum
        if outsum:
            self._outsum(inst)
            return

        # elif outsum_move:
        #     _outsum_move(inst)
        #     return

        if outsum_move or outsum:
            assert False, "This should not happen!"

        dst_offset = self.read_general_reg(inst.reg_out_addr)

        internel_buffer = self.memory_space.get_memory_by_name(
            "internel_macro_output_reg_buffer"
        )
        src_offset, size = internel_buffer.offset, internel_buffer.size

        self.memory_space.check_memory_type(
            src_offset, size, ["rf", "reg_buffer", "sram"]
        )
        self.memory_space.check_memory_type(
            dst_offset, size, ["rf", "reg_buffer", "sram"]
        )

        data = self.memory_space.read(src_offset, size)
        self.memory_space.write(data, dst_offset, size)

        internel_buffer.clear()

    def _outsum(self, inst):
        out_n = self.read_general_reg(inst.reg_out_n)
        assert out_n % 8 == 0
        out_mask_addr = self.read_general_reg(inst.reg_out_mask_addr)
        out_mask = self.memory_space.read_as(out_mask_addr, out_n // 8, np.int8)
        out_mask = tensor_int8_to_bits(out_mask)
        out_mask = out_mask.reshape(-1)

        width_bw = self.read_special_reg(SpecialReg.WEIGHT_BIT_WIDTH)
        output_bw = self.read_special_reg(SpecialReg.OUTPUT_BIT_WIDTH)
        output_byte = output_bw // 8
        group_size = self.read_special_reg(SpecialReg.GROUP_SIZE)
        n_group = self.macro_config.n_macro // group_size
        n_macro_per_group = group_size

        src_offset = self.memory_space.get_base_of("internel_macro_output_reg_buffer")
        src_group_step = (
            self.macro_config.n_vcol(width_bw) * n_macro_per_group * output_bw // 8
        )

        dst_offset = self.read_general_reg(inst.reg_out_addr)
        dst_group_step = src_group_step
        for g in range(n_group):
            src_group_offset = src_offset + g * src_group_step
            dst_group_offset = dst_offset + g * dst_group_step
            group_data_size = out_n * output_byte

            data = self.memory_space.read_as(
                src_group_offset, group_data_size, np.int32
            )
            assert data.size == out_n
            for i in range(out_n):
                if out_mask[i] == 1:
                    assert i + 1 < len(out_mask)
                    assert out_mask[i + 1] == 0
                    data[i] = data[i] + data[i + 1]

            self.memory_space.write(data, dst_group_offset, group_data_size)

        internel_buffer = self.memory_space.get_memory_by_name(
            "internel_macro_output_reg_buffer"
        )
        internel_buffer.clear()

    def _run_pim_class_pim_transfer_type_inst(self, inst):

        src_addr = self.read_general_reg(inst.reg_src_addr)
        output_num = self.read_general_reg(inst.reg_out_n)
        output_mask_addr = self.read_general_reg(inst.reg_out_mask_addr)
        dst_addr = self.read_general_reg(inst.reg_dst_addr)
        output_bw = self.read_special_reg(SpecialReg.OUTPUT_BIT_WIDTH)
        output_byte = output_bw // 8

        output_mask = self.memory_space.read_as(
            output_mask_addr, output_num // 8, np.int8
        )
        output_mask = tensor_int8_to_bits(output_mask)
        output_mask = output_mask.reshape(-1)

        group_size = self.read_special_reg(SpecialReg.GROUP_SIZE)
        n_group = self.macro_config.n_macro // group_size

        data = self.memory_space.read_as(src_addr, output_num * output_byte, np.int32)
        assert data.size == output_mask.size
        # logger.debug(f"{data=}")
        # logger.debug(f"{output_mask=}")
        filtered_data = data[output_mask == 1]
        # logger.debug(f"{filtered_data=}")
        # import pdb; pdb.set_trace()
        assert (
            filtered_data.size == output_mask.sum()
        ), f"{filtered_data.size=}, {data.sum()=}"
        self.memory_space.write(
            filtered_data, dst_addr, filtered_data.size * output_byte
        )

    def _run_pim_class_pim_set_type_inst(self, inst):

        assert inst.flag_group_broadcast == 1, "Only support group broadcast"
        mask_addr = self.read_general_reg(inst.reg_mask_addr)

        group_size = self.read_special_reg(SpecialReg.GROUP_SIZE)
        vcol = self.read_special_reg(SpecialReg.WEIGHT_BIT_WIDTH)
        n_vcol_per_group = self.macro_config.n_vcol(vcol) * group_size
        mask_size = int(math.ceil(n_vcol_per_group / 8))

        mask_data = self.memory_space.read_as(mask_addr, mask_size, np.int8)
        mask_data = tensor_int8_to_bits(mask_data)
        mask_data = mask_data.reshape(-1)
        mask_data = mask_data[:n_vcol_per_group]
        self.stats_util.record_pimset_mask(mask_data.tolist(), vcol)

        mask_data = mask_data.astype(bool)
        mask_data = ~mask_data
        self.set_pimset_mask(mask_data.copy())

    def _run_debug_class_inst(self, inst):
        if isinstance(inst, PrintInst):  # print
            rs = inst.reg
            val = self.read_general_reg(rs)
            self.print_record.append(val)
            logger.info(f"[{self.core_id}] general_reg[{rs}] = {val}")
        elif isinstance(inst, DebugInst):
            import pdb

            pdb.set_trace()
            if self.debug_hook is not None:
                self.debug_hook(simulator=self)
        else:
            assert False, "Not support yet."

    def _run_simd_class_vector_vector_inst(self, inst):

        opcode = inst.opcode
        assert inst.input_num == 2, f"{inst.input_num=}, {opcode=}"

        # Prepare input
        input_size = self.read_general_reg(inst.reg_size)

        input1_addr = self.read_general_reg(inst.reg_in1)
        input1_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input1_byte_size = input1_bitwidth * input_size // 8
        # self.memory_space.check_memory_type(input1_addr, input1_byte_size, "sram")

        input2_addr = self.read_general_reg(inst.reg_in2)
        input2_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH)
        input2_byte_size = input2_bitwidth * input_size // 8
        # self.memory_space.check_memory_type(input2_addr, input2_byte_size, "sram")

        output_addr = self.read_general_reg(inst.reg_out)
        output_bitwidth = self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))

        input1_data = self.memory_space.read_as(
            input1_addr, input1_byte_size, get_dtype_from_bitwidth(input1_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        )
        input2_data = self.memory_space.read_as(
            input2_addr, input2_byte_size, get_dtype_from_bitwidth(input2_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        )

        # Compute
        if opcode == 0x00:
            # assert input1_bitwidth in [8,32]
            # assert input2_bitwidth in [8,32]
            # assert output_bitwidth in [8,32]
            output_data = input1_data.astype(output_dtype) + input2_data.astype(
                output_dtype
            )
        elif opcode == 0x02:
            # assert input1_bitwidth == 8
            # assert input2_bitwidth == 8
            # assert output_bitwidth == 32
            output_data = input1_data.astype(output_dtype) * input2_data.astype(
                output_dtype
            )
        elif opcode == 6 :
            # vvmax
            assert input1_bitwidth == 8
            assert input2_bitwidth == 8
            assert output_bitwidth == 8
            output_data = np.maximum(input1_data, input2_data)

        else:
            assert False, f"Not support: {opcode=}"
        # import pdb; pdb.set_trace()
        # Save output
        output_byte_size = output_data.size * output_bitwidth // 8
        # self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")

        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_scalar_vector_inst(self, inst):
        
        opcode = inst.opcode
        assert inst.input_num == 2

        # Prepare input
        input_size = self.read_general_reg(inst.reg_size)

        input1_addr = self.read_general_reg(inst.reg_in1)
        input1_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input1_byte_size = input1_bitwidth * input_size // 8
        self.memory_space.check_memory_type(input1_addr, input1_byte_size, "sram")
        input1_dtype = get_dtype_from_bitwidth(input1_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))

        input2_addr = self.read_general_reg(inst.reg_in2)
        input2_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH)
        input2_byte_size = input2_bitwidth // 8
        self.memory_space.check_memory_type(input2_addr, input2_byte_size, "sram")
        input2_dtype = get_dtype_from_bitwidth(input2_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))

        output_addr = self.read_general_reg(inst.reg_out)
        output_bitwidth = self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))

        input1_data = self.memory_space.read_as(
            input1_addr, input1_byte_size, input1_dtype
        )
        input2_data = self.memory_space.read_as(
            input2_addr, input2_byte_size, input2_dtype
        )

        if opcode==1:
            # Compute
            output_data = input2_data.astype(output_dtype) + input1_data.astype(
                output_dtype
            )

        elif opcode==7:
            # vsmul
            # assert input1_bitwidth == 32
            # assert input2_bitwidth == 32
            # assert output_bitwidth == 32
            # assert input2_dtype == np.float32
            # Compute
            output_data = (input1_data * input2_data).astype(output_dtype)

        elif opcode==9:
            # Compute
            scalar = input2_data[0]
            output_data = np.full(input_size, scalar, dtype=output_dtype)
        elif opcode == 14: # VS_DIV
            output_data = (
                input1_data.astype(output_dtype) / input2_data.astype(output_dtype)
            ).astype(output_dtype)
        elif opcode == 15: # VS_SUB
            output_data = (
                input1_data.astype(output_dtype) - input2_data.astype(output_dtype)
            ).astype(output_dtype)
        else:
            assert False, f"Not support: {opcode=}"

        # Save output
        output_byte_size = output_data.size * output_bitwidth // 8
        self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")
        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_vector_inst(self, inst):
        assert inst.input_num == 1

        input_addr = self.read_general_reg(inst.reg_in1)
        input_size = self.read_general_reg(inst.reg_size)
        output_addr = self.read_general_reg(inst.reg_out)

        input_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input_byte_size = input_bitwidth * input_size // 8
        input_dtype = get_dtype_from_bitwidth(input_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        output_bitwidth = self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        self.memory_space.check_memory_type(input_addr, input_byte_size, "sram")
        input_data = self.memory_space.read_as(
            input_addr, 
            input_byte_size, 
            input_dtype
        )
        if inst.opcode == 12: # vector exp
            output_data = np.exp(input_data.astype(output_dtype)).astype(output_dtype)
        else:
            assert False, f"Not support: {inst.opcode=}"

        output_byte_size = output_data.size * output_bitwidth // 8
        self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")
        self.memory_space.write(output_data, output_addr, output_byte_size)
    
    def _run_simd_class_sqrt_inst(self, inst):
        assert inst.input_num == 1

        input_addr = self.read_general_reg(inst.reg_in1)
        input_size = self.read_general_reg(inst.reg_size)
        output_addr = self.read_general_reg(inst.reg_out)
        input_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input_byte_size = input_bitwidth * input_size // 8
        input_dtype = get_dtype_from_bitwidth(input_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        output_bitwidth = self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        self.memory_space.check_memory_type(input_addr, input_byte_size, "sram")
        input_data = self.memory_space.read_as(
            input_addr, 
            input_byte_size, 
            input_dtype
        )
        output_data = np.sqrt(input_data.astype(output_dtype)).astype(output_dtype)
        output_byte_size = output_data.size * output_bitwidth // 8
        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_gelu_inst(self, inst):
        assert inst.input_num == 1

        input_addr = self.read_general_reg(inst.reg_in1)
        input_size = self.read_general_reg(inst.reg_size)
        output_addr = self.read_general_reg(inst.reg_out)
        input_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input_byte_size = input_bitwidth * input_size // 8
        input_dtype = get_dtype_from_bitwidth(input_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        output_bitwidth = self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        self.memory_space.check_memory_type(input_addr, input_byte_size, "sram")
        input_data = self.memory_space.read_as(
            input_addr, 
            input_byte_size, 
            input_dtype
        )
        torch_tensor = torch.tensor(input_data.astype(np.float32))
        gelu_output = F.gelu(torch_tensor)
        output_data = gelu_output.numpy().astype(output_dtype)
        output_byte_size = output_data.size * output_bitwidth // 8
        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_softmax_inst(self, inst):
        assert inst.input_num == 1

        input_addr = self.read_general_reg(inst.reg_in1)
        output_addr = self.read_general_reg(inst.reg_out)
        input_size = self.read_general_reg(inst.reg_size)
        input_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input_byte_size = input_bitwidth * input_size // 8
        input_dtype = get_dtype_from_bitwidth(input_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        output_bitwidth = self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        self.memory_space.check_memory_type(input_addr, input_byte_size, "sram")
        input_data = self.memory_space.read_as(
            input_addr, 
            input_byte_size, 
            input_dtype
        )

        def softmax(x, axis=-1):
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
            
        output_data = softmax(input_data.astype(output_dtype)).astype(output_dtype)
        output_byte_size = output_data.size * output_bitwidth // 8
        self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")
        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_reduce_inst(self, inst):
        assert inst.input_num == 1

        input_addr = self.read_general_reg(inst.reg_in1)
        input_size = self.read_general_reg(inst.reg_size)
        output_addr = self.read_general_reg(inst.reg_out)
        
        # dtypes
        input_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input_dtype = get_dtype_from_bitwidth(input_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        output_bitwidth = self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth, is_float=self.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        
        input_byte_size = input_bitwidth * input_size // 8
        self.memory_space.check_memory_type(input_addr, input_byte_size, "sram")
        input_data = self.memory_space.read_as(input_addr, input_byte_size, input_dtype)
        assert input_data.shape[0] == input_size

        if inst.opcode == 11:
            # reduce max
            output_data = np.max(input_data.astype(output_dtype)).reshape(-1)
        elif inst.opcode == 13:
            # reduce sum
            # output_data = np.sum(input_data.astype(output_dtype)).reshape(-1)
            output_data = self.reduce_sum_util.reduce_sum(input_data)
            logger.debug(f"reduce sum {input_addr=}, {output_addr=}, {output_bitwidth=}, reduce {input_data} to {output_data}")
        else:
            assert False, f"Not support: {inst.opcode=}"

        output_byte_size = output_data.size * output_bitwidth // 8
        self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")
        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_floor_inst(self, inst):
        assert inst.input_num == 1

        input_addr = self.read_general_reg(inst.reg_in1)
        output_addr = self.read_general_reg(inst.reg_out)
        input_size = self.read_general_reg(inst.reg_size)
        input_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input_byte_size = input_bitwidth * input_size // 8
        self.memory_space.check_memory_type(input_addr, input_byte_size, "sram")
        input_data = self.memory_space.read_as(input_addr, input_byte_size, np.float32)
        output_data = np.floor(input_data).astype(np.int32)
        output_byte_size = output_data.size * 32 // 8
        self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")
        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_quantify_inst(self, inst):
        input_addr = self.read_general_reg(inst.reg_in1)
        bias_scale_addr = self.read_general_reg(inst.reg_in2)
        out_zp_addr = self.read_special_reg(
            SpecialReg.SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_1
        )
        input_size = self.read_general_reg(inst.reg_size)
        output_addr = self.read_general_reg(inst.reg_out)
        # clip_min = 0 if "relu" in inst and inst["relu"] else -128
        # clip_min = 0 if inst.relu else -128
        clip_min = -128
        clip_max = 127
        # print(f"{clip_min=}")
        # import pdb; pdb.set_trace()

        assert self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH) == 32
        assert self.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH) == 64
        assert self.read_special_reg(SpecialReg.SIMD_INPUT_3_BIT_WIDTH) == 8
        assert self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH) == 8

        input_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input_byte_size = input_bitwidth * input_size // 8
        self.memory_space.check_memory_type(input_addr, input_byte_size, "sram")
        input_data = self.memory_space.read_as(
            input_addr, input_byte_size, get_dtype_from_bitwidth(input_bitwidth)
        )

        # read bias and scale
        bias_scale_byte_size = input_size * 2 * 4
        bias_data = self.memory_space.read_as(
            bias_scale_addr, bias_scale_byte_size, np.int32
        )[0::2]
        scale_data = self.memory_space.read_as(
            bias_scale_addr, bias_scale_byte_size, np.float32
        )[1::2]

        # read out_zp
        out_zp_byte_size = 4
        out_zp_data = self.memory_space.read_as(out_zp_addr, out_zp_byte_size, np.int32)

        # calculate
        output_data = input_data + bias_data
        output_data = banker_round(output_data * scale_data) + out_zp_data
        output_data = banker_round(np.clip(output_data, clip_min, clip_max))
        # output_data = banker_round(np.clip(output_data, 0, 127))
        output_data = output_data.astype("int8")

        # save back
        output_byte_size = output_data.size
        # import pdb; pdb.set_trace()
        self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")
        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_resadd_quantify_inst(self, inst):
        assert inst.input_num == 4, inst.input_num
        input_1_addr = self.read_general_reg(inst.reg_in1)
        input_2_addr = self.read_general_reg(inst.reg_in2)
        bias_scale_addr = self.read_special_reg(
            SpecialReg.SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_1
        )
        out_zp_addr = self.read_special_reg(
            SpecialReg.SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_2
        )
        input_size = self.read_general_reg(inst.reg_size)
        output_addr = self.read_general_reg(inst.reg_out)
        clip_min = -128
        clip_max = 127
        # print(f"{clip_min=}")
        # import pdb; pdb.set_trace()

        assert self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH) == 8
        assert self.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH) == 8
        assert self.read_special_reg(SpecialReg.SIMD_INPUT_3_BIT_WIDTH) == 128
        assert self.read_special_reg(SpecialReg.SIMD_INPUT_4_BIT_WIDTH) == 32
        assert self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH) == 8

        input_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input_byte_size = input_bitwidth * input_size // 8
        self.memory_space.check_memory_type(input_1_addr, input_byte_size, "sram")
        input_1_data = self.memory_space.read_as(
            input_1_addr, input_byte_size, get_dtype_from_bitwidth(input_bitwidth)
        )

        input_2_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH)
        input_2_byte_size = input_2_bitwidth * input_size // 8
        self.memory_space.check_memory_type(input_2_addr, input_2_byte_size, "sram")
        input_2_data = self.memory_space.read_as(
            input_2_addr, input_2_byte_size, get_dtype_from_bitwidth(input_2_bitwidth)
        )

        # read bias and scale
        bias_scale_byte_size = 4 * 4
        bias_data = self.memory_space.read_as(
            bias_scale_addr, bias_scale_byte_size, np.int32
        )[0::2]
        scale_data = self.memory_space.read_as(
            bias_scale_addr, bias_scale_byte_size, np.float32
        )[1::2]

        # read out_zp
        out_zp_byte_size = 4
        out_zp_data = self.memory_space.read_as(out_zp_addr, out_zp_byte_size, np.int32)

        # calculate
        output_1_data = (input_1_data + bias_data[0]) * scale_data[0]
        output_2_data = (input_2_data + bias_data[1]) * scale_data[1]
        output_data = output_1_data + output_2_data
        output_data = banker_round(output_data)

        output_data = np.clip(output_data, clip_min, clip_max)
        output_data = output_data.astype("int8")

        # save back
        output_byte_size = output_data.size
        # import pdb; pdb.set_trace()
        self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")
        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_resmul_quantify_inst(self, inst):
        assert inst.input_num == 4, inst.input_num
        input_1_addr = self.read_general_reg(inst.reg_in1)
        input_2_addr = self.read_general_reg(inst.reg_in2)
        bias_scale_addr = self.read_special_reg(
            SpecialReg.SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_1
        )
        out_zp_addr = self.read_special_reg(
            SpecialReg.SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_2
        )
        input_size = self.read_general_reg(inst.reg_size)
        output_addr = self.read_general_reg(inst.reg_out)
        clip_min = -128
        clip_max = 127
        # print(f"{clip_min=}")
        # import pdb; pdb.set_trace()

        assert self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH) == 8
        assert self.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH) == 8
        assert self.read_special_reg(SpecialReg.SIMD_INPUT_3_BIT_WIDTH) == 64
        assert self.read_special_reg(SpecialReg.SIMD_INPUT_4_BIT_WIDTH) == 32
        assert self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH) == 8

        input_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input_byte_size = input_bitwidth * input_size // 8
        self.memory_space.check_memory_type(input_1_addr, input_byte_size, "sram")
        input_1_data = self.memory_space.read_as(
            input_1_addr, input_byte_size, get_dtype_from_bitwidth(input_bitwidth)
        )

        input_2_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH)
        input_2_byte_size = input_2_bitwidth * input_size // 8
        self.memory_space.check_memory_type(input_2_addr, input_2_byte_size, "sram")
        input_2_data = self.memory_space.read_as(
            input_2_addr, input_2_byte_size, get_dtype_from_bitwidth(input_2_bitwidth)
        )

        # read bias and scale
        bias_scale_byte_size = 2 * 4
        bias_data = self.memory_space.read_as(
            bias_scale_addr, bias_scale_byte_size, np.int32
        )[0::2]
        scale_data = self.memory_space.read_as(
            bias_scale_addr, bias_scale_byte_size, np.float32
        )[1::2]

        # read out_zp
        out_zp_byte_size = 4
        out_zp_data = self.memory_space.read_as(out_zp_addr, out_zp_byte_size, np.int32)

        # calculate
        output_data = input_1_data.astype(np.int32) * input_2_data.astype(np.int32) * scale_data
        output_data = banker_round(output_data)

        output_data = np.clip(output_data, clip_min, clip_max)
        output_data = output_data.astype("int8")

        # save back
        output_byte_size = output_data.size
        # import pdb; pdb.set_trace()
        self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")
        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_send_inst(self, inst):
        assert self.pipes is not None

        dst_core = self.read_general_reg(inst.reg_dst_core)
        transfer_id = self.read_general_reg(inst.reg_transfer_id)
        src_addr = self.read_general_reg(inst.reg_src_addr)
        dst_addr = self.read_general_reg(inst.reg_dst_addr)
        size = self.read_general_reg(inst.reg_size)
        data = self.memory_space.read(src_addr, size)
        data_np = np.frombuffer(data, dtype=np.float16)
        logger.info(f"[{self.core_id}] send to {dst_core}, data: {data_np}")

        assert dst_core < len(self.pipes)
        assert self.pipes[dst_core] is not None
        self.pipes[dst_core].send((data, (src_addr, dst_addr, transfer_id)))

        # Wait for acknowledgment
        ack = self.pipes[dst_core].recv()
        assert ack == "ACK", "Did not receive acknowledgment from receiver"

    def _run_recv_inst(self, inst):
        assert self.pipes is not None
        src_core = self.read_general_reg(inst.reg_src_core)
        transfer_id = self.read_general_reg(inst.reg_transfer_id)
        dst_addr = self.read_general_reg(inst.reg_dst_addr)
        src_addr = self.read_general_reg(inst.reg_src_addr)
        size = self.read_general_reg(inst.reg_size)
        
        assert src_core < len(self.pipes)
        assert self.pipes[src_core] is not None
        data, (_src_addr, _dst_addr, _transfer_id) = self.pipes[src_core].recv()

        data_np = np.frombuffer(data, dtype=np.float16)
        logger.info(f"[{self.core_id}] recv from {src_core}, data: {data_np}")
        
        self.pipes[src_core].send("ACK")
        assert _src_addr == src_addr
        assert _dst_addr == dst_addr
        assert _transfer_id == transfer_id
        
        self.memory_space.write(data, dst_addr, size)
