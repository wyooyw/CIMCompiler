import copy
import cProfile
import json
import logging
from enum import Enum

import numpy as np
from tqdm import tqdm

from simulator.data_type import get_bitwidth_from_dtype, get_dtype_from_bitwidth
from simulator.flat_inst_util import FlatInstUtil
from simulator.macro_utils import MacroConfig, MacroUtil
from simulator.mask_utils import MaskConfig, MaskUtil
from simulator.meta_utils import MetaUtil
from simulator.stats_util import StatsUtil
from utils.df_layout import tensor_int8_to_bits
from utils.round import banker_round

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


class InstClass(Enum):
    PIM_CLASS = 0  # 0b00
    SIMD_CLASS = 1  # 0b01
    SCALAR_CLASS = 2  # 0b10
    TRANS_CLASS = 6  # 0b110
    CTR_CLASS = 7  # 0b111
    DEBUG_CLASS = -1


class PIMInstType(Enum):
    PIM_COMPUTE = 0  # 0b00
    PIM_SET = 1  # 0b01
    PIM_OUTPUT = 2  # 0b10
    PIM_TRANSFER = 3  # 0b11


class ScalarInstType(Enum):
    RR = 0  # 0b00
    RI = 1  # 0b01
    LOAD_STORE = 2  # 0b10
    OTHER = 3  # 0b11


class ControlInstType(Enum):
    EQ_BR = 0  # 0b000
    NE_BR = 1  # 0b001
    GT_BR = 2  # 0b010
    LT_BR = 3  # 0b011
    JUMP = 4  # 0b100


class TransInstType(Enum):
    TRANS = 0  # 0b0


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
        for memory in self.memory_space:
            if offset >= memory.offset and (
                offset + size <= memory.offset + memory.size
            ):
                assert (
                    memory.memtype in memtype
                ), f"require {memtype=}, but get {memory.memtype=}. {offset=}, {size=}"
                return
        assert False, f"can not find memory! {offset=}, {size=}, {memtype=}"

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
        for memory in self.memory_space:
            if memory.name == name:
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
        memory_config_path="/home/wangyiou/project/cim_compiler_frontend/playground/config/config.json",
    ):
        with open(memory_config_path, "r") as f:
            memory_config = json.load(f)
        memory_space = cls()
        for memory in memory_config["memory_list"]:
            name = memory["name"]
            memtype = memory["type"]
            offset = memory["addressing"]["offset_byte"]
            size = memory["addressing"]["size_byte"]
            logging.debug(f"Add memory: {name=}, {memtype=}, {offset=}, {size=}")
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
        safe_time=999999999,
        mask_memory_name="mask",
    ):
        super().__init__()
        self.general_rf = np.zeros([64], dtype=np.int32)
        self.special_rf = np.zeros([32], dtype=np.int32)
        self.memory_space = memory_space
        self.macro_config = macro_config
        self.mask_config = mask_config
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
        self.jump_offset = None
        self.safe_time = safe_time

        self.debug_hook = None

        self._int_data_type = {8: np.int8, 16: np.int16, 32: np.int32}

        self.print_record = list()

        self._add_internel_macro_output_buffer()

        self._read_reg_value_directly = False

    def _add_internel_macro_output_buffer(self):
        """
        This is an internal memory for doing accumulate for macro's output
        """
        if self.memory_space.get_memory_by_name("pim_output_reg_buffer") is None:
            logging.debug(
                "[Warning] Can't find pim_output_reg_buffer. Make sure the code has no macro-related instruction."
            )
            return
        end_memory = self.memory_space.memory_space[-1]
        end_offset = end_memory.offset + end_memory.size
        output_buffer_size = self.memory_space.get_memory_by_name(
            "pim_output_reg_buffer"
        ).size
        internel_macro_output_buffer = Memory(
            "internel_macro_output_reg_buffer",
            "reg_buffer",
            offset=end_offset,
            size=output_buffer_size,
        )
        logging.debug(f"{end_offset=}")
        self.memory_space.add_memory(internel_macro_output_buffer)

    @classmethod
    def from_config(
        cls,
        config_path="/home/wangyiou/project/cim_compiler_frontend/playground/config/config.json",
    ):
        with open(config_path, "r") as f:
            config = json.load(f)
        memory_space = MemorySpace.from_memory_config(config_path)
        macro_config = MacroConfig.from_config(config_path)
        mask_config = MaskConfig.from_config(config_path)
        if "mask_memory_name" in config:
            return cls(
                memory_space,
                macro_config,
                mask_config,
                mask_memory_name=config["mask_memory_name"],
            )
        else:
            return cls(memory_space, macro_config, mask_config)

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
            self.stats_util.record_reg_status(pc + 1, cnt, self.general_rf)
            if record_flat:
                self.flat_inst_util.flat_inst(inst, cnt)

            inst_class = inst["class"]
            if inst_class == InstClass.PIM_CLASS.value:
                self._run_pim_class_inst(inst)
            elif inst_class == InstClass.SIMD_CLASS.value:
                self._run_simd_class_inst(inst)
            elif inst_class == InstClass.SCALAR_CLASS.value:
                self._run_scalar_class_inst(inst)
            elif inst_class == InstClass.TRANS_CLASS.value:
                self._run_trans_class_inst(inst)
            elif inst_class == InstClass.CTR_CLASS.value:
                self._run_control_class_inst(inst)
            elif inst_class == InstClass.DEBUG_CLASS.value:
                self._run_debug_class_inst(inst)
            else:
                assert False, f"Not support {inst_class=}"

            if self.jump_offset is not None:
                pc += self.jump_offset
                self.jump_offset = None
            else:
                pc += 1

            cnt += 1

        print(f"{self.pimcompute_cnt=}, {total_pim_compute_count=}")
        self.pbar.close()

        if pc == len(code):
            logging.debug("Run finish!")
            return self.FINISH, self.stats_util, self.flat_inst_util
        elif pc < len(code) and cnt == self.safe_time:
            logging.debug("Meet safe time!")
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

    def _run_pim_class_inst(self, inst):
        inst_type = inst["type"]
        if inst_type == PIMInstType.PIM_COMPUTE.value:
            self._run_pim_class_pim_compute_type_inst(inst)
        elif inst_type == PIMInstType.PIM_OUTPUT.value:
            self._run_pim_class_pim_output_type_inst(inst)
        elif inst_type == PIMInstType.PIM_TRANSFER.value:
            self._run_pim_class_pim_transfer_type_inst(inst)
        elif inst_type == PIMInstType.PIM_SET.value:
            self._run_pim_class_pim_set_type_inst(inst)
        else:
            assert False, f"Not support"

    def _run_simd_class_inst(self, inst):
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
        opcode = inst["opcode"]
        if opcode in [0x00, 0x02]:  # vec add
            self._run_simd_class_vector_vector_inst(inst)
        elif opcode == 0b01:  # scalar add
            self._run_simd_class_scalar_vector_inst(inst)
        elif opcode == 3:
            self._run_simd_class_quantify_inst(inst)
        else:
            assert False, f"Not support {opcode=} yet."

    def _run_scalar_class_inst(self, inst):
        inst_type = inst["type"]
        # import pdb; pdb.set_trace()
        if inst_type == ScalarInstType.RR.value:
            self._run_scalar_class_rr_type_inst(inst)
        elif inst_type == ScalarInstType.RI.value:
            self._run_scalar_class_ri_type_inst(inst)
        elif inst_type == ScalarInstType.LOAD_STORE.value:
            self._run_scalar_class_load_store_type_inst(inst)
        elif inst_type == ScalarInstType.OTHER.value:
            self._run_scalar_class_other_type_inst(inst)
        else:
            assert False, f"Not support"

    def _run_control_class_inst(self, inst):
        inst_type = inst["type"]
        if inst_type in [
            ControlInstType.EQ_BR.value,
            ControlInstType.NE_BR.value,
            ControlInstType.GT_BR.value,
            ControlInstType.LT_BR.value,
        ]:
            self._run_control_class_br_type_inst(inst)
        elif inst_type == ControlInstType.JUMP.value:
            self._run_control_class_jump_type_inst(inst)
        else:
            assert False, f"Not support"

    def _run_trans_class_inst(self, inst):
        inst_type = inst["type"]
        if inst_type == TransInstType.TRANS.value:
            self._run_trans_class_trans_type_inst(inst)
        else:
            assert False, f"Not support"

    """
    Types
    """

    def _run_scalar_class_rr_type_inst(self, inst):
        """
        - [31, 30]，2bit：class，指令类别码，值为10
        - [29, 28]，2bit：type，指令类型码，值为00
        - [27, 26]，2bit：reserve，保留字段
        - [25, 21]，5bit：rs1，通用寄存器1，表示运算数1的值
        - [20, 16]，5bit：rs2，通用寄存器2，表示运算数2的值
        - [15, 11]，5bit：rd，通用寄存器3，即运算结果写回的寄存器
        - [10, 3]，8bit：reserve，保留字段
        - [2, 0]，3bit：opcode，操作类别码，表示具体计算的类型
        - 000：add，整型加法
        - 001：sub，整型减法
        - 010：mul，整型乘法，结果寄存器仅保留低32位
        - 011：div，整型除法，结果寄存器仅保留商
        - 100：sll，逻辑左移
        - 101：srl，逻辑右移
        - 110：sra，算数右移
        """
        value1 = self.read_general_reg(inst["rs1"])
        value2 = self.read_general_reg(inst["rs2"])
        opcode = inst["opcode"]
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
        else:
            assert False, f"Not support {opcode=}."
        self.write_general_reg(inst["rd"], result)

    def _run_scalar_class_ri_type_inst(self, inst):
        """
        R-I型整数运算指令：scalar-RI
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为10
        - [29, 28]，2bit：type，指令类型码，值为01
        - [27, 26]，2bit：opcode，操作类别码，表示具体计算的类型
        - 00：addi，整型立即数加法
        - 01：muli，整型立即数乘法，结果寄存器仅保留低32位
        - 10：lui，高16位立即数赋值
        - [25, 21]，5bit：rs，通用寄存器1，表示运算数1的值
        - [20, 16]，5bit：rd，通用寄存器2，即运算结果写回的寄存器
        - [15, 0]，16bit：imm，立即数，表示运算数2的值
        """
        value = self.read_general_reg(inst["rs"])
        imm = inst["imm"]
        opcode = inst["opcode"]
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
        self.write_general_reg(inst["rd"], result)

    def _run_scalar_class_load_store_type_inst(self, inst):
        """
        Load/Store指令：scalar-SL
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为10
        - [29, 28]，2bit：type，指令类型码，值为10
        - [27, 26]，2bit：opcode，操作类别码，表示具体操作的类型
        - 00：本地存储load至寄存器
        - 01：寄存器值store至本地存储
        - 10：全局存储load至寄存器
        - 11：寄存器值store至全局存储
        - [25, 21]，5bit：rs1，通用寄存器1，即寻址的基址寄存器base
        - [20, 16]，5bit：rs2，通用寄存器2，即存储load/store值的寄存器
        - [15, 0]，16bit：offset，立即数，表示寻址的偏移值
        - 地址计算公式：$rs + offset
        """
        opcode = inst["opcode"]
        if opcode == 0b00:  # load

            addr = self.read_general_reg(inst["rs1"])
            offset = inst["offset"]
            addr += offset
            value = self.memory_space.read_as(addr, 4, np.int32).item()
            self.write_general_reg(inst["rs2"], value)

        elif opcode == 0b01:  # store

            addr = self.read_general_reg(inst["rs1"])
            value = self.read_general_reg(inst["rs2"])
            offset = inst["offset"]
            addr += offset
            self.memory_space.write(np.array([value], dtype=np.int32), addr, 4)

        else:
            assert False, f"Not support {opcode=}."

    def _run_scalar_class_other_type_inst(self, inst):
        assert inst["class"] == 0b10
        assert inst["type"] == 0b11
        if inst["opcode"] in [0b00, 0b01]:
            self._run_scalar_class_other_type_li_inst(inst)
        elif inst["opcode"] in [0b10, 0b11]:
            self._run_scalar_class_other_type_special_general_assign_inst(inst)
        else:
            assert False, f"Not support {inst['opcode']=}"

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
        rd = inst["rd"]
        imm = inst["imm"]
        opcode = inst["opcode"]
        if opcode == 0b00:  # 通用寄存器
            rf = self.general_rf
        elif opcode == 0b01:  # 专用寄存器
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
        if opcode == 0b10:
            value = self.read_general_reg(inst["rs1"])
            self.write_special_reg(inst["rs2"], value)
        elif opcode == 0b11:
            value = self.read_special_reg(inst["rs2"])
            self.write_general_reg(inst["rs1"], value)
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
        src_base = self.read_general_reg(inst["rs1"])
        dst_base = self.read_general_reg(inst["rd"])
        offset = inst["offset"]
        src_offset_mask = inst["source_offset_mask"]
        dst_offset_mask = inst["destination_offset_mask"]
        size = self.read_general_reg(inst["rs2"])

        src_addr = src_base + src_offset_mask * offset
        dst_addr = dst_base + dst_offset_mask * offset

        logging.debug(
            "Trans: from {}({}) to {}({}), {} bytes".format(
                str(src_addr),
                self.memory_space.get_memory_by_address(src_addr).name,
                str(dst_addr),
                self.memory_space.get_memory_by_address(dst_addr).name,
                str(size),
            )
        )

        src_data = self.memory_space.read(src_addr, size)
        self.memory_space.write(src_data, dst_addr, size)

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
        val1 = self.read_general_reg(inst["rs1"])
        val2 = self.read_general_reg(inst["rs2"])
        inst_type = inst["type"]
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
            self.jump_offset = inst["offset"]

    def _run_control_class_jump_type_inst(self, inst):
        """
        无条件跳转指令：jmp
        指令字段划分：
        - [31, 29]，3bit：class，指令类别码，值为111
        - [28, 26]，3bit：type，指令类型码，值为100
        - [25, 0]，26bit：offset，立即数，表示跳转指令地址相对于该指令的偏移值
        """
        self.jump_offset = inst["offset"]

    def _run_scalar_class_other_type_inst(self, inst):
        opcode = inst["opcode"]
        if opcode == 0b00:  # general-li
            self.write_general_reg(inst["rd"], inst["imm"])
        elif opcode == 0b01:  # special-li
            self.write_special_reg(inst["rd"], inst["imm"])
        elif opcode == 0b10:  # general-to-special
            val = self.read_general_reg(inst["rs1"])
            self.write_special_reg(inst["rs2"], val)
        elif opcode == 0b11:  # special-to-general
            val = self.read_special_reg(inst["rs2"])
            self.write_general_reg(inst["rs1"], val)
        else:
            assert False, "Not support yet"

    def _run_pim_class_pim_compute_type_inst(self, inst):
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
        使用的专用寄存器：
        - input bit width：输入的bit长度
        - output bit width：输出的bit长度
        - weight bit width：权重的bit长度
        - group size：macro group的大小，即包含多少个macro，仅允许设置为config文件里设置的数值之一
        - activation group num：激活的group的数量
        - activation element col num：每个group内激活的element列的数量
        - group input step/offset addr：每一组输入向量的起始地址相对于上一组的增量（step），或相对于rs1的偏移量的地址（offset addr）
        - value sparse mask addr：值稀疏掩码Mask的起始地址
        - bit sparse meta addr：Bit级稀疏Meta数据的起始地址
        """
        # assert inst["group"] == 0, "Not support group yet."
        # if inst["value_sparse"] == 1 and inst["bit_sparse"] == 1:
        #     self._run_pim_class_pim_compute_type_inst_value_bit_sparse(inst)
        # elif inst["value_sparse"] == 1:
        #     self._run_pim_class_pim_compute_type_inst_value_sparse(inst)
        if inst["bit_sparse"] == 1:
            self._run_pim_class_pim_compute_type_inst_bit_sparse(inst)
        else:
            self._run_pim_class_pim_compute_type_inst_dense(inst)
        self.pbar.update(1)
        self.pimcompute_cnt += 1

    """
    Diffenet Pim Compute
    """

    def _run_pim_class_pim_compute_type_inst_value_bit_sparse(self, inst):
        assert False, "Executor not support value & bit sparse yet."

    def _run_pim_class_pim_compute_type_inst_value_sparse(self, inst):
        assert False, "Executor not support value sparse yet."

    def _stats_macro_util(self, macro_id, group_size, n_use_comp, width_bw):
        pimset_mask = self.pimset_mask.reshape(group_size, -1)
        n_comp = self.macro_config.n_comp
        n_col = self.macro_config.n_bcol

        macro_pimset_mask = ~pimset_mask[macro_id]
        n_use_col = (macro_pimset_mask.sum() * width_bw).item()
        if n_use_col > 0 and n_use_comp > 0:
            self.stats_util.record_macro_ultilize(n_use_comp, n_use_col, n_comp * n_col)

    def _value_sparsity_compute(self, inst, input_data, weight_data):
        if inst["value_sparse"] == 0:
            return input_data
        output_bw = self.read_special_reg(SpecialReg.OUTPUT_BIT_WIDTH)
        width_bw = self.read_special_reg(SpecialReg.WEIGHT_BIT_WIDTH)
        group_size = self.read_special_reg(SpecialReg.GROUP_SIZE)
        logging.debug(f"{group_size=}, {self.macro_config.n_macro=}")
        logging.debug(f"old {weight_data.shape=}")
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
        logging.debug(f"new {weight_data.shape=}")
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
        # logging.debug(f"{mask_addr=}")
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
        # logging.debug(f"{mask_data.shape=}")
        # logging.debug(mask_data.astype(np.int8))
        # logging.debug(f"{weight_data.shape=}")
        # logging.debug(weight_data.reshape(self.macro_config.n_comp, -1))
        # import pdb; pdb.set_trace()
        pimset_mask = self.pimset_mask.reshape(group_size, -1)
        n_comp = self.macro_config.n_comp
        n_col = self.macro_config.n_bcol

        output_list = []
        out_dtype = get_dtype_from_bitwidth(output_bw)
        for macro_id in range(group_size):
            # get macro input data
            macro_mask = mask_data[macro_id]
            # import pdb; pdb.set_trace()
            macro_input_data = input_data[macro_mask]
            # logging.debug(f"{input_data=}, {macro_mask=}, {macro_input_data=}")
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
            # logging.debug(f"{macro_input_data=}, {macro_weight=}, {macro_output=}")
            output_list.append(macro_output)

            n_use_comp = macro_mask.sum().item()
            # macro_pimset_mask = ~pimset_mask[macro_id]
            # n_use_col = (macro_pimset_mask.sum() * width_bw).item()
            # if n_use_col > 0:
            #     self.stats_util.record_macro_ultilize(n_use_comp, n_use_col, n_comp * n_col)
            self._stats_macro_util(macro_id, group_size, n_use_comp, width_bw)
        # import pdb; pdb.set_trace()
        output_data = np.concatenate(output_list)
        # logging.debug(f"{output_data=}")
        return output_data

    def _run_pim_class_pim_compute_type_inst_dense(self, inst):
        input_offset = self.read_general_reg(inst["rs1"])
        input_size = self.read_general_reg(inst["rs2"])
        activate_row = self.read_general_reg(inst["rs3"])
        # output_offset = self.read_general_reg(inst["rd"])
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
        group_input_step = self.read_special_reg(SpecialReg.GROUP_INPUT_STEP)
        assert inst.get("group", -1) == 1
        assert inst.get("group_input_mode", -1) == 0
        logging.debug(f"{group_num=}")
        # logging.debug(f"{self.macro_config.n_macro=}")
        # logging.debug(f"{self.macro_config.n_macro=}")

        value_sparsity = inst["value_sparse"]
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
            width_bw,
            group_num,
            activate_element_row_num,
            activation_element_col_num,
            activation_group_num,
        )  # shape: [compartment, group, vcolumn]
        logging.debug(f"{weight_data.shape=}")
        group_weight_data = []
        for group_id in range(activation_group_num):
            group_weight_data.append(weight_data[:, group_id, :])

        # compute
        group_output_data = []
        for group_id in range(activation_group_num):
            input_data = group_input_data[group_id]
            weight_data = group_weight_data[group_id]
            # logging.debug(f"{input_data=}, {weight_data=}")

            # use pimset to mask weight
            assert self.pimset_mask is not None
            assert (
                len(self.pimset_mask) == weight_data.shape[1]
            ), f"{len(self.pimset_mask)=}, {weight_data.shape[1]=}"
            assert self.pimset_mask.dtype == bool, f"{self.pimset_mask.dtype=}"
            weight_data[:, self.pimset_mask] = 0
            # import pdb; pdb.set_trace()

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
            # import pdb; pdb.set_trace()
            if inst["accumulate"] == 1:
                output_data_ori = self.memory_space.read_as(
                    group_output_offset, output_byte_size, out_dtype
                )
                output_data = output_data + output_data_ori
            # else:
            #     assert False
            self.memory_space.write(output_data, group_output_offset, output_byte_size)

    def _run_pim_class_pim_compute_type_inst_bit_sparse(self, inst):
        input_offset = self.read_general_reg(inst["rs1"])
        input_size = self.read_general_reg(inst["rs2"])
        activate_row = self.read_general_reg(inst["rs3"])
        # output_offset = self.read_general_reg(inst["rd"])
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
        group_input_step = self.read_special_reg(SpecialReg.GROUP_INPUT_STEP)
        assert inst.get("group", -1) == 1
        assert inst.get("group_input_mode", -1) == 0
        logging.debug(f"{group_num=}")
        # logging.debug(f"{self.macro_config.n_macro=}")
        # logging.debug(f"{self.macro_config.n_macro=}")
        value_sparsity = inst["value_sparse"]
        assert "bit_sparse" in inst and inst["bit_sparse"] == 1, str(inst)
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
            group_num,
            activate_element_row_num,
            self.macro_config.n_vcol(8) * group_size,  # activation_element_col_num,
            activation_group_num,
        )  # shape: [compartment, group, vcolumn]
        logging.debug(f"{weight_data.shape=}")
        group_weight_data = []
        for group_id in range(activation_group_num):
            _weight = weight_data[:, group_id, :]
            _weight = self.meta_util.recover_weight(meta_addr, _weight, group_num)
            group_weight_data.append(_weight)

        # compute
        group_output_data = []
        for group_id in range(activation_group_num):
            input_data = group_input_data[group_id]
            weight_data = group_weight_data[group_id]
            # logging.debug(f"{input_data=}, {weight_data=}")

            # use pimset to mask weight
            assert self.pimset_mask is not None
            assert (
                len(self.pimset_mask) == weight_data.shape[1]
            ), f"{len(self.pimset_mask)=}, {weight_data.shape[1]=}"
            assert self.pimset_mask.dtype == bool, f"{self.pimset_mask.dtype=}"
            weight_data[:, self.pimset_mask] = 0

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
            if inst["accumulate"] == 1:
                output_data_ori = self.memory_space.read_as(
                    group_output_offset, output_byte_size, out_dtype
                )
                output_data = output_data + output_data_ori
            # else:
            #     assert False
            self.memory_space.write(output_data, group_output_offset, output_byte_size)

    def _run_pim_class_pim_output_type_inst(self, inst):
        outsum_move = inst["outsum_move"]
        outsum = inst["outsum"]
        if outsum:
            self._outsum(inst)
            return

        # elif outsum_move:
        #     _outsum_move(inst)
        #     return

        if outsum_move or outsum:
            assert False, "This should not happend!"

        # out_n = self.read_general_reg(inst["rs1"])
        dst_offset = self.read_general_reg(inst["rd"])

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
        out_n = self.read_general_reg(inst["rs1"])
        assert out_n % 8 == 0
        out_mask_addr = self.read_general_reg(inst["rs2"])
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

        dst_offset = self.read_general_reg(inst["rd"])
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
        """
        pim数据传输：pim-transfer
        该指令针对【使用”基于CSD编码的bit-level sparsity“算法的pim运算结果】，在阈值有1和2的情况下，在output reg buffer中不规则、不连续的问题，专门用于搬运pim运算结果，且该指令需要使用缓冲区
        指令字段划分：
        - [31, 30]，2bit：class，指令类别码，值为00
        - [29, 28]，2bit：type，指令类型码，值为11
        - [19, 15]，5bit：rs1，通用寄存器1，src addr，表示源本地存储器的地址
        - [14, 10]，5bit：rs2，通用寄存器2，output num，表示output的数量，包含有效值和无效值，也即掩码的长度
        - [9, 5]，5bit：rs3，通用寄存器3，output mask，表示掩码的存储地址，掩码的每一bit表示对应的output是否有效，掩码长度由rs2指定
        - [4, 0]，5bit：rd，通用寄存器4，dst addr，表示目的本地存储器的地址
        使用的专用寄存器：
        - output bit width：输出的bit长度
        """
        src_addr = self.read_general_reg(inst["rs1"])
        output_num = self.read_general_reg(inst["rs2"])
        output_mask_addr = self.read_general_reg(inst["rs3"])
        dst_addr = self.read_general_reg(inst["rd"])
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
        # logging.debug(f"{data=}")
        # logging.debug(f"{output_mask=}")
        filtered_data = data[output_mask == 1]
        # logging.debug(f"{filtered_data=}")
        # import pdb; pdb.set_trace()
        assert (
            filtered_data.size == output_mask.sum()
        ), f"{filtered_data.size=}, {data.sum()=}"
        self.memory_space.write(
            filtered_data, dst_addr, filtered_data.size * output_byte
        )

    def _run_pim_class_pim_set_type_inst(self, inst):
        """
        pim设置：pim-set
        设置pim单元的一些参数，以每个MacroGroup为单位进行设置，设置的参数包括每个macro激活的element列等
        - [31, 30]，2bit：class，指令类别码，值为00
        - [29, 28]，2bit：type，指令类型码，值为01
        - [27, 21]，7bit：reserve，保留字段
        - [20, 20]，1bit：group broadcast，表示是否进行设置的组广播
        - 0：不进行组广播，即仅对单个MacroGroup进行设置，MacroGroup编号由寄存器rs1给出
        - 1：进行组广播，即对所有MacroGroup进行该次设置，此时忽略寄存器rs1
        - [19, 15]，5bit：rs1，通用寄存器1，表示单播时设置的MacroGroup编号
        - [14, 10]，5bit：rs2，通用寄存器2，表示一个MacroGroup内所有Macro激活element列的掩码mask地址
        - 每个element列对应1bit mask，0表示不激活，1表示激活
        - 每个Macro的mask从前到后依次排布，连续存储
        - [9, 0]，10bit：reserve，保留字段
        """
        assert inst["group_broadcast"] == 1, "Only support group broadcast"
        mask_addr = self.read_general_reg(inst["rs2"])

        group_size = self.read_special_reg(SpecialReg.GROUP_SIZE)
        vcol = self.read_special_reg(SpecialReg.WEIGHT_BIT_WIDTH)
        n_vcol_per_group = self.macro_config.n_vcol(vcol) * group_size
        mask_size = n_vcol_per_group // 8

        mask_data = self.memory_space.read_as(mask_addr, mask_size, np.int8)
        mask_data = tensor_int8_to_bits(mask_data)
        mask_data = mask_data.reshape(-1)
        self.stats_util.record_pimset_mask(mask_data.tolist(), vcol)

        # import pdb; pdb.set_trace()
        mask_data = mask_data.astype(bool)
        mask_data = ~mask_data
        self.pimset_mask = mask_data.copy()

    def _run_debug_class_inst(self, inst):
        if inst["type"] == 0:  # print
            rs = inst["rs"]
            val = self.read_general_reg(rs)
            self.print_record.append(val)
            logging.info(f" general_reg[{rs}] = {val}")
        elif inst["type"] == 1:
            import pdb

            pdb.set_trace()
            if self.debug_hook is not None:
                self.debug_hook(simulator=self)
        else:
            assert False, "Not support yet."

    def _run_simd_class_vector_vector_inst(self, inst):
        """
        support: 1.vec add; 2.vec mul
        """
        opcode = inst["opcode"]

        # Prepare input
        input_size = self.read_general_reg(inst["rs3"])

        input1_addr = self.read_general_reg(inst["rs1"])
        input1_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input1_byte_size = input1_bitwidth * input_size // 8
        # self.memory_space.check_memory_type(input1_addr, input1_byte_size, "sram")

        input2_addr = self.read_general_reg(inst["rs2"])
        input2_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH)
        input2_byte_size = input2_bitwidth * input_size // 8
        # self.memory_space.check_memory_type(input2_addr, input2_byte_size, "sram")

        output_addr = self.read_general_reg(inst["rd"])
        output_bitwidth = self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth)

        input1_data = self.memory_space.read_as(
            input1_addr, input1_byte_size, get_dtype_from_bitwidth(input1_bitwidth)
        )
        input2_data = self.memory_space.read_as(
            input2_addr, input2_byte_size, get_dtype_from_bitwidth(input2_bitwidth)
        )

        # Compute
        if opcode == 0x00:
            assert input1_bitwidth == 32
            assert input2_bitwidth == 32
            assert output_bitwidth == 32
            output_data = input1_data.astype(output_dtype) + input2_data.astype(
                output_dtype
            )
        elif opcode == 0x02:
            assert input1_bitwidth == 8
            assert input2_bitwidth == 8
            assert output_bitwidth == 32
            output_data = input1_data.astype(output_dtype) * input2_data.astype(
                output_dtype
            )
        else:
            assert False, f"Not support: {opcode=}"

        # Save output
        output_byte_size = output_data.size * output_bitwidth // 8
        # self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")

        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_scalar_vector_inst(self, inst):
        """
        support: scalar-vec add
        """
        opcode = inst["opcode"]

        # Prepare input
        input_size = self.read_general_reg(inst["rs3"])

        input1_addr = self.read_general_reg(inst["rs1"])
        input1_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input1_byte_size = input1_bitwidth * input_size // 8
        self.memory_space.check_memory_type(input1_addr, input1_byte_size, "sram")

        input2_value = self.read_general_reg(inst["rs2"])
        input2_bitwidth = self.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH)
        input2_dtype = get_dtype_from_bitwidth(input2_bitwidth)

        output_addr = self.read_general_reg(inst["rd"])
        output_bitwidth = self.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth)

        input1_data = self.memory_space.read_as(
            input1_addr, input1_byte_size, get_dtype_from_bitwidth(input1_bitwidth)
        )
        input2_data = np.array([input2_value], dtype=output_dtype)

        # Compute
        output_data = input2_data.astype(output_dtype) + input1_data.astype(
            output_dtype
        )

        # Save output
        output_byte_size = output_data.size * output_bitwidth // 8
        self.memory_space.check_memory_type(output_addr, output_byte_size, "sram")

        self.memory_space.write(output_data, output_addr, output_byte_size)

    def _run_simd_class_quantify_inst(self, inst):
        input_addr = self.read_general_reg(inst["rs1"])
        bias_scale_addr = self.read_special_reg(
            SpecialReg.SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_1
        )
        out_zp_addr = self.read_general_reg(inst["rs2"])
        input_size = self.read_general_reg(inst["rs3"])
        output_addr = self.read_general_reg(inst["rd"])
        clip_min = 0 if inst["relu"] else -128
        clip_max = 127
        # print(f"{clip_min=}")
        # print(f"{inst['relu']=}")
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
