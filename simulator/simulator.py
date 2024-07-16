from enum import Enum
import numpy as np
from simulator.macro_utils import MacroUtil, MacroConfig
import copy
from simulator.data_type import get_dtype_from_bitwidth, get_bitwidth_from_dtype

class SpecialReg(Enum):

    # pim special reg
    INPUT_BIT_WIDTH = 0
    OUTPUT_BIT_WIDTH = 1
    WEIGHT_BIT_WIDTH = 2
    GROUP_SIZE = 3
    ACTIVATION_GROUP_NUM = 4
    ACTIVATION_ELEMENT_COL_NUM = 5
    GROUP_INPUT_STEP_ADDR = 6
    VALUE_SPARSE_MASK_ADDR = 7
    BIT_SPARSE_META_ADDR = 8

    # simd special reg
    SIMD_INPUT_1_BIT_WIDTH = 16
    SIMD_INPUT_2_BIT_WIDTH = 17
    SIMD_INPUT_3_BIT_WIDTH = 18
    SIMD_INPUT_4_BIT_WIDTH = 19
    SIMD_OUTPUT_BIT_WIDTH = 20

class InstClass(Enum):
    PIM_CLASS = 0 # 0b00
    SIMD_CLASS = 1 # 0b01
    SCALAR_CLASS = 2 # 0b10
    TRANS_CLASS = 6 # 0b110
    CTR_CLASS = 7 # 0b111
    DEBUG_CLASS = -1

class PIMInstType(Enum):
    PIM_COMPUTE = 0 # 0b0
    PIM_BATCH = 1 # 0b1

class ScalarInstType(Enum):
    RR = 0          # 0b00
    RI = 1             # 0b01
    LOAD_STORE = 2      # 0b10
    OTHER = 3           # 0b11

class ControlInstType(Enum):
    EQ_BR = 0       # 0b000
    NE_BR = 1       # 0b001
    GT_BR = 2       # 0b010
    LT_BR = 3       # 0b011
    JUMP = 4        # 0b100

class TransInstType(Enum):
    TRANS = 0 # 0b0

class Memory:
    def __init__(self, name, memtype, offset, size):
        self.name = name
        self.memtype = memtype
        self.offset = offset
        self.size = size
        self._data = bytearray(np.zeros((size,) , dtype=np.int8))

    def _check_range(self, offset, size):
        return (offset >= self.offset) and (offset + size <= self.offset + self.size)

    def read(self, offset, size):
        assert self._check_range(offset, size)
        offset = offset - self.offset
        return copy.copy(self._data[offset: offset+size])

    def read_all(self):
        return copy.copy(self._data)

    def write(self, data, offset, size):
        assert self._check_range(offset, size)
        assert type(data) in [np.array, np.ndarray, bytearray], f"{type(data)=}"
        if type(data) in [np.array, np.ndarray]:
            data = bytearray(data)
        assert len(data) == size, f"{len(data)=}, {size=}"

        offset = offset - self.offset
        self._data[offset: offset+size] = data
        assert len(self._data) == self.size, f"{len(self._data)=}, {self.size=}, {offset=}, "


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
            if offset >= memory.offset and (offset+size <= memory.offset+memory.size):
                return memory
        assert False, 'can not find memory!'

    def check_memory_type(self, offset, size, memtype):
        """
        check [offset, offset + size) is in one memory
        """
        for memory in self.memory_space:
            if offset >= memory.offset and (offset+size <= memory.offset+memory.size):
                assert memory.memtype==memtype, f"require {memtype=}, but get {memory.memtype=}"
                return
        assert False, f'can not find memory! {offset=}, {size=}, {memtype=}'

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
            if memory.memtype=="macro":
                return memory
        return None

    def get_base_of(self, name):
        for memory in self.memory_space:
            if memory.name==name:
                return memory.offset
        assert False, f"Can not find {name=}"
        return None

class Simulator:
    FINISH = 0
    TIMEOUT = 1
    ERROR = 2
    def __init__(self, memory_space, macro_config, safe_time=999999):
        super().__init__()
        self.general_rf = np.zeros([32], dtype=np.int32)
        self.special_rf = np.zeros([32], dtype=np.int32)
        self.memory_space = memory_space
        self.macro_util = MacroUtil(self.memory_space.get_macro_memory(), macro_config)

        self.jump_offset = None
        self.safe_time = safe_time

        self._int_data_type = {
            8: np.int8,
            16: np.int16,
            32: np.int32
        }
    
    def get_dtype(self, bitwidth):
        assert bitwidth in self._int_data_type
        return self._int_data_type[bitwidth]

    def run_code(self, code: list[dict]):
        pc = 0
        cnt = 0
        
        while pc < len(code) and cnt < self.safe_time:
            inst = code[pc]
            inst_class = inst["class"]
            if inst_class==InstClass.PIM_CLASS.value:
                self._run_pim_class_inst(inst)
            elif inst_class==InstClass.SIMD_CLASS.value:
                self._run_simd_class_inst(inst)
            elif inst_class==InstClass.SCALAR_CLASS.value:
                self._run_scalar_class_inst(inst)
            elif inst_class==InstClass.TRANS_CLASS.value:
                self._run_trans_class_inst(inst)
            elif inst_class==InstClass.CTR_CLASS.value:
                self._run_control_class_inst(inst)
            elif inst_class==InstClass.DEBUG_CLASS.value:
                self._run_debug_class_inst(inst)
            if self.jump_offset is not None:
                pc += self.jump_offset
                self.jump_offset = None
            else:
                pc += 1
            
            cnt += 1

        if pc == len(code):
            print("Run finish!")
            return self.FINISH
        elif pc < len(code) and cnt == self.safe_time:
            print("Meet safe time!")
            return self.TIMEOUT
        else:
            print(f"Strange exit situation! {pc=}, {len(code)=}, {cnt=}, {self.safe_time=}")
            return self.ERROR
    
    def read_general_reg(self, regid):
        return self.read_reg(self.general_rf, regid)

    def write_general_reg(self, regid, value):
        self.write_reg(self.general_rf, regid, value)

    def read_special_reg(self, regid):
        if type(regid)==SpecialReg:
            regid = regid.value
        assert type(regid)==int, f"{regid=}"
        return self.read_reg(self.special_rf, regid)

    def write_special_reg(self, regid, value):
        if type(regid)==SpecialReg:
            regid = regid.value
        assert type(regid)==int, f"{regid=}"
        self.write_reg(self.special_rf, regid, value)

    def read_reg(self, rf, regid):
        assert 0 <= regid and regid < rf.shape[0], f"{regid=}"
        return rf[regid]

    def write_reg(self, rf, regid, value):
        assert 0 <= regid and regid < rf.shape[0]
        # TODO: check value is in range of int32
        rf[regid] = value

    """
    Classes
    """
    def _run_pim_class_inst(self, inst):
        inst_type = inst["type"]
        if inst_type==PIMInstType.PIM_COMPUTE.value:
            self._run_pim_class_pim_compute_type_inst(inst)
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
        pass


    def _run_scalar_class_inst(self, inst):
        inst_type = inst["type"]
        # import pdb; pdb.set_trace()
        if inst_type==ScalarInstType.RR.value:
            self._run_scalar_class_rr_type_inst(inst)
        # elif inst_type==ScalarInstType.RI:
        #     self._run_scalar_class_ri_type_inst(inst)
        # elif inst_type==ScalarInstType.LOAD_STORE:
        #     self._run_scalar_class_load_store_type_inst(inst)
        elif inst_type==ScalarInstType.OTHER.value:
            self._run_scalar_class_other_type_inst(inst)
        else:
            assert False, f"Not support"

    def _run_control_class_inst(self, inst):
        inst_type = inst["type"]
        if inst_type in [ControlInstType.EQ_BR.value, 
                            ControlInstType.NE_BR.value, 
                            ControlInstType.GT_BR.value, 
                            ControlInstType.LT_BR.value]:
            self._run_control_class_br_type_inst(inst)
        elif inst_type==ControlInstType.JUMP.value:
            self._run_control_class_jump_type_inst(inst)
        else:
            assert False, f"Not support"

    def _run_trans_class_inst(self, inst):
        inst_type = inst["type"]
        if inst_type==TransInstType.TRANS.value:
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
        if opcode==0b000: # add
            result = value1 + value2
        elif opcode==0b001: # sub
            result = value1 - value2
        elif opcode==0b010: # mul
            result = value1 * value2
        elif opcode==0b011: # div
            result = value1 // value2
        elif opcode==0b100: # sll
            assert False, "Not support sll yet"
        elif opcode==0b101: # srl
            assert False, "Not support srl yet"
        elif opcode==0b110: # sra
            assert False, "Not support sra yet"
        else:
            assert False, f"Not support {opcode=}."
        self.write_general_reg(inst["rd"], result)

    def _run_scalar_class_other_type_inst(self, inst):
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
        if opcode==0b00: # 通用寄存器
            rf = self.general_rf
        elif opcode==0b01: # 专用寄存器
            rf = self.special_rf
        self.write_reg(rf, rd, imm)

    def _run_trans_class_trans_type_inst(self, inst):
        """
        核内数据传输指令：trans
        指令字段划分：
        - [31, 29]，3bit：class，指令类别码，值为110
        - [28, 28]，1bit：type，指令类型码，值为0
        - [27, 26]，1bit：offset mask，偏移值掩码，0表示该地址不使用偏移值，1表示使用偏移值
        - [27]，1bit：source offset mask，源地址偏移值掩码
        - [26]，1bit：destination offset mask，目的地址偏移值掩码
        - [25, 21]，5bit：rs，通用寄存器1，表示传输源地址的基址
        - [20, 16]，5bit：rd，通用寄存器2，表示传输目的地址的基址
        - [15, 0]，16bit：offset，立即数，表示寻址的偏移值
            - 源地址计算公式：$rs + offset * [27]
            - 目的地址计算公式：$rd + offset * [26]
        """
        src_base = self.read_general_reg(inst["rs"])
        dst_base = self.read_general_reg(inst["rd"])
        offset = self.read_general_reg(inst["offset"])
        src_offset_mask = self.read_general_reg(inst["source_offset_mask"])
        dst_offset_mask = self.read_general_reg(inst["destination_offset_mask"])
        size = self.read_general_reg(inst["size"])

        src_addr = src_base + src_offset_mask * offset
        dst_addr = dst_base + dst_offset_mask * offset

        src_data = self.memory_space.read(src_addr, size)
        self.memory_space.write(src_data, dst_addr, size)

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
        if inst_type==0b000: # equals
            cond = (val1 == val2)
        elif inst_type==0b001: # not equals
            cond = not (val1 == val2)
        elif inst_type==0b010: # greater than
            cond = (val1 > val2)
        elif inst_type==0b011: # less than
            cond = (val1 < val2)
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
        if opcode==0b00: # general-li
            self.write_general_reg(inst["rd"], inst["imm"])
        elif opcode==0b01: # special-li
            self.write_special_reg(inst["rd"], inst["imm"])
        elif opcode==0b10: # general-to-special
            val = self.read_general_reg(inst["rs1"])
            self.write_special_reg(inst["rs2"], val)
        elif opcode==0b11: # special-to-general
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
        assert inst["group"] == 0, "Not support group yet."
        if inst["value_sparse"] == 1 and inst["bit_sparse"] == 1:
            self._run_pim_class_pim_compute_type_inst_value_bit_sparse(inst)
        elif inst["value_sparse"] == 1:
            self._run_pim_class_pim_compute_type_inst_value_sparse(inst)
        elif inst["bit_sparse"] == 1:
            self._run_pim_class_pim_compute_type_inst_bit_sparse(inst)
        else:
            self._run_pim_class_pim_compute_type_inst_dense(inst)

    """
    Diffenet Pim Compute
    """
    def _run_pim_class_pim_compute_type_inst_value_bit_sparse(self, inst):
        assert False, "Executor not support value & bit sparse yet."
    
    def _run_pim_class_pim_compute_type_inst_value_sparse(self, inst):
        assert False, "Executor not support value sparse yet."

    def _run_pim_class_pim_compute_type_inst_bit_sparse(self, inst):
        assert False, "Executor not support bit sparse yet."

    def _run_pim_class_pim_compute_type_inst_dense(self, inst):
        input_offset = self.read_general_reg(inst["rs1"])
        input_size = self.read_general_reg(inst["rs2"])
        activate_row = self.read_general_reg(inst["rs3"])
        output_offset = self.read_general_reg(inst["rd"])
        input_bw = self.read_special_reg(SpecialReg.INPUT_BIT_WIDTH)
        output_bw = self.read_special_reg(SpecialReg.OUTPUT_BIT_WIDTH)
        width_bw = self.read_special_reg(SpecialReg.WEIGHT_BIT_WIDTH)
        activation_element_col_num = self.read_special_reg(SpecialReg.ACTIVATION_ELEMENT_COL_NUM)

        # Get input vector
        input_byte_size = input_size * input_bw // 8
        self.memory_space.check_memory_type(input_offset, input_byte_size, "rf")
        input_data = self.memory_space.read_as(input_offset, input_byte_size, self.get_dtype(input_bw))
        print(f"{input_size=}, {input_bw=}, {self.get_dtype(input_bw)=}, {input_data=}")
        # Get weight matrix
        activate_element_row_num = input_size
        weight_data = self.macro_util.get_macro_data(activate_row, width_bw, activate_element_row_num, activation_element_col_num)
        
        assert input_data.ndim==1
        assert weight_data.ndim==2
        assert input_data.shape[0] == weight_data.shape[0], f"{input_data.shape=}, {weight_data.shape=}"
        out_dtype = get_dtype_from_bitwidth(output_bw)
        output_data = np.dot(input_data.astype(out_dtype), weight_data.astype(out_dtype))
        
        # Save output
        output_byte_size = output_data.size * output_bw // 8
        self.memory_space.check_memory_type(output_offset, output_byte_size, "rf")

        # Accumulate
        if inst["accumulate"] == 1:
            output_data_ori = self.memory_space.read_as(output_offset, output_byte_size, out_dtype)
            output_data = output_data + output_data_ori
        self.memory_space.write(output_data, output_offset, output_byte_size)

    def _run_debug_class_inst(self, inst):
        rs = inst['rs']
        val = self.read_general_reg(rs)
        print(f"[debug] general_reg[{rs}] = {val}")