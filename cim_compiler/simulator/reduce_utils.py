import numpy as np
import json
from cim_compiler.simulator.special_regs import SpecialReg
from cim_compiler.simulator.data_type import get_dtype_from_bitwidth
from cim_compiler.utils.logger import get_logger
from dataclasses import dataclass

logger = get_logger(__name__)

class ReduceSumConfig:
    def __init__(self, reduce_len, reduce_num):
        self.reduce_len = reduce_len
        self.reduce_num = reduce_num
        logger.debug(f"Mask config: {reduce_len=}, {reduce_num=}")

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config.get("reduce_sum", {}).get("reduce_len", None), config.get("reduce_sum", {}).get("reduce_num", None))

class ReduceSumUtil:
    def __init__(self, reduce_sum_config):
        self.reduce_sum_config = reduce_sum_config
        if reduce_sum_config is not None:
            assert isinstance(reduce_sum_config, ReduceSumConfig)
            self.reduce_len = reduce_sum_config.reduce_len
            self.reduce_num = reduce_sum_config.reduce_num
        else:
            self.reduce_len = None
            self.reduce_num = None

    def reduce_sum(self, src_vector):
        if self.reduce_sum_config is None or self.reduce_len is None or self.reduce_num is None:
            assert False, "Reduce sum config is not set"
        assert len(src_vector.shape) == 1
        assert src_vector.shape[0] <= self.reduce_len * self.reduce_num
        # pad src_vector to the nearest multiple of reduce_len
        N = src_vector.shape[0]
        pad_len = (self.reduce_len - N % self.reduce_len) % self.reduce_len
        src_vector = np.pad(src_vector, (0, pad_len), mode='constant')
        N = src_vector.shape[0]
        src_vector = src_vector.reshape(N // self.reduce_len, self.reduce_len)
        dst_vector = src_vector.sum(axis=1)
        dst_vector = dst_vector.reshape(-1)
        return dst_vector


class ReduceMaxConfig:
    def __init__(self, reduce_len, reduce_num):
        self.reduce_len = reduce_len
        self.reduce_num = reduce_num
        logger.debug(f"Mask config: {reduce_len=}, {reduce_num=}")

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config.get("reduce_max", {}).get("reduce_len", None), config.get("reduce_max", {}).get("reduce_num", None))

class ReduceMaxUtil:
    def __init__(self, reduce_max_config):
        self.reduce_max_config = reduce_max_config
        if reduce_max_config is not None:
            assert isinstance(reduce_max_config, ReduceMaxConfig)
            self.reduce_len = reduce_max_config.reduce_len
            self.reduce_num = reduce_max_config.reduce_num
        else:
            self.reduce_len = None
            self.reduce_num = None

    def reduce_max(self, src_vector):
        if self.reduce_max_config is None or self.reduce_len is None or self.reduce_num is None:
            assert False, "Reduce max config is not set"
        assert len(src_vector.shape) == 1
        assert src_vector.shape[0] <= self.reduce_len * self.reduce_num
        # pad src_vector to the nearest multiple of reduce_len
        N = src_vector.shape[0]
        pad_len = (self.reduce_len - N % self.reduce_len) % self.reduce_len
        src_vector = np.pad(src_vector, (0, pad_len), mode='constant')
        N = src_vector.shape[0]
        src_vector = src_vector.reshape(N // self.reduce_len, self.reduce_len)
        dst_vector = src_vector.max(axis=1)
        dst_vector = dst_vector.reshape(-1)
        return dst_vector

class ReduceConfig:
    def __init__(self, reduce_list):
        self.reduce_list = reduce_list
        for i, op_name in enumerate(reduce_list):
            assert not hasattr(self, op_name), f"op_name {op_name} already exists"
            setattr(self, op_name, i)

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config["reduce"])

@dataclass
class ReduceOpInput:
    input_size: int
    input_addr: int
    input_bitwidth: int
    input_byte_size: int
    input_data: np.ndarray
    output_addr: int
    output_bitwidth: int
    output_dtype: np.dtype

class ReduceUtil:
    def __init__(self, reduce_config, simulator):
        self.reduce_config = reduce_config
        self.simulator = simulator

    def get_reduce_op_name(self, opcode):
        if opcode >= len(self.reduce_config.reduce_list):
            raise ValueError(f"Invalid opcode: {opcode}")
        return self.reduce_config.reduce_list[opcode]
    
    def run(self, inst):
        op_name = self.get_reduce_op_name(inst.opcode)
        op = getattr(self, f"_run_{op_name}")
        op(inst)

    
    def _prepare_input_for_reduce_op(self, inst):
        input_addr = self.simulator.read_general_reg(inst.reg_in)
        input_size = self.simulator.read_general_reg(inst.reg_size)
        output_addr = self.simulator.read_general_reg(inst.reg_out)

        input_bitwidth = self.simulator.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input_byte_size = input_bitwidth * input_size // 8
        input_dtype = get_dtype_from_bitwidth(input_bitwidth, is_float=self.simulator.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        output_bitwidth = self.simulator.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth, is_float=self.simulator.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        self.simulator.memory_space.check_memory_type(input_addr, input_byte_size, "sram")
        input_data = self.simulator.memory_space.read_as(
            input_addr, 
            input_byte_size, 
            input_dtype
        )
        return ReduceOpInput(
            input_size=input_size,
            input_addr=input_addr,
            input_bitwidth=input_bitwidth,
            input_byte_size=input_byte_size,
            input_data=input_data,
            output_addr=output_addr,
            output_bitwidth=output_bitwidth,
            output_dtype=output_dtype,
        )
    
    def _all_reduce_op(self, inst, compute_fn):
        params = self._prepare_input_for_reduce_op(inst)
        output_data = compute_fn(
            params.input_data.astype(params.output_dtype)
        ).astype(params.output_dtype).reshape(-1)
        output_byte_size = output_data.size * params.output_bitwidth // 8
        self.simulator.memory_space.write(output_data, params.output_addr, output_byte_size)
    

    def _run_reduce_sum(self, inst):
        self._all_reduce_op(inst, lambda x: self.simulator.reduce_sum_util.reduce_sum(x))
    
    def _run_reduce_max(self, inst):
        # self._all_vec_op(inst, lambda x: np.max(x))
        self._all_reduce_op(inst, lambda x: self.simulator.reduce_max_util.reduce_max(x))