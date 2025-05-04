import json
from cim_compiler.simulator.special_regs import SpecialReg
from cim_compiler.simulator.data_type import get_dtype_from_bitwidth
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

class SIMDConfig:
    def __init__(self, simd_list):
        self.simd_list = simd_list
        for i, op_name in enumerate(simd_list):
            assert not hasattr(self, op_name), f"op_name {op_name} already exists"
            setattr(self, op_name, i)

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config["simd"])

@dataclass
class VecVecOpInput:
    input_size: int
    input1_addr: int
    input1_bitwidth: int
    input1_byte_size: int
    input1_data: np.ndarray
    input2_addr: int
    input2_bitwidth: int
    input2_byte_size: int
    input2_data: np.ndarray
    output_addr: int
    output_bitwidth: int
    output_dtype: np.dtype

@dataclass
class VecScalarOpInput(VecVecOpInput):
    pass

@dataclass
class VecOpInput:
    input_size: int
    input_addr: int
    input_bitwidth: int
    input_byte_size: int
    input_data: np.ndarray
    output_addr: int
    output_bitwidth: int
    output_dtype: np.dtype

class SIMDUtil:
    def __init__(self, simd_config, simulator):
        self.simd_config = simd_config
        self.simulator = simulator

    def get_simd_op_name(self, opcode):
        if opcode >= len(self.simd_config.simd_list):
            raise ValueError(f"Invalid opcode: {opcode}")
        return self.simd_config.simd_list[opcode]

    def run(self, inst):
        op_name = self.get_simd_op_name(inst.opcode)
        op = getattr(self, f"_run_{op_name}")
        op(inst)

    def _prepare_input_for_vec_vec_op(self, inst):
        # Prepare input
        input_size = self.simulator.read_general_reg(inst.reg_size)

        input1_addr = self.simulator.read_general_reg(inst.reg_in1)
        input1_bitwidth = self.simulator.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input1_byte_size = input1_bitwidth * input_size // 8

        input2_addr = self.simulator.read_general_reg(inst.reg_in2)
        input2_bitwidth = self.simulator.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH)
        input2_byte_size = input2_bitwidth * input_size // 8

        output_addr = self.simulator.read_general_reg(inst.reg_out)
        output_bitwidth = self.simulator.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth, is_float=self.simulator.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))

        input1_data = self.simulator.memory_space.read_as(
            input1_addr, input1_byte_size, get_dtype_from_bitwidth(input1_bitwidth, is_float=self.simulator.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        )
        input2_data = self.simulator.memory_space.read_as(
            input2_addr, input2_byte_size, get_dtype_from_bitwidth(input2_bitwidth, is_float=self.simulator.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        )

        return VecVecOpInput(
            input_size=input_size,
            input1_addr=input1_addr,
            input1_bitwidth=input1_bitwidth,
            input1_byte_size=input1_byte_size,
            input1_data=input1_data,
            input2_addr=input2_addr,
            input2_bitwidth=input2_bitwidth,
            input2_byte_size=input2_byte_size,
            input2_data=input2_data,
            output_addr=output_addr,
            output_bitwidth=output_bitwidth,
            output_dtype=output_dtype,
        )
    
    def _all_vv_op(self, inst, compute_fn):
        assert inst.input_num == 2, f"{inst.input_num=}"
        params = self._prepare_input_for_vec_vec_op(inst)
        output_data = compute_fn(
            params.input1_data.astype(params.output_dtype), 
            params.input2_data.astype(params.output_dtype)
        ).astype(params.output_dtype)
        output_byte_size = output_data.size * params.output_bitwidth // 8
        self.simulator.memory_space.write(output_data, params.output_addr, output_byte_size)
    
    def _run_vv_add(self, inst):
        self._all_vv_op(inst, lambda x, y: x + y)

    def _run_vv_mul(self, inst):
        self._all_vv_op(inst, lambda x, y: x * y)

    def _run_vv_max(self, inst):
        self._all_vv_op(inst, lambda x, y: np.maximum(x, y))

    def _prepare_input_for_vec_scalar_op(self, inst):
        # Prepare input
        input_size = self.simulator.read_general_reg(inst.reg_size)

        input1_addr = self.simulator.read_general_reg(inst.reg_in1)
        input1_bitwidth = self.simulator.read_special_reg(SpecialReg.SIMD_INPUT_1_BIT_WIDTH)
        input1_byte_size = input1_bitwidth * input_size // 8

        input2_addr = self.simulator.read_general_reg(inst.reg_in2)
        input2_bitwidth = self.simulator.read_special_reg(SpecialReg.SIMD_INPUT_2_BIT_WIDTH)
        input2_byte_size = input2_bitwidth // 8

        output_addr = self.simulator.read_general_reg(inst.reg_out)
        output_bitwidth = self.simulator.read_special_reg(SpecialReg.SIMD_OUTPUT_BIT_WIDTH)
        output_dtype = get_dtype_from_bitwidth(output_bitwidth, is_float=self.simulator.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))

        input1_data = self.simulator.memory_space.read_as(
            input1_addr, input1_byte_size, get_dtype_from_bitwidth(input1_bitwidth, is_float=self.simulator.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        )
        input2_data = self.simulator.memory_space.read_as(
            input2_addr, input2_byte_size, get_dtype_from_bitwidth(input2_bitwidth, is_float=self.simulator.read_special_reg(SpecialReg.DTYPE_SIMD_IS_FLOAT))
        )

        return VecScalarOpInput(
            input_size=input_size,
            input1_addr=input1_addr,
            input1_bitwidth=input1_bitwidth,
            input1_byte_size=input1_byte_size,
            input1_data=input1_data,
            input2_addr=input2_addr,
            input2_bitwidth=input2_bitwidth,
            input2_byte_size=input2_byte_size,
            input2_data=input2_data,
            output_addr=output_addr,
            output_bitwidth=output_bitwidth,
            output_dtype=output_dtype,
        )
    
    def _all_vs_op(self, inst, compute_fn):
        assert inst.input_num == 2, f"{inst.input_num=}"
        params = self._prepare_input_for_vec_scalar_op(inst)
        output_data = compute_fn(
            params.input1_data.astype(params.output_dtype), 
            params.input2_data.astype(params.output_dtype)
        ).astype(params.output_dtype)
        output_byte_size = output_data.size * params.output_bitwidth // 8
        self.simulator.memory_space.write(output_data, params.output_addr, output_byte_size)

    
    def _run_vs_add(self, inst):
        self._all_vs_op(inst, lambda x, y: x + y)

    def _run_vs_mul(self, inst):
        self._all_vs_op(inst, lambda x, y: x * y)

    def _run_vs_div(self, inst):
        self._all_vs_op(inst, lambda x, y: x / y)

    def _run_vs_sub(self, inst):
        self._all_vs_op(inst, lambda x, y: x - y)

    def _run_vs_max(self, inst):
        self._all_vs_op(inst, lambda x, y: np.maximum(x, y))


    def _prepare_input_for_vec_op(self, inst):
        input_addr = self.simulator.read_general_reg(inst.reg_in1)
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
        return VecOpInput(
            input_size=input_size,
            input_addr=input_addr,
            input_bitwidth=input_bitwidth,
            input_byte_size=input_byte_size,
            input_data=input_data,
            output_addr=output_addr,
            output_bitwidth=output_bitwidth,
            output_dtype=output_dtype,
        )
    
    def _all_vec_op(self, inst, compute_fn):
        assert inst.input_num == 1, f"{inst.input_num=}"
        params = self._prepare_input_for_vec_op(inst)
        output_data = compute_fn(
            params.input_data.astype(params.output_dtype)
        ).astype(params.output_dtype).reshape(-1)
        output_byte_size = output_data.size * params.output_bitwidth // 8
        self.simulator.memory_space.write(output_data, params.output_addr, output_byte_size)
    
    def _run_v_exp(self, inst):
        self._all_vec_op(inst, lambda x: np.exp(x))
    
    def _run_v_sqrt(self, inst):
        self._all_vec_op(inst, lambda x: np.sqrt(x))
    
    def _run_v_gelu(self, inst):
        self._all_vec_op(inst, lambda x: F.gelu(torch.tensor(x.astype(np.float32))).numpy())
    
    def _run_reduce_sum(self, inst):
        self._all_vec_op(inst, lambda x: self.simulator.reduce_sum_util.reduce_sum(x))
    
    def _run_reduce_max(self, inst):
        # self._all_vec_op(inst, lambda x: np.max(x))
        self._all_vec_op(inst, lambda x: self.simulator.reduce_max_util.reduce_max(x))