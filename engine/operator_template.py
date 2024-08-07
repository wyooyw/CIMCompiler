import pytest
from test.simulator.utils import InstUtil
from simulator.simulator import MemorySpace, Memory, Simulator, SpecialReg
from simulator.macro_utils import MacroConfig
from simulator.mask_utils import MaskConfig
import numpy as np
import subprocess
import os
import json
from functools import partial
import numpy as np
from utils.predict_pimcompute_count import predict_pimcompute_count_for_conv2d_dense
from engine.operator_cim import (
    Operator, 
    DenseConv2dOperator,
    BitSparseConv2dOperator,
    ValueSparseConv2dOperator,
    ValueBitSparseConv2dOperator,

    DenseConv2dQuantifyOperator,
    ValueSparseConv2dQuantifyOperator,
    BitSparseConv2dQuantifyOperator,
    ValueBitSparseConv2dQuantifyOperator,
    DenseLinearQuantifyOperator,
    BitSparseLinearQuantifyOperator
)


class OperatorTemplate:
    def __init__(self, config_path, template_path):
        self.config_path = config_path
        self.simulator = Simulator.from_config(self.config_path)
        self.memory_space = self.simulator.memory_space
        self.macro_config = self.simulator.macro_config
        self.mask_config = self.simulator.mask_config
        self.template_path = template_path

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse):
        assert False, "Not implemented"

    def raw_layer_to_op_config(self, raw_layer):
        assert False, "Not implemented"

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return Operator(self.config_path, self.template_path, op_config)

class Conv2dTemplate(OperatorTemplate):
    def __init__(self, template_path):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/config/config.json", 
            template_path,
        )

    def raw_layer_to_op_config(self, raw_layer):

        in_hw = raw_layer["input_row"]
        ker_size = raw_layer["weight_row"]
        out_channel = raw_layer["output_channel"]
        in_channel = raw_layer["input_channel"]

        if raw_layer["padding_mode"] == "SAME":
            if raw_layer["weight_row"] == 3:
                padding = 1
            elif raw_layer["weight_row"] == 1:
                padding = 0
            
            out_hw = in_hw
        else:
            padding = 0
            out_hw = in_hw - ker_size + 1

        input_buffer_size_per_group = 128
        # if in_channel >= 128:
        #     input_buffer_size_per_group = 128
        # elif in_channel < 128 and 128 % in_channel == 0:
        #     input_buffer_size_per_group = 128
        # else:
        #     input_buffer_size_per_group = in_channel
        
        return {
            "out_channel": out_channel,
            "in_channel": in_channel, 
            "ker_size": ker_size, 
            "in_hw": in_hw,
            "out_hw": out_hw, 
            "input_buffer_size_per_group": input_buffer_size_per_group,
            "padding": padding
        }

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        """
        Conditions:
            1.input_row==input_col, and input_row and input_col should be multiple of 2
            2.either 
                input_channel % 128 == 0 
            or 
                input_channel < 128 and input_channel % 16 == 0
            3.
                weight_row == weight_col
            4.
                depthwise==False
            5.
                padding_mode==SAME or VALID
            6.
                stride==1    
        """

        if not raw_layer.get("type", None)=="CONV":
            return False

        if not raw_layer["input_row"]==raw_layer["input_col"]:
            return False

        if not (raw_layer["input_row"] % 2 == 0):
            return False

        # if not (
        #     (raw_layer["input_channel"] % 128 == 0) or
        #     (raw_layer["input_channel"] < 128 and raw_layer["input_channel"] % 16 == 0)
        # ):
        #     return False
        
        if not raw_layer["weight_row"]==raw_layer["weight_col"]:
            return False

        if not raw_layer.get("depthwise", False)==False:
            return False

        if not raw_layer["padding_mode"] in ["SAME", "VALID"]:
            return False

        if not raw_layer["stride"]==1:
            return False

        return True

class Conv2dBaseTemplate(Conv2dTemplate):
    def __init__(self, template_path):
        super().__init__(
            template_path,
        )
    
    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (quantify==False):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

class Conv2dQuantifyTemplate(Conv2dTemplate):
    def __init__(self, template_path):
        super().__init__(
            template_path,
        )
    
    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (quantify==True):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

class DenseConv2dTemplate(Conv2dBaseTemplate):
    def __init__(self):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/pimcompute/dense/dense_conv2d_group",
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse==False and bit_sparse==False):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)
    
    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return DenseConv2dOperator(self.config_path, self.template_path, op_config)

class BitSparseConv2dTemplate(Conv2dBaseTemplate):
    def __init__(self):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/pimcompute/bit_sparse/bit_sparse_conv2d_group",
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse==False and bit_sparse==True):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)
    
    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return BitSparseConv2dOperator(self.config_path, self.template_path, op_config)

class ValueSparseConv2dTemplate(Conv2dBaseTemplate):
    def __init__(self):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/pimcompute/value_sparse/value_sparse_group_longer",
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse==True and bit_sparse==False):
            return False
        if not ((raw_layer["input_channel"] % 128 == 0) or (128 % raw_layer["input_channel"] == 0)):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)
    
    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ValueSparseConv2dOperator(self.config_path, self.template_path, op_config)

class ValueBitSparseConv2dTemplate(Conv2dBaseTemplate):
    def __init__(self):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/pimcompute/value_bit_sparse/value_bit_sparse_base",
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse==True and bit_sparse==True):
            return False
        if not ((raw_layer["input_channel"] % 128 == 0) or (128 % raw_layer["input_channel"] == 0)):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)
    
    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ValueBitSparseConv2dOperator(self.config_path, self.template_path, op_config)

class DenseConv2dQuantifyTemplate(Conv2dQuantifyTemplate):
    def __init__(self):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/pimcompute/dense/dense_conv2d_group_quantify",
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse==False and bit_sparse==False):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)
    
    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return DenseConv2dQuantifyOperator(self.config_path, self.template_path, op_config)

class ValueSparseConv2dQuantifyTemplate(Conv2dQuantifyTemplate):
    def __init__(self):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/pimcompute/value_sparse/value_sparse_group_longer_quantify",
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse==True and bit_sparse==False):
            return False
        # if not ((raw_layer["input_channel"] % 128 == 0) or (128 % raw_layer["input_channel"] == 0)):
        #     return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)
    
    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ValueSparseConv2dQuantifyOperator(self.config_path, self.template_path, op_config)

class BitSparseConv2dQuantifyTemplate(Conv2dQuantifyTemplate):
    def __init__(self):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/pimcompute/bit_sparse/bit_sparse_conv2d_group_quantify",
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse==False and bit_sparse==True):
            return False
        if not (raw_layer["input_channel"] % 16 == 0):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)
    
    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return BitSparseConv2dQuantifyOperator(self.config_path, self.template_path, op_config)

class ValueBitSparseConv2dQuantifyTemplate(Conv2dQuantifyTemplate):
    def __init__(self):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/pimcompute/value_bit_sparse/value_bit_sparse_quantify",
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse==True and bit_sparse==True):
            return False
        # if not ((raw_layer["input_channel"] % 128 == 0) or (128 % raw_layer["input_channel"] == 0)):
        #     return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)
    
    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ValueBitSparseConv2dQuantifyOperator(self.config_path, self.template_path, op_config)


class LinearTemplate(Conv2dTemplate):
    def __init__(self, template_path):
        super().__init__(
            template_path,
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        """
        Conditions:
            1.input_row==input_col, and input_row and input_col should be multiple of 2
            2.either 
                input_channel % 128 == 0 
            or 
                input_channel < 128 and input_channel % 16 == 0
            3.
                weight_row == weight_col
            4.
                depthwise==False
            5.
                padding_mode==SAME or VALID
            6.
                stride==1    
        """

        if not raw_layer.get("type", None)=="FCN":
            return False

        if not (raw_layer["input_row"]==raw_layer["input_col"] and raw_layer["input_col"]==1):
            return False

        # if not (
        #     (raw_layer["input_channel"] % 128 == 0) or
        #     (raw_layer["input_channel"] < 128 and raw_layer["input_channel"] % 16 == 0)
        # ):
        #     return False
        
        if not (raw_layer["weight_row"]==raw_layer["weight_col"] and raw_layer["weight_col"]==1):
            return False

        return True

class DenseLinearQuantifyTemplate(LinearTemplate):
    def __init__(self):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/pimcompute/linear/dense_quantify",
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse==False and bit_sparse==False):
            return False
        if not (quantify==True):
            return
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)
    
    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return DenseLinearQuantifyOperator(self.config_path, self.template_path, op_config)

    
class BitSparseLinearQuantifyTemplate(LinearTemplate):
    def __init__(self):
        super().__init__(
            "/home/wangyiou/project/cim_compiler_frontend/playground/test/compiler/pimcompute/linear/bit_sparse_quantify",
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse==False and bit_sparse==True):
            return False
        if not (quantify==True):
            return
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)
    
    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return BitSparseLinearQuantifyOperator(self.config_path, self.template_path, op_config)