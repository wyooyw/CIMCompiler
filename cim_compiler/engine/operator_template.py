import json
import os
import subprocess
from functools import partial

import numpy as np
import pytest

from cim_compiler.engine.operator_cim import (
    BitSparseConv2dOperator,
    BitSparseConv2dQuantifyOperator,
    BitSparseLinearOperator,
    BitSparseLinearQuantifyOperator,
    DenseConv2dOperator,
    DenseConv2dQuantifyOperator,
    DenseLinearOperator,
    DenseLinearQuantifyOperator,
    DepthWiseConv2dQuantifyOperator,
    Operator,
    ValueBitSparseConv2dOperator,
    ValueBitSparseConv2dQuantifyOperator,
    ValueSparseConv2dOperator,
    ValueSparseConv2dQuantifyOperator,
    ResAddQuantizeOperator,
    ResMulQuantizeOperator,
    ReLUOperator,
    MaxPoolingOperator,
    AvgPoolingQuantizeOperator,
)
from cim_compiler.simulator.macro_utils import MacroConfig
from cim_compiler.simulator.mask_utils import MaskConfig
from cim_compiler.simulator.simulator import Memory, MemorySpace, Simulator, SpecialReg
from cim_compiler.utils.predict_pimcompute_count import predict_pimcompute_count_for_conv2d_dense


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

class ResAddQuantizeTemplate(OperatorTemplate):
    def __init__(self):
        super().__init__(
            os.environ["CONFIG_PATH"],
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/resadd_quantize"),
        )

    def raw_layer_to_op_config(self, raw_layer):

        in_hw = raw_layer["input_row"]
        in_channel = raw_layer["input_channel"]
        
        return {
            "in_channel": in_channel,
            "in_hw": in_hw,
        }

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ResAddQuantizeOperator(self.config_path, self.template_path, op_config)

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        """
        """

        if not raw_layer.get("type", None) == "RESADD":
            return False

        if not (
            "input_row" in raw_layer and
            "input_col" in raw_layer and
            "input_channel" in raw_layer
        ):
            return False

        if not raw_layer["input_row"] == raw_layer["input_col"]:
            return False

        return True

class ResMulQuantizeTemplate(OperatorTemplate):
    def __init__(self):
        super().__init__(
            os.environ["CONFIG_PATH"],
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/dwconv2d/simd"),
        )

    def raw_layer_to_op_config(self, raw_layer):

        in_hw = raw_layer["input_row"]
        in_channel = raw_layer["input_channel"]
        
        return {
            "out_channel": in_channel,
            "in_channel": in_channel,
            "ker_size": 1,
            "in_hw": in_hw,
            "out_hw": in_hw,
            "padding": 0,
            "stride": 1,
        }

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ResMulQuantizeOperator(self.config_path, self.template_path, op_config)

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        """
        """

        if not raw_layer.get("type", None) == "MULT":
            return False

        if not (
            "input_row" in raw_layer and
            "input_col" in raw_layer and
            "input_channel" in raw_layer
        ):
            return False

        if not raw_layer["input_row"] == raw_layer["input_col"]:
            return False

        return True


class ReLUTemplate(OperatorTemplate):
    def __init__(self):
        super().__init__(
            os.environ["CONFIG_PATH"],
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/relu"),
        )

    def raw_layer_to_op_config(self, raw_layer):
        return {
            "in_channel": raw_layer["input_channel"],
            "in_hw": raw_layer["input_row"],
        }

    def get_operator(self, raw_layer):  
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ReLUOperator(self.config_path, self.template_path, op_config)

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        """
        """

        if not raw_layer.get("type", None) == "RELU":
            return False

        return True

class MaxPoolingTemplate(OperatorTemplate):
    def __init__(self):
        super().__init__(
            os.environ["CONFIG_PATH"],
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/max_pooling"),
        )

    def raw_layer_to_op_config(self, raw_layer):
        return {
            "in_channel": raw_layer["input_channel"],
            "in_hw": raw_layer["input_row"],
            "out_hw": raw_layer["input_row"] // raw_layer["weight_row"],
            "ker_size": raw_layer["weight_row"],
        }

    def get_operator(self, raw_layer):  
        op_config = self.raw_layer_to_op_config(raw_layer)
        return MaxPoolingOperator(self.config_path, self.template_path, op_config)

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        """
        """

        if not (
            raw_layer.get("type", None) == "POOLING" and
            raw_layer.get("pooling_type", None) == "MAX"
        ):
            return False

        return True

class AvgPoolingQuantizeTemplate(OperatorTemplate):
    def __init__(self):
        super().__init__(
            os.environ["CONFIG_PATH"],
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/avg_pooling"),
        )

    def raw_layer_to_op_config(self, raw_layer):
        return {
            "in_channel": raw_layer["input_channel"],
            "in_hw": raw_layer["input_row"],
            "out_hw": raw_layer["input_row"] // raw_layer["weight_row"],
            "ker_size": raw_layer["weight_row"],
        }

    def get_operator(self, raw_layer):  
        op_config = self.raw_layer_to_op_config(raw_layer)
        return AvgPoolingQuantizeOperator(self.config_path, self.template_path, op_config)

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        """
        """

        if not (
            raw_layer.get("type", None) == "POOLING" and
            raw_layer.get("pooling_type", None) == "MEAN"
        ):
            return False

        if not quantify:
            return False

        return True

class Conv2dTemplate(OperatorTemplate):
    def __init__(self, template_path):
        super().__init__(
            os.environ["CONFIG_PATH"],
            template_path,
        )

    def raw_layer_to_op_config(self, raw_layer):

        in_hw = raw_layer["input_row"]
        ker_size = raw_layer["weight_row"]
        out_channel = raw_layer["output_channel"]
        in_channel = raw_layer["input_channel"]
        stride = raw_layer["stride"]

        if raw_layer["padding_mode"] == "SAME":
            if raw_layer["weight_row"] == 3:
                padding = 1
            elif raw_layer["weight_row"] == 1:
                padding = 0

        else:
            padding = 0

        out_hw = (in_hw + 2 * padding - ker_size) // stride + 1

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
            "padding": padding,
            "stride": stride,
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

        if not raw_layer.get("type", None) == "CONV":
            return False

        if not raw_layer["input_row"] == raw_layer["input_col"]:
            return False

        if not (raw_layer["input_row"] % 2 == 0 or raw_layer["input_row"] == 1):
            return False

        # if not (
        #     (raw_layer["input_channel"] % 128 == 0) or
        #     (raw_layer["input_channel"] < 128 and raw_layer["input_channel"] % 16 == 0)
        # ):
        #     return False

        if not raw_layer["weight_row"] == raw_layer["weight_col"]:
            return False

        if not raw_layer.get("depthwise", False) == False:
            return False

        if not raw_layer["padding_mode"] in ["SAME", "VALID"]:
            return False

        if not (raw_layer["stride"] == 1 or raw_layer["stride"] == 2):
            return False

        return True


class LinearTemplate(Conv2dTemplate):
    def __init__(self, template_path):
        super().__init__(
            template_path,
        )

    def is_dense(self):
        return False

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

        if not (raw_layer.get("type", None) == "FCN"):
            return False

        if not (
            raw_layer["input_row"] == raw_layer["input_col"]
            and raw_layer["input_col"] == 1
        ):
            return False

        # if not (
        #     (raw_layer["input_channel"] % 128 == 0) or
        #     (raw_layer["input_channel"] < 128 and raw_layer["input_channel"] % 16 == 0)
        # ):
        #     return False

        if not (
            raw_layer["weight_row"] == raw_layer["weight_col"]
            and raw_layer["weight_col"] == 1
        ):
            return False

        return True


class Conv2dBaseTemplate(Conv2dTemplate):
    def __init__(self, template_path):
        super().__init__(
            template_path,
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (quantify == False):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)


class Conv2dQuantifyTemplate(Conv2dTemplate):
    def __init__(self, template_path):
        super().__init__(
            template_path,
        )

    def is_dense(self):
        return False

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (quantify == True):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)


class DenseConv2dTemplate(Conv2dBaseTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/conv2d/dense/normal")
        )

    def is_dense(self):
        return True

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == False and bit_sparse == False):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return DenseConv2dOperator(self.config_path, self.template_path, op_config)


class BitSparseConv2dTemplate(Conv2dBaseTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(
                os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/conv2d/bs/normal"
            )
        )

    def is_dense(self):
        return False

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == False and bit_sparse == True):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return BitSparseConv2dOperator(self.config_path, self.template_path, op_config)


class ValueSparseConv2dTemplate(Conv2dBaseTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/conv2d/vs/normal")
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == True and bit_sparse == False):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ValueSparseConv2dOperator(
            self.config_path, self.template_path, op_config
        )


class ValueBitSparseConv2dTemplate(Conv2dBaseTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/conv2d/vs_bs/normal")
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == True and bit_sparse == True):
            return False
        # if not ((raw_layer["input_channel"] % 128 == 0) or (128 % raw_layer["input_channel"] == 0)):
        #     return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ValueBitSparseConv2dOperator(
            self.config_path, self.template_path, op_config
        )


class DenseLinearTemplate(LinearTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/linear/dense/normal"),
        )

    def is_dense(self):
        return True

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == False and bit_sparse == False):
            return False
        if not (quantify == False):
            return
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return DenseLinearOperator(self.config_path, self.template_path, op_config)


class BitSparseLinearTemplate(LinearTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/linear/bs/normal"),
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == False and bit_sparse == True):
            return False
        if not (quantify == False):
            return
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return BitSparseLinearOperator(self.config_path, self.template_path, op_config)


class DenseConv2dQuantifyTemplate(Conv2dQuantifyTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/conv2d/dense/quantize"),
        )

    def is_dense(self):
        return True

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == False and bit_sparse == False):
            return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return DenseConv2dQuantifyOperator(
            self.config_path, self.template_path, op_config
        )


class ValueSparseConv2dQuantifyTemplate(Conv2dQuantifyTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/conv2d/vs/quantize"),
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == True and bit_sparse == False):
            return False
        # if not ((raw_layer["input_channel"] % 128 == 0) or (128 % raw_layer["input_channel"] == 0)):
        #     return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ValueSparseConv2dQuantifyOperator(
            self.config_path, self.template_path, op_config
        )


class BitSparseConv2dQuantifyTemplate(Conv2dQuantifyTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/conv2d/bs/quantize"),
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == False and bit_sparse == True):
            return False
        # if not (raw_layer["input_channel"] % 16 == 0):
        #     return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return BitSparseConv2dQuantifyOperator(
            self.config_path, self.template_path, op_config
        )


class ValueBitSparseConv2dQuantifyTemplate(Conv2dQuantifyTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/conv2d/vs_bs/quantize"),
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == True and bit_sparse == True):
            return False
        # if not ((raw_layer["input_channel"] % 128 == 0) or (128 % raw_layer["input_channel"] == 0)):
        #     return False
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return ValueBitSparseConv2dQuantifyOperator(
            self.config_path, self.template_path, op_config
        )


class DenseLinearQuantifyTemplate(LinearTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/linear/dense/quantize"),
        )

    def is_dense(self):
        return True

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == False and bit_sparse == False):
            return False
        if not (quantify == True):
            return
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return DenseLinearQuantifyOperator(
            self.config_path, self.template_path, op_config
        )


class BitSparseLinearQuantifyTemplate(LinearTemplate):
    def __init__(self):
        super().__init__(
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/linear/bs/quantize"),
        )

    def check_raw_layer(self, raw_layer, value_sparse, bit_sparse, quantify):
        if not (value_sparse == False and bit_sparse == True):
            return False
        if not (quantify == True):
            return
        return super().check_raw_layer(raw_layer, value_sparse, bit_sparse, quantify)

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return BitSparseLinearQuantifyOperator(
            self.config_path, self.template_path, op_config
        )


class DepthWiseConv2dQuantifyTemplate(OperatorTemplate):
    def __init__(self):
        super().__init__(
            os.environ["CONFIG_PATH"],
            os.path.join(os.environ["CIM_COMPILER_BASE"], "cim_compiler/op/dwconv2d/simd"),
        )

    def is_dense(self):
        return True

    def raw_layer_to_op_config(self, raw_layer):

        in_hw = raw_layer["input_row"]
        ker_size = raw_layer["weight_row"]
        out_channel = raw_layer["output_channel"]
        in_channel = raw_layer["input_channel"]
        stride = raw_layer["stride"]

        if raw_layer["padding_mode"] == "SAME":
            if raw_layer["weight_row"] == 3:
                padding = 1
            elif raw_layer["weight_row"] == 1:
                padding = 0
        else:
            padding = 0

        out_hw = (in_hw + 2 * padding - ker_size) // stride + 1

        input_buffer_size_per_group = 128
        # if in_channel >= 128:
        #     input_buffer_size_per_group = 128
        # elif in_channel < 128 and 128 % in_channel == 0:
        #     input_buffer_size_per_group = 128
        # else:
        #     input_buffer_size_per_group = in_channel

        return {
            "out_channel": out_channel,
            "in_channel": out_channel,
            "ker_size": ker_size,
            "in_hw": in_hw,
            "out_hw": out_hw,
            "input_buffer_size_per_group": input_buffer_size_per_group,
            "padding": padding,
            "stride": stride,
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

        if not raw_layer.get("type", None) == "CONV":
            return False

        if not raw_layer["input_row"] == raw_layer["input_col"]:
            return False

        if not (raw_layer["input_row"] % 2 == 0):
            return False

        # if not (
        #     (raw_layer["input_channel"] % 128 == 0) or
        #     (raw_layer["input_channel"] < 128 and raw_layer["input_channel"] % 16 == 0)
        # ):
        #     return False

        if (
            not raw_layer["weight_row"] == raw_layer["weight_col"]
            and raw_layer["weight_col"] == 3
        ):
            return False

        if not raw_layer.get("depthwise", False) == True:
            return False

        if not raw_layer["padding_mode"] in ["SAME", "VALID"]:
            return False

        if not raw_layer["stride"] in [1, 2]:
            return False

        if not (value_sparse == False and bit_sparse == False):
            return False
        if not (quantify == True):
            return

        return True

    def get_operator(self, raw_layer):
        op_config = self.raw_layer_to_op_config(raw_layer)
        return DepthWiseConv2dQuantifyOperator(
            self.config_path, self.template_path, op_config
        )
