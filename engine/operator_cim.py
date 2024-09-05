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
from utils.bit_sparse_weight_transform import find_nonzero_filter, argsort_filters_threshold
from utils.round import banker_round
class Operator:
    def __init__(self, config_path, template_path, op_config):
        self.config_path = config_path
        self.simulator = Simulator.from_config(self.config_path)
        self.memory_space = self.simulator.memory_space
        self.macro_config = self.simulator.macro_config
        self.mask_config = self.simulator.mask_config
        self.template_path = template_path
        self.op_config = op_config
        self.helper = self.get_helper(template_path, op_config)

    def get_helper(self, helper_dir, op_config):
        # Get helper
        helper_path = os.path.join(helper_dir, "helper.py")
        with open(helper_path, 'r') as file:
            code = file.read()
        local_namespace = {}
        exec(code, {}, local_namespace)
        Helper = local_namespace["TestHelper"]
        helper = Helper(op_config)
        return helper


    def compile(self, code_dir, image_kwargs={}):
        op_config = self.op_config

        case_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.template_path)
        assert os.path.exists(case_dir), f"{case_dir} not exists"
        assert os.path.isdir(case_dir), f"{case_dir} is not a directory"

        # Prepare path
        input_template_path = os.path.join(case_dir, "code_template.cim")
        input_path = os.path.join(case_dir, "code.cim")
        test_helper_path = os.path.join(case_dir, "helper.py")
        assert os.path.exists(input_template_path), f"{input_template_path} not exists"
        assert os.path.exists(test_helper_path), f"{test_helper_path} not exists"
        
        # output_dir = os.path.join(case_dir, ".result")
        os.makedirs(code_dir, exist_ok=True)

        # If there is already files in .result, remove them, to make sure not execute old codes.
        for filename in os.listdir(code_dir):
            file_path = os.path.join(code_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # return

        # load image
        image = self.helper.get_image(self.simulator, **image_kwargs)
        global_memory_base = self.simulator.memory_space.get_base_of("global")
        self.simulator.memory_space.write(image, global_memory_base, len(image))
        with open(os.path.join(code_dir, 'global_image'), 'wb') as file:
            file.write(image)

        # fill code template
        self.helper.fill_template(input_template_path, input_path, self.simulator)
        # return

        # register debug hook
        # self.simulator.debug_hook = partial(debug_hook, helper=helper)

        # run compiler
        cmd = f"bash compile.sh isa {input_path} {code_dir} {self.config_path}"
        result = subprocess.run(cmd.split(" "), capture_output=True, text=True)
        print('输出:', result.stdout)
        print('错误:', result.stderr)
        assert result.returncode==0
        return image

    def run(self, code_dir, image):
        op_config = self.op_config

        # get output code
        output_path = os.path.join(code_dir, "final_code.json")
        with open(output_path, "r") as f:
            code = json.load(f)

        # run code in simulator
        pimcompute_count = predict_pimcompute_count_for_conv2d_dense(self.macro_config, op_config, group_size=16)
        print(f"{pimcompute_count=}")

        status,stats,flat = self.simulator.run_code(code, total_pim_compute_count = pimcompute_count)
        assert status==self.simulator.FINISH, status
        stats.dump(code_dir)
        flat.dump(code_dir)
        output = self.helper.get_output(self.simulator.memory_space)

        # get flat code
        flat_code = flat.get_flat_code()
        self.simulator.clear()
        global_memory_base = self.simulator.memory_space.get_base_of("global")
        self.simulator.memory_space.write(image, global_memory_base, len(image))
        self.simulator._read_reg_value_directly = True
        status,stats,flat = self.simulator.run_code(flat_code, total_pim_compute_count = pimcompute_count, record_flat=False)
        self.simulator._read_reg_value_directly = False
        assert status==self.simulator.FINISH
        stats.dump(code_dir, prefix="flat_")

        # check result
        return output

    def compile_and_run(self, code_dir, image_kwargs):
        image = self.compile(code_dir, image_kwargs)
        return self.run(code_dir, image)
        # return None

    def compile_and_run_with_mock_data(self, code_dir):
        self.compile(code_dir)
        output = self.run(code_dir)

    def compile_run_and_check_with_mock_data(self, code_dir):
        self.compile_and_run_with_mock_data(code_dir)
        self.helper.check_image(self.simulator.memory_space)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result):
        assert False, "Not implemented"

class DenseConv2dOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        # read data
        input = np.loadtxt(os.path.join(df_dir, "conv_input_feature.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        golden = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden = golden.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        bias = bias.reshape(-1,1,1)
        golden = golden - bias
        golden = np.transpose(golden, (1,2,0))

        
        # compile and run, get output
        output = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})

        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()
            # correct = np.array_equal(golden, output)
            correct_percent = (golden==output).sum() / golden.size
            # import pdb; pdb.set_trace()
            return output, correct_percent

        return output, None

class BitSparseConv2dOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        
        # read data
        input = np.loadtxt(os.path.join(df_dir, "conv_input_feature.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        golden = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden = golden.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        bias = bias.reshape(-1,1,1)
        golden = golden - bias

        # reorder by threshold
        sort_index = argsort_filters_threshold(weight)
        weight = weight[sort_index]
        golden = golden[sort_index]

        golden = np.transpose(golden, (1,2,0))

        
        # compile and run, get output
        output = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})

        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()
            # correct = np.array_equal(golden, output)
            correct_percent = (golden==output).sum() / golden.size
            return output, correct_percent

        return output, None

class ValueSparseConv2dOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        
        # read data
        input = np.loadtxt(os.path.join(df_dir, "conv_input_feature.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        golden = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden = golden.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        bias = bias.reshape(-1,1,1)
        golden = golden - bias
        golden = np.transpose(golden, (1,2,0))

        
        # compile and run, get output
        output = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})

        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()
            correct = np.array_equal(golden, output)
            correct_percent = (golden==output).sum() / golden.size
            return output, correct_percent

        return output, None

class ValueBitSparseConv2dOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        
        # read data
        input = np.loadtxt(os.path.join(df_dir, "conv_input_feature.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        golden = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden = golden.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        bias = bias.reshape(-1,1,1)
        golden = golden - bias
        golden = np.transpose(golden, (1,2,0))

        
        # compile and run, get output
        output = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})

        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()
            correct_percent = (golden==output).sum() / golden.size
            return output, correct_percent

        return output, None

class DenseLinearOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        # read data
        input = np.loadtxt(os.path.join(df_dir, "input.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        scale = np.loadtxt(os.path.join(df_dir, "scale.txt"), dtype=np.float32).reshape(-1)
        out_zp = np.loadtxt(os.path.join(df_dir, "qo.zero_point.txt"), dtype=np.int32).reshape(1)
        golden = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int32)
        golden_i8 = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int8)
        # bias = bias.reshape(-1,1,1)
        # golden = golden - bias
        # golden_i8 = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int8)
        relu = (golden_i8 >= 0).all()

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        # golden_i8 = golden_i8.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        # golden_i8 = np.transpose(golden_i8, (1,2,0))
        # assert scale.size==1
        # scale = scale.repeat(self.op_config["out_channel"], axis=0)

        # compile and run, get output
        # compile and run, get output
        output = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})

        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()

            output_data = output + bias
            output_data = banker_round(output_data * scale) + out_zp
            clip_min = 0 if relu else -128
            clip_max = 127
            output_data = banker_round(np.clip(output_data, clip_min, clip_max))
            output_data = output_data.astype("int8")
            output = output_data
            
            correct_percent = (golden==output).sum() / golden.size
            return output, correct_percent

        return output, None


class BitSparseLinearOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        # read data
        input = np.loadtxt(os.path.join(df_dir, "input.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        scale = np.loadtxt(os.path.join(df_dir, "scale.txt"), dtype=np.float32).reshape(-1)
        out_zp = np.loadtxt(os.path.join(df_dir, "qo.zero_point.txt"), dtype=np.int32).reshape(1)
        # golden_i32 = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)
        golden_i8 = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int8)
        relu = (golden_i8 >= 0).all()

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden_i8 = golden_i8.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i8 = np.transpose(golden_i8, (1,2,0))
        assert scale.size==1
        # scale = scale.repeat(self.op_config["out_channel"], axis=0)

        # hack for zero-threshold filter
        keep_oc = find_nonzero_filter(weight)
        weight = np.take(weight, keep_oc, axis=0)
        bias = np.take(bias, keep_oc, axis=0)
        # scale = np.take(scale, keep_oc, axis=0)
        golden_i8 = np.take(golden_i8, keep_oc, axis=2)
        self.op_config["out_channel"] = weight.shape[0]
        self.helper.out_channel = weight.shape[0]
        
        # compile and run, get output
        output = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})

        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()

            output_data = output + bias.reshape(-1)
            output_data = banker_round(output_data * scale.reshape(-1)[0]) + out_zp
            clip_min = 0 if relu else -128
            clip_max = 127
            output_data = banker_round(np.clip(output_data, clip_min, clip_max))
            output_data = output_data.astype("int8")
            output = output_data
            
            correct_percent = (golden_i8==output).sum() / golden_i8.size
            return output, correct_percent

        return output, None

class DenseConv2dQuantifyOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        # read data
        input = np.loadtxt(os.path.join(df_dir, "conv_input_feature.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        scale = np.loadtxt(os.path.join(df_dir, "scale.txt"), dtype=np.float32)
        out_zp = np.loadtxt(os.path.join(df_dir, "qo.zero_point.txt"), dtype=np.int32).reshape(1)
        golden_i32 = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)
        golden_i8 = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int8)

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden_i8 = golden_i8.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i8 = np.transpose(golden_i8, (1,2,0))
        relu = (golden_i8 >= 0).all()

        golden_i32 = golden_i32.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i32 = golden_i32 - bias.reshape(-1,1,1)
        golden_i32 = np.transpose(golden_i32, (1,2,0))

        
        # compile and run, get output
        output_i8 = self.compile_and_run(code_dir, image_kwargs={
            "input": input, 
            "weight": weight, 
            "bias":bias, 
            "scale": scale, 
            "out_zp": out_zp,
            "relu": relu
        })
        # output_i32 = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})
        # output_i32 = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight, "bias":bias, "scale": scale, "out_zp": out_zp})
        # import pdb; pdb.set_trace()
        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()
            # correct = np.array_equal(golden_i8, output_i8)
            # assert correct
            correct_percent = (golden_i8==output_i8).sum() / golden_i8.size
            return output_i8, correct_percent

        return output_i8, None

class ValueSparseConv2dQuantifyOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        # read data
        input = np.loadtxt(os.path.join(df_dir, "conv_input_feature.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        scale = np.loadtxt(os.path.join(df_dir, "scale.txt"), dtype=np.float32)
        out_zp = np.loadtxt(os.path.join(df_dir, "qo.zero_point.txt"), dtype=np.int32).reshape(1)
        golden_i32 = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)
        golden_i8 = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int8)

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden_i8 = golden_i8.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i8 = np.transpose(golden_i8, (1,2,0))
        relu = (golden_i8 >= 0).all()

        golden_i32 = golden_i32.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i32 = golden_i32 - bias.reshape(-1,1,1)
        golden_i32 = np.transpose(golden_i32, (1,2,0))

        
        # compile and run, get output
        output_i8 = self.compile_and_run(code_dir, image_kwargs={
            "input": input, 
            "weight": weight, 
            "bias":bias, 
            "scale": scale, 
            "out_zp": out_zp,
            "relu": relu
        })
        # output_i32 = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})
        # output_i32 = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight, "bias":bias, "scale": scale, "out_zp": out_zp})
        # import pdb; pdb.set_trace()
        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()
            # correct = np.array_equal(golden_i8, output_i8)
            correct_percent = (golden_i8==output_i8).sum() / golden_i8.size
            # assert correct
            return output_i8, correct_percent

        return output_i8, None

class BitSparseConv2dQuantifyOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        # read data
        input = np.loadtxt(os.path.join(df_dir, "conv_input_feature.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        scale = np.loadtxt(os.path.join(df_dir, "scale.txt"), dtype=np.float32)
        out_zp = np.loadtxt(os.path.join(df_dir, "qo.zero_point.txt"), dtype=np.int32).reshape(1)
        golden_i32 = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)
        golden_i8 = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int8)

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden_i8 = golden_i8.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i8 = np.transpose(golden_i8, (1,2,0))
        relu = (golden_i8 >= 0).all()

        golden_i32 = golden_i32.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i32 = golden_i32 - bias.reshape(-1,1,1)
        golden_i32 = np.transpose(golden_i32, (1,2,0))

        
        # compile and run, get output
        output_i8 = self.compile_and_run(code_dir, image_kwargs={
            "input": input, 
            "weight": weight, 
            "bias":bias, 
            "scale": scale, 
            "out_zp": out_zp,
            "relu": relu
        })
        # output_i32 = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})
        # output_i32 = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight, "bias":bias, "scale": scale, "out_zp": out_zp})
        # import pdb; pdb.set_trace()
        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()
            # correct = np.array_equal(golden_i8, output_i8)
            correct_percent = (golden_i8==output_i8).sum() / golden_i8.size
            
            # assert correct
            return output_i8, correct_percent

        return output_i8, None


class ValueBitSparseConv2dQuantifyOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        # read data
        input = np.loadtxt(os.path.join(df_dir, "conv_input_feature.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        scale = np.loadtxt(os.path.join(df_dir, "scale.txt"), dtype=np.float32)
        out_zp = np.loadtxt(os.path.join(df_dir, "qo.zero_point.txt"), dtype=np.int32).reshape(1)
        golden_i32 = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)
        golden_i8 = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int8)

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden_i8 = golden_i8.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i8 = np.transpose(golden_i8, (1,2,0))
        relu = (golden_i8 >= 0).all()

        golden_i32 = golden_i32.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i32 = golden_i32 - bias.reshape(-1,1,1)
        golden_i32 = np.transpose(golden_i32, (1,2,0))

        
        # compile and run, get output
        output_i8 = self.compile_and_run(code_dir, image_kwargs={
            "input": input, 
            "weight": weight, 
            "bias":bias, 
            "scale": scale, 
            "out_zp": out_zp,
            "relu": relu
        })
        # output_i32 = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})
        # output_i32 = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight, "bias":bias, "scale": scale, "out_zp": out_zp})
        # import pdb; pdb.set_trace()
        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()
            # correct = np.array_equal(golden_i8, output_i8)
            correct_percent = (golden_i8==output_i8).sum() / golden_i8.size
            # assert correct
            return output_i8, correct_percent

        return output_i8, None


class DenseLinearQuantifyOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        # read data
        input = np.loadtxt(os.path.join(df_dir, "input.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        scale = np.loadtxt(os.path.join(df_dir, "scale.txt"), dtype=np.float32).reshape(-1)
        out_zp = np.loadtxt(os.path.join(df_dir, "qo.zero_point.txt"), dtype=np.int32).reshape(1)
        # golden_i32 = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)
        golden_i8 = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int8)
        relu = (golden_i8 >= 0).all()

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden_i8 = golden_i8.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i8 = np.transpose(golden_i8, (1,2,0))
        assert scale.size==1
        scale = scale.repeat(self.op_config["out_channel"], axis=0)

        # compile and run, get output
        output_i8 = self.compile_and_run(code_dir, image_kwargs={
            "input": input, 
            "weight": weight, 
            "bias":bias, 
            "scale": scale, 
            "out_zp": out_zp,
            "relu": relu
        })
        if check_result:

            helper_golden = self.helper._calculate_golden()
            correct = np.array_equal(golden_i8, output_i8)
            correct_percent = (golden_i8==output_i8).sum() / golden_i8.size
            # if correct_percent < 0.9:
            #     import pdb; pdb.set_trace()
            # assert correct
            return output_i8, correct_percent

        return output_i8, None


class BitSparseLinearQuantifyOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        # read data
        input = np.loadtxt(os.path.join(df_dir, "input.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        scale = np.loadtxt(os.path.join(df_dir, "scale.txt"), dtype=np.float32).reshape(-1)
        out_zp = np.loadtxt(os.path.join(df_dir, "qo.zero_point.txt"), dtype=np.int32).reshape(1)
        # golden_i32 = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)
        golden_i8 = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int8)
        relu = (golden_i8 >= 0).all()

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["in_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        weight = np.transpose(weight, (0,2,3,1))
        golden_i8 = golden_i8.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i8 = np.transpose(golden_i8, (1,2,0))
        assert scale.size==1
        scale = scale.repeat(self.op_config["out_channel"], axis=0)

        # hack for zero-threshold filter
        keep_oc = find_nonzero_filter(weight)
        weight = np.take(weight, keep_oc, axis=0)
        bias = np.take(bias, keep_oc, axis=0)
        scale = np.take(scale, keep_oc, axis=0)
        golden_i8 = np.take(golden_i8, keep_oc, axis=2)
        self.op_config["out_channel"] = weight.shape[0]
        self.helper.out_channel = weight.shape[0]
        
        # compile and run, get output
        output_i8 = self.compile_and_run(code_dir, image_kwargs={
            "input": input, 
            "weight": weight, 
            "bias":bias, 
            "scale": scale, 
            "out_zp": out_zp,
            "relu": relu
        })
        if check_result:

            helper_golden = self.helper._calculate_golden()
            correct = np.array_equal(golden_i8, output_i8)
            correct_percent = (golden_i8==output_i8).sum() / golden_i8.size
            # if correct_percent < 0.9:
            # assert correct
            return output_i8, correct_percent

        return output_i8, None

class DepthWiseConv2dQuantifyOperator(Operator):
    def __init__(self, config_path, template_path, op_config):
        super().__init__(config_path, template_path, op_config)

    def compile_and_run_from_dataflow_dir(self, df_dir, code_dir, check_result=False):
        # read data
        input = np.loadtxt(os.path.join(df_dir, "conv_input_feature.txt"), dtype=np.int8)
        weight = np.loadtxt(os.path.join(df_dir, "weight.txt"), dtype=np.int8)
        bias = np.loadtxt(os.path.join(df_dir, "bias.txt"), dtype=np.int32)
        scale = np.loadtxt(os.path.join(df_dir, "scale.txt"), dtype=np.float32)
        out_zp = np.loadtxt(os.path.join(df_dir, "qo.zero_point.txt"), dtype=np.int32).reshape(1)
        golden_i32 = np.loadtxt(os.path.join(df_dir, "output_feature.txt"), dtype=np.int32)
        golden_i8 = np.loadtxt(os.path.join(df_dir, "output.txt"), dtype=np.int8)

        # transform data
        #   input & golden: C,H,W -> H,W,C
        #   weight: O,I,H,W -> O,H,W,I
        #   golden output: 
        #       C,H,W -> H,W,C
        #       - bias (currently not support bias)
        input = input.reshape(self.op_config["in_channel"], self.op_config["in_hw"], self.op_config["in_hw"])
        # input = np.transpose(input, (1,2,0))
        weight = weight.reshape(self.op_config["out_channel"], self.op_config["ker_size"], self.op_config["ker_size"])
        # weight = np.transpose(weight, (0,2,3,1))
        golden_i8 = golden_i8.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i8 = np.transpose(golden_i8, (1,2,0))
        relu = (golden_i8 >= 0).all()

        golden_i32 = golden_i32.reshape(self.op_config["out_channel"], self.op_config["out_hw"], self.op_config["out_hw"])
        golden_i32 = golden_i32 - bias.reshape(-1,1,1)
        golden_i32 = np.transpose(golden_i32, (1,2,0))

        
        # compile and run, get output
        output_i8 = self.compile_and_run(code_dir, image_kwargs={
            "input": input, 
            "weight": weight, 
            "bias":bias, 
            "scale": scale, 
            "out_zp": out_zp,
            "relu": relu
        })
        # output_i32 = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight})
        # output_i32 = self.compile_and_run(code_dir, image_kwargs={"input": input, "weight": weight, "bias":bias, "scale": scale, "out_zp": out_zp})
        # import pdb; pdb.set_trace()
        # check result
        if check_result:

            helper_golden = self.helper._calculate_golden()
            # correct = np.array_equal(golden_i8, output_i8)
            correct_percent = (golden_i8==output_i8).sum() / golden_i8.size
            # import pdb; pdb.set_trace()
            # assert correct
            return output_i8, correct_percent

        return output_i8, None

if __name__=="__main__":
    pass
    # code_dir = "/home/wangyiou/project/cim_compiler_frontend/playground/.result"
    # dense_conv2d_template = DenseConv2dTemplate()
    # dense_conv2d_operator = dense_conv2d_template.get_operator(None)
    # dense_conv2d_operator.compile_and_run(code_dir, image_kwargs={"input": , "weight": })