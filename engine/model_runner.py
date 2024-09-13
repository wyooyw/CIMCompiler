import os
import json
import logging
from engine.operator_template import (
    DenseConv2dTemplate,
    BitSparseConv2dTemplate,
    ValueSparseConv2dTemplate,
    ValueBitSparseConv2dTemplate,
    DenseLinearTemplate,
    BitSparseLinearTemplate,
    # quantify
    DenseConv2dQuantifyTemplate,
    ValueSparseConv2dQuantifyTemplate,
    BitSparseConv2dQuantifyTemplate,
    ValueBitSparseConv2dQuantifyTemplate,
    DenseLinearQuantifyTemplate,
    BitSparseLinearQuantifyTemplate,
    DepthWiseConv2dQuantifyTemplate
)
from utils.logger import get_logger

from datetime import datetime
import shutil


formatted_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_base_dir = "/home/wangyiou/project/cim_compiler_frontend/playground/.result"
save_base_dir = os.path.join(save_base_dir, formatted_now)
os.makedirs(save_base_dir, exist_ok=False)
logger = get_logger(__name__, logging.DEBUG, os.path.join(save_base_dir, "model_runner.log"))

OP_TEMPLATE_LIST = [
    DenseConv2dTemplate(),
    BitSparseConv2dTemplate(),
    ValueSparseConv2dTemplate(),
    ValueBitSparseConv2dTemplate(),
    DenseLinearTemplate(),
    BitSparseLinearTemplate(),
    # quantify
    # DenseConv2dQuantifyTemplate(),
    # ValueSparseConv2dQuantifyTemplate(),
    # BitSparseConv2dQuantifyTemplate(),
    # ValueBitSparseConv2dQuantifyTemplate(),
    # DenseLinearQuantifyTemplate(),
    # BitSparseLinearQuantifyTemplate(),
    # DepthWiseConv2dQuantifyTemplate()
]
dense_cache = dict()
class ModelRunner:
    def __init__(self, model_name, model_path, code_dir, is_bit_sparse=False, is_value_sparse=False, quantify=False):
        self.model_name = model_name
        self.model_path = model_path
        self.is_bit_sparse = is_bit_sparse
        self.is_value_sparse = is_value_sparse
        self.quantify = quantify
        self.code_dir = code_dir

        logger.info(f"{model_name=}")
        logger.info(f"{model_path=}")
        logger.info(f"{is_bit_sparse=}")
        logger.info(f"{is_value_sparse=}")
        logger.info(f"{quantify=}")

    def raw_layers_from_json(self):
        json_path = os.path.join(self.model_path, f"{self.model_name}.json")
        with open(json_path, 'r') as f:
            model = json.load(f)
        layers = model["layers"]
        return layers

    def show_layers(self):
        for layer in self.raw_layers_from_json():
            print(layer)

    def run_layers_np(self):
        pass

    def find_first_valid_op_template(self, raw_layer, is_value_sparse=None, is_bit_sparse=None):
        for op_template in OP_TEMPLATE_LIST:
            if op_template.check_raw_layer(raw_layer, is_value_sparse, is_bit_sparse, self.quantify):
                return op_template
        if is_value_sparse or is_bit_sparse:
            if is_bit_sparse and is_value_sparse:
                fallback_value_sparse = False
                fallback_bit_sparse = True
            else:
                fallback_value_sparse = False
                fallback_bit_sparse = False
            logger.info(f"Layer {raw_layer['name']}: can't find op template with value_sparse={is_value_sparse} and bit_sparse={is_bit_sparse}. Fallback to value_sparse={fallback_value_sparse} and bit_sparse={fallback_bit_sparse}.")
            return self.find_first_valid_op_template(raw_layer, fallback_value_sparse, fallback_bit_sparse)
        else :
            return None

    def make_dense_cache_key(self, raw_layer):
        key = (
            raw_layer["input_row"],
            raw_layer["input_col"],
            raw_layer["input_channel"],
            raw_layer["output_channel"],
            raw_layer["weight_row"],
            raw_layer["weight_col"],
            raw_layer["stride"],
            raw_layer["padding_mode"],
            raw_layer["depthwise"],
        )
        return key

    def find_dense_cache(self, op_template, raw_layer):
        global dense_cache
        return None
        if not op_template.is_dense():
            return None

        key = self.make_dense_cache_key(raw_layer)
        return dense_cache.get(key, None)

    def fill_dense_cache(self, op_template, raw_layer, code_dir):
        global dense_cache
        if not op_template.is_dense():
            return None

        key = self.make_dense_cache_key(raw_layer)
        dense_cache[key] = code_dir

    def run_raw_layer(self, raw_layer):
        # prepare op
        op_template = self.find_first_valid_op_template(raw_layer, self.is_value_sparse, self.is_bit_sparse)
        if op_template is None:
            logger.info(f"Layer {raw_layer['name']} fail: no valid op template")
            return
        else:
            logger.info(f"Layer {raw_layer['name']} use op template: {op_template.__class__.__name__}")
        
        if self.is_bit_sparse and self.is_value_sparse:
            mode_name = "bit_value_sparse"
        elif self.is_bit_sparse:
            mode_name = "bit_sparse"
        elif self.is_value_sparse:
            mode_name = "value_sparse"
        else:
            mode_name = "dense"
        code_dir = os.path.join(self.code_dir, self.model_name, mode_name, raw_layer["name"])
        os.makedirs(code_dir)
        df_dir = os.path.join(self.model_path, raw_layer["name"])

        # cache_code_dir = self.find_dense_cache(op_template, raw_layer)
        # if cache_code_dir is not None:
        #     logger.info(f"Layer {raw_layer['name']} success: cache hit,  copy from {cache_code_dir} to {code_dir}")
        #     shutil.copytree(cache_code_dir, code_dir, dirs_exist_ok=True)
        #     return

        op = op_template.get_operator(raw_layer)
        
        logger.info(f"Layer {raw_layer['name']} begin")
        output, check_result = op.compile_and_run_from_dataflow_dir(df_dir, code_dir, check_result=False)

        # show layer name and check result
        logger.info(f"Layer {raw_layer['name']} success: {check_result}")

        # self.fill_dense_cache(op_template, raw_layer, code_dir)

    def run_layers_cim(self, num_layers=1000000, run_layers_name=None):
        if run_layers_name is None:
            run_layers_name = []
        elif isinstance(run_layers_name, str):
            run_layers_name = [run_layers_name]
        assert isinstance(run_layers_name, list)

        for idx,raw_layer in enumerate(self.raw_layers_from_json()):
            if idx > num_layers:
                break
            if len(run_layers_name)>0 and raw_layer["name"] not in run_layers_name:
                continue
            self.run_raw_layer(raw_layer)

def compile_for_model(
    model_name,
    model_path_dense,
    model_path_bit_sparse,
    model_path_value_sparse,
    model_path_value_bit_sparse,
    quantify=True,
    num_layers=1000000,
    run_layers_name=None
    ):
    plan_list = [
        [model_path_value_bit_sparse, True, True],
        [model_path_dense, False, False], # bit, value
        [model_path_bit_sparse, True, False],
        [model_path_value_sparse, False, True],
    ]
    for model_path_list,is_bit_sparse,is_value_sparse in plan_list:
        if type(model_path_list)==str:
            model_path_list = [model_path_list]
        for model_path in model_path_list:
            model_runner = ModelRunner(
                model_name=model_name, 
                model_path=model_path,
                code_dir=save_base_dir,
                quantify=quantify,
                is_value_sparse=is_value_sparse,
                is_bit_sparse=is_bit_sparse
            )
            model_runner.run_layers_cim(num_layers, run_layers_name)

if __name__=="__main__":
    # compile_for_model(
    #     model_name="AlexNet",
    #     model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_ori_data_0525",
    #     model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_csd_th2_data_0729",
    #     model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_0.6_csd_th2_data_0522",
    #     model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_0.6_csd_th2_data_0522",
    #     quantify=False,
    #     # run_layers_name=["0_conv"]
    # )
    # compile_for_model(
    #     model_name="VGG19",
    #     model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vggnet/VGG19_ori_data_0513",
    #     model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vggnet/VGG19_csd_th2_data_0803",
    #     model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vggnet/VGGNet_0.6_data_0731",
    #     model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vggnet/VGGNet_0.6_csd_th2_data_0717",
    #     quantify=False
    #     # run_layers_name=["12_conv"]
    # )
    # compile_for_model(
    #     model_name="EfficientNet",
    #     model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_ori_data_0730",
    #     model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_csd_th2_0803_data",
    #     model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_0.6_csd_th2_0616",
    #     model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_0.6_csd_th2_0616",
    #     quantify=False,
    # )
    # compile_for_model(
    #     model_name="MobileNetV2",
    #     model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/mobilenet/MobileNet-ori-data-0801",
    #     model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/mobilenet/MobileNet_csd_th2_data_0801",
    #     model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/mobilenet/MobileNet_0.6_csd_th2_data_0518",
    #     model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/mobilenet/MobileNet_0.6_csd_th2_data_0518",
    #     quantify=False,
    #     # run_layers_name=["4_pwconv"]
    # )
    
    # compile_for_model(
    #     model_name="ResNet18",
    #     model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet18_ori_data_0731",
    #     model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_csd_th2_data_0619",
    #     model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_0.6_data_0725",
    #     model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_0.6_csd_th2_data_0703",
    #     quantify = False,
    #     # run_layers_name=["22_conv"]
    # )


    compile_for_model(
        model_name="VGG19",
        model_path_dense = [],
        model_path_bit_sparse = [],
        model_path_value_sparse = [],
        model_path_value_bit_sparse = [
            # "/home/wangyiou/project/cim_compiler_frontend/playground/models/vggnet/VGGNet_0.2_csd_th2_data_0525"
            "/home/wangyiou/project/cim_compiler_frontend/playground/models/vggnet/VGGNet_0.4_csd_th2_data_0526"
        ],
        quantify=False
    )
    compile_for_model(
        model_name="MobileNet",
        model_path_dense = [],
        model_path_bit_sparse = [],
        model_path_value_sparse = [],
        model_path_value_bit_sparse = [
            # "/home/wangyiou/project/cim_compiler_frontend/playground/models/mobilenet/MobileNet_0.2_csd_th2_data_0516"
            "/home/wangyiou/project/cim_compiler_frontend/playground/models/mobilenet/MobileNet_0.4_csd_th2_data_0518"
        ],
        quantify=False,
        # run_layers_name=["4_pwconv"]
    )
    
    compile_for_model(
        model_name="ResNet18",
        model_path_dense = [],
        model_path_bit_sparse = [],
        model_path_value_sparse = [],
        model_path_value_bit_sparse = [
            # "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet_/ResNet_0.2_csd_th2_data_0620"
            "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet_/ResNet_0.4_csd_th2_data_0518"
        ],
        quantify = False,
        # run_layers_name=["22_conv"]
    )