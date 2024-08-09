import os
import json
import logging
from engine.operator_template import (
    DenseConv2dTemplate,
    BitSparseConv2dTemplate,
    ValueSparseConv2dTemplate,
    ValueBitSparseConv2dTemplate,
    # quantify
    DenseConv2dQuantifyTemplate,
    ValueSparseConv2dQuantifyTemplate,
    BitSparseConv2dQuantifyTemplate,
    ValueBitSparseConv2dQuantifyTemplate,
    DenseLinearQuantifyTemplate,
    BitSparseLinearQuantifyTemplate
)
from utils.logger import get_logger

from datetime import datetime
formatted_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_base_dir = "/home/wangyiou/project/cim_compiler_frontend/playground/.result"
save_base_dir = os.path.join(save_base_dir, formatted_now)
os.makedirs(save_base_dir, exist_ok=False)
logger = get_logger(__name__, logging.DEBUG, os.path.join(save_base_dir, "model_runner.log"))

OP_TEMPLATE_LIST = [
    # DenseConv2dTemplate(),
    # BitSparseConv2dTemplate(),
    # ValueSparseConv2dTemplate(),
    # ValueBitSparseConv2dTemplate(),
    # quantify
    DenseConv2dQuantifyTemplate(),
    ValueSparseConv2dQuantifyTemplate(),
    BitSparseConv2dQuantifyTemplate(),
    ValueBitSparseConv2dQuantifyTemplate(),
    DenseLinearQuantifyTemplate(),
    BitSparseLinearQuantifyTemplate()
]

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

    def find_first_valid_op_template(self, raw_layer):
        for op_template in OP_TEMPLATE_LIST:
            if op_template.check_raw_layer(raw_layer, self.is_value_sparse, self.is_bit_sparse, self.quantify):
                return op_template
        return None


    def run_raw_layer(self, raw_layer):
        # prepare op
        op_template = self.find_first_valid_op_template(raw_layer)
        if op_template is None:
            logger.info(f"Layer {raw_layer['name']} fail: no valid op template")
            return
        op = op_template.get_operator(raw_layer)
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
        logger.info(f"Layer {raw_layer['name']} begin")
        output, check_result = op.compile_and_run_from_dataflow_dir(df_dir, code_dir, check_result=True)

        # show layer name and check result
        logger.info(f"Layer {raw_layer['name']} success: {check_result}")

    def run_layers_cim(self):
        for raw_layer in self.raw_layers_from_json():
            self.run_raw_layer(raw_layer)

def compile_for_model(
    model_name,
    model_path_dense,
    model_path_bit_sparse,
    model_path_value_sparse,
    model_path_value_bit_sparse
    ):
    plan_list = [
        [model_path_dense, False, False], # bit, value
        [model_path_bit_sparse, True, False],
        [model_path_value_sparse, False, True],
        [model_path_value_bit_sparse, True, True],
    ]
    for model_path,is_bit_sparse,is_value_sparse in plan_list:
        model_runner = ModelRunner(
            model_name=model_name, 
            model_path=model_path,
            code_dir=save_base_dir,
            quantify=True,
            is_value_sparse=is_value_sparse,
            is_bit_sparse=is_bit_sparse
        )
        model_runner.run_layers_cim()

if __name__=="__main__":
    # compile_for_model(
    #     model_name="AlexNet",
    #     model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_ori_data_0525",
    #     model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_csd_th2_data_0525",
    #     model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_0.6_data_0725",
    #     model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_0.6_csd_th2_data_0522",
    # )

    # compile_for_model(
    #     model_name="VGG19",
    #  model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vgg19/VGG19_ori_data_0513",
    # model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_csd_th2_data_0619",
    # model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vgg19/VGGNet_0.6_data_0731",
    # model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vgg19/VGGNet_0.6_csd_th2_data_0717"
        
    #     )

    compile_for_model(
        model_name="ResNet18",
        model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet18_ori_data_0731",
        model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_csd_th2_data_0619",
        model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_0.6_data_0725",
        model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_0.6_csd_th2_data_0703"
    )
    # model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet18_ori_data_0731"
    # model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_csd_th2_data_0619"
    # model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_0.6_data_0725"
    # model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_0.6_csd_th2_data_0703"
    # model_runner = ModelRunner(
    #     model_name="ResNet18", 
    #     model_path=model_path_bit_sparse,
    #     code_dir="/home/wangyiou/project/cim_compiler_frontend/playground/.result",
    #     is_bit_sparse=True,
    #     # is_value_sparse=True,
    #     quantify=False
    # )
    # model_runner.run_layers_cim()

    # model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vgg19/VGG19_ori_data_0513"
    # model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet_csd_th2_data_0619"
    # model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vgg19/VGGNet_0.6_data_0731"
    # model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vgg19/VGGNet_0.6_csd_th2_data_0717"
    # model_runner = ModelRunner(
    #     model_name="VGG19", 
    #     model_path=model_path_value_bit_sparse,
    #     code_dir="/home/wangyiou/project/cim_compiler_frontend/playground/.result",
    #     is_value_sparse=True,
    #     is_bit_sparse=True
    # )
    # model_runner.run_layers_cim()

    # model_path_dense = "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_ori_data_0730"
    # model_path_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_csd_th2_data_0705"
    # model_path_value_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_0.6_data_0726"
    # model_path_value_bit_sparse = "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_0.6_csd_th2_0616"
    # model_runner = ModelRunner(
    #     model_name="EfficientNet", 
    #     model_path=model_path_value_bit_sparse,
    #     code_dir="/home/wangyiou/project/cim_compiler_frontend/playground/.result",
    #     is_bit_sparse=True,
    #     is_value_sparse=True
    # )
    # model_runner.run_layers_cim()