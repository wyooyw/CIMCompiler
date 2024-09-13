import json
import math

def conv_max_out_memory_use(layer):
    input_row = layer["input_row"] + 2
    input_channel = layer["input_channel"]
    kernel_size = layer["weight_row"]
    # return kernel_size * 2 * input_row * input_channel
    return kernel_size * input_row * input_row * input_channel

def model_max_out_memory_use(model_config):
    max_out_memory_use = 0
    for layer in model_config["layers"]:
        if (layer["type"] == "CONV" or layer["type"] == "FC") and layer["depthwise"]==False:
            max_out_memory_use = max(max_out_memory_use, conv_max_out_memory_use(layer))
    return max_out_memory_use

def main():
    model_config_list = [
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_ori_data_0525/AlexNet.json",
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/vggnet/VGG19_ori_data_0513/VGG19.json",
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet18_ori_data_0731/ResNet18.json",
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/mobilenet/MobileNet-ori-data-0801/MobileNetV2.json",
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_ori_data_0730/EfficientNet.json"
    ]
    for model_config in model_config_list:
        with open(model_config, "r") as f:
            model_config = json.load(f)
        memory = model_max_out_memory_use(model_config)
        # print in B,KB,
        print(f"{memory}B, {math.ceil(memory/1024)}KB")

if __name__ == "__main__":
    main()