import json
import math


def get_padding(raw_layer):
    if raw_layer["padding_mode"] == "SAME":
        if raw_layer["weight_row"] == 3:
            padding = 1
        elif raw_layer["weight_row"] == 1:
            padding = 0
    else:
        padding = 0
    return padding


def conv_max_out_memory_use(layer):
    if layer["depthwise"] == True:
        return (0, 0)
    input_row = layer["input_row"]
    stride = layer["stride"]
    ker_size = layer["weight_row"]

    padding = get_padding(layer)

    output_row = (input_row + 2 * padding - ker_size) // stride + 1
    output_channel = layer["output_channel"]
    reduce_size = layer["input_channel"] * layer["weight_row"] * layer["weight_col"]
    return (output_row * output_row * min(64, output_channel) * 4, 0)
    # if reduce_size <= 256:
    #     return (0, output_row * output_row * output_channel)
    # else:
    #     return (
    #         output_row * output_row * min(64,output_channel) * 4,             # i32 partial sum
    #         0
    #         # ,output_row * output_row * output_channel    # i8 result
    #     )


def model_max_out_memory_use(model_config):
    max_out_memory_use = 0
    for layer in model_config["layers"]:
        if layer["type"] == "CONV" or layer["type"] == "FC":
            layer_use_i32, layer_use_i8 = conv_max_out_memory_use(layer)
            print(
                f"{layer['name']} max out memory use: i32: {layer_use_i32}B, {math.ceil(layer_use_i32/1024)}KB; i8: {layer_use_i8}B, {math.ceil(layer_use_i8/1024)}KB"
            )
            layer_use = layer_use_i32 + layer_use_i8
            print(
                f"{layer['name']} max out memory use: {layer_use}B, {math.ceil(layer_use/1024)}KB"
            )
            max_out_memory_use = max(max_out_memory_use, layer_use)
    return max_out_memory_use


def main():
    model_config_list = [
        # "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_ori_data_0525/AlexNet.json",
        # "/home/wangyiou/project/cim_compiler_frontend/playground/models/vggnet/VGG19_ori_data_0513/VGG19.json",
        # "/home/wangyiou/project/cim_compiler_frontend/playground/models/mobilenet/MobileNet-ori-data-0801/MobileNetV2.json",
        # "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_ori_data_0730/EfficientNet.json",
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet18_ori_data_0731/ResNet18.json"
    ]
    for model_config in model_config_list:
        with open(model_config, "r") as f:
            model_config = json.load(f)
        print("\n-----------------------------------------------------\n")
        memory = model_max_out_memory_use(model_config)
        # print in B,KB,
        print(f"{memory}B, {math.ceil(memory/1024)}KB")


if __name__ == "__main__":
    main()
