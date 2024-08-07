import json
import math
def conv_max_out_memory_use(layer):
    input_row = layer["input_row"]
    output_row = input_row
    output_channel = layer["output_channel"]
    reduce_size = layer["input_channel"] * layer["weight_row"] * layer["weight_col"]
    if reduce_size <= 1024:
        return output_row * output_row * output_channel
    else:
        return (
            output_row * output_row * 128 * 4             # i32 partial sum
            + output_row * output_row * output_channel    # i8 result
        )

def model_max_out_memory_use(model_config):
    max_out_memory_use = 0
    for layer in model_config["layers"]:
        if layer["type"] == "CONV" or layer["type"] == "FC":
            max_out_memory_use = max(max_out_memory_use, conv_max_out_memory_use(layer))
    return max_out_memory_use

def main():
    model_config_list = [
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_ori_data_0525/AlexNet.json",
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/vgg19/VGG19_ori_data_0513/VGG19.json"
    ]
    for model_config in model_config_list:
        with open(model_config, "r") as f:
            model_config = json.load(f)
        memory = model_max_out_memory_use(model_config)
        # print in B,KB,
        print(f"{memory}B, {math.ceil(memory/1024)}KB")

if __name__ == "__main__":
    main()