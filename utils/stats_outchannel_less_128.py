import json
def main():
    model_config_list = [
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_ori_data_0525/AlexNet.json",
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/vggnet/VGG19_ori_data_0513/VGG19.json",
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/mobilenet/MobileNet-ori-data-0801/MobileNetV2.json",
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_ori_data_0730/EfficientNet.json",
        "/home/wangyiou/project/cim_compiler_frontend/playground/models/resnet18/ResNet18_ori_data_0731/ResNet18.json"
    ]
    for model_config in model_config_list:
        with open(model_config, "r") as f:
            model = json.load(f)
        model_name = model["name"]
        layers = model["layers"]
        oc_threshold = 128
        conv_fc_layers = [layer for layer in layers if layer["type"] in ["CONV","FC"] and layer["depthwise"]==False]
        num_total_layers = len(conv_fc_layers)
        num_oc_less_layers = len([layer for layer in conv_fc_layers if layer["output_channel"] < oc_threshold])
        print(f"{model_name=}, \n\t{num_total_layers=}, \n\tnum_out_channel_less_than_{oc_threshold}_layers={num_oc_less_layers}")

if __name__ == "__main__":
    main()