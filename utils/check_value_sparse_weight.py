import json
import math
import os

import numpy as np


def get_model_info(path, model_name):
    json_path = os.path.join(path, f"{model_name}.json")
    with open(json_path, "r") as f:
        model_info = json.load(f)
    return model_info


def get_weight(model_path, layer_info, layer_name):
    weight_path = os.path.join(model_path, layer_name, "weight.txt")
    weight = np.loadtxt(weight_path, dtype=np.int8).reshape(
        layer_info["output_channel"],
        layer_info["input_channel"],
        layer_info["weight_row"],
        layer_info["weight_col"],
    )
    weight = np.transpose(weight, (0, 2, 3, 1))
    return weight


def get_value_index(weight):
    from data_processor.dense import convert_value_sparse_conv2d_weight

    bitwidth = 8
    n_group = 4
    config = {
        "n_row": 64,
        "n_vcol": 2,
        "n_group": n_group,
        "n_macro": 64,
        "n_comp": 16,
        "n_value_sparse_from": 128,
        "n_value_sparse_to": 16,
    }
    result = convert_value_sparse_conv2d_weight(weight, config)
    mapping_from_to_row = result["mapping_from_to_row"]
    return mapping_from_to_row


def filter_zero_ratio(weight):
    out_channel = weight.shape[0]
    for oc in range(out_channel):
        nero_cnt = (weight[oc] == 0).sum()
        zero_ratio = nero_cnt / weight[oc].size

        print(f"Filter {oc} has zero count {nero_cnt} ratio {zero_ratio}")
        if oc % 2 == 1:
            common_nero_cnt = np.logical_and(weight[oc - 1] == 0, weight[oc] == 0).sum()
            common_zero_ratio = common_nero_cnt / weight[oc].size
            print(
                f"    Filter [{oc-1}, {oc}] has common zero count {common_nero_cnt}, ratio {common_zero_ratio}"
            )
        # import pdb; pdb.set_trace()


def twin_filter_zero_ratio(weight):
    out_channel = weight.shape[0]
    for oc in range(0, out_channel, 2):
        nero_cnt = np.logical_and(weight[oc] == 0, weight[oc + 1] == 0).sum()
        zero_ratio = nero_cnt / weight[oc].size
        print(
            f"Filter [{oc}, {oc+1}] has common zero count {nero_cnt}, ratio {zero_ratio}"
        )


if __name__ == "__main__":
    model_path = "/home/wangyiou/project/cim_compiler_frontend/playground/models/vggnet/VGGNet_0.4_csd_th2_data_0526"
    model_name = "VGG19"
    model_info = get_model_info(path=model_path, model_name=model_name)
    np.set_printoptions(threshold=np.inf)
    for idx, layer_info in enumerate(model_info["layers"]):
        if layer_info["type"] not in ["CONV"]:  # , "FCN"]:
            continue
        print("\n------------------------------------------------")
        print(f"Layer {layer_info['name']}")
        print("------------------------------------------------\n")
        weight = get_weight(model_path, layer_info, layer_info["name"])
        mapping_from_to_row = get_value_index(weight)
        print(len(mapping_from_to_row))
        print(mapping_from_to_row)

        weight = weight.reshape(weight.shape[0], -1)
        filter_zero_ratio(weight)
    # twin_filter_zero_ratio(weight)
    exit()
    print("------------------------------------------------")
    filter_begin = 0
    filter_end = 32
    for j in range(0, weight.shape[1] // 128 * 128, 128):
        print(f"Filter({filter_begin}-{filter_end}) element {j} to {j + 128}")
        subweight = weight[filter_begin:filter_end, j : j + 128]
        print(subweight.shape)
        subweight = subweight.reshape(16, 2, 128)
        num_zero_min = 9999
        for i in range(16):
            weight_for_one_macrp = subweight[i]
            num_zero = np.logical_and(
                weight_for_one_macrp[0] == 0, weight_for_one_macrp[1] == 0
            ).sum()
            print(f"macro {i} has {num_zero} zeros")
            num_zero_min = min(num_zero_min, num_zero)
        num_16_in_128 = math.ceil((128 - num_zero_min) / 16)
        print(f"{num_16_in_128=}")
