import numpy as np
import os
# os.environ["PYTHONPATH"] = "/cpfs/2926428ee2463e44/user/wangyiou/code/CIMCompiler"

def banker_round(x):

    half = 0.5
    one = 1
    two = 2

    rounded = np.ceil(x - half)
    bankers_mask = one - (np.ceil(x + half) - np.floor(x + half))
    non_even = np.abs(np.mod(rounded, two))
    return rounded + (bankers_mask * non_even)

def check_resadd_quantize(resadd_path):
    input1 = np.loadtxt(os.path.join(resadd_path, "input1.txt"), dtype=np.int8)
    input2 = np.loadtxt(os.path.join(resadd_path, "input2.txt"), dtype=np.int8)
    scale1 = np.loadtxt(os.path.join(resadd_path, "scale1.txt"), dtype=np.float64)
    scale2 = np.loadtxt(os.path.join(resadd_path, "scale2.txt"), dtype=np.float64)
    output = np.loadtxt(os.path.join(resadd_path, "output.txt"), dtype=np.int8)
    # check [22633 30378]
    input1_scaled = input1 * scale1
    input2_scaled = input2 * scale2
    add_result_1 = input1_scaled + input2_scaled
    add_result_2 = banker_round(add_result_1)
    add_result = np.clip(add_result_2, -128, 127).astype(np.int8)
    # import pdb; pdb.set_trace()
    # print(f"{add_result[:10]}")
    # print(f"{output[:10]}")
    # num of different elements
    # different_num = np.sum(add_result != output)
    # print(f"different num: {different_num}")

    # diff position and values
    # diff_position = np.where(add_result != output)[0]
    # diff_value = add_result[diff_position] - output[diff_position]
    # print(f"diff position: {diff_position}")
    # print(f"diff value: {add_result[diff_position]=}, {output[diff_position]=}")
    assert np.sum(add_result == output) / len(add_result) > 0.99, "quantize error"

    # return input1, input2, output

def check_res_add_need_input_memory_all(model_path):
    # get all sub folder ends with _add
    res_add_paths = [os.path.join(model_path, i) for i in os.listdir(model_path) if i.endswith("_add")]
    for res_add_path in res_add_paths:
        print(f"test {res_add_path}")
        check_resadd_quantize(res_add_path)

check_res_add_need_input_memory_all("/cpfs/2926428ee2463e44/user/wangyiou/code/CIMCompiler/models/mobilenet/MobileNet-ori-data-0801")
# 
# check_resadd_quantize("/cpfs/2926428ee2463e44/user/wangyiou/code/CIMCompiler/models/mobilenet/MobileNet-ori-data-0801/70`_add")
# print(banker_round(0.5000026))
# print(round(0.50000001))
# print(round(1.50000001))