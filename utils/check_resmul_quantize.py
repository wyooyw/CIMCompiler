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

def get_shapes(resmul_path):
    """
    info.txt is something like:
    input1: shape=(1, 16, 16, 16)
    input2: shape=(1, 16, 1, 1)
    output: shape=(1, 16, 16, 16)

    get shape of input1, input2, output
    """
    with open(os.path.join(resmul_path, "info.txt"), "r") as f:
        info = f.read()
    
    input1_shape = info.split("input1: shape=(")[1].split(")")[0]
    input1_shape = [int(i) for i in input1_shape.split(",")]
    input2_shape = info.split("input2: shape=(")[1].split(")")[0]
    input2_shape = [int(i) for i in input2_shape.split(",")]
    output_shape = info.split("output: shape=(")[1].split(")")[0]
    output_shape = [int(i) for i in output_shape.split(",")]
    return input1_shape, input2_shape, output_shape

def check_resmul_quantize(resmul_path):
    input1_shape, input2_shape, output_shape = get_shapes(resmul_path)


    input1 = np.loadtxt(os.path.join(resmul_path, "input1.txt"), dtype=np.int8).reshape(input1_shape)
    input2 = np.loadtxt(os.path.join(resmul_path, "input2.txt"), dtype=np.int8).reshape(input2_shape)
    scale = np.loadtxt(os.path.join(resmul_path, "scale.txt"), dtype=np.float64)
    output = np.loadtxt(os.path.join(resmul_path, "output.txt"), dtype=np.int8).reshape(output_shape)
    # check [22633 30378]
    input1_scaled = input1
    input2_scaled = input2
    # import pdb; pdb.set_trace()
    mul_result_1 = input1_scaled.astype(np.int32) * input2_scaled.astype(np.int32)
    mul_result_2 = banker_round(mul_result_1 * scale)
    mul_result = np.clip(mul_result_2, -128, 127).astype(np.int8)
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
    
    if not np.sum(mul_result == output) / mul_result.size > 0.99:
        print("quantize error")
        import pdb; pdb.set_trace()
        pass

    # return input1, input2, output

def check_res_add_need_input_memory_all(model_path):
    # get all sub folder ends with _add
    res_mul_paths = [os.path.join(model_path, i) for i in os.listdir(model_path) if i.endswith("_mult")]
    for res_mul_path in res_mul_paths:
        print(f"test {res_mul_path}")
        check_resmul_quantize(res_mul_path)

check_res_add_need_input_memory_all("/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_ori_data_0730")
# print(get_shapes("/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_ori_data_0730/7_mult"))
# 
# check_resadd_quantize("/cpfs/2926428ee2463e44/user/wangyiou/code/CIMCompiler/models/mobilenet/MobileNet-ori-data-0801/70`_add")
# print(banker_round(0.5000026))
# print(round(0.50000001))
# print(round(1.50000001))