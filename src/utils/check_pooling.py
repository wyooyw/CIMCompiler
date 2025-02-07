import os
import re
import numpy as np
def get_shapes(path):
    """
    info.txt is something like:
    input: shape=torch.Size([1, 96, 32, 32])
    kernel_size: 2
    stride: 2
    padding: 0
    output: shape=(1, 96, 16, 16)


    get shape of input, output
    """
    with open(os.path.join(path, "info.txt"), "r") as f:
        info = f.read()
    
        # 解析输入形状
    input_match = re.search(r'input: shape=.*?\[(.+?)\]', info)
    if input_match:
        input_shape = tuple(map(int, input_match.group(1).split(',')))
    else:
        raise ValueError("Cannot find input shape in info.txt")

    # 解析输出形状
    output_match = re.search(r'output: shape=\((.+?)\)', info)
    if output_match:
        output_shape = tuple(map(int, output_match.group(1).split(',')))
    else:
        raise ValueError("Cannot find output shape in info.txt")


    return input_shape, output_shape

def get_kernel_size(path):
    with open(os.path.join(path, "info.txt"), "r") as f:
        info = f.read()
    # kernel size
    kernel_size = int(re.search(r'kernel_size: (\d+)', info).group(1))


    return kernel_size

def check_max_pooling(path):
    input_shape, output_shape = get_shapes(path)
    kernel_size = get_kernel_size(path)
    input_tensor = np.loadtxt(os.path.join(path, "input.txt"), dtype=np.int8).reshape(input_shape)
    output_tensor = np.loadtxt(os.path.join(path, "output.txt"), dtype=np.int8).reshape(output_shape)

    b,c,h,w = input_shape
    input_tensor = input_tensor.reshape(
        b, c, h//kernel_size, kernel_size, w//kernel_size, kernel_size
    )
    # input_tensor = np.transpose(input_tensor, (0, 1, 2, 4, 3, 5))
    input_tensor = input_tensor.max(axis=(3,5))
    assert np.sum(input_tensor == output_tensor) / output_tensor.size > 0.99, "quantize error"

def banker_round(x):

    half = 0.5
    one = 1
    two = 2

    rounded = np.ceil(x - half)
    bankers_mask = one - (np.ceil(x + half) - np.floor(x + half))
    non_even = np.abs(np.mod(rounded, two))
    return rounded + (bankers_mask * non_even)

def check_mean_pooling(path):
    input_shape, output_shape = get_shapes(path)
    b,c,h,w = input_shape
    assert (
        output_shape[0]==b 
        and output_shape[1]==c 
        and output_shape[2]==1 
        and output_shape[3]==1
    ), "output shape error"

    input_tensor = np.loadtxt(os.path.join(path, "input.txt"), dtype=np.int8).reshape(input_shape)
    output_tensor = np.loadtxt(os.path.join(path, "output.txt"), dtype=np.int8).reshape(output_shape)
    scale = np.loadtxt(os.path.join(path, "scale.txt"), dtype=np.float32)

    input_tensor = input_tensor.mean(axis=(2,3))
    input_tensor = np.floor(input_tensor)
    input_tensor = input_tensor * scale
    input_tensor = banker_round(input_tensor)
    input_tensor = np.clip(input_tensor, -128, 127).astype(np.int8)
    
    output_tensor = output_tensor.reshape(input_tensor.shape)
    # print(input_tensor.shape, output_tensor.shape)
    # print(input_tensor)
    # print(output_tensor)
    assert np.sum(input_tensor == output_tensor) / output_tensor.size > 0.99, "mean pooling error"

def check_max_pooling_all(model_path):
    # get all sub folder ends with _add
    pool_paths = [os.path.join(model_path, i) for i in os.listdir(model_path) if i.endswith("_pool")]
    for pool_path in pool_paths:
        print(f"test {pool_path}")
        check_max_pooling(pool_path)

def check_mean_pooling_all(model_path):
    # get all sub folder ends with _add
    pool_paths = [os.path.join(model_path, i) for i in os.listdir(model_path) if i.endswith("_pool")]
    for pool_path in pool_paths:
        print(f"test {pool_path}")
        check_mean_pooling(pool_path)

if __name__=="__main__":
    # path = "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_ori_data_0525/1_pool/"
    # check_max_pooling(path) 
    path_max_pooling = "/home/wangyiou/project/cim_compiler_frontend/playground/models/alexnet/AlexNet_ori_data_0525/"
    check_max_pooling_all(path_max_pooling)
    path_mean_pooling = "/home/wangyiou/project/cim_compiler_frontend/playground/models/efficientnet/EfficientNet_csd_th2_0803_data/"
    check_mean_pooling_all(path_mean_pooling)
