import os
from functools import reduce

def check_res_add_need_input_memory(res_add_path):
    with open(os.path.join(res_add_path, "info.txt"), "r") as f:
        info = f.read()
    
    """
    info.txt is something like:

    input1: shape=(1, 24, 32, 32)
    input2: shape=(1, 24, 32, 32)
    output: shape=(1, 24, 32, 32)

    parse these shapes into list of ints

    """

    input1_shape = info.split("input1: shape=(")[1].split(")")[0]
    input1_shape = [int(i) for i in input1_shape.split(",")]

    input_size = 2 * reduce(lambda x, y: x * y, input1_shape)
    
    return input_size

def human_readable_size(size):
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024}KB"
    else:
        return f"{size / 1024 / 1024}MB"

def check_res_add_need_input_memory_all(model_path):
    # get all sub folder ends with _add
    res_add_paths = [os.path.join(model_path, i) for i in os.listdir(model_path) if i.endswith("_add")]
    for res_add_path in res_add_paths:
        input_size = check_res_add_need_input_memory(res_add_path)
        print(f"{res_add_path}: {human_readable_size(input_size)}")

if __name__ == "__main__":
    check_res_add_need_input_memory_all("/cpfs/2926428ee2463e44/user/wangyiou/code/CIMCompiler/models/resnet18")
