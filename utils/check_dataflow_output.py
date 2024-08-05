import numpy as np

def conv2d(input, weight, padding, bias=None):
    input_pad = np.pad(input, ((0,0),(padding,padding),(padding,padding)), mode='constant', constant_values=0)
    input_hw = input.shape[1]
    ker_size = weight.shape[2]
    
    output_hw = input_hw + 2 * padding - ker_size + 1
    output_c = weight.shape[0]

    weight = weight.reshape(output_c, -1)
    
    output = np.zeros((output_c, output_hw, output_hw), dtype=np.int32)
    for row in range(output_hw):
        for col in range(output_hw):
            input = input_pad[:,row:row+ker_size,col:col+ker_size].reshape(-1,1)
            print(weight.shape, input.shape)
            output_pixel = np.matmul(weight.astype(np.int32), input.astype(np.int32))
            output[:,row,col] = output_pixel.reshape(-1)
    if bias is not None:
        output += bias.reshape(-1,1,1)
    return output

def conv2d2(input, weight, padding, bias=None):
    input = np.transpose(input, [1,2,0])

    input_pad = np.pad(input, ((padding,padding),(padding,padding),(0,0)), mode='constant', constant_values=0)
    input_hw = input.shape[1]
    ker_size = weight.shape[2]
    
    output_hw = input_hw + 2 * padding - ker_size + 1
    output_c = weight.shape[0]

    # weight = np.transpose(weight, [0,2,3,1])
    weight = weight.reshape(output_c, -1)
    
    output = np.zeros((output_hw, output_hw, output_c), dtype=np.int32)

    for row in range(output_hw):
        for col in range(output_hw):
            input = input_pad[row:row+ker_size,col:col+ker_size,:].reshape(-1,1)
            golden = np.matmul(weight.astype(np.int32), input.astype(np.int32))
            output[row,col,:] = golden.reshape(-1)

    output = np.transpose(output, [2,0,1])
    return output

    if bias is not None:
        output += bias.reshape(-1,1,1)
    return output

def main():
    input_path = "models/alexnet/AlexNet_ori_data_0525/2_conv/conv_input_feature.txt"
    weight_path = "models/alexnet/AlexNet_ori_data_0525/2_conv/weight.txt"
    bias_path = "models/alexnet/AlexNet_ori_data_0525/2_conv/bias.txt"
    output_path = "models/alexnet/AlexNet_ori_data_0525/2_conv/output_feature.txt"

    input = np.loadtxt(input_path, dtype=np.int8).reshape(96,16,16)
    weight = np.loadtxt(weight_path, dtype=np.int8).reshape(256, 96, 3, 3)
    bias = np.loadtxt(bias_path, dtype=np.int32)
    golden = np.loadtxt(output_path, dtype=np.int32).reshape(256, 16, 16)
    golden = golden - bias.reshape(-1,1,1)

    output = conv2d(input, weight, 1)
    print(output)
    print(golden)
    print(f"{np.array_equal(output, golden)=}")

    output2 = conv2d2(input, weight, 1)
    print(f"{np.array_equal(output2, golden)=}")
    # print(f"{output-golden=}")

if __name__=="__main__":
    main()