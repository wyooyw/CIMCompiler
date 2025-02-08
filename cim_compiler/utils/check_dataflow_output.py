import numpy as np

from cim_compiler.utils.round import banker_round


def conv2d(input, weight, padding, bias=None, stride=1):
    input_pad = np.pad(
        input,
        ((0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )
    input_hw = input.shape[1]
    ker_size = weight.shape[2]

    output_hw = (input_hw + 2 * padding - ker_size) // stride + 1
    output_c = weight.shape[0]

    weight = weight.reshape(output_c, -1)

    output = np.zeros((output_c, output_hw, output_hw), dtype=np.int32)
    for row in range(output_hw):
        for col in range(output_hw):
            input = input_pad[
                :,
                stride * row : stride * row + ker_size,
                stride * col : stride * col + ker_size,
            ].reshape(-1, 1)
            print(weight.shape, input.shape)
            output_pixel = np.matmul(weight.astype(np.int32), input.astype(np.int32))
            output[:, row, col] = output_pixel.reshape(-1)
    if bias is not None:
        output += bias.reshape(-1, 1, 1)
    return output


def conv2d2(input, weight, padding, bias=None):
    input = np.transpose(input, [1, 2, 0])

    input_pad = np.pad(
        input,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    input_hw = input.shape[1]
    ker_size = weight.shape[2]

    output_hw = input_hw + 2 * padding - ker_size + 1
    output_c = weight.shape[0]

    # weight = np.transpose(weight, [0,2,3,1])
    weight = weight.reshape(output_c, -1)

    output = np.zeros((output_hw, output_hw, output_c), dtype=np.int32)

    for row in range(output_hw):
        for col in range(output_hw):
            input = input_pad[row : row + ker_size, col : col + ker_size, :].reshape(
                -1, 1
            )
            golden = np.matmul(weight.astype(np.int32), input.astype(np.int32))
            output[row, col, :] = golden.reshape(-1)

    output = np.transpose(output, [2, 0, 1])
    return output

    if bias is not None:
        output += bias.reshape(-1, 1, 1)
    return output


def main():
    input_path = "models/resnet18/ResNet18_ori_data_0731/7_conv/conv_input_feature.txt"
    weight_path = "models/resnet18/ResNet18_ori_data_0731/7_conv/weight.txt"
    bias_path = "models/resnet18/ResNet18_ori_data_0731/7_conv/bias.txt"
    output_path = "models/resnet18/ResNet18_ori_data_0731/7_conv/output_feature.txt"

    input = np.loadtxt(input_path, dtype=np.int8).reshape(64, 32, 32)
    weight = np.loadtxt(weight_path, dtype=np.int8).reshape(128, 64, 3, 3)
    bias = np.loadtxt(bias_path, dtype=np.int32)
    golden = np.loadtxt(output_path, dtype=np.int32).reshape(128, 16, 16)
    golden = golden - bias.reshape(-1, 1, 1)

    output = conv2d(input, weight, 1, stride=2)
    print(output)
    print(golden)
    print(f"{np.array_equal(output, golden)=}")
    print(golden[:, 0, 0])

    # output2 = conv2d2(input, weight, 1)
    # print(f"{np.array_equal(output2, golden)=}")
    # print(f"{output-golden=}")


def check_zeropoint():
    out_wh = 16
    out_channel = 128
    output_feature_path = (
        "models/resnet18/ResNet18_ori_data_0731/7_conv/output_feature.txt"
    )
    output_feature = np.loadtxt(output_feature_path, dtype=np.int32).reshape(
        out_channel, out_wh, out_wh
    )
    output_path = "models/resnet18/ResNet18_ori_data_0731/7_conv/output.txt"
    output_golden = np.loadtxt(output_path, dtype=np.int8).reshape(
        out_channel, out_wh, out_wh
    )
    output = np.zeros((out_channel, out_wh, out_wh), dtype=np.int8)
    bias = np.loadtxt(
        "models/resnet18/ResNet18_ori_data_0731/7_conv/bias.txt", dtype=np.int32
    )
    scale = np.loadtxt(
        "models/resnet18/ResNet18_ori_data_0731/7_conv/scale.txt", dtype=np.float32
    )
    out_zp_data = 0
    for r in range(out_wh):
        for c in range(out_wh):
            input_data = output_feature[:, r, c]
            output_data = input_data  # + bias
            # print("1:",output_data)
            output_data = banker_round(output_data * scale) + out_zp_data
            # print("2:",output_data)
            output_data = banker_round(np.clip(output_data, 0, 127))
            # print("3:",output_data)
            # exit()
            output_data = output_data.astype("int8")
            output[:, r, c] = output_data
    print(f"{np.array_equal(output, output_golden)=}")
    print(f"output[:,0,0]:\n{output[:,0,0]}")
    print(f"output_golden[:,0,0]:\n{output_golden[:,0,0]}")


if __name__ == "__main__":
    main()
    # check_zeropoint()
