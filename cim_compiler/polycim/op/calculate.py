import numpy as np


def conv2d(input, weight, dilation=1, stride=1):
    assert len(input.shape) == 4, f"{input.shape=}"
    assert len(weight.shape) == 4, f"{weight.shape=}"
    b, ic, ih, iw = input.shape  # nb: batch size, ic: input channels
    oc, kc, kh, kw = weight.shape  # oc: output channels, kc: kernel channels
    assert ic == kc  # 输入通道数需要匹配卷积核的通道数

    effective_kh = kh + (kh - 1) * (dilation - 1)
    effective_kw = kw + (kw - 1) * (dilation - 1)

    oh = (ih - effective_kh) // stride + 1
    ow = (iw - effective_kw) // stride + 1

    output = np.zeros((b, oc, oh, ow), dtype=np.int32)
    for _b in range(b):
        for _oc in range(oc):
            for _oh in range(oh):
                for _ow in range(ow):
                    input_window = input[
                        _b,
                        :,
                        _oh * stride : _oh * stride + kh * dilation : dilation,
                        _ow * stride : _ow * stride + kw * dilation : dilation,
                    ]
                    weight_window = weight[_oc, :, :, :]
                    output[_b, _oc, _oh, _ow] = np.sum(
                        input_window.astype(np.int32) * weight_window.astype(np.int32)
                    )

    return output


def depth_wise_conv2d(input, weight, dilation=1, stride=1):
    assert len(input.shape) == 3
    assert len(weight.shape) == 3
    ic, ih, iw = input.shape
    kc, kh, kw = weight.shape
    assert ic == kc

    effective_kh = kh + (kh - 1) * (dilation - 1)
    effective_kw = kw + (kw - 1) * (dilation - 1)
    oh = (ih - effective_kh) // stride + 1
    ow = (iw - effective_kw) // stride + 1
    oc = ic
    output = np.zeros((oc, oh, ow), dtype=np.int32)
    # for _oc in range(oc):
    for _oh in range(oh):
        for _ow in range(ow):
            input_window = input[
                :,
                _oh : _oh + kh * dilation : dilation,
                _ow : _ow + kw * dilation : dilation,
            ]
            weight_window = weight[:, :, :]
            output[:, _oh, _ow] += np.sum(
                input_window.astype(np.int32) * weight_window.astype(np.int32),
                axis=(1, 2),
            )
    return output


def depth_wise_conv3d(input, weight, dilation=1, stride=1):
    assert len(input.shape) == 4  # (channel, depth, height, width)
    assert len(weight.shape) == 4  # (channel, kernel_d, kernel_h, kernel_w)
    ic, id, ih, iw = input.shape
    kc, kd, kh, kw = weight.shape
    assert ic == kc

    effective_kd = kd + (kd - 1) * (dilation - 1)
    effective_kh = kh + (kh - 1) * (dilation - 1)
    effective_kw = kw + (kw - 1) * (dilation - 1)

    od = (id - effective_kd) // stride + 1
    oh = (ih - effective_kh) // stride + 1
    ow = (iw - effective_kw) // stride + 1
    oc = ic

    output = np.zeros((oc, od, oh, ow), dtype=np.int32)

    for _od in range(od):
        for _oh in range(oh):
            for _ow in range(ow):
                input_window = input[
                    :,
                    _od : _od + kd * dilation : dilation,
                    _oh : _oh + kh * dilation : dilation,
                    _ow : _ow + kw * dilation : dilation,
                ]
                weight_window = weight[:, :, :, :]
                output[:, _od, _oh, _ow] += np.sum(
                    input_window.astype(np.int32) * weight_window.astype(np.int32),
                    axis=(1, 2, 3),
                )
    return output


if __name__ == "__main__":
    # Add test for conv2d
    input_conv2d = (
        np.arange(32).reshape(1, 2, 4, 4).astype(np.float32)
    )  # batch=1, channels=2, height=4, width=4
    weight_conv2d = (
        np.arange(18).reshape(3, 2, 2, 3).astype(np.float32)
    )  # out_channels=3, in_channels=2, kernel_h=2, kernel_w=3
    output_conv2d = conv2d(input_conv2d, weight_conv2d, dilation=1, stride=1)
    print("Conv2D Output:")
    print(output_conv2d)
    print("Output shape:", output_conv2d.shape)

    # Add test for 3D convolution
    input_3d = np.arange(128).reshape(2, 4, 4, 4).astype(np.int32)
    weight_3d = np.arange(16).reshape(2, 2, 2, 2).astype(np.int32)
    output_3d = depth_wise_conv3d(input_3d, weight_3d, dilation=1)
    print("3D Depthwise Convolution Output:")
    print(output_3d)

    # Original 2D test
    input = np.arange(64).reshape(1, 8, 8).astype(np.int32)
    weight = np.arange(9).reshape(1, 3, 3).astype(np.int32)
    output = depth_wise_conv2d(input, weight, dilation=2)
    print("\n2D Depthwise Convolution Output:")
    print(output)
