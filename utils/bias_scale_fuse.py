import numpy as np


def bias_scale_fuse(bias, scale):
    assert len(bias.shape) == 1
    assert len(scale.shape) == 1
    assert bias.shape[0] == scale.shape[0]
    assert bias.dtype == np.int32
    assert scale.dtype == np.float32
    fuse = bytearray()
    for i in range(bias.shape[0]):
        fuse = fuse + bytearray(bias[i : i + 1])
        fuse = fuse + bytearray(scale[i : i + 1])
        # print(f"{i=}, {len(fuse)=} {bias[i]=}, {len(bytearray(bias[i]))=}")
    assert len(fuse) == bias.shape[0] * 8, f"{len(fuse)=}, {bias.shape[0]=}"
    return fuse
