import numpy as np


def get_dtype_from_bitwidth(bitwidth, is_float=False):
    if bitwidth == 8:
        assert not is_float, f"8-bit is integer"
        return np.int8
    elif bitwidth == 16:
        return np.float16 if is_float else np.int16
    elif bitwidth == 32:
        return np.float32 if is_float else np.int32
    else:
        assert False, f"Unsupport {bitwidth=}"


def get_bitwidth_from_dtype(dtype):
    if dtype == np.int8:
        return 8
    elif dtype in [np.int16, np.float16]:
        return 16
    elif dtype in [np.int32, np.float32]:
        return 32
    else:
        assert False, f"Unsupport {dtype=}, {type(dtype)=}"
