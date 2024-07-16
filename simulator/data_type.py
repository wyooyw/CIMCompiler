import numpy as np
def get_dtype_from_bitwidth(bitwidth):
    if bitwidth==8:
        return np.int8
    elif bitwidth==16:
        return np.int16
    elif bitwidth==32:
        return np.int32
    else:
        assert False, f"Unsupport {bitwidth=}"

def get_bitwidth_from_dtype(dtype):
    if dtype==np.int8:
        return 8
    elif dtype==np.int16:
        return 16
    elif dtype==np.int32:
        return 32
    else:
        assert False, f"Unsupport {dtype=}"