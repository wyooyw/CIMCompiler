import numpy as np

from data_processor.dense import convert_value_sparse_conv2d_weight
from utils.bit_sparse_weight_transform import *
from utils.bit_value_sparse_weight_transform import (
    convert_value_bit_sparse_conv2d_weight,
)


def test_case1():
    weight = generate_valid_weight([4, 2, 2, 6], 2).astype(np.int8)
    # weight = np.arange(4*24).astype(np.int8).reshape(4,-1)
    macro_config = {
        "n_row": 4,
        "n_bcol": 16,
        # "n_vcol": 2,
        "n_group": 1,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    value_sparse_golden = {
        "mapping_reduce_to_macro": [2],
        "mapping_macro_to_from": [2, 1],
        "mapping_from_to_row": [2, 2, 2],
        "mapping_macro_to_row": [4, 2],
    }
    value_sparse_golden = {
        key: np.array(value).astype(np.int32)
        for key, value in value_sparse_golden.items()
    }

    bit_sparse_golden = {
        "outsum_mask": [
            [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ],
        "transfer_mask": [
            [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ],
    }
    bit_sparse_golden = {
        key: np.array(value).astype(np.int8) for key, value in bit_sparse_golden.items()
    }
    result = convert_value_bit_sparse_conv2d_weight(weight, macro_config)

    # check value sparse result
    value_sparse_result = result["value_sparse_result"]
    for key in value_sparse_golden.keys():
        assert np.array_equal(
            value_sparse_result[key], value_sparse_golden[key]
        ), f"{key=}, {value_sparse_result[key]=}, {value_sparse_golden[key]=}"

    # check bit sparse result
    bit_sparse_result = result["bit_sparse_result"]
    assert bit_sparse_result["outsum_mask"].shape[0] == 1
    assert bit_sparse_result["transfer_mask"].shape[0] == 1

    assert np.array_equal(
        bit_sparse_result["outsum_mask"], bit_sparse_golden["outsum_mask"]
    ), f"{key=}, {bit_sparse_golden['outsum_mask']=}, {bit_sparse_golden['outsum_mask']=}"
    assert np.array_equal(
        bit_sparse_result["transfer_mask"], bit_sparse_golden["transfer_mask"]
    ), f"{key=}, {bit_sparse_golden['transfer_mask']=}, {bit_sparse_golden['transfer_mask']=}"


if __name__ == "__main__":
    test_case1()
