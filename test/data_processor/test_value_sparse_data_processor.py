import os

import numpy as np
import pytest

from data_processor.dense import (
    convert_value_sparse_conv2d_weight,
    extrace_mask_and_data,
)


def test_extrace_mask_and_data():
    weight = np.array(
        [
            [[0, 0], [41, 51]],
            [[0, 21], [41, 51]],
            [[0, 0], [0, 0]],
            [[11, 21], [41, 51]],
            [[11, 21], [41, 51]],
            [[11, 21], [0, 0]],
            [[0, 0], [41, 51]],
            [[11, 21], [41, 51]],
        ]
    )
    # n_from, n_macro, n_vcol -> n_vcol, n_from, n_macro
    weight = np.transpose(weight, (2, 0, 1))
    print(weight.shape)
    mask, data = extrace_mask_and_data(weight, 8, 4, False, False)
    print(mask)
    print(data)

    # compare with goldens
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    golden_mask = np.load(
        os.path.join(curr_dir, "golden/test_extrace_mask_and_data/mask.npy")
    )
    golden_data = np.load(
        os.path.join(curr_dir, "golden/test_extrace_mask_and_data/data.npy")
    )
    assert np.array_equal(mask, golden_mask), f"{mask=}, {golden_mask=}"
    assert np.array_equal(data, golden_data), f"{data=}, {golden_data=}"


def test_convert_value_sparse_conv2d_weight():
    # 2 * 2 * 8 * n
    # out_channel = 4
    # reduce_size = 8
    weight = np.array(
        [
            [1, 2, 3, 4, 0, 0, 0, 0],
            [11, 12, 13, 14, 0, 0, 0, 0],
            [0, 0, 23, 24, 25, 26, 0, 0],
            [0, 0, 33, 34, 35, 36, 0, 0],
        ]
    )

    macro_config = {
        "n_vcol": 2,
        "n_group": 2,
        "n_macro": 4,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }

    converted_weight, mask, index, _ = convert_value_sparse_conv2d_weight(
        weight, macro_config
    )
    print(
        f"{converted_weight.shape=} ([time, n_to, n_group, n_macro_per_group, n_vcol])"
    )
    print(converted_weight)
    time, n_to, n_group, n_macro_per_group, n_vcol = converted_weight.shape
    converted_weight = converted_weight.reshape(time, n_to, n_group, -1)
    print(
        f"{converted_weight.shape=} ([time, n_to, n_group, n_macro_per_group * n_vcol])"
    )
    print(converted_weight)
    print(f"{mask.shape=}")
    print(mask)
    print(f"{index.shape=}")
    print(index)

    # compare with goldens
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    golden_weight = np.load(
        os.path.join(
            curr_dir,
            "golden/test_convert_value_sparse_conv2d_weight/converted_weight.npy",
        )
    )
    golden_index = np.load(
        os.path.join(
            curr_dir, "golden/test_convert_value_sparse_conv2d_weight/index.npy"
        )
    )
    golden_mask = np.load(
        os.path.join(
            curr_dir, "golden/test_convert_value_sparse_conv2d_weight/mask.npy"
        )
    )
    assert np.array_equal(
        converted_weight, golden_weight
    ), f"{converted_weight=}, {golden_weight=}"
    assert np.array_equal(index, golden_index), f"{index=}, {golden_index=}"
    assert np.array_equal(mask, golden_mask), f"{mask=}, {golden_mask=}"


def test_convert_value_sparse_conv2d_weight2():
    # 2 * 2 * 8 * n
    # out_channel = 4
    # reduce_size = 8
    weight = np.array(
        [
            [1, 2, 3, 4, 5, 0, 0, 0],
            [11, 12, 13, 14, 15, 0, 0, 0],
            [0, 0, 23, 0, 25, 26, 0, 28],
            [0, 0, 33, 0, 35, 36, 0, 38],
        ]
    )

    macro_config = {
        "n_vcol": 2,
        "n_group": 2,
        "n_macro": 4,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }

    converted_weight, mask, index, _ = convert_value_sparse_conv2d_weight(
        weight, macro_config
    )
    print(
        f"{converted_weight.shape=} ([time, n_to, n_group, n_macro_per_group, n_vcol])"
    )
    print(converted_weight)
    time, n_to, n_group, n_macro_per_group, n_vcol = converted_weight.shape
    converted_weight = converted_weight.reshape(time, n_to, n_group, -1)
    print(
        f"{converted_weight.shape=} ([time, n_to, n_group, n_macro_per_group * n_vcol])"
    )
    print(converted_weight)
    print(f"{mask.shape=}")
    print(mask)
    print(f"{index.shape=}")
    print(index)

    # compare with goldens
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    golden_weight = np.load(
        os.path.join(
            curr_dir,
            "golden/test_convert_value_sparse_conv2d_weight2/converted_weight.npy",
        )
    )
    golden_index = np.load(
        os.path.join(
            curr_dir, "golden/test_convert_value_sparse_conv2d_weight2/index.npy"
        )
    )
    golden_mask = np.load(
        os.path.join(
            curr_dir, "golden/test_convert_value_sparse_conv2d_weight2/mask.npy"
        )
    )
    assert np.array_equal(
        converted_weight, golden_weight
    ), f"{converted_weight=}, {golden_weight=}"
    assert np.array_equal(index, golden_index), f"{index=}, {golden_index=}"
    assert np.array_equal(mask, golden_mask), f"{mask=}, {golden_mask=}"
