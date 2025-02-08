import numpy as np

from cim_compiler.data_processor.dense import convert_value_sparse_conv2d_weight


def test_case1():
    weight = np.arange(4 * 24).astype(np.int8).reshape(4, -1)
    macro_config = {
        "n_row": 4,
        "n_vcol": 2,
        "n_group": 1,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    golden = {
        "mapping_reduce_to_macro": [2],
        "mapping_macro_to_from": [2, 1],
        "mapping_from_to_row": [2, 2, 2],
        "mapping_macro_to_row": [4, 2],
    }
    golden = {key: np.array(value).astype(np.int32) for key, value in golden.items()}

    result = convert_value_sparse_conv2d_weight(weight, macro_config)
    for key in golden.keys():
        assert np.array_equal(
            result[key], golden[key]
        ), f"{key=}, {result[key]=}, {golden[key]=}"


def test_case2():
    mask = np.array(
        [
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        ]
    ).astype(np.int8)
    weight = np.arange(4 * 16).astype(np.int8).reshape(4, -1)
    weight = weight * mask
    macro_config = {
        "n_row": 4,
        "n_vcol": 2,
        "n_group": 1,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    golden = {
        "mapping_reduce_to_macro": [1],
        "mapping_macro_to_from": [2],
        "mapping_from_to_row": [1, 1],
        "mapping_macro_to_row": [2],
    }
    golden = {key: np.array(value).astype(np.int32) for key, value in golden.items()}

    result = convert_value_sparse_conv2d_weight(weight, macro_config)
    for key in golden.keys():
        assert np.array_equal(
            result[key], golden[key]
        ), f"{key=}, {result[key]=}, {golden[key]=}"


def test_case3():
    mask = np.array(
        [
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        ]
    ).astype(np.int8)
    weight = np.arange(4 * 16).astype(np.int8).reshape(4, -1)
    weight = weight * mask
    macro_config = {
        "n_row": 4,
        "n_vcol": 2,
        "n_group": 1,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    golden = {
        "mapping_reduce_to_macro": [1],
        "mapping_macro_to_from": [2],
        "mapping_from_to_row": [1, 2],
        "mapping_macro_to_row": [3],
    }
    golden = {key: np.array(value).astype(np.int32) for key, value in golden.items()}

    result = convert_value_sparse_conv2d_weight(weight, macro_config)
    assert len(golden.keys()) > 0
    for key in golden.keys():
        assert np.array_equal(
            result[key], golden[key]
        ), f"{key=}, {result[key]=}, {golden[key]=}"


def test_case4():
    mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        ]
    ).astype(np.int8)
    weight = np.arange(4 * 16).astype(np.int8).reshape(4, -1)
    weight = weight * mask
    macro_config = {
        "n_row": 4,
        "n_vcol": 2,
        "n_group": 1,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    golden = {
        "mapping_reduce_to_macro": [1],
        "mapping_macro_to_from": [2],
        "mapping_from_to_row": [0, 1],
        "mapping_macro_to_row": [1],
    }
    golden = {key: np.array(value).astype(np.int32) for key, value in golden.items()}

    result = convert_value_sparse_conv2d_weight(weight, macro_config)
    for key in golden.keys():
        print(f"{key}: {result[key]}")
    assert len(golden.keys()) > 0
    for key in golden.keys():
        assert np.array_equal(
            result[key], golden[key]
        ), f"{key=}, {result[key]=}, {golden[key]=}"


def test_case5():
    mask = np.array(
        [
            [
                1,
                0,
                1,
                0,
                0,
                0,
                0,
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
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                0,
                1,
                0,
                0,
                0,
                0,
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
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                0,
                1,
                0,
                0,
                0,
                0,
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
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                0,
                1,
                0,
                0,
                0,
                0,
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
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
        ]
    ).astype(np.int8)
    weight = np.arange(4 * 32).astype(np.int8).reshape(4, -1)
    weight = weight * mask
    macro_config = {
        "n_row": 4,
        "n_vcol": 2,
        "n_group": 1,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    golden = {
        "mapping_reduce_to_macro": [1],
        "mapping_macro_to_from": [4],
        "mapping_from_to_row": [1, 1, 1, 1],
        "mapping_macro_to_row": [4],
    }
    golden = {key: np.array(value).astype(np.int32) for key, value in golden.items()}

    result = convert_value_sparse_conv2d_weight(weight, macro_config)
    assert len(golden.keys()) > 0
    for key in golden.keys():
        assert np.array_equal(
            result[key], golden[key]
        ), f"{key=}, {result[key]=}, {golden[key]=}"


def test_case6():
    mask = np.array(
        [
            [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
        ]
    ).astype(np.int8)
    weight = np.arange(4 * 32).astype(np.int8).reshape(4, -1)
    weight = weight * mask
    macro_config = {
        "n_row": 4,
        "n_vcol": 2,
        "n_group": 1,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    golden = {
        "mapping_reduce_to_macro": [2],
        "mapping_macro_to_from": [3, 1],
        "mapping_from_to_row": [2, 1, 1, 1],
        "mapping_macro_to_row": [4, 1],
    }
    golden = {key: np.array(value).astype(np.int32) for key, value in golden.items()}

    result = convert_value_sparse_conv2d_weight(weight, macro_config)
    for key in golden.keys():
        print(f"{key}: {result[key]}")
    assert len(golden.keys()) > 0
    for key in golden.keys():
        assert np.array_equal(
            result[key], golden[key]
        ), f"{key=}, {result[key]=}, {golden[key]=}"


def test_case7():
    mask = np.array(
        [
            [
                1,
                0,
                1,
                0,
                1,
                0,
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
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                0,
                1,
                0,
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
                1,
                0,
                0,
                1,
                0,
                1,
                0,
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
            ],
            [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
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
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
        ]
    ).astype(np.int8)
    weight = np.arange(4 * 32).astype(np.int8).reshape(4, -1)
    weight = weight * mask
    macro_config = {
        "n_row": 4,
        "n_vcol": 2,
        "n_group": 1,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    golden = {
        "mapping_reduce_to_macro": [1],
        "mapping_macro_to_from": [4],
        "mapping_from_to_row": [1, 1, 1, 1],
        "mapping_macro_to_row": [4],
    }
    golden = {key: np.array(value).astype(np.int32) for key, value in golden.items()}

    result = convert_value_sparse_conv2d_weight(weight, macro_config)
    for key in golden.keys():
        print(f"{key}: {result[key]}")
    assert len(golden.keys()) > 0
    for key in golden.keys():
        assert np.array_equal(
            result[key], golden[key]
        ), f"{key=}, {result[key]=}, {golden[key]=}"


def test_case8():
    mask = np.array(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
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
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
        ]
    ).astype(np.int8)
    weight = np.arange(4 * 32).astype(np.int8).reshape(4, -1)
    weight = weight * mask
    macro_config = {
        "n_row": 4,
        "n_vcol": 2,
        "n_group": 1,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    golden = {
        "mapping_reduce_to_macro": [2],
        "mapping_macro_to_from": [3, 1],
        "mapping_from_to_row": [2, 1, 1, 2],
        "mapping_macro_to_row": [4, 2],
    }
    golden = {key: np.array(value).astype(np.int32) for key, value in golden.items()}

    result = convert_value_sparse_conv2d_weight(weight, macro_config)
    for key in golden.keys():
        print(f"{key}: {result[key]}")
    assert len(golden.keys()) > 0
    for key in golden.keys():
        assert np.array_equal(
            result[key], golden[key]
        ), f"{key=}, {result[key]=}, {golden[key]=}"


def test_case8():
    mask = np.array(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
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
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
        ]
    ).astype(np.int8)
    weight = np.arange(4 * 32).astype(np.int8).reshape(4, -1)
    weight = weight * mask
    macro_config = {
        "n_row": 4,
        "n_vcol": 2,
        "n_group": 2,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    golden = {
        "mapping_reduce_to_macro": [2, 2],
        "mapping_macro_to_from": [3, 1, 3, 1],
        "mapping_from_to_row": [2, 1, 1, 2, 2, 1, 1, 1],
        "mapping_macro_to_row": [4, 2, 4, 1],
    }
    golden = {key: np.array(value).astype(np.int32) for key, value in golden.items()}

    result = convert_value_sparse_conv2d_weight(weight, macro_config)
    for key in golden.keys():
        print(f"{key}: {result[key]}")
    assert len(golden.keys()) > 0
    for key in golden.keys():
        assert np.array_equal(
            result[key], golden[key]
        ), f"{key=}, {result[key]=}, {golden[key]=}"


def test_case9():
    weight = np.ones((4, 512), dtype=np.int8)
    weight[0, :16] = 0
    weight[1, 1:17] = 0
    weight[2, 2:18] = 0
    weight[3, 3:19] = 0

    macro_config = {
        "n_row": 4,
        "n_vcol": 2,
        "n_group": 1,
        "n_macro": 2,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4,
    }
    golden = {
        "mapping_reduce_to_macro": [2, 2],
        "mapping_macro_to_from": [3, 1, 3, 1],
        "mapping_from_to_row": [2, 1, 1, 2, 2, 1, 1, 1],
        "mapping_macro_to_row": [4, 2, 4, 1],
    }
    golden = {key: np.array(value).astype(np.int32) for key, value in golden.items()}

    result = convert_value_sparse_conv2d_weight(weight, macro_config)
    for key in golden.keys():
        print(f"{key}: {result[key]}")
    # assert len(golden.keys()) > 0
    # for key in golden.keys():
    #     assert np.array_equal(result[key], golden[key]), f"{key=}, {result[key]=}, {golden[key]=}"


if __name__ == "__main__":
    # test_case1()
    # test_case2()
    # test_case3()
    # test_case4()
    # test_case5()
    # test_case6()
    # test_case7()
    # test_case8()
    test_case9()
