import pytest

from utils.bit_sparse_weight_transform import (
    generate_valid_weight,
    int_to_csd_nonzero_count,
    int_to_csd_nonzero_count_tensor,
    weight_transform,
)


@pytest.mark.parametrize(
    "shape", [[8, 4, 3, 3], [1, 3, 3, 3], [8, 1, 3, 3], [8, 4, 1, 1]]
)
@pytest.mark.parametrize("threshold", [1, 2])
def test_generate_valid_weight_fix_threshold(shape, threshold):
    weight = generate_valid_weight(shape, threshold)
    nonzero_count = int_to_csd_nonzero_count_tensor(weight)
    assert (nonzero_count == threshold).all()


@pytest.mark.parametrize(
    "shape", [[8, 4, 3, 3], [1, 3, 3, 3], [8, 1, 3, 3], [8, 4, 1, 1]]
)
def test_generate_valid_weight_random_threshold(shape):
    weight = generate_valid_weight(shape)
    nonzero_count = int_to_csd_nonzero_count_tensor(weight)
    leading_nonzero_count = nonzero_count[:, :1, :1, :1]
    assert (nonzero_count == leading_nonzero_count).all()


if __name__ == "__main__":
    test_generate_valid_weight_fix_threshold([8, 4, 3, 3], 1)
