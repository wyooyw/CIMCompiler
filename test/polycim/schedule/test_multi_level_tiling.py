import itertools

import islpy as isl
import pytest

from cim_compiler.polycim.op.base_operator import BasicOperator
from cim_compiler.polycim.passes.multi_level_tiling_pass import (
    combine_tilesize_by_symmetry_info,
    multi_level_splitting_var_level,
    multi_level_tiling,
)
from cim_compiler.polycim.utils.utils import get_box_hull_shape


def get_op_simple_conv2d(oh, ow, kh, kw):
    op = BasicOperator(
        domain=isl.BasicSet(
            f"{{ [oh,ow,kh,kw]: 0<=oh<{oh} and 0<=ow<{ow} and 0<=kh<{kh} and 0<=kw<{kw} }}"
        ),
        access_I=isl.BasicMap(f"{{ [oh,ow,kh,kw] -> I[oh + kh, ow + kw] }}"),
        access_O=isl.BasicMap("{ [oh,ow,kh,kw] -> O[oh, ow] }"),
        access_W=isl.BasicMap("{ [oh,ow,kh,kw] -> W[kh, kw] }"),
    )
    return op


def test_multi_level_tiling_basic():
    op = get_op_simple_conv2d(8, 8, 3, 3)
    tiling_factors = [
        [2, 4],
        [2, 4],
        [1, 3],
        [1, 3],
    ]
    new_op = multi_level_tiling(op, 2, tiling_factors)
    domain_shape = get_box_hull_shape(new_op.domain)
    golden_domain_shape = [2, 2, 1, 1, 4, 4, 3, 3]
    assert domain_shape == golden_domain_shape


@pytest.mark.parametrize(
    "tiling_factors, golden_domain_shape",
    [
        ([[2, 4], [2, 2, 2], [3], [3]], [2, 4, 2, 2, 2, 3, 3]),
        ([[2, 4], [2, 2, 2], [1, 3], [3, 1]], [2, 4, 2, 2, 2, 1, 3, 3, 1]),
        ([[8], [8], [3], [3]], [8, 8, 3, 3]),
    ],
)
def test_multi_level_tiling_var_level(tiling_factors, golden_domain_shape):
    op = get_op_simple_conv2d(8, 8, 3, 3)
    new_op = multi_level_splitting_var_level(op, tiling_factors)
    domain_shape = get_box_hull_shape(new_op.domain)
    assert domain_shape == golden_domain_shape


@pytest.mark.parametrize(
    "dim_factors, symmetry_info, golden_tile_sizes",
    [
        (  # test case 1
            (  # dim_factors
                ((3, 2), (2, 3), (6,)),  # dim 0
                ((3, 2), (2, 3), (6,)),  # dim 1
            ),
            # symmetry_info
            (
                (0,),
                (1,),
            ),
            # golden
            6,
        ),
        (  # test case 2
            (  # dim_factors
                ((3, 2), (2, 3), (6,)),  # dim 0
                ((3, 2), (2, 3), (6,)),  # dim 1
                ((3, 2), (2, 3), (6,)),  # dim 2
            ),
            # symmetry_info
            (
                (0,),
                (1,),
                (2,),
            ),
            # golden
            10,
        ),
        (  # test case 3
            (  # dim_factors
                ((3, 2), (2, 3), (6,)),  # dim 0
                ((3, 2), (2, 3), (6,)),  # dim 1
                ((2, 2), (4,)),  # dim 2
                ((2, 2), (4,)),  # dim 3
            ),
            # symmetry_info
            (
                (0, 2),
                (1, 3),
            ),
            # golden
            21,
        ),
        (  # test case 4
            (  # dim_factors
                ((3, 2), (2, 3), (6,)),  # dim 0
                ((3, 2), (2, 3), (6,)),  # dim 1
                ((3, 2), (2, 3), (6,)),  # dim 2
                ((2, 2), (4,)),  # dim 3
                ((2, 2), (4,)),  # dim 4
                ((2, 2), (4,)),  # dim 5
            ),
            # symmetry_info
            (
                (0, 3),
                (1, 4),
                (2, 5),
            ),
            # golden
            56,
        ),
    ],
)
def test_combine_tilesize_by_symmetry_info(
    dim_factors, symmetry_info, golden_tile_sizes
):
    tile_sizes = combine_tilesize_by_symmetry_info(dim_factors, symmetry_info)

    all_combinations = list(itertools.product(*dim_factors))
    assert set(tile_sizes).issubset(set(all_combinations))

    if golden_tile_sizes is None:
        pass
    elif type(golden_tile_sizes) == int:
        assert len(tile_sizes) == golden_tile_sizes
    elif type(golden_tile_sizes) == tuple:
        assert tile_sizes == golden_tile_sizes
    else:
        raise ValueError(f"Invalid golden_tile_sizes type: {type(golden_tile_sizes)}")


if __name__ == "__main__":
    test_multi_level_tiling_var_level()
