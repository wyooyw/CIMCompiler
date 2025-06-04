from collections import OrderedDict
from functools import partial

import polycim.op.benchmark as benchmark
from polycim.op.calculate import conv2d, depth_wise_conv2d, depth_wise_conv3d


def get_op_list():

    symmetry_info_for_dwconv2d = ((1, 3), (2, 4))
    dim_types_for_dwconv2d = ["c", "oh", "ow", "kh", "kw"]

    op_list = OrderedDict()
    op_list["conv2d_b1o8i1h8w8k3"] = {
        "op": benchmark.get_op_conv2d(
            b=1, oc=8, ic=1, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=False
        ),
        "symmetry_info": ((3, 5), (4, 6)),
        "dim_types": ["b", "oc", "ic", "oh", "ow", "kh", "kw"],
        "verify_fn": conv2d,
    }
    op_list["conv2d_b1o8i8h8w8k3"] = {
        "op": benchmark.get_op_conv2d(
            b=1, oc=8, ic=8, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=False
        ),
        "symmetry_info": ((3, 5), (4, 6)),
        "dim_types": ["b", "oc", "ic", "oh", "ow", "kh", "kw"],
        "verify_fn": conv2d,
    }
    op_list["conv2d_b1o16i8h8w8k3"] = {
        "op": benchmark.get_op_conv2d(
            b=1, oc=16, ic=8, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=False
        ),
        "symmetry_info": ((3, 5), (4, 6)),
        "dim_types": ["b", "oc", "ic", "oh", "ow", "kh", "kw"],
        "verify_fn": conv2d,
    }
    op_list["conv2d_b2o16i8h8w8k3"] = {
        "op": benchmark.get_op_conv2d(
            b=2, oc=16, ic=8, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=False
        ),
        "symmetry_info": ((3, 5), (4, 6)),
        "dim_types": ["b", "oc", "ic", "oh", "ow", "kh", "kw"],
        "verify_fn": conv2d,
    }
    op_list["conv2d_b1o326i256h8w8k3s2"] = {
        "op": benchmark.get_op_conv2d(
            b=1, oc=32, ic=256, oh=4, ow=4, kh=1, kw=1, stride=2, virtual_axis=False
        ),
        "symmetry_info": ((3, 5), (4, 6)),
        "dim_types": ["b", "oc", "ic", "oh", "ow", "kh", "kw"],
        "verify_fn": partial(conv2d, stride=2),
    }
    op_list["test"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=8, ow=8, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["d2h4"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=4, ow=4, kh=3, kw=3, stride=1, dilation=2, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "max_tiling_level": 3,
        "verify_fn": partial(depth_wise_conv2d, dilation=2),
    }
    op_list["test3d"] = {
        "op": benchmark.get_op_dwconv3d(
            ic=4, ox=4, oy=4, oz=4, kx=3, ky=3, kz=3, stride=1
        ),
        "symmetry_info": ((1, 4), (2, 5), (3, 6)),
        "dim_types": ["c", "ox", "oy", "oz", "kx", "ky", "kz"],
        "max_tiling_level": 2,
        "verify_fn": depth_wise_conv3d,
    }
    op_list["C1"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=112, ow=112, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C2"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=56, ow=56, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C3"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C4"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=14, ow=14, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C5"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=14, ow=14, kh=5, kw=5, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C6"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=7, ow=7, kh=5, kw=5, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C7"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=7, ow=7, kh=3, kw=3, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C8"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=56, ow=56, kh=7, kw=7, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C9"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=7, ow=7, kh=7, kw=7, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C10"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=56, ow=56, kh=51, kw=51, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "not_tiling": [3, 4],
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C11"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=7, ow=7, kh=13, kw=13, stride=1, dilation=1, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "verify_fn": depth_wise_conv2d,
    }
    op_list["C12"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=28, ow=28, kh=3, kw=3, stride=1, dilation=2, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "max_tiling_level": 3,
        "verify_fn": partial(depth_wise_conv2d, dilation=2),
    }
    op_list["C13"] = {
        "op": benchmark.get_op_dwconv2d(
            ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=2, virtual_axis=False
        ),
        "symmetry_info": symmetry_info_for_dwconv2d,
        "dim_types": dim_types_for_dwconv2d,
        "max_tiling_level": 3,
        "verify_fn": partial(depth_wise_conv2d, dilation=2),
    }
    # # op_list["C14"] = benchmark.get_op_dwconv2d(b=1, oc=1, ic=1, oh=28, ow=28, kh=5, kw=5, stride=1, dilation=2)
    op_list["C15"] = {
        "op": benchmark.get_op_dwconv3d(
            ic=1, ox=28, oy=28, oz=28, kx=5, ky=5, kz=5, stride=1
        ),
        "symmetry_info": ((1, 4), (2, 5), (3, 6)),
        "dim_types": ["c", "ox", "oy", "oz", "kx", "ky", "kz"],
        "max_tiling_level": 2,
        "verify_fn": depth_wise_conv3d,
    }
    return op_list
