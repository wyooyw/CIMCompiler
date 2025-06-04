import islpy as isl

import polycim.utils.utils as utils
from polycim.utils.dominate import get_dominate_iters_of_pw_multi_aff
from polycim.utils.draw import (
    FrameInfo,
    _extract_frame_info,
    extract_frame_info,
    extract_time_list,
    extract_val_from_singleton_set,
    get_macro_hash,
)
from polycim.utils.utils import get_mpf_lb_up_from_domain


def count_minimal_needed_macro(op, cim_cfg):
    domain = op.domain
    n_dim = domain.dim(isl.dim_type.set)
    shape = utils.get_box_hull_shape(domain)
    intra_macro_iters = [n_dim - 2, n_dim - 1]
    inter_macro_iters = [n_dim - 4, n_dim - 3]
    cnt = shape[inter_macro_iters[0]] * shape[inter_macro_iters[1]]

    access_W = op.access_W
    dominate_weight_iters = get_dominate_iters_of_pw_multi_aff(
        access_W.as_pw_multi_aff(), return_name=False
    )
    dominate_weight_iters = (
        set(dominate_weight_iters) - set(intra_macro_iters) - set(inter_macro_iters)
    )
    for dominate_weight_iter in dominate_weight_iters:
        cnt *= shape[dominate_weight_iter]

    return cnt
