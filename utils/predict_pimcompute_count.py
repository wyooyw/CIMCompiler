from types import SimpleNamespace
import math
def predict_pimcompute_count_for_conv2d_dense(macro_config, op_config, group_size=16):
    if type(op_config) == dict:
        op_config = SimpleNamespace(**op_config)
    
    n_group = macro_config.n_macro // group_size
    sliding_window_count = op_config.out_hw * op_config.out_hw // n_group

    reduce_len = op_config.ker_size * op_config.ker_size * op_config.in_channel
    weight_reduce_count = math.ceil(reduce_len/ macro_config.n_comp)

    n_vcol_per_group = group_size * macro_config.n_vcol(8)
    weight_spatial_count = math.ceil(op_config.out_channel/ n_vcol_per_group)

    total = sliding_window_count * weight_reduce_count * weight_spatial_count
    return total