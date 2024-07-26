import numpy as np
def convert_dense_conv2d_weight(weight, macro_config):
    """
    weight: [oc,ic,kh,kw]
    converted_weight: [out_spatial_tile, out_reduce_tile, n_comp, n_group, n_group_vcol]

    if num_group > 1, weight should be replicated
    """

    oc, ic, kh, kw = weight.shape
    reduce_size = ic * kh * kw
    weight = weight.reshape(oc, reduce_size)

    n_vcol = macro_config["n_vcol"]
    n_group = macro_config["n_group"]
    n_macro_per_group = macro_config["n_macro"] // n_group
    n_group_vcol = n_macro_per_group * n_vcol
    n_comp = macro_config["n_comp"] # * macro_config["n_row"]

    # padding weights
    spatial_pad_size = int(math.ceil(oc / n_group_vcol)) * n_group_vcol - oc
    reduce_pad_size = int(math.ceil(reduce_size / n_comp)) * n_comp - reduce_size
    weight = np.pad(weight, ((0,spatial_pad_size),(0, reduce_pad_size)), mode='constant', constant_values=0)
    total_spatial_size = weight.shape[0]
    total_reduce_size = weight.shape[1]
    assert total_spatial_size % n_group_vcol == 0
    assert total_reduce_size % n_comp == 0

    # replicate weight
    weight = weight.reshape(total_spatial_size, total_reduce_size, 1)
    weight = np.repeat(weight, group_num, axis=2)
    assert weight.shape==[total_spatial_size, total_reduce_size, n_group]

    # tile the weight
    out_spatial_tile = total_spatial_size // n_group_vcol
    out_reduce_tile = total_reduce_size // n_comp
    weight = weight.reshape(out_spatial_tile, n_group_vcol, out_reduce_tile, n_comp, n_group)
    weight = np.transpose(weight, (0, 2, 3, 4, 1))
    # weight = weight.reshape(out_spatial_tile, out_reduce_tile, n_comp, n_group, n_group_vcol)

    return weight

def extrace_mask_and_data(weight2d, n_from, n_to, concat=True, bit_to_byte=True):
    """
    weight2d: batch * n_from * n_macro_per_group
    all 

    return :
    weight2d: t * n_to * n_macro_per_group
    mask: t * n_to * n_macro_per_group
    """
    if bit_to_byte:
        assert concat

    mask = []
    data = []

    n_from, n_macro_per_group = weight2d.shape

    non_zero_mask = (weight2d!=0)
    prefix_sum = np.cumsum(non_zero_mask, axis=0) - 1
    position = np.repeat(np.arange(n_from).reshape(-1,1), n_macro_per_group, axis=1)
    begin_idx = 0
    max_mask_cnt = prefix_sum.max() + 1
    for begin_idx in range(0, max_mask_cnt, n_to):
        end_idx = begin_idx + n_to
        to_mask = np.logical_and(prefix_sum >= begin_idx, prefix_sum < end_idx)
        to_mask = np.logical_and(to_mask, non_zero_mask)
        mask.append(to_mask)
        
        to_mask_position = (position + 1) * to_mask
        extended_row = n_from + 1
        to_mask_position_extend = np.where(to_mask_position>0, to_mask_position, extended_row)
        sorted_position = np.sort(to_mask_position_extend, axis=0)
        sorted_position = sorted_position - 1
        sorted_position = sorted_position[:n_to, :]
        extended_weight2d = np.pad(weight2d, ((0,1),(0,0)), mode='constant', constant_values=0)
        extracted_data = np.take_along_axis(extended_weight2d, sorted_position, axis=0)
        data.append(extracted_data)

    if concat:
        mask = np.concatenate([m.reshape(1,-1) for m in mask])
        data = np.concatenate([d.reshape(1,-1) for d in data])

        if bit_to_byte:
            # turn mask into bits, i.e. a int8 tensor mask_bits, where mask_bits.size=mask.size / 8
            pass

    return mask, data



def convert_value_sparse_conv2d_weight(weight, macro_config):
    """
    weight: [oc,ic,kh,kw]

    converted_weight: [out_spatial_tile, out_reduce_tile, n_comp, n_group, n_group_vcol] 1byte
    mask: [n_sparse_time, n_comp, n_macro] 1bit   "n_sparse_time" is the combination of "out_spatial_tile" and "out_reduce_tile", it is a sparse axis.

    return: converted_weight, mask, index
    """
    
    oc, ic, kh, kw = weight.shape
    reduce_size = ic * kh * kw
    weight = weight.reshape(oc, reduce_size)

    n_vcol = macro_config["n_vcol"]
    n_group = macro_config["n_group"]
    n_macro_per_group = macro_config["n_macro"] // n_group
    n_group_vcol = n_macro_per_group * n_vcol
    n_comp = macro_config["n_comp"] # * macro_config["n_row"]
    n_from = macro_config["n_value_sparse_from"] # 128
    n_to = macro_config["n_value_sparse_to"] # 16
    assert n_from==128 and n_to==16 and n_to==n_comp

    # padding weights
    spatial_pad_size = int(math.ceil(oc / n_group_vcol)) * n_group_vcol - oc
    reduce_pad_size = int(math.ceil(reduce_size / n_from)) * n_from - reduce_size
    weight = np.pad(weight, ((0,spatial_pad_size),(0, reduce_pad_size)), mode='constant', constant_values=0)
    total_spatial_size = weight.shape[0]
    total_reduce_size = weight.shape[1]
    assert total_spatial_size % n_group_vcol == 0
    assert total_reduce_size % n_from == 0

    # tile the weight
    out_spatial_tile = total_spatial_size // n_group_vcol
    out_reduce_tile = total_reduce_size // n_from
    weight = weight.reshape(out_spatial_tile, n_macro_per_group, n_vcol, out_reduce_tile, n_from)
    weight = np.transpose(weight, (0, 3, 2, 4, 1))
    assert weight.shape[-2]==n_macro_per_group and weight.shape[-1]==n_from
    weight = weight.reshape(-1, n_vcol, n_from, n_macro_per_group)
    
    # extract n_to non-zero element from n_from
    index_list = []
    mask_list = []
    weight_list = []
    for t in range(weight.shape[0]):
        subweight = weight[t, 0, :, :] # n_from, n_macro_per_group
        submask, subweight = extrace_mask_and_data(subweight,  n_from, n_to, concat=True, bit_to_byte=True) # t, n_to, n_macro_per_group
        index_list.append(submask.shape[0])
        mask_list.append(submask)
        weight_list.append(subweight)
    converted_weight = np.concatenate(weight_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    index = np.concatenate(index_list, axis=0)

    # duplicate weight for groups
    converted_weight = np.repeat(converted_weight.reshape(-1, n_from, 1, n_vcol_macro_per_group), n_group, axis=2)
    assert converted_weight.shape[2]==n_group

    return converted_weight, mask, index
    
    
if __name__=="__main__":
    weight = np.array([
        [21,0,41,0],
        [21,31,41,51],
        [0,31,0,51],
        [0,0,41,0],
        [0,0,0,51],
        [21,31,0,0],
        [21,31,0,129],
        [21,0,41,128],
    ])
    mask,data = extrace_mask(weight, 8, 4)
    print(mask)
    print(data)
    # for m,d in zip(mask, data):
    #     print(m)
    #     print(d)
    #     print("\n")
    