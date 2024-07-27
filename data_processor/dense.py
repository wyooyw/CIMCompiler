import numpy as np
import math
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

def extract_non_zero_mask_2d(weight3d):
    """
    weight3d: batch * n_from * n_macro_per_group

    return : 
    non_zero_mask: n_from * n_macro_per_group
    """
    assert len(weight3d.shape)==3
    non_zero_mask_3d = weight3d != 0
    non_zero_mask_2d = non_zero_mask_3d.sum(axis=0) != 0
    return non_zero_mask_2d

def extrace_mask_and_data(weight3d, n_from, n_to, concat=True, bit_to_byte=True):
    """
    weight2d: n_vcol * n_from * n_macro_per_group
    all 

    return :
    weight2d: t * n_to * n_macro_per_group
    mask: t * n_to * n_macro_per_group
    """
    if bit_to_byte:
        assert concat

    mask = []
    data = []

    n_vcol, n_from, n_macro_per_group = weight3d.shape

    non_zero_mask = extract_non_zero_mask_2d(weight3d)
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
        sorted_position_3d = np.repeat(np.expand_dims(sorted_position, axis=0), n_vcol, axis=0)
        extended_weight_3d = np.pad(weight3d, ((0,0),(0,1),(0,0)), mode='constant', constant_values=0)
        extracted_data_3d = np.take_along_axis(extended_weight_3d, sorted_position_3d, axis=1)
        assert extracted_data_3d.dtype==np.int8
        data.append(extracted_data_3d)

    if concat:
        if len(mask)==0:
            mask = np.zeros((0, n_from, n_macro_per_group), dtype=np.int8)
            data = np.zeros((0, n_vcol, n_to, n_macro_per_group), dtype=np.int8)
            return mask, data
        mask = np.concatenate([np.expand_dims(m,0) for m in mask])
        data = np.concatenate([np.expand_dims(d,0) for d in data])

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
    if len(weight.shape)==4:
        oc, ic, kh, kw = weight.shape
        spatial_size = oc
        reduce_size = ic * kh * kw
        weight = weight.reshape(oc, reduce_size)
    elif len(weight.shape)==2:
        spatial_size, reduce_size = weight.shape
    else:
        assert False
    assert weight.dtype==np.int8

    n_vcol = macro_config["n_vcol"]
    n_group = macro_config["n_group"]
    n_macro_per_group = macro_config["n_macro"] // n_group
    n_group_vcol = n_macro_per_group * n_vcol
    n_comp = macro_config["n_comp"] # * macro_config["n_row"]
    n_from = macro_config["n_value_sparse_from"] # 128
    n_to = macro_config["n_value_sparse_to"] # 16
    assert n_from > n_to and n_to==n_comp

    # padding weights
    spatial_pad_size = int(math.ceil(spatial_size / n_group_vcol)) * n_group_vcol - spatial_size
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
    assert weight.shape[-2]==n_from and weight.shape[-1]==n_macro_per_group
    weight = weight.reshape(out_spatial_tile, out_reduce_tile, n_vcol, n_from, n_macro_per_group)
    
    # extract n_to non-zero element from n_from
    index_list = []
    mask_list = []
    weight_list = []
    out_spatial_tile_size_list = []
    for ost in range(out_spatial_tile):
        out_spatial_tile_size = 0
        for ort in range(out_reduce_tile):
            subweight = weight[ost, ort, :, :, :] # n_vcol, n_from, n_macro_per_group
            submask, subweight = extrace_mask_and_data(subweight,  n_from, n_to, concat=True, bit_to_byte=True) 
            # subweight: t, n_vcol, n_to, n_macro_per_group
            # submask: t, n_to, n_macro_per_group
            assert len(subweight.shape)==4 and subweight.shape[1]==n_vcol and subweight.shape[2]==n_to and subweight.shape[3]==n_macro_per_group
            assert len(submask.shape)==3 and submask.shape[1]==n_from and submask.shape[2]==n_macro_per_group, f"{submask.shape=}"
            assert subweight.shape[0]==submask.shape[0]
            
            index_list.append(submask.shape[0])
            mask_list.append(submask)
            weight_list.append(subweight)
            out_spatial_tile_size += submask.shape[0]
        out_spatial_tile_size_list.append(out_spatial_tile_size)

    converted_weight = np.concatenate(weight_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)

    # filter zero in index_list
    index_list = [i for i in index_list if i>0]
    index = np.array(index_list, dtype=np.int32)
    assert mask.shape[0]==converted_weight.shape[0] and converted_weight.shape[0]==index.sum()
    assert len(subweight.shape)==4
    
    # duplicate weight for groups
    converted_weight = np.repeat(converted_weight.reshape(-1, n_vcol, n_to, 1, n_macro_per_group), n_group, axis=3)
    assert converted_weight.shape[3]==n_group
    converted_weight = np.transpose(converted_weight, (0,2,3,4,1))
    # [time, n_to, n_group, n_macro_per_group, n_vcol]
    
    out_spatial_tile_size_list = np.array(out_spatial_tile_size_list, dtype=np.int32)

    # mask.shape : [t, n_from, n_macro_per_group] -> [t, n_macro_per_group, n_from]
    mask = np.transpose(mask, [0,2,1])
    return converted_weight, mask, index, out_spatial_tile_size_list

def test_extrace_mask_and_data():
    weight = np.array([
        [[0,0],[41,51]],
        [[0,21],[41,51]],
        [[0,0],[0,0]],
        [[11,21],[41,51]],
        [[11,21],[41,51]],
        [[11,21],[0,0]],
        [[0,0],[41,51]],
        [[11,21],[41,51]],
    ])
    # n_from, n_macro, n_vcol -> n_vcol, n_from, n_macro
    weight = np.transpose(weight, (2, 0, 1))
    print(weight.shape)
    mask,data = extrace_mask_and_data(weight, 8, 4, False, False)
    print(mask)
    print(data)
    np.save("test/data_processor/golden/test_extrace_mask_and_data/mask.npy", mask)
    np.save("test/data_processor/golden/test_extrace_mask_and_data/data.npy", data)


def test_convert_value_sparse_conv2d_weight():
    # 2 * 2 * 8 * n
    # out_channel = 4
    # reduce_size = 8
    weight = np.array([
        [1,2,3,4,0,0,0,0],
        [11,12,13,14,0,0,0,0],
        [0,0,23,24,25,26,0,0],
        [0,0,33,34,35,36,0,0],
    ])

    macro_config = {
        "n_vcol": 2,
        "n_group": 2,
        "n_macro": 4,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4
    }

    converted_weight, mask, index = convert_value_sparse_conv2d_weight(weight, macro_config)
    print(f"{converted_weight.shape=} ([time, n_to, n_group, n_macro_per_group, n_vcol])")
    print(converted_weight)
    time, n_to, n_group, n_macro_per_group, n_vcol = converted_weight.shape
    converted_weight = converted_weight.reshape(time, n_to, n_group, -1)
    print(f"{converted_weight.shape=} ([time, n_to, n_group, n_macro_per_group * n_vcol])")
    print(converted_weight)
    print(f"{mask.shape=}")
    print(mask)
    print(f"{index.shape=}")
    print(index)
    np.save("test/data_processor/golden/test_convert_value_sparse_conv2d_weight/converted_weight.npy", converted_weight)
    np.save("test/data_processor/golden/test_convert_value_sparse_conv2d_weight/mask.npy", mask)
    np.save("test/data_processor/golden/test_convert_value_sparse_conv2d_weight/index.npy", index)


def test_convert_value_sparse_conv2d_weight2():
    # 2 * 2 * 8 * n
    # out_channel = 4
    # reduce_size = 8
    weight = np.array([
        [1,2,3,4,5,0,0,0],
        [11,12,13,14,15,0,0,0],
        [0,0,23,0,25,26,0,28],
        [0,0,33,0,35,36,0,38],
    ])

    macro_config = {
        "n_vcol": 2,
        "n_group": 2,
        "n_macro": 4,
        "n_comp": 4,
        "n_value_sparse_from": 8,
        "n_value_sparse_to": 4
    }

    converted_weight, mask, index = convert_value_sparse_conv2d_weight(weight, macro_config)
    print(f"{converted_weight.shape=} ([time, n_to, n_group, n_macro_per_group, n_vcol])")
    print(converted_weight)
    time, n_to, n_group, n_macro_per_group, n_vcol = converted_weight.shape
    converted_weight = converted_weight.reshape(time, n_to, n_group, -1)
    print(f"{converted_weight.shape=} ([time, n_to, n_group, n_macro_per_group * n_vcol])")
    print(converted_weight)
    print(f"{mask.shape=}")
    print(mask)
    print(f"{index.shape=}")
    print(index)
    np.save("test/data_processor/golden/test_convert_value_sparse_conv2d_weight2/converted_weight.npy", converted_weight)
    np.save("test/data_processor/golden/test_convert_value_sparse_conv2d_weight2/mask.npy", mask)
    np.save("test/data_processor/golden/test_convert_value_sparse_conv2d_weight2/index.npy", index)

if __name__=="__main__":
    test_extrace_mask_and_data()