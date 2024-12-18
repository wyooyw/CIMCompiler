"""
Do bit sparse and value sparse transform simultaneously

input:
    weight
    cim_config

output:
    transformed_weight

    meta
    outsum_mask
    transfer_mask

    mask
    mapping_reduce_to_macro
    mapping_macro_to_from
    mapping_from_to_row
    mapping_macro_to_row
"""

import math

from data_processor.dense import *
from utils.bit_sparse_weight_transform import *


def convert_value_bit_sparse_conv2d_weight(weight, macro_config):
    macro_config["n_vcol"] = macro_config["n_bcol"] // 2
    value_sparse_result = convert_value_sparse_conv2d_weight(weight, macro_config)
    value_sparse_weight = value_sparse_result["converted_weight"]

    # print(f"{value_sparse_weight.shape=}")
    assert value_sparse_weight.shape[-1] == 8
    # bit_sparse_converted_weight.shape = [time, n_to, n_macro_per_group, n_vcol]

    bit_sparse_result = convert_bit_sparse_conv2d_weight(
        weight,
        value_sparse_weight,
        macro_config,
        value_sparse_result["mapping_reduce_to_macro"],
        value_sparse_result["mapping_macro_to_row"],
    )
    value_bit_sparse_weight = bit_sparse_result["converted_weight"]
    del value_sparse_result["converted_weight"]
    del bit_sparse_result["converted_weight"]

    # group duplicate
    value_bit_sparse_weight = value_bit_sparse_weight.reshape(
        *value_bit_sparse_weight.shape[0:2],
        1,
        *value_bit_sparse_weight.shape[2:],
    )
    value_bit_sparse_weight = np.repeat(
        value_bit_sparse_weight, macro_config["n_group"], axis=2
    )

    return {
        "value_bit_sparse_weight": value_bit_sparse_weight,
        "value_sparse_result": value_sparse_result,
        "bit_sparse_result": bit_sparse_result,
    }


def convert_bit_sparse_conv2d_weight(
    weight, value_sparse_weight, cim_cfg, mapping_reduce_to_macro, mapping_macro_to_row
):
    # assert weight is OHWI layout
    out_channel, ker_height, ker_width, in_channel = weight.shape
    ele_in_filter = in_channel * ker_height * ker_width
    weight = np.transpose(weight, (1, 2, 3, 0))  # OHWI -> HWIO

    n_row = cim_cfg["n_row"]
    n_vcol = cim_cfg["n_vcol"]
    n_bcol = cim_cfg["n_bcol"]
    n_group = cim_cfg["n_group"]
    n_macro_per_group = cim_cfg["n_macro"] // n_group
    n_vcol_per_group = n_macro_per_group * n_vcol
    n_comp = cim_cfg["n_comp"]  # * macro_config["n_row"]
    out_spatial_time = math.ceil(out_channel / n_vcol_per_group)
    n_filter_per_macro = n_vcol
    n_filter_per_group = n_filter_per_macro * n_macro_per_group

    # step 1: get threshold, outsum_mask, transfer_mask
    weight_bit_num = int_to_csd_nonzero_count_tensor(weight)
    total_threshold = []
    total_outsum_mask = np.zeros(
        (out_spatial_time, n_macro_per_group, n_bcol), dtype=np.int8
    )
    total_transfer_mask = np.zeros(
        (out_spatial_time, n_macro_per_group, n_bcol), dtype=np.int8
    )
    total_pimset_mask = np.zeros(
        (out_spatial_time, n_macro_per_group, n_bcol), dtype=np.int8
    )
    for outer_oc in range(0, out_channel, n_filter_per_group):
        group_threshold = []
        outsum_mask_per_group = []
        transfer_mask_per_group = []
        pimset_mask_per_group = []

        for middle_oc in range(0, n_filter_per_group, n_filter_per_macro):
            # macro_threshold = []
            outsum_mask_per_macro = []
            transfer_mask_per_macro = []
            pimset_mask_per_macro = []

            for inner_oc in range(0, n_filter_per_macro):
                oc = outer_oc + middle_oc + inner_oc
                if oc >= out_channel:
                    total_threshold.append(1)
                    outsum_mask_per_macro.append(0)
                    transfer_mask_per_macro.append(0)
                    pimset_mask_per_macro.append(0)
                    continue
                one_filter_threshold = weight_bit_num[:, :, :, oc].max()
                assert one_filter_threshold > 0
                valid = np.logical_or(
                    (weight_bit_num[:, :, :, oc] == 0),
                    (weight_bit_num[:, :, :, oc] == one_filter_threshold),
                ).all()
                assert valid, "Element in one filter should have same threshold."
                total_threshold.append(one_filter_threshold)
                assert one_filter_threshold in [1, 2]
                if one_filter_threshold == 2:
                    outsum_mask_per_macro.append(1)
                    outsum_mask_per_macro.append(0)

                    transfer_mask_per_macro.append(1)
                    transfer_mask_per_macro.append(0)

                    pimset_mask_per_macro.append(1)
                    pimset_mask_per_macro.append(1)

                elif one_filter_threshold == 1:
                    outsum_mask_per_macro.append(0)
                    transfer_mask_per_macro.append(1)
                    pimset_mask_per_macro.append(1)
                else:
                    assert False
            # group_threshold.append(macro_threshold)
            assert len(outsum_mask_per_macro) == len(transfer_mask_per_macro)
            assert len(outsum_mask_per_macro) == len(pimset_mask_per_macro)
            assert len(outsum_mask_per_macro) <= n_bcol
            outsum_mask_per_macro = np.array(outsum_mask_per_macro, dtype=np.int8)
            outsum_mask_per_macro = np.pad(
                outsum_mask_per_macro,
                (0, n_bcol - len(outsum_mask_per_macro)),
                mode="constant",
                constant_values=0,
            )
            outsum_mask_per_group.append(outsum_mask_per_macro)

            transfer_mask_per_macro = np.array(transfer_mask_per_macro, dtype=np.int8)
            transfer_mask_per_macro = np.pad(
                transfer_mask_per_macro,
                (0, n_bcol - len(transfer_mask_per_macro)),
                mode="constant",
                constant_values=0,
            )
            transfer_mask_per_group.append(transfer_mask_per_macro)

            pimset_mask_per_macro = np.array(pimset_mask_per_macro, dtype=np.int8)
            pimset_mask_per_macro = np.pad(
                pimset_mask_per_macro,
                (0, n_bcol - len(pimset_mask_per_macro)),
                mode="constant",
                constant_values=0,
            )
            pimset_mask_per_group.append(pimset_mask_per_macro)

        outsum_mask_per_group = np.stack(outsum_mask_per_group, axis=0)
        outsum_mask_per_group = np.pad(
            outsum_mask_per_group,
            ((0, n_macro_per_group - len(outsum_mask_per_group)), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        assert len(outsum_mask_per_group.shape) == 2
        assert (
            outsum_mask_per_group.shape[0] == n_macro_per_group
            and outsum_mask_per_group.shape[1] == n_bcol
        )

        transfer_mask_per_group = np.stack(transfer_mask_per_group, axis=0)
        transfer_mask_per_group = np.pad(
            transfer_mask_per_group,
            ((0, n_macro_per_group - len(transfer_mask_per_group)), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        assert len(transfer_mask_per_group.shape) == 2
        assert (
            transfer_mask_per_group.shape[0] == n_macro_per_group
            and transfer_mask_per_group.shape[1] == n_bcol
        )

        pimset_mask_per_group = np.stack(pimset_mask_per_group, axis=0)
        pimset_mask_per_group = np.pad(
            pimset_mask_per_group,
            ((0, n_macro_per_group - len(pimset_mask_per_group)), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        assert len(pimset_mask_per_group.shape) == 2
        assert (
            pimset_mask_per_group.shape[0] == n_macro_per_group
            and pimset_mask_per_group.shape[1] == n_bcol
        )

        outer_oc_idx = outer_oc // n_filter_per_group
        # print(f"{total_outsum_mask.shape=}, {outsum_mask_per_group.shape=}")
        total_outsum_mask[outer_oc_idx, :, :] = outsum_mask_per_group
        total_transfer_mask[outer_oc_idx, :, :] = transfer_mask_per_group
        total_pimset_mask[outer_oc_idx, :, :] = pimset_mask_per_group

    assert len(total_threshold) == int(
        math.ceil(out_channel / n_filter_per_group) * n_filter_per_group
    )

    total_outsum_mask = total_outsum_mask.reshape(total_outsum_mask.shape[0], -1)
    total_transfer_mask = total_transfer_mask.reshape(total_transfer_mask.shape[0], -1)
    total_pimset_mask = total_pimset_mask.reshape(total_pimset_mask.shape[0], -1)

    # step 2: transform value-sparse weight to value-bit-sparse weight
    time, n_to, n_macro_per_group, n_vcol = value_sparse_weight.shape

    filter_idx = 0
    # Elements in 'new_weight' and 'info' are 0 or 1.
    # Datatype of these tensors is bit.
    # Setting dtype='int8' just for save memory.
    new_weight = np.zeros((time, n_to, n_macro_per_group, n_bcol), dtype="int8")
    info = np.zeros((time, n_to, n_macro_per_group, n_bcol, 3), dtype="int8")
    mapping_macro_to_row_begin = 0
    row_begin = 0
    for i_reduce_to_macro in range(len(mapping_reduce_to_macro)):
        mapping_macro_to_row_end = (
            mapping_macro_to_row_begin + mapping_reduce_to_macro[i_reduce_to_macro]
        )

        for i_macro_to_row in range(
            mapping_macro_to_row_begin, mapping_macro_to_row_end
        ):
            time_id = mapping_macro_to_row[i_macro_to_row]
            row_end = row_begin + time_id
            for i_row in range(row_begin, row_end):
                # import pdb; pdb.set_trace()
                for macro_id in range(n_macro_per_group):
                    filter_col_id = 0
                    for filter_id in range(n_filter_per_macro):
                        # value_sparse_weight.shape = [time, n_to, n_macro_per_group, n_vcol]
                        # import pdb; pdb.set_trace()
                        # print(f"{value_sparse_weight.shape=}")
                        filter_weight = value_sparse_weight[
                            i_row, :, macro_id, filter_id
                        ].reshape(
                            -1
                        )  # n_to
                        abs_filter_id = (
                            i_reduce_to_macro * n_filter_per_group
                            + macro_id * n_filter_per_macro
                            + filter_id
                        )
                        filter_threshold = total_threshold[abs_filter_id]

                        # check threshold
                        assert filter_threshold in [1, 2]
                        _filter_threshold = int_to_csd_nonzero_count_tensor(
                            filter_weight
                        )
                        assert np.logical_or(
                            (_filter_threshold == 0),
                            (_filter_threshold == filter_threshold),
                        ).all()

                        # value : [threshold, elem_in_filter,1], each value is 0 or 1
                        # sign : [threshold, elem_in_filter,1], each value is 0 or 1
                        # location : [threshold, elem_in_filter, 2], each value is in [0,1]
                        value, sign, location = parse_tensor(
                            filter_weight, filter_threshold
                        )

                        # Put 'value' into 'new_weight',
                        # Put 'sign' and 'location' into 'info'
                        for bit in range(filter_threshold):
                            new_weight[i_row, :, macro_id, filter_col_id] = value[bit]
                            info[i_row, :, macro_id, filter_col_id, 0] = sign[bit]
                            info[i_row, :, macro_id, filter_col_id, 1:3] = location[bit]
                            filter_col_id += 1
            row_begin = row_end

        mapping_macro_to_row_begin = mapping_macro_to_row_end

    # import pdb; pdb.set_trace()
    # Turn 'new_weight' and 'info' to tensor of bytes.
    assert n_bcol % 8 == 0
    # import pdb; pdb.set_trace()
    new_weight = new_weight.reshape(time, n_to, n_macro_per_group, n_bcol // 8, 8)
    info = info.reshape(time, n_to, n_macro_per_group, n_bcol * 3 // 8, 8)

    new_weight = tensor_bits_to_int8(new_weight)
    info = tensor_bits_to_int8(info)

    total_outsum_mask = total_outsum_mask.astype(np.int8)
    total_transfer_mask = total_transfer_mask.astype(np.int8)
    total_pimset_mask = total_pimset_mask.astype(np.int8)

    return {
        "converted_weight": new_weight,
        "meta": info,
        "outsum_mask": total_outsum_mask,
        "transfer_mask": total_transfer_mask,
        "pimset_mask": total_pimset_mask,
    }


def convert_value_sparse_conv2d_weight(weight, macro_config):
    """
    weight: [oc,kh,kw,ic]

    converted_weight: [out_spatial_tile, out_reduce_tile, n_comp, n_group, n_group_vcol] 1byte
    mask: [n_sparse_time, n_comp, n_macro] 1bit   "n_sparse_time" is the combination of "out_spatial_tile" and "out_reduce_tile", it is a sparse axis.

    return: converted_weight, mask, index
    """
    if len(weight.shape) == 4:
        oc, kh, kw, ic = weight.shape
        spatial_size = oc
        reduce_size = ic * kh * kw
        weight = weight.reshape(oc, reduce_size)
    elif len(weight.shape) == 2:
        spatial_size, reduce_size = weight.shape
    else:
        assert False
    assert weight.dtype == np.int8

    n_row = macro_config["n_row"]
    n_vcol = macro_config["n_vcol"]
    n_group = macro_config["n_group"]
    n_macro_per_group = macro_config["n_macro"] // n_group
    n_group_vcol = n_macro_per_group * n_vcol
    n_comp = macro_config["n_comp"]  # * macro_config["n_row"]
    n_from = macro_config["n_value_sparse_from"]  # 128
    n_to = macro_config["n_value_sparse_to"]  # 16
    assert n_from > n_to and n_to == n_comp

    # padding weights
    spatial_pad_size = (
        int(math.ceil(spatial_size / n_group_vcol)) * n_group_vcol - spatial_size
    )
    reduce_pad_size = int(math.ceil(reduce_size / n_from)) * n_from - reduce_size
    weight = np.pad(
        weight,
        ((0, spatial_pad_size), (0, reduce_pad_size)),
        mode="constant",
        constant_values=0,
    )
    total_spatial_size = weight.shape[0]
    total_reduce_size = weight.shape[1]
    assert total_spatial_size % n_group_vcol == 0
    assert total_reduce_size % n_from == 0

    # tile the weight
    out_spatial_tile = total_spatial_size // n_group_vcol
    out_reduce_tile = total_reduce_size // n_from
    weight = weight.reshape(
        out_spatial_tile, n_macro_per_group, n_vcol, out_reduce_tile, n_from
    )
    weight = np.transpose(weight, (0, 3, 2, 4, 1))
    assert weight.shape[-2] == n_from and weight.shape[-1] == n_macro_per_group
    weight = weight.reshape(
        out_spatial_tile, out_reduce_tile, n_vcol, n_from, n_macro_per_group
    )

    # extract n_to non-zero element from n_from
    mapping_reduce_to_macro = []
    mapping_macro_to_from = []
    mapping_from_to_row = []
    mapping_macro_to_row = []

    weight_list = []
    mask_list = []
    for ost in range(out_spatial_tile):
        macro_in_reduce = 0
        total_row = 0
        i_outer_reduce = 0
        reduce_element = 0

        while reduce_element < total_reduce_size:
            macro_fill = False
            from_in_macro = 0
            row_in_macro = 0
            while row_in_macro < n_row and i_outer_reduce < out_reduce_tile:
                ort = i_outer_reduce

                subweight = weight[
                    ost, ort, :, :, :
                ]  # n_vcol, n_from, n_macro_per_group
                submask, subweight = extrace_mask_and_data(
                    subweight,
                    n_from,
                    n_to,
                    concat=True,
                    bit_to_byte=True,
                    strict_align=True,
                )
                # subweight: t, n_vcol, n_to, n_macro_per_group
                # submask: t, n_to, n_macro_per_group
                assert (
                    len(subweight.shape) == 4
                    and subweight.shape[1] == n_vcol
                    and subweight.shape[2] == n_to
                    and subweight.shape[3] == n_macro_per_group
                )
                assert (
                    len(submask.shape) == 3
                    and submask.shape[1] == n_from
                    and submask.shape[2] == n_macro_per_group
                ), f"{submask.shape=}"
                assert subweight.shape[0] == submask.shape[0]

                delta_row = submask.shape[0]
                if row_in_macro + delta_row > n_row:
                    macro_fill = True
                    break

                mapping_from_to_row.append(delta_row)
                total_row += delta_row
                row_in_macro += delta_row
                from_in_macro += 1
                i_outer_reduce += 1

                mask_list.append(submask)
                weight_list.append(subweight)
            if from_in_macro > 0:
                mapping_macro_to_from.append(from_in_macro)
                mapping_macro_to_row.append(row_in_macro)
                macro_in_reduce += 1
                reduce_element += from_in_macro * n_from
            # if macro_fill:
            #     break

        mapping_reduce_to_macro.append(macro_in_reduce)

    converted_weight = np.concatenate(weight_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)

    mapping_reduce_to_macro = np.array(mapping_reduce_to_macro, np.int32)
    mapping_macro_to_from = np.array(mapping_macro_to_from, np.int32)
    mapping_from_to_row = np.array(mapping_from_to_row, np.int32)
    mapping_macro_to_row = np.array(mapping_macro_to_row, np.int32)

    converted_weight = np.transpose(converted_weight, (0, 2, 3, 1))
    # [time, n_to, n_macro_per_group, n_vcol]

    # mask.shape : [t, n_from, n_macro_per_group] -> [t, n_macro_per_group, n_from]
    mask = np.transpose(mask, [0, 2, 1]).astype(np.int8)
    return {
        "converted_weight": converted_weight,
        "mask": mask,
        "mapping_reduce_to_macro": mapping_reduce_to_macro,
        "mapping_macro_to_from": mapping_macro_to_from,
        "mapping_from_to_row": mapping_from_to_row,
        "mapping_macro_to_row": mapping_macro_to_row,
    }


if __name__ == "__main__":
    test()
