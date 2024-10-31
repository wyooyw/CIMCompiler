from op.helper import BitSparseConv2dTestHelper, QuantizeHelper


class TestHelper(BitSparseConv2dTestHelper, QuantizeHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        import numpy as np

        self.output_bytes = 1
        self.output_dtype = np.int8

    def _get_mock_weight(self):
        from utils.bit_sparse_weight_transform import generate_valid_weight

        weight = generate_valid_weight(
            [self.out_channel, self.ker_size, self.ker_size, self.in_channel]
        )
        return weight

    def _make_bit_sparse_data(self, weight, simulator):
        from types import SimpleNamespace

        import numpy as np

        from utils.bit_sparse_weight_transform import (
            outsum_mask_to_transfer_mask,
            parse_out_begin_channel,
            parse_out_mask,
            parse_out_mask_and_transfer_mask,
            weight_transform_group,
        )
        from utils.df_layout import tensor_bits_to_int8

        macro_config = simulator.macro_config
        bitwidth = 8
        n_group = 4
        n_comp = macro_config.n_comp
        n_bcol = macro_config.n_bcol
        n_macro_per_group = macro_config.n_macro // n_group
        n_group_bcol = n_macro_per_group * n_bcol
        op_cfg = SimpleNamespace(
            out_channel=weight.shape[0],
        )
        cim_cfg = SimpleNamespace(
            bits_column=n_bcol,
            n_macro=simulator.macro_config.n_macro // n_group,  # n_macro_per_group
            n_group=n_group,
            n_comp=n_comp,
        )
        bit_sparse_weight, meta, fold = weight_transform_group(
            weight, cim_cfg, op_cfg, "OHWI"
        )
        ele_in_filter = bit_sparse_weight.shape[1]
        assert ele_in_filter % n_comp == 0
        bit_sparse_weight = bit_sparse_weight.reshape(
            bit_sparse_weight.shape[0],
            ele_in_filter // n_comp,
            n_comp,
            bit_sparse_weight.shape[2],
            bit_sparse_weight.shape[3],
        )
        outsum_mask, transfer_mask, pimset_mask = parse_out_mask_and_transfer_mask(
            fold, n_group_bcol
        )
        out_begin_channel = parse_out_begin_channel(fold)

        assert len(bit_sparse_weight.shape) == 5, f"{bit_sparse_weight.shape=}"
        assert (
            bit_sparse_weight.shape[2] == macro_config.n_comp
        ), f"{bit_sparse_weight.shape=}, {macro_config.n_comp=}"
        assert (
            bit_sparse_weight.shape[3] == n_group
        ), f"{bit_sparse_weight.shape=}, {n_group=}"
        assert bit_sparse_weight.shape[4] == (
            n_group_bcol // 8
        ), f"{bit_sparse_weight.shape=}, {(n_group_bcol//8)=}"
        assert outsum_mask.shape[1] == (
            n_group_bcol // 8
        ), f"{outsum_mask.shape=}, {(n_group_bcol//8)=}"
        assert transfer_mask.shape[1] == (
            n_group_bcol // 8
        ), f"{outsum_mask.shape=}, {(n_group_bcol//8)=}"
        assert pimset_mask.shape[1] == (
            n_group_bcol // 8
        ), f"{outsum_mask.shape=}, {(n_group_bcol//8)=}"
        assert (
            outsum_mask.shape[0] == bit_sparse_weight.shape[0]
        ), f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"
        assert (
            transfer_mask.shape[0] == bit_sparse_weight.shape[0]
        ), f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"
        assert (
            pimset_mask.shape[0] == bit_sparse_weight.shape[0]
        ), f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"

        assert bit_sparse_weight.dtype == np.int8, f"{bit_sparse_weight.dtype=}"
        assert outsum_mask.dtype == np.int8, f"{outsum_mask.dtype=}"
        assert transfer_mask.dtype == np.int8, f"{transfer_mask.dtype=}"
        assert pimset_mask.dtype == np.int8, f"{pimset_mask.dtype=}"
        assert meta.dtype == np.int8, f"{meta.dtype=}"

        assert len(out_begin_channel.shape) == 1
        assert (
            out_begin_channel.shape[0] == bit_sparse_weight.shape[0] + 1
        ), f"{out_begin_channel.shape=}, {bit_sparse_weight.shape=}"
        assert (
            out_begin_channel[-1] == self.out_channel
        ), f"{out_begin_channel[-1]=}, {self.out_channel=}"
        assert out_begin_channel.dtype == np.int32, f"{out_begin_channel.dtype=}"

        return (
            bit_sparse_weight,
            meta,
            outsum_mask,
            transfer_mask,
            pimset_mask,
            out_begin_channel,
        )

    def _calculate_golden(self):
        return self._calculate_golden_quantize()

    def get_image(
        self,
        simulator,
        input=None,
        weight=None,
        bias=None,
        scale=None,
        out_zp=None,
        relu=False,
    ):
        import numpy as np

        from utils.bias_scale_fuse import bias_scale_fuse

        quantify_image = self.get_image_quantify(simulator, bias, scale, out_zp, relu)
        origin_image = super().get_image(simulator, input, weight)
        image = origin_image + quantify_image
        self.output_offset = len(image)
        return image

    def _make_template_config(self, simulator):
        context = super()._make_template_config(simulator)
        context["RELU"] = int(self.relu)
        return context
