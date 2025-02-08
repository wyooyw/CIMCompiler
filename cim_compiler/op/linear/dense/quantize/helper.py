from cim_compiler.op.helper import DenseConv2dTestHelper, QuantizeHelper


class TestHelper(DenseConv2dTestHelper, QuantizeHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        import numpy as np

        self.output_bytes = 1
        self.output_dtype = np.int8

    def _make_dense_data(self, weight, simulator):
        from cim_compiler.data_processor.dense import convert_dense_conv2d_weight

        macro_config = simulator.macro_config
        bitwidth = 8
        n_group = macro_config.n_group
        n_vcol = macro_config.n_bcol // bitwidth
        n_macro_per_group = macro_config.n_macro_per_group
        n_group_vcol = n_macro_per_group * n_vcol
        config = {
            "n_vcol": n_vcol,
            "n_group": n_group,
            "n_macro": macro_config.n_macro,
            "n_comp": macro_config.n_comp,
        }
        converted_weight, pimset_mask = convert_dense_conv2d_weight(weight, config)

        assert len(converted_weight.shape) == 5, f"{converted_weight.shape=}"
        assert (
            converted_weight.shape[2] == macro_config.n_comp
        ), f"{converted_weight.shape=}, {macro_config.n_comp=}"
        assert (
            converted_weight.shape[3] == n_group
        ), f"{converted_weight.shape=}, {n_group=}"
        assert (
            converted_weight.shape[4] == n_group_vcol
        ), f"{converted_weight.shape=}, {n_group_vcol=}"
        converted_weight = converted_weight[:, :, :, 0:1, :]
        out_spatial_tile, out_reduce_tile, _, _, _ = converted_weight.shape
        converted_weight = converted_weight.reshape(
            out_spatial_tile, out_reduce_tile, -1
        )

        print(f"{converted_weight.shape=}, {converted_weight.dtype=}")
        # print(converted_weight)

        return converted_weight, pimset_mask

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

        from cim_compiler.utils.bias_scale_fuse import bias_scale_fuse

        quantify_image = self.get_image_quantify(simulator, bias, scale, out_zp, relu)
        origin_image = super().get_image(simulator, input, weight)
        image = origin_image + quantify_image
        self.output_offset = len(image)
        return image

    def _make_template_config(self, simulator):
        context = super()._make_template_config(simulator)
        context["RELU"] = int(self.relu)
        return context
