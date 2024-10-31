from test.compiler.pimcompute.helper import DenseConv2dTestHelper


class TestHelper(DenseConv2dTestHelper):
    def __init__(self, op_config):
        super().__init__(op_config)

    def _make_dense_data(self, weight, simulator):
        from data_processor.dense import convert_dense_conv2d_weight

        macro_config = simulator.macro_config
        bitwidth = 8
        n_group = 1
        n_vcol = macro_config.n_bcol // bitwidth
        n_macro_per_group = macro_config.n_macro // n_group
        n_group_vcol = n_macro_per_group * n_vcol
        config = {
            "n_vcol": n_vcol,
            "n_group": n_group,
            "n_macro": macro_config.n_macro,
            "n_comp": macro_config.n_comp,
        }
        converted_weight = convert_dense_conv2d_weight(weight, config)

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

        out_spatial_tile, out_reduce_tile, _, _, _ = converted_weight.shape
        converted_weight = converted_weight.reshape(
            out_spatial_tile, out_reduce_tile, -1
        )

        print(f"{converted_weight.shape=}, {converted_weight.dtype=}")
        # print(converted_weight)

        return converted_weight
