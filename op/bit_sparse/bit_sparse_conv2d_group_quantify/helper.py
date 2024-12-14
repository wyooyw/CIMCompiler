from op.helper import BitSparseConv2dTestHelper, QuantizeHelper


class TestHelper(BitSparseConv2dTestHelper, QuantizeHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        import numpy as np

        self.output_bytes = 1
        self.output_dtype = np.int8

        # a special hack for efficient net
        if self.in_hw == 1 and self.out_hw == 1 and self.ker_size == 1:
            self.n_use_group = 1
        else:
            self.n_use_group = 4

        self.im2col = True

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
        import os
        
        context = super()._make_template_config(simulator)
        context["RELU"] = int(self.relu)
        context["SINGLE_OUTER_REDUCE"] = int(
            context["OUT_REDUCE_TILE"] <= simulator.macro_config.n_row
        )
        context["N_USE_GROUP"] = self.n_use_group
        context["IM2COL"] = self.im2col
        if self.im2col:
            context["IM2COL_SIZE_0"] = self.input_data_im2col.shape[0]
            context["IM2COL_SIZE_1"] = self.input_data_im2col.shape[1]
            if context["IM2COL_SIZE_0"] > 1:
                context["IM2COL_SMALL_INPUT_MEMORY"] = bool(
                    int(os.environ.get("IM2COL_SMALL_INPUT_MEMORY"))
                )
            else:
                context["IM2COL_SMALL_INPUT_MEMORY"] = False
        context["FAST_MODE"] = bool(int(os.environ.get("FAST_MODE")))
        
        # output fill memory: whether output can fill output_memory
        i32_output_size = 4 * context["OUTPUT_ROW"] * context["OUTPUT_COL"] * context["MAX_OUTER_SPATIEL_TILE_SIZE"]
        i8_output_size = context["OUTPUT_ROW"] * context["OUTPUT_COL"] * context["OUTPUT_CHANNEL"]
        output_size = i32_output_size + i8_output_size

        output_memory_size = simulator.memory_space.get_memory_by_name("output_memory").size
        output_fill_memory = output_size <= output_memory_size
        context["FAST_MODE_OUTPUT_FILL_MEMORY"] = context["FAST_MODE"] and output_fill_memory and context["SINGLE_OUTER_REDUCE"] == 0
        
        return context
