from test.compiler.pimcompute.helper import DenseConv2dTestHelper, QuantizeHelper
class TestHelper(DenseConv2dTestHelper, QuantizeHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        import numpy as np
        self.output_bytes = 1
        self.output_dtype = np.int8

        # a special hack for efficient net
        if self.in_hw==1 and self.out_hw==1 and self.ker_size==1:
            self.n_use_group = 1
        else:
            self.n_use_group = 4

    def _calculate_golden(self):
        return self._calculate_golden_quantize()

    def get_image(self, simulator, input=None, weight=None, bias=None, scale=None, out_zp=None, relu=False):
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
        context["SINGLE_OUTER_REDUCE"] = int(context["OUT_REDUCE_TILE"] <= simulator.macro_config.n_row)
        context["N_USE_GROUP"] = self.n_use_group
        return context