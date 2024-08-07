from test.compiler.pimcompute.helper import ValueBitSparseConv2dTestHelper, QuantizeHelper
class TestHelper(ValueBitSparseConv2dTestHelper, QuantizeHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        import numpy as np
        self.output_bytes = 1
        self.output_dtype = np.int8

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
        context["SINGLE_OUTER_REDUCE"] = (self.mapping_reduce_to_macro==1).all()
        return context