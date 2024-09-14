from test.compiler.pimcompute.helper import BitSparseConv2dTestHelper, QuantizeHelper
class TestHelper(BitSparseConv2dTestHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        import numpy as np
        self.output_bytes = 4
        self.output_dtype = np.int32

        # a special hack for efficient net
        if self.in_hw==1 and self.out_hw==1 and self.ker_size==1:
            self.n_use_group = 1
        else:
            self.n_use_group = 4
        
        self.im2col = True

    # def _get_mock_weight(self):
    #     import numpy as np
    #     from utils.bit_sparse_weight_transform import generate_valid_weight

    #     weight = np.zeros([self.out_channel, self.ker_size, self.ker_size, self.in_channel], dtype=np.int8) + 3
    #     return weight

    # def _get_mock_input(self):
    #     import numpy as np
    #     input_data = np.ones((self.in_hw,self.in_hw, self.in_channel), dtype=np.int8) # .reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
    #     for row in range(self.in_hw):
    #         for col in range(self.in_hw):
    #             input_data[row,col,:] = row * self.in_hw + col
    #     # assert input_data.shape==(self.in_hw,self.in_hw,self.in_channel), f"{input_data.shape=}"
    #     return input_data

    # def get_image(self, simulator, input=None, weight=None, bias=None, scale=None, out_zp=None, relu=False):
    #     import numpy as np
    #     from utils.bias_scale_fuse import bias_scale_fuse

    #     quantify_image = self.get_image_quantify(simulator, bias, scale, out_zp, relu)
    #     origin_image = super().get_image(simulator, input, weight)
    #     image = origin_image + quantify_image
    #     self.output_offset = len(image)
    #     return image

    def _make_template_config(self, simulator):
        import os
        
        context = super()._make_template_config(simulator)
        # context["RELU"] = int(self.relu)
        context["SINGLE_OUTER_REDUCE"] = int(context["OUT_REDUCE_TILE"] <= simulator.macro_config.n_row)
        context["N_USE_GROUP"] = self.n_use_group
        context["IM2COL"] = self.im2col
        if self.im2col:
            context["IM2COL_SIZE_0"] = self.input_data_im2col.shape[0]
            context["IM2COL_SIZE_1"] = self.input_data_im2col.shape[1]
            if context["IM2COL_SIZE_0"] > 1:
                context["IM2COL_SMALL_INPUT_MEMORY"] = bool(int(os.environ.get("IM2COL_SMALL_INPUT_MEMORY")))
            else:
                context["IM2COL_SMALL_INPUT_MEMORY"] = False
                
        context["FAST_MODE"] = bool(int(os.environ.get("FAST_MODE")))
        return context