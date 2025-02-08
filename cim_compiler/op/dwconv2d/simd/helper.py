from cim_compiler.op.helper import DenseConv2dTestHelper, QuantizeHelper


class TestHelper(DenseConv2dTestHelper, QuantizeHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        import numpy as np

        self.output_bytes = 1
        self.output_dtype = np.int8

        assert op_config["in_channel"] == op_config["out_channel"], f"{op_config=}"

        self.im2col = True

    def _get_mock_weight(self):
        import numpy as np

        weight = np.random.randint(
            -1, 3, size=(self.out_channel, self.ker_size, self.ker_size), dtype=np.int8
        )
        # weight = np.ones((self.out_channel, self.ker_size, self.ker_size), dtype=np.int8)
        # weight = np.arange(1, self.out_channel*self.ker_size*self.ker_size+1, dtype=np.int8).reshape(self.out_channel, self.ker_size, self.ker_size)
        # import pdb; pdb.set_trace()
        return weight

    def _get_mock_input(self):
        import numpy as np

        input_data = np.random.randint(
            -1, 3, size=(self.in_channel, self.in_hw, self.in_hw), dtype=np.int8
        )  # .reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        # input_data = np.ones((self.in_channel,self.in_hw,self.in_hw), dtype=np.int8)# .reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        # input_data = np.arange(1, self.in_channel*self.in_hw*self.in_hw+1, dtype=np.int8).reshape(self.in_channel, self.in_hw, self.in_hw)
        return input_data

    def _get_mock_bias(self):
        import numpy as np

        bias = np.random.randint(-4,5,size=(self.out_channel), dtype=np.int32)
        # bias = np.zeros((self.out_channel,), dtype=np.int32)
        return bias

    def _get_mock_scale(self):
        import numpy as np

        scale = np.random.rand(self.out_channel).astype(np.float32)
        # scale = np.ones((self.out_channel,)).astype(np.float32)
        return scale

    def _get_mock_out_zp(self):
        import numpy as np

        out_zp = np.zeros((1,), dtype=np.int32)
        return out_zp

    def _assert_check_input_and_weight_shape(self, input, weight):
        """
        assert input.shape is [in_hw, in_hw, in_channel]
        assert weight.shape is [out_channel, ker_size, ker_size, in_channel]
        """
        assert len(input.shape) == 3, f"{input.shape=}"
        assert (
            input.shape[1] == input.shape[2] and input.shape[1] == self.in_hw
        ), f"{input.shape=}"
        assert input.shape[0] == self.in_channel, f"{input.shape=}"

        assert len(weight.shape) == 3, f"{weight.shape=}"
        assert weight.shape[0] == self.out_channel, f"{weight.shape=}"
        assert (
            weight.shape[1] == weight.shape[2] and weight.shape[2] == self.ker_size
        ), f"{weight.shape=}"

    def _apply_padding(self, input_data):
        """
        input.shape: C,H,W
        """
        import numpy as np

        input_data = np.pad(
            input_data,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
            constant_values=0,
        )
        return input_data

    def _make_dense_data(self, weight, simulator):
        import numpy as np
        # C, H, W -> H, W, C
        weight = np.transpose(weight, (1, 2, 0))

        # empty pimset mask
        pimset_mask = np.zeros((0,), dtype=np.int8)

        return weight, pimset_mask

    def _calculate_golden(self):
        import numpy as np

        from cim_compiler.utils.round import banker_round

        output_h = output_w = self.out_hw
        output_c = self.out_channel

        output = np.zeros((output_h, output_w, output_c), dtype=np.int32)
        weight = self.weight_data.reshape(self.weight_data.shape[0], -1)
        stride = self.stride
        out_channel = self.out_channel
        for channel in range(out_channel):
            for row in range(output_h):
                for col in range(output_w):
                    input = self.input_data[
                        channel,
                        stride * row : stride * row + self.ker_size,
                        stride * col : stride * col + self.ker_size,
                    ].reshape(-1)
                    use_weight = weight[channel, :]
                    golden = np.dot(use_weight.astype(np.int32), input.astype(np.int32))
                    output[row, col, channel] = golden.reshape(-1)

        # quantify
        clip_min = 0 if self.relu else -128
        output_quantify = np.zeros((output_h, output_w, output_c), dtype=np.int8)
        for row in range(output_h):
            for col in range(output_w):
                input_data = output[row, col, :]
                output_data = input_data + self.bias
                output_data = banker_round(output_data * self.scale) + self.out_zp
                output_data = banker_round(np.clip(output_data, clip_min, 127))
                output_data = output_data.astype("int8")
                output_quantify[row, col, :] = output_data
        # print(output_quantify)
        return output_quantify

    def _apply_im2col(self, input_data):
        import numpy as np

        in_channel, in_hw, _ = input_data.shape
        ker_size = self.ker_size
        stride = self.stride
        out_hw = self.out_hw
        im2col_input = np.zeros((out_hw, in_hw, ker_size, in_channel), dtype=np.int8)
        for ic in range(in_channel):
            for ow in range(out_hw):
                iw = ow * stride
                for ih in range(in_hw):
                    im2col_input[ow, ih, :, ic] = input_data[ic, ih, iw : iw + ker_size]
        im2col_input = im2col_input.reshape(out_hw, -1, in_channel)
        return im2col_input

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
        context["WINDOW_SIZE"] = self.ker_size # * self.ker_size
        context["IM2COL"] = self.im2col
        if self.im2col:
            context["IM2COL_SIZE_0"] = self.input_data_im2col.shape[0]
            context["IM2COL_SIZE_1"] = self.input_data_im2col.shape[1]
        return context
