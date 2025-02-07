# from op.helper import TestHelper


class TestHelper:
    def __init__(self, op_config):
        import numpy as np

        self.output_bytes = 1
        self.output_dtype = np.int8

        self.in_channel = op_config["in_channel"] 
        self.in_hw = op_config["in_hw"]

        self.im2col = True

    def _get_mock_input1(self):
        import numpy as np

        input_data = np.random.randint(
            -1, 3, size=(self.in_channel, self.in_hw, self.in_hw), dtype=np.int8
        ) 
        # input_data = np.ones((self.in_channel, self.in_hw, self.in_hw), dtype=np.int8)# .reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        # input_data = np.arange(1, self.in_channel*self.in_hw*self.in_hw+1, dtype=np.int8).reshape(self.in_channel, self.in_hw, self.in_hw)
        return input_data

    def _get_mock_input2(self):
        import numpy as np

        input_data = np.random.randint(
            -1, 3, size=(self.in_channel,), dtype=np.int8
        ) 
        # input_data = np.ones((self.input_size), dtype=np.int8)# .reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        # input_data = np.arange(1, self.in_channel+1, dtype=np.int8).reshape(1, 1, self.in_channel)
        return input_data

    def _get_mock_scale(self):
        import numpy as np

        scale = np.random.rand(1).astype(np.float32)
        # scale = np.ones((self.out_channel,)).astype(np.float32)
        # scale = np.ones((1,), dtype=np.float32)
        return scale

    def _calculate_golden(self):
        import numpy as np

        from utils.round import banker_round
        
        mul_result_1 = self.input1.astype(np.int32) * self.input2.astype(np.int32)
        mul_result_2 = banker_round(mul_result_1 * self.scale)
        mul_result = np.clip(mul_result_2, -128, 127).astype(np.int8)   

        return mul_result   

    def get_image_quantify(self, simulator, scale):
        import numpy as np

        from utils.bias_scale_fuse import bias_scale_fuse


        if scale is None:
            scale = self._get_mock_scale()

        assert scale.size==1

        self.scale = scale
        
        bias = np.zeros((1,), dtype=np.int32)
        bias_scale_data = bias_scale_fuse(bias, scale)

        out_zp = np.zeros((1,), dtype=np.int32)
        out_zp_data = bytearray(out_zp)
        
        return bias_scale_data + out_zp_data

    def get_image(
        self,
        simulator,
        input1=None,
        input2=None,
        scale=None,
    ):
        import numpy as np

        if input1 is None:
            self.input1 = self._get_mock_input1()
        else:
            self.input1 = input1

        self.input1 = np.transpose(self.input1, (1, 2, 0))

        if input2 is None:
            self.input2 = self._get_mock_input2()
        else:
            self.input2 = input2

        assert len(self.input2.shape) == 1
        self.input2 = self.input2.reshape(1, 1, self.in_channel)

        

        origin_image = bytearray(self.input1) + bytearray(self.input2)
        quantify_image = self.get_image_quantify(simulator, scale)
        
        image = origin_image + quantify_image
        print(f"{len(origin_image)=}")
        print(f"{len(quantify_image)=}")
        self.output_offset = len(image)
        
        return image

    def _make_template_config(self, simulator):
        context = {"INPUT_CHANNEL": self.in_channel, "INPUT_COL": self.in_hw, "INPUT_ROW": self.in_hw}
        return context

    def fill_template(self, src_path, dst_path, simulator):
        import os

        from jinja2 import Environment, FileSystemLoader, StrictUndefined

        src_folder, src_file = os.path.split(src_path)

        # 创建 Jinja2 环境和加载器
        env = Environment(
            loader=FileSystemLoader([src_folder, os.environ["CIM_COMPILER_BASE"]]), 
            undefined=StrictUndefined
        )

        # 加载模板
        template = env.get_template(src_file)

        context = self._make_template_config(simulator)

        # 渲染模板
        output = template.render(context)

        with open(dst_path, "w") as f:
            f.write(output)

    def get_output(self, memory_space):
        import numpy as np

        """
        image should have:
        input (4 * 1 byte)
        weight (32 * 1 byte)
        output (8 * 4 byte)
        """
        global_offset = memory_space.get_base_of("global")
        output_offset = global_offset + self.output_offset
        output_byte_size = (
            self.in_channel * self.in_hw * self.in_hw * self.output_bytes
        )
        output = memory_space.read_as(
            output_offset, output_byte_size, self.output_dtype
        )
        output = output.reshape(self.in_hw, self.in_hw, self.in_channel)
        return output

    def check_image(self, memory_space):
        import numpy as np

        """
        image should have:
        input (4 * 1 byte)
        weight (32 * 1 byte)
        output (8 * 4 byte)
        """
        output = self.get_output(memory_space)
        golden = self._calculate_golden()
        assert np.array_equal(output, golden), f"{output=}, {golden=}"
