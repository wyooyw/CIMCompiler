# from cim_compiler.op.helper import TestHelper


class TestHelper:
    def __init__(self, op_config):
        import numpy as np

        self.output_bytes = 1
        self.output_dtype = np.int8

        # assert op_config["in_channel"] == op_config["out_channel"], f"{op_config=}"
        self.input_size = op_config["in_channel"] * op_config["in_hw"] * op_config["in_hw"]
        self.in_channel = op_config["in_channel"]
        self.in_hw = op_config["in_hw"]
        self.out_hw = op_config["out_hw"]
        self.ker_size = op_config["ker_size"]
        assert self.in_hw % self.ker_size == 0, f"{self.in_hw=}, {self.ker_size=}"
        assert self.in_hw // self.ker_size == self.out_hw, f"{self.in_hw=}, {self.ker_size=}, {self.out_hw=}"


    def _get_mock_input(self):
        import numpy as np

        input_data = np.random.randint(
            -10, 100, size=(self.in_channel, self.in_hw, self.in_hw), dtype=np.int8
        )
        # input_data = np.arange(self.in_channel * self.in_hw * self.in_hw).reshape(self.in_channel, self.in_hw, self.in_hw).astype(np.int8)
        # input_data = np.ones((self.in_channel, self.in_hw, self.in_hw), dtype=np.int8)
        return input_data
    
    def _get_mock_scale(self):
        import numpy as np

        scale = np.random.rand(1).astype(np.float32)
        # scale = np.ones((1,)).astype(np.float32)
        return scale

    def _get_mock_out_zp(self):
        import numpy as np

        out_zp = np.zeros((1,), dtype=np.int32)
        return out_zp

    def _calculate_golden(self):
        import numpy as np
        from cim_compiler.utils.round import banker_round

        input_tensor = self.input_.reshape(
            self.in_hw//self.ker_size, self.ker_size, self.in_hw//self.ker_size, self.ker_size, self.in_channel
        )
        output_tensor = np.floor(input_tensor.mean(axis=(1,3))).astype(np.int32)
        
        clip_min = -128
        output_quantify = np.zeros((self.out_hw, self.out_hw, self.in_channel), dtype=np.int8)
        for row in range(self.out_hw):
            for col in range(self.out_hw):
                input_data = output_tensor[row, col, :]
                output_data = banker_round(input_data * self.scale) + self.out_zp
                output_data = banker_round(np.clip(output_data, clip_min, 127))
                output_data = output_data.astype("int8")
                output_quantify[row, col, :] = output_data

        return output_quantify

    def get_image(
        self,
        simulator,
        input_=None,
        scale=None,
        out_zp=None
    ):
        import numpy as np
        from cim_compiler.utils.bias_scale_fuse import bias_scale_fuse
        
        self.input_ = self._get_mock_input() if input_ is None else input_
        self.mul_factor = np.array([1 / (self.ker_size * self.ker_size)], dtype=np.float32)
        self.scale = self._get_mock_scale() if scale is None else scale
        self.out_zp = self._get_mock_out_zp() if out_zp is None else out_zp

        self.input_ = np.transpose(self.input_, (1, 2, 0))
        self.scale = np.repeat(self.scale, self.in_channel)
        bias = np.zeros((self.in_channel,), dtype=np.int32)
        bias_scale_data = bias_scale_fuse(bias, self.scale)

        image = (
            bytearray(self.input_) +
            bytearray(self.mul_factor) +
            bytearray(bias_scale_data) +
            bytearray(self.out_zp)
        )
        self.output_offset = len(image)
        
        return image

    def _make_template_config(self, simulator):
        context = {
            "INPUT_ROW": self.in_hw,
            "INPUT_COL": self.in_hw,
            "INPUT_CHANNEL": self.in_channel,
            "OUTPUT_ROW": self.out_hw,
            "OUTPUT_COL": self.out_hw,
            "KERNEL_SIZE": self.ker_size,
        }
        return context

    def fill_template(self, src_path, dst_path, simulator):
        import os

        from jinja2 import Environment, FileSystemLoader, StrictUndefined

        src_folder, src_file = os.path.split(src_path)

        # 创建 Jinja2 环境和加载器
        env = Environment(
            loader=FileSystemLoader([
                src_folder, 
                os.environ["CIM_COMPILER_BASE"],
                os.environ.get(os.environ["CIM_COMPILER_BASE"], "cim_compiler")
            ]),
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

        global_offset = memory_space.get_base_of("global")
        output_offset = global_offset + self.output_offset
        output_byte_size = (
            self.out_hw * self.out_hw * self.in_channel * self.output_bytes
        )
        output = memory_space.read_as(
            output_offset, output_byte_size, self.output_dtype
        )
        output = output.reshape(self.out_hw, self.out_hw, self.in_channel)
        return output

    def check_image(self, memory_space):
        import numpy as np

        output = self.get_output(memory_space)
        golden = self._calculate_golden()
        assert np.array_equal(output, golden), f"{output=}, {golden=}"
