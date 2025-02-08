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
            -127, 128, size=(self.in_channel, self.in_hw, self.in_hw), dtype=np.int8
        )
        # input_data = np.zeros((self.in_channel, self.in_hw, self.in_hw), dtype=np.int8)
        # input_data[0, 0, 0] = 1
        # input_data[0, 0, 2] = 1
        # input_data[0, 2, 0] = 1
        # input_data[0, 2, 2] = 1
        # print(f"{input_data=}")
        return input_data

    def _calculate_golden(self):
        import numpy as np

        input_tensor = self.input_.reshape(
            self.in_hw//self.ker_size, self.ker_size, self.in_hw//self.ker_size, self.ker_size, self.in_channel
        )
        output_tensor = input_tensor.max(axis=(1,3))
        # output_tensor = np.transpose(output_tensor, (1, 2, 0))

        return output_tensor

    def get_image(
        self,
        simulator,
        input_=None,
    ):
        import numpy as np

        if input_ is None:
            self.input_ = self._get_mock_input()
        else:
            self.input_ = input_

        self.input_ = np.transpose(self.input_, (1, 2, 0))

        image = bytearray(self.input_)
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
