class TestHelper:
    def __init__(self):
        self.out_channel = 32
        self.ker_size = 1
        self.in_channel = 1024
        self.in_hw = 2
        self.out_hw = 2

    def _prepare_weight_data(self):
        import numpy as np
        """
        weight: 32 * 32 * 3 * 3
        input: 32 * 8 * 8
        """
        # make a weight
        # weight = np.zeros((self.out_channel, self.ker_size * self.ker_size * self.in_channel), dtype=np.int8)
        # for i in range(0,self.out_channel,2):
        #     weight[i:i+2, i:i+99*2:2] = np.arange(1+i,1+i+99, dtype=np.int8)
            # weight[i:i+2, 256:3*3*32] = np.arange(0,3*3*32-256, dtype=np.int8)

        # print(weight.shape)
        # print(weight)
        # weight = np.ones((self.out_channel, self.ker_size * self.ker_size * self.in_channel), dtype=np.int8)
        # weight = np.repeat(weight, self.out_channel, axis=0)
        weight = np.random.randint(-100, 100, size=(self.out_channel, self.ker_size * self.ker_size * self.in_channel), dtype=np.int8)

        return weight

    def _make_dense_data(self, weight, simulator):
        from data_processor.dense import convert_dense_conv2d_weight

        macro_config = simulator.macro_config
        bitwidth = 8
        n_group = 4
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

        assert len(converted_weight.shape)==5, f"{converted_weight.shape=}"
        assert converted_weight.shape[2]==macro_config.n_comp, f"{converted_weight.shape=}, {macro_config.n_comp=}"
        assert converted_weight.shape[3]==n_group, f"{converted_weight.shape=}, {n_group=}"
        assert converted_weight.shape[4]==n_group_vcol, f"{converted_weight.shape=}, {n_group_vcol=}"
        
        out_spatial_tile, out_reduce_tile, _, _, _ = converted_weight.shape
        converted_weight = converted_weight.reshape(out_spatial_tile, out_reduce_tile, -1)

        print(f"{converted_weight.shape=}, {converted_weight.dtype=}")
        # print(converted_weight)
        
        return converted_weight

    def _prepare_input_data(self):
        import numpy as np
        input_data = np.arange(0,self.in_hw*self.in_hw, dtype=np.int8).reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        assert input_data.shape==(self.in_hw,self.in_hw,self.in_channel), f"{input_data.shape=}"
        return input_data

    def _calculate_golden(self):
        import numpy as np
        output_h = output_w = self.out_hw
        output_c = self.out_channel

        output = np.zeros((output_h, output_w, output_c), dtype=np.int32)
        for row in range(output_h):
            for col in range(output_w):
                input = self.input_data[row:row+self.ker_size,col:col+self.ker_size,:].reshape(-1,1)
                weight = self.weight_data
                golden = np.matmul(weight.astype(np.int32), input.astype(np.int32))
                output[row,col,:] = golden.reshape(-1)
        return output

    def get_image(self, simulator):
        import numpy as np
        """
        
        """
        self.input_data = self._prepare_input_data()
        self.weight_data = self._prepare_weight_data()


        self.converted_weight = self._make_dense_data(self.weight_data, simulator)

        assert self.input_data.dtype==np.int8, f"{self.input_data.dtype=}"
        assert self.converted_weight.dtype==np.int8, f"{self.converted_weight.dtype=}"

        input_data = bytearray(self.input_data)
        converted_weight_bytes = bytearray(self.converted_weight)

        print(f"{self.input_data.shape=}, {self.input_data.dtype=}, byte_size={len(input_data)}")
        print(f"{self.converted_weight.shape=}, {self.converted_weight.dtype=}, byte_size={len(converted_weight_bytes)}")

        # import pdb; pdb.set_trace()
        image = input_data + converted_weight_bytes
        self.output_offset = len(image)

        return image
    
    def check_image(self, memory_space):
        import numpy as np
        """
        image should have:
        input (4 * 1 byte)
        weight (32 * 1 byte)
        output (8 * 4 byte)
        """
        global_offset = memory_space.get_base_of("global")
        output_offset = global_offset + self.output_offset
        output_byte_size = self.out_hw * self.out_hw * self.out_channel * 4
        output = memory_space.read_as(output_offset, output_byte_size, np.int32)
        output = output.reshape(self.out_hw, self.out_hw, self.out_channel)
        # output = np.frombuffer(image[self.output_offset: self.output_offset+6*6*32*4], dtype=np.int32)
        golden = self._calculate_golden()
        assert np.array_equal(output,golden), f"{output=}, {golden=}"

    def fill_template(self, src_path, dst_path, simulator):
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
        import os

        src_folder,src_file = os.path.split(src_path)

        # 创建 Jinja2 环境和加载器
        env = Environment(loader=FileSystemLoader(src_folder), undefined=StrictUndefined)

        # 加载模板
        template = env.get_template(src_file)

        macro_config = simulator.macro_config
        mask_config = simulator.mask_config

        # 准备模板上下文
        n_group = 4
        bitwidth = 8
        n_macro = macro_config.n_macro
        n_macro_per_group = n_macro // n_group
        n_vcol = macro_config.n_vcol(bitwidth)
        n_group_vcol = n_macro_per_group * n_vcol
        n_row = macro_config.n_row
        n_comp = macro_config.n_comp
        n_macro_reduce = n_row * n_comp
        mask_base = simulator.memory_space.get_base_of("pim_mask_data_reg_buffer")
        context = {
            'OUTPUT_CHANNEL': self.out_channel,
            'INPUT_ROW': self.in_hw,
            'INPUT_COL': self.in_hw,
            'INPUT_CHANNEL': self.in_channel,
            'OUTPUT_ROW': self.out_hw,
            'OUTPUT_COL': self.out_hw,
            'KERNEL_SIZE': self.ker_size,
            'PADDING': 0,
            'STRIDE': 1,

            'N_MACRO': macro_config.n_macro,
            'N_MACRO_PER_GROUP': n_macro_per_group,
            'N_VCOL': n_vcol,
            'N_GROUP_VCOL': n_group_vcol,
            'N_GROUP': n_group,
            'N_ROW': n_row,
            'N_COMP': n_comp,
            'N_MACRO_REDUCE': n_macro_reduce,
            'INPUT_BUFFER_SIZE_PER_GROUP': 128,
            'OUT_SPATIAL_TILE': self.converted_weight.shape[0],
            'OUT_REDUCE_TILE': self.converted_weight.shape[1],

        }

        # 渲染模板
        output = template.render(context)

        with open(dst_path, "w") as f:
            f.write(output)