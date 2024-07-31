class TestHelper:
    def __init__(self, op_config):
        self.out_channel = op_config["out_channel"]
        self.ker_size = op_config["ker_size"]
        self.in_channel = op_config["in_channel"]
        self.in_hw = op_config["in_hw"]
        self.out_hw = op_config["out_hw"]
        if "input_buffer_size_per_group" in op_config:
            self.input_buffer_size_per_group = op_config["input_buffer_size_per_group"]
            assert self.input_buffer_size_per_group % 16 == 0
            assert self.input_buffer_size_per_group <= 128
        else:
            self.input_buffer_size_per_group = 128
            
    def _prepare_weight_data(self):
        import numpy as np
        """
        weight: 32 * 32 * 3 * 3
        input: 32 * 8 * 8
        """
        # make a weight
        # weight = np.arange(self.ker_size * self.ker_size, dtype=np.int8).reshape(1, -1).repeat(self.out_channel * self.in_channel, axis=0)
        # weight = np.ones((self.out_channel, self.ker_size * self.ker_size * self.in_channel), dtype=np.int8)
        # weight[:,:16] = 0
        # weight[0,:16] = 0
        # weight[1,1:17] = 0
        # weight[2,2:18] = 0
        # weight[3,3:19] = 0
        # weight = weight.reshape(self.out_channel, self.ker_size * self.ker_size * self.in_channel)
        # weight[:,::2] = 0
        # for i in range(0,self.out_channel,2):
        #     weight[i:i+2, i:i+99*2:2] = np.arange(1+i,1+i+99, dtype=np.int8)
            # weight[i:i+2, 256:3*3*32] = np.arange(0,3*3*32-256, dtype=np.int8)

        # print(weight.shape)
        # print(weight)
        weight = np.random.randint(-8, 8, size=(self.out_channel, self.ker_size * self.ker_size * self.in_channel), dtype=np.int8)
        import random
        np.random.seed(4)
        random.seed(4)
        mask = np.random.randint(-1, 2, size=(self.out_channel, self.ker_size * self.ker_size * self.in_channel), dtype=np.int8)
        weight = weight * mask
        mask = np.random.randint(-1, 2, size=(self.out_channel, self.ker_size * self.ker_size * self.in_channel), dtype=np.int8)
        weight = weight * mask
        mask = np.random.randint(-1, 2, size=(self.out_channel, self.ker_size * self.ker_size * self.in_channel), dtype=np.int8)
        weight = weight * mask
        return weight

    def _make_value_sparse_data(self, weight, simulator):
        from data_processor.dense import convert_value_sparse_conv2d_weight

        macro_config = simulator.macro_config
        n_from = simulator.mask_config.n_from
        n_to = simulator.mask_config.n_to
        bitwidth = 8
        n_group = 4
        config = {
            "n_row": macro_config.n_row,
            "n_vcol": macro_config.n_bcol // bitwidth,
            "n_group": n_group,
            "n_macro": macro_config.n_macro,
            "n_comp": macro_config.n_comp,
            "n_value_sparse_from": n_from,
            "n_value_sparse_to": n_to
        }
        result = convert_value_sparse_conv2d_weight(weight, config)
        
        converted_weight = result["converted_weight"]
        time, n_to, n_group, n_macro_per_group, n_vcol = converted_weight.shape
        converted_weight = converted_weight.reshape(time, n_to, n_group, -1)
        result["converted_weight"] = converted_weight

        print(f"{converted_weight.shape=}, {converted_weight.dtype=}")
        # print(converted_weight)
        # print(f"{index.shape=}, {index.dtype=}")
        # print(index)
        # print(f"{tile_list.shape=}, {tile_list.dtype=}")
        # print(tile_list)
        # print(f"mask: {mask.shape=}, {mask.dtype=}")
        # print(mask)
        # import pdb; pdb.set_trace()
        return result

    def _prepare_input_data(self):
        import numpy as np
        input_data = np.arange(1,self.in_hw*self.in_hw+1, dtype=np.int8).reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
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
        from utils.df_layout import tensor_bits_to_int8
        """
        
        """
        self.input_data = self._prepare_input_data()
        self.weight_data = self._prepare_weight_data()


        result = self._make_value_sparse_data(self.weight_data, simulator)
        self.converted_weight = result["converted_weight"]
        self.mask = result["mask"]
        self.mapping_reduce_to_macro = result["mapping_reduce_to_macro"]
        self.mapping_macro_to_from = result["mapping_macro_to_from"]
        self.mapping_from_to_row = result["mapping_from_to_row"]
        self.mapping_macro_to_row = result["mapping_macro_to_row"]

        print(f"{self.mapping_reduce_to_macro=}")
        print(f"{self.mapping_macro_to_from=}")
        print(f"{self.mapping_from_to_row=}")
        print(f"{self.mapping_macro_to_row=}")

        assert self.input_data.dtype==np.int8, f"{self.input_data.dtype=}"
        assert self.converted_weight.dtype==np.int8, f"{self.converted_weight.dtype=}"
        assert self.mask.dtype==np.int8, f"{self.mask.dtype=}"
        assert self.mapping_reduce_to_macro.dtype==np.int32, f"{self.mapping_reduce_to_macro.dtype=}"
        assert self.mapping_macro_to_from.dtype==np.int32, f"{self.mapping_macro_to_from.dtype=}"
        assert self.mapping_from_to_row.dtype==np.int32, f"{self.mapping_from_to_row.dtype=}"
        assert self.mapping_macro_to_row.dtype==np.int32, f"{self.mapping_macro_to_row.dtype=}"

        assert (self.mask.sum(axis=2) <= simulator.mask_config.n_to).all(), f"{self.mask.sum(axis=2)=}, {simulator.mask_config.n_to=}"

        assert self.mask.shape[-1] % 8 == 0, f"{self.mask.shape=}"
        mask_bits = self.mask.reshape(*self.mask.shape[:2], self.mask.shape[-1]//8, 8)
        mask_bits = tensor_bits_to_int8(mask_bits)

        input_data = bytearray(self.input_data)
        converted_weight_bytes = bytearray(self.converted_weight)
        mask_bytes = bytearray(mask_bits)
        mapping_reduce_to_macro_bytes = bytearray(self.mapping_reduce_to_macro)
        mapping_macro_to_from_bytes = bytearray(self.mapping_macro_to_from)
        mapping_from_to_row_bytes = bytearray(self.mapping_from_to_row)
        mapping_macro_to_row_bytes = bytearray(self.mapping_macro_to_row)

        print(f"{self.input_data.shape=}, {self.input_data.dtype=}, byte_size={len(input_data)}")
        print(f"{self.converted_weight.shape=}, {self.converted_weight.dtype=}, byte_size={len(converted_weight_bytes)}")
        print(f"{self.mask.shape=}, {self.mask.dtype=}, byte_size={len(mask_bytes)}")
        print(f"{self.mapping_reduce_to_macro.shape=}, {self.mapping_reduce_to_macro.dtype=}, byte_size={len(mapping_reduce_to_macro_bytes)}")
        print(f"{self.mapping_macro_to_from.shape=}, {self.mapping_macro_to_from.dtype=}, byte_size={len(mapping_macro_to_from_bytes)}")
        print(f"{self.mapping_from_to_row.shape=}, {self.mapping_from_to_row.dtype=}, byte_size={len(mapping_from_to_row_bytes)}")
        print(f"{self.mapping_macro_to_row.shape=}, {self.mapping_macro_to_row.dtype=}, byte_size={len(mapping_macro_to_row_bytes)}")
        # import pdb; pdb.set_trace()
        image = (input_data 
            + converted_weight_bytes 
            + mask_bytes 
            + mapping_reduce_to_macro_bytes 
            + mapping_macro_to_from_bytes
            + mapping_macro_to_row_bytes
            + mapping_from_to_row_bytes
        )
        self.output_offset = len(image)
        # image = input_data + index_bytes + tile_list_bytes
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
            'INPUT_BUFFER_SIZE_PER_GROUP': self.input_buffer_size_per_group,
            'TIME': self.converted_weight.shape[0],
            'N_FROM':  mask_config.n_from,
            'N_TO':  mask_config.n_to,

            # 'FROM_OVER_TO': mask_config.n_from // mask_config.n_to,
            'VALUE_SPARSE_MASK_BASE_ADDR': mask_base,
            'MAPPING_REDUCE_TO_MACRO_LENGTH': self.mapping_reduce_to_macro.shape[0],
            'MAPPING_MACRO_TO_FROM_LENGTH': self.mapping_macro_to_from.shape[0],
            'MAPPING_MACRO_TO_ROW_LENGTH': self.mapping_macro_to_row.shape[0],
            'MAPPING_FROM_TO_ROW_LENGTH': self.mapping_from_to_row.shape[0],
        }

        # 渲染模板
        output = template.render(context)

        with open(dst_path, "w") as f:
            f.write(output)