class TestHelper:
    def _prepare_weight_data(self):
        import numpy as np
        """
        weight: 32 * 32 * 3 * 3
        input: 32 * 8 * 8
        """
        # make a weight
        weight = np.zeros((32, 3 * 3 * 32), dtype=np.int8)
        for i in range(0,32,2):
            weight[i:i+2, i:i+64] = np.arange(1+i,1+i+64, dtype=np.int8)

        # print(weight.shape)
        # print(weight)

        return weight

    def _make_value_sparse_data(self, weight, simulator):
        from data_processor.dense import convert_value_sparse_conv2d_weight

        macro_config = simulator.macro_config
        n_from = simulator.mask_config.n_from
        n_to = simulator.mask_config.n_to
        bitwidth = 8
        n_group = 4
        config = {
            "n_vcol": macro_config.n_bcol // bitwidth,
            "n_group": n_group,
            "n_macro": macro_config.n_macro,
            "n_comp": macro_config.n_comp,
            "n_value_sparse_from": n_from,
            "n_value_sparse_to": n_to
        }
        converted_weight, mask, index, tile_list = convert_value_sparse_conv2d_weight(weight, config)

        time, n_to, n_group, n_macro_per_group, n_vcol = converted_weight.shape
        converted_weight = converted_weight.reshape(time, n_to, n_group, -1)

        print(f"{converted_weight.shape=}, {converted_weight.dtype=}")
        # print(converted_weight)
        print(f"{index.shape=}, {index.dtype=}")
        # print(index)
        print(f"{tile_list.shape=}, {tile_list.dtype=}")
        # print(tile_list)
        # import pdb; pdb.set_trace()
        return converted_weight, mask, index, tile_list

    def _prepare_input_data(self):
        import numpy as np
        input_data = np.arange(0,64, dtype=np.int8).reshape(8,8,1).repeat(32, axis=2)
        assert input_data.shape==(8,8,32), f"{input_data.shape=}"
        return input_data

    def get_image(self, simulator):
        import numpy as np
        """
        
        """
        self.input_data = self._prepare_input_data()
        self.weight_data = self._prepare_weight_data()


        self.converted_weight, self.mask, self.index, self.tile_list = self._make_value_sparse_data(self.weight_data, simulator)

        assert self.input_data.dtype==np.int8, f"{self.input_data.dtype=}"
        assert self.converted_weight.dtype==np.int8, f"{self.converted_weight.dtype=}"
        assert self.mask.dtype==np.int8, f"{self.mask.dtype=}"
        assert self.index.dtype==np.int32, f"{self.index.dtype=}"
        assert self.tile_list.dtype==np.int32, f"{self.tile_list.dtype=}"

        assert (self.mask.sum(axis=2) <= simulator.mask_config.n_to).all(), f"{self.mask.sum(axis=2)=}, {simulator.mask_config.n_to=}"


        input_data = bytearray(self.input_data)
        converted_weight_bytes = bytearray(self.converted_weight)
        mask_bytes = bytearray(self.mask)
        index_bytes = bytearray(self.index)
        tile_list_bytes = bytearray(self.tile_list)

        print(f"{self.input_data.shape=}, {self.input_data.dtype=}, byte_size={len(input_data)}")
        print(f"{self.converted_weight.shape=}, {self.converted_weight.dtype=}, byte_size={len(converted_weight_bytes)}")
        print(f"{self.mask.shape=}, {self.mask.dtype=}, byte_size={len(mask_bytes)}")
        print(f"{self.index.shape=}, {self.index.dtype=}, byte_size={len(index_bytes)}")
        print(f"{self.tile_list.shape=}, {self.tile_list.dtype=}, byte_size={len(tile_list_bytes)}")
        # import pdb; pdb.set_trace()
        image = input_data + converted_weight_bytes + mask_bytes + index_bytes + tile_list_bytes
        # image = input_data + index_bytes + tile_list_bytes
        return image
    
    def check_image(self, image):
        import numpy as np
        """
        image should have:
        input (4 * 1 byte)
        weight (32 * 1 byte)
        output (8 * 4 byte)
        """
        return
        # output = np.frombuffer(image[36:68], dtype=np.int32)
        # golden = np.dot(self.input.astype(np.int32), self.weight.astype(np.int32))
        # assert np.array_equal(output,golden), f"{output=}, {golden=}"

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
            'OUTPUT_CHANNEL': 32,
            'INPUT_ROW': 8,
            'INPUT_COL': 8,
            'INPUT_CHANNEL': 32,
            'OUTPUT_ROW': 6,
            'OUTPUT_COL': 6,
            'KERNEL_SIZE': 3,
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
            'TIME': self.converted_weight.shape[0],
            'N_FROM':  mask_config.n_from,
            'N_TO':  mask_config.n_to,

            'INDEX_LENGTH': self.index.shape[0],
            'FROM_OVER_TO': mask_config.n_from // mask_config.n_to,
            'VALUE_SPARSE_MASK_BASE_ADDR': mask_base
        }

        # 渲染模板
        output = template.render(context)

        with open(dst_path, "w") as f:
            f.write(output)