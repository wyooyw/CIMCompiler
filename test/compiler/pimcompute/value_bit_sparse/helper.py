class TestHelper:
    def __init__(self, op_config):
        self.out_channel = op_config["out_channel"]
        self.ker_size = op_config["ker_size"]
        self.in_channel = op_config["in_channel"]
        self.in_hw = op_config["in_hw"]
        self.out_hw = op_config["out_hw"]
        
        if "padding" in op_config:
            self.padding = op_config["padding"]
        else:
            self.padding = 0

        if "input_buffer_size_per_group" in op_config:
            self.input_buffer_size_per_group = op_config["input_buffer_size_per_group"]
            assert self.input_buffer_size_per_group % 16 == 0
            assert self.input_buffer_size_per_group <= 128
        else:
            self.input_buffer_size_per_group = 128
            
    def _get_mock_weight(self):
        import numpy as np
        """
        weight: 32 * 32 * 3 * 3
        input: 32 * 8 * 8
        """
        
        from utils.bit_sparse_weight_transform import generate_valid_weight
        import random
        np.random.seed(4)
        random.seed(4)

        # weight = generate_valid_weight([self.out_channel, self.ker_size, self.ker_size, self.in_channel], threshold=2)
        # weight = np.zeros((self.out_channel, self.ker_size, self.ker_size, self.in_channel), dtype=np.int8) + 3
        weight = generate_valid_weight([self.out_channel, self.ker_size, self.ker_size, self.in_channel], threshold=2)
        for filter_begin in range(0, self.out_channel, 8):
            filter_end = min(filter_begin+8, self.out_channel)
            mask = np.random.randint(0, 2, size=weight.shape[1:], dtype=np.int8)
            weight[filter_begin:filter_end, :, :, :] = weight[filter_begin:filter_end, :, :, :] * mask
            mask = np.random.randint(0, 2, size=weight.shape[1:], dtype=np.int8)
            weight[filter_begin:filter_end, :, :, :] = weight[filter_begin:filter_end, :, :, :] * mask
            mask = np.random.randint(0, 2, size=weight.shape[1:], dtype=np.int8)
            weight[filter_begin:filter_end, :, :, :] = weight[filter_begin:filter_end, :, :, :] * mask
        
        # weight = weight * mask
        # mask = np.random.randint(0, 2, size=weight.shape, dtype=np.int8)
        # weight = weight * mask
        # mask = np.random.randint(0, 2, size=weight.shape, dtype=np.int8)
        # weight = weight * mask
        # weight[:,:,:,24:] = 0
        return weight

    def _get_mock_input(self):
        import numpy as np
        input_data = np.arange(1,self.in_hw*self.in_hw+1, dtype=np.int8).reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        # input_data = np.ones((self.in_hw,self.in_hw,self.in_channel)).astype(np.int8)
        assert input_data.shape==(self.in_hw,self.in_hw,self.in_channel), f"{input_data.shape=}"
        return input_data

    def _make_value_bit_sparse_data(self, weight, simulator):
        from utils.bit_value_sparse_weight_transform import convert_value_bit_sparse_conv2d_weight

        macro_config = simulator.macro_config
        n_from = simulator.mask_config.n_from
        n_to = simulator.mask_config.n_to
        bitwidth = 8
        n_group = 4
        config = {
            "n_row": macro_config.n_row,
            "n_bcol": macro_config.n_bcol,
            "n_group": n_group,
            "n_macro": macro_config.n_macro,
            "n_comp": macro_config.n_comp,
            "n_value_sparse_from": n_from,
            "n_value_sparse_to": n_to
        }
        result = convert_value_bit_sparse_conv2d_weight(weight, config)
        
        # converted_weight = result["value_bit_sparse_weight"]
        # time, n_to, n_group, n_macro_per_group, n_vcol = converted_weight.shape
        # converted_weight = converted_weight.reshape(time, n_to, n_group, -1)
        # result["converted_weight"] = converted_weight

        # print(f"{converted_weight.shape=}, {converted_weight.dtype=}")
        # print(converted_weight)
        # print(f"{index.shape=}, {index.dtype=}")
        # print(index)
        # print(f"{tile_list.shape=}, {tile_list.dtype=}")
        # print(tile_list)
        # print(f"mask: {mask.shape=}, {mask.dtype=}")
        # print(mask)
        # import pdb; pdb.set_trace()
        return result

    def _calculate_golden(self):
        import numpy as np
        output_h = output_w = self.out_hw
        output_c = self.out_channel
        weight = self.weight_data.reshape(self.weight_data.shape[0], -1)
        output = np.zeros((output_h, output_w, output_c), dtype=np.int32)
        for row in range(output_h):
            for col in range(output_w):
                input = self.input_data[row:row+self.ker_size,col:col+self.ker_size,:].reshape(-1,1)
                
                golden = np.matmul(weight.astype(np.int32), input.astype(np.int32))
                output[row,col,:] = golden.reshape(-1)
        return output

    def _assert_check_input_and_weight_shape(self, input, weight):
        """
        assert input.shape is [in_hw, in_hw, in_channel]
        assert weight.shape is [out_channel, in_channel, ker_size, ker_size]
        """
        assert len(input.shape)==3, f"{input.shape=}"
        assert input.shape[0]==input.shape[1] and input.shape[0]==self.in_hw, f"{input.shape=}"
        assert input.shape[2]==self.in_channel, f"{input.shape=}"

        assert len(weight.shape)==4, f"{weight.shape=}"
        assert weight.shape[0]==self.out_channel, f"{weight.shape=}"
        assert weight.shape[3]==self.in_channel, f"{weight.shape=}"
        assert weight.shape[1]==weight.shape[2] and weight.shape[2]==self.ker_size, f"{weight.shape=}"
        
    def _apply_padding(self, input_data):
        """
        input.shape: H,W,C
        """
        import numpy as np
        print("apply_padding")
        input_data = np.pad(input_data, ((self.padding,self.padding),(self.padding,self.padding),(0,0)), mode='constant', constant_values=0)
        return input_data

    def get_image(self, simulator, input=None, weight=None):
        import numpy as np
        from utils.df_layout import tensor_bits_to_int8
        """
        
        """
        if input is None:
            self.input_data = self._get_mock_input()
        else:
            self.input_data = input

        if weight is None:
            self.weight_data = self._get_mock_weight()
        else:
            self.weight_data = weight

        self._assert_check_input_and_weight_shape(self.input_data, self.weight_data)
        self.input_data = self._apply_padding(self.input_data)


        result = self._make_value_bit_sparse_data(self.weight_data, simulator)
        converted_weight = result["value_bit_sparse_weight"]
        self.converted_weight = converted_weight
        value_sparse_result = result["value_sparse_result"]
        bit_sparse_result = result["bit_sparse_result"]

        mask = value_sparse_result["mask"]
        self.mapping_reduce_to_macro = value_sparse_result["mapping_reduce_to_macro"]
        self.mapping_macro_to_from = value_sparse_result["mapping_macro_to_from"]
        self.mapping_from_to_row = value_sparse_result["mapping_from_to_row"]
        self.mapping_macro_to_row = value_sparse_result["mapping_macro_to_row"]

        meta = bit_sparse_result["meta"]
        outsum_mask = bit_sparse_result["outsum_mask"]
        transfer_mask = bit_sparse_result["transfer_mask"]

        print(f"{self.mapping_reduce_to_macro=}")
        print(f"{self.mapping_macro_to_from=}")
        print(f"{self.mapping_from_to_row=}")
        print(f"{self.mapping_macro_to_row=}")

        assert self.input_data.dtype==np.int8, f"{self.input_data.dtype=}"
        assert converted_weight.dtype==np.int8, f"{converted_weight.dtype=}"
        assert mask.dtype==np.int8, f"{mask.dtype=}"
        assert self.mapping_reduce_to_macro.dtype==np.int32, f"{self.mapping_reduce_to_macro.dtype=}"
        assert self.mapping_macro_to_from.dtype==np.int32, f"{self.mapping_macro_to_from.dtype=}"
        assert self.mapping_from_to_row.dtype==np.int32, f"{self.mapping_from_to_row.dtype=}"
        assert self.mapping_macro_to_row.dtype==np.int32, f"{self.mapping_macro_to_row.dtype=}"
        assert (mask.sum(axis=2) <= simulator.mask_config.n_to).all(), f"{mask.sum(axis=2)=}, {simulator.mask_config.n_to=}"

        macro_config = simulator.macro_config
        bitwidth = 8
        n_group = 4
        n_comp = macro_config.n_comp
        n_bcol = macro_config.n_bcol
        n_macro_per_group = macro_config.n_macro // n_group
        n_group_bcol = n_macro_per_group * n_bcol

        assert outsum_mask.shape[1]==(n_group_bcol), f"{outsum_mask.shape=}, {(n_group_bcol)=}"
        assert transfer_mask.shape[1]==(n_group_bcol), f"{outsum_mask.shape=}, {(n_group_bcol)=}"
        assert outsum_mask.shape[0]==self.mapping_reduce_to_macro.shape[0], f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"
        assert transfer_mask.shape[0]==self.mapping_reduce_to_macro.shape[0], f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"

        assert mask.shape[-1] % 8 == 0, f"{mask.shape=}"
        mask_bits = mask.reshape(*mask.shape[:-1], mask.shape[-1]//8, 8)
        mask_bits = tensor_bits_to_int8(mask_bits)
        assert outsum_mask.shape[-1] % 8 == 0, f"{outsum_mask.shape=}"
        outsum_mask_bits = outsum_mask.reshape(*outsum_mask.shape[:-1], outsum_mask.shape[-1]//8, 8)
        outsum_mask_bits = tensor_bits_to_int8(outsum_mask_bits)
        assert transfer_mask.shape[-1] % 8 == 0, f"{transfer_mask.shape=}"
        transfer_mask_bits = transfer_mask.reshape(*transfer_mask.shape[:-1], transfer_mask.shape[-1]//8, 8)
        transfer_mask_bits = tensor_bits_to_int8(transfer_mask_bits)

        input_data = bytearray(self.input_data)
        converted_weight_bytes = bytearray(converted_weight)
        mask_bytes = bytearray(mask_bits)
        mapping_reduce_to_macro_bytes = bytearray(self.mapping_reduce_to_macro)
        mapping_macro_to_from_bytes = bytearray(self.mapping_macro_to_from)
        mapping_from_to_row_bytes = bytearray(self.mapping_from_to_row)
        mapping_macro_to_row_bytes = bytearray(self.mapping_macro_to_row)
        meta_bytes = bytearray(meta)
        outsum_offset_bytes = bytearray(outsum_mask_bits)
        transfer_offset_bytes = bytearray(transfer_mask_bits)

        print(f"{self.input_data.shape=}, {self.input_data.dtype=}, byte_size={len(input_data)}")
        print(f"{converted_weight.shape=}, {converted_weight.dtype=}, byte_size={len(converted_weight_bytes)}")
        print(f"{mask.shape=}, {mask.dtype=}, byte_size={len(mask_bytes)}")
        print(f"{self.mapping_reduce_to_macro.shape=}, {self.mapping_reduce_to_macro.dtype=}, byte_size={len(mapping_reduce_to_macro_bytes)}")
        print(f"{self.mapping_macro_to_from.shape=}, {self.mapping_macro_to_from.dtype=}, byte_size={len(mapping_macro_to_from_bytes)}")
        print(f"{self.mapping_from_to_row.shape=}, {self.mapping_from_to_row.dtype=}, byte_size={len(mapping_from_to_row_bytes)}")
        print(f"{self.mapping_macro_to_row.shape=}, {self.mapping_macro_to_row.dtype=}, byte_size={len(mapping_macro_to_row_bytes)}")
        print(f"{meta.shape=}, {meta.dtype=}, byte_size={len(meta_bytes)}")
        print(f"{outsum_mask.shape=}, {outsum_mask.dtype=}, byte_size={len(outsum_offset_bytes)}")
        # print(f"{transfer_mask.shape=}, {transfer_mask.dtype=}, byte_size={len(transfer_offset_bytes)}")
        # import pdb; pdb.set_trace()
        data_for_test = np.array([12345], dtype=np.int32)
        data_for_test_bytes = bytearray(data_for_test)
        image = (input_data 
            + converted_weight_bytes 
            + mask_bytes 
            + mapping_reduce_to_macro_bytes 
            + mapping_macro_to_from_bytes
            + mapping_macro_to_row_bytes
            + mapping_from_to_row_bytes
            + meta_bytes
            + outsum_offset_bytes
            + transfer_offset_bytes
        )
        self.output_offset = len(image)
        # image = input_data + index_bytes + tile_list_bytes
        return image

    def get_output(self, memory_space):
        import numpy as np

        global_offset = memory_space.get_base_of("global")
        output_offset = global_offset + self.output_offset
        output_byte_size = self.out_hw * self.out_hw * self.out_channel * 4
        output = memory_space.read_as(output_offset, output_byte_size, np.int32)
        output = output.reshape(self.out_hw, self.out_hw, self.out_channel)
        return output
    
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
        n_group_bcol = n_macro_per_group * macro_config.n_bcol
        n_row = macro_config.n_row
        n_comp = macro_config.n_comp
        n_macro_reduce = n_row * n_comp
        mask_base = simulator.memory_space.get_base_of("pim_mask_data_reg_buffer")
        meta_base = simulator.memory_space.get_base_of("pim_meta_data_reg_buffer")
        in_hw_padding = self.in_hw + 2 * self.padding
        context = {
            'OUTPUT_CHANNEL': self.out_channel,
            'INPUT_ROW': in_hw_padding,
            'INPUT_COL': in_hw_padding,
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
            'N_GROUP_BCOL': n_group_bcol,
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

            'BIT_SPARSE_META_BASE_ADDR': meta_base,
            'OUT_SPATIAL_TILE': self.mapping_reduce_to_macro.shape[0],
            'N_FILTER_PER_GROUP': 128


        }

        # 渲染模板
        output = template.render(context)

        with open(dst_path, "w") as f:
            f.write(output)