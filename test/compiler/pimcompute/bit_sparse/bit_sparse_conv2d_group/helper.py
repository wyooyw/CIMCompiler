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
        from utils.bit_sparse_weight_transform import generate_valid_weight

        weight = generate_valid_weight([self.out_channel, self.ker_size, self.ker_size, self.in_channel])
        return weight

    def _get_mock_input(self):
        import numpy as np
        # input_data = np.arange(0,self.in_hw*self.in_hw, dtype=np.int8).reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        # input_data = np.random.randint(-32,32,size=(self.in_hw,self.in_hw), dtype=np.int8).reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        input_data = np.ones((self.in_hw,self.in_hw), dtype=np.int8).reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        assert input_data.shape==(self.in_hw,self.in_hw,self.in_channel), f"{input_data.shape=}"
        return input_data

    def _make_bit_sparse_data(self, weight, simulator):
        from utils.bit_sparse_weight_transform import weight_transform_group, outsum_mask_to_transfer_mask, parse_out_mask, parse_out_mask_and_transfer_mask, parse_out_begin_channel
        from types import SimpleNamespace
        from utils.df_layout import tensor_bits_to_int8
        import numpy as np

        macro_config = simulator.macro_config
        bitwidth = 8
        n_group = 4
        n_comp = macro_config.n_comp
        n_bcol = macro_config.n_bcol
        n_macro_per_group = macro_config.n_macro // n_group
        n_group_bcol = n_macro_per_group * n_bcol
        op_cfg = SimpleNamespace(
            out_channel = weight.shape[0],
        )
        cim_cfg = SimpleNamespace(
            bits_column=n_bcol,
            n_macro=simulator.macro_config.n_macro // n_group,   # n_macro_per_group
            n_group=n_group
        )
        bit_sparse_weight, meta, fold = weight_transform_group(weight, cim_cfg, op_cfg, "OHWI")
        ele_in_filter = bit_sparse_weight.shape[1]
        assert ele_in_filter % n_comp == 0
        bit_sparse_weight = bit_sparse_weight.reshape(
            bit_sparse_weight.shape[0], 
            ele_in_filter // n_comp,
            n_comp,
            bit_sparse_weight.shape[2],
            bit_sparse_weight.shape[3]
            )
        outsum_mask, transfer_mask = parse_out_mask_and_transfer_mask(fold, n_group_bcol)
        out_begin_channel = parse_out_begin_channel(fold)

        assert len(bit_sparse_weight.shape)==5, f"{bit_sparse_weight.shape=}"
        assert bit_sparse_weight.shape[2]==macro_config.n_comp, f"{bit_sparse_weight.shape=}, {macro_config.n_comp=}"
        assert bit_sparse_weight.shape[3]==n_group, f"{bit_sparse_weight.shape=}, {n_group=}"
        assert bit_sparse_weight.shape[4]==(n_group_bcol // 8), f"{bit_sparse_weight.shape=}, {(n_group_bcol//8)=}"
        assert outsum_mask.shape[1]==(n_group_bcol//8), f"{outsum_mask.shape=}, {(n_group_bcol//8)=}"
        assert transfer_mask.shape[1]==(n_group_bcol//8), f"{outsum_mask.shape=}, {(n_group_bcol//8)=}"
        assert outsum_mask.shape[0]==bit_sparse_weight.shape[0], f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"
        assert transfer_mask.shape[0]==bit_sparse_weight.shape[0], f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"
        
        assert bit_sparse_weight.dtype==np.int8, f"{bit_sparse_weight.dtype=}"
        assert outsum_mask.dtype==np.int8, f"{outsum_mask.dtype=}"
        assert transfer_mask.dtype==np.int8, f"{transfer_mask.dtype=}"
        assert meta.dtype==np.int8, f"{meta.dtype=}"

        assert len(out_begin_channel.shape)==1
        assert out_begin_channel.shape[0]==bit_sparse_weight.shape[0] + 1, f"{out_begin_channel.shape=}, {bit_sparse_weight.shape=}"
        assert out_begin_channel[-1] == self.out_channel, f"{out_begin_channel[-1]=}, {self.out_channel=}"
        assert out_begin_channel.dtype==np.int32, f"{out_begin_channel.dtype=}"
        
        return bit_sparse_weight, meta, outsum_mask, transfer_mask, out_begin_channel

    def _calculate_golden(self):
        import numpy as np
        output_h = output_w = self.out_hw
        output_c = self.out_channel
        weight = self.weight_data.reshape(self.out_channel, -1)
        output = np.zeros((output_h, output_w, output_c), dtype=np.int32)
        for row in range(output_h):
            for col in range(output_w):
                input = self.input_data[row:row+self.ker_size,col:col+self.ker_size,:].reshape(-1,1)
                # weight = self.weight_data
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

        if weight is None:
            self.weight_data = self._get_mock_weight()
        else:
            self.weight_data = weight
        assert self.weight_data.shape[0]

        if input is None:
            self.input_data = self._get_mock_input()
        else:
            self.input_data = input

        self._assert_check_input_and_weight_shape(self.input_data, self.weight_data)
        self.input_data = self._apply_padding(self.input_data)

        bit_sparse_weight, meta, outsum_mask, transfer_mask, out_begin_channel = self._make_bit_sparse_data(self.weight_data, simulator)

        self.converted_weight = bit_sparse_weight
        self.meta = meta
        self.outsum_mask = outsum_mask
        self.transfer_mask = transfer_mask
        self.out_begin_channel = out_begin_channel

        input_data = bytearray(self.input_data)
        converted_weight_bytes = bytearray(self.converted_weight)
        meta_bytes = bytearray(self.meta)
        outsum_offset_bytes = bytearray(self.outsum_mask)
        transfer_offset_bytes = bytearray(self.transfer_mask)
        out_begin_channel_bytes = bytearray(self.out_begin_channel)

        print(f"{self.input_data.shape=}, {self.input_data.dtype=}, byte_size={len(input_data)}")
        print(f"{self.converted_weight.shape=}, {self.converted_weight.dtype=}, byte_size={len(converted_weight_bytes)}")
        print(f"{self.meta.shape=}, {self.meta.dtype=}, byte_size={len(meta_bytes)}")
        print(f"{self.outsum_mask.shape=}, {self.outsum_mask.dtype=}, byte_size={len(outsum_offset_bytes)}")
        print(f"{self.transfer_mask.shape=}, {self.transfer_mask.dtype=}, byte_size={len(transfer_offset_bytes)}")
        print(f"{self.out_begin_channel.shape=}, {self.out_begin_channel.dtype=}, byte_size={len(out_begin_channel_bytes)}")

        # import pdb; pdb.set_trace()
        image = (
            input_data 
            + converted_weight_bytes 
            + meta_bytes 
            + outsum_offset_bytes 
            + transfer_offset_bytes
            + out_begin_channel_bytes
        )
        
        self.output_offset = len(image)

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
        print("Check image pass")

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
        n_bcol = macro_config.n_bcol
        n_group_vcol = n_macro_per_group * n_vcol
        n_group_bcol = n_macro_per_group * n_bcol
        n_row = macro_config.n_row
        n_comp = macro_config.n_comp
        n_macro_reduce = n_row * n_comp
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
            'OUT_SPATIAL_TILE': self.converted_weight.shape[0],
            'OUT_REDUCE_TILE': self.converted_weight.shape[1],

            'BIT_SPARSE_META_BASE_ADDR': meta_base,
            "OUT_CHANNEL_BEGIN_LEN": self.out_begin_channel.shape[0],

        }
        # print(f"{self.out_begin_channel=}")
        # print(f"{self.out_begin_channel.shape=}")
        # print(f"{self.converted_weight.shape=}")
        # import pdb; pdb.set_trace()

        # 渲染模板
        output = template.render(context)

        with open(dst_path, "w") as f:
            f.write(output)