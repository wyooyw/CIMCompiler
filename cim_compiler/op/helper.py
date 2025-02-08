import numpy as np


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
        assert False, "Not implemented"

    def _get_mock_input(self):
        assert False, "Not implemented"

    def _calculate_golden(self):
        assert False, "Not implemented"

    def _assert_check_input_and_weight_shape(self, input, weight):
        """
        assert input.shape is [in_hw, in_hw, in_channel]
        assert weight.shape is [out_channel, ker_size, ker_size, in_channel]
        """
        assert len(input.shape) == 3, f"{input.shape=}"
        assert (
            input.shape[0] == input.shape[1] and input.shape[0] == self.in_hw
        ), f"{input.shape=}"
        assert input.shape[2] == self.in_channel, f"{input.shape=}"

        assert len(weight.shape) == 4, f"{weight.shape=}"
        assert weight.shape[0] == self.out_channel, f"{weight.shape=}"
        assert weight.shape[3] == self.in_channel, f"{weight.shape=}"
        assert (
            weight.shape[1] == weight.shape[2] and weight.shape[2] == self.ker_size
        ), f"{weight.shape=}"

    def _apply_padding(self, input_data):
        """
        input.shape: H,W,C
        """
        import numpy as np

        input_data = np.pad(
            input_data,
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return input_data

    def get_image(self, simulator, **image_kwargs):
        assert False, "Not implemented"

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
            self.out_hw * self.out_hw * self.out_channel * self.output_bytes
        )
        output = memory_space.read_as(
            output_offset, output_byte_size, self.output_dtype
        )
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
        output = self.get_output(memory_space)
        golden = self._calculate_golden()
        assert np.array_equal(output, golden), f"{output=}, {golden=}"

    def fill_template(self, src_path, dst_path, simulator):
        assert False, "Not implemented"


class Conv2dTestHelper(TestHelper):
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

        if "stride" in op_config:
            self.stride = op_config["stride"]
        else:
            self.stride = 1

        should_output_hw = (
            self.in_hw + 2 * self.padding - self.ker_size
        ) // self.stride + 1
        assert should_output_hw == self.out_hw, f"{should_output_hw=}, {self.out_hw=}"

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
        # make a weight
        weight = np.random.randint(
            -100,
            100,
            size=(self.out_channel, self.ker_size, self.ker_size, self.in_channel),
            dtype=np.int8,
        )

        return weight

    def _get_mock_input(self):
        import numpy as np

        input_data = np.random.randint(
            -126, 126, size=(self.in_hw, self.in_hw, self.in_channel), dtype=np.int8
        )  # .reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        # input_data = np.ones((self.in_hw,self.in_hw, self.in_channel), dtype=np.int8)# .reshape(self.in_hw,self.in_hw,1).repeat(self.in_channel, axis=2)
        assert input_data.shape == (
            self.in_hw,
            self.in_hw,
            self.in_channel,
        ), f"{input_data.shape=}"
        return input_data

    def _calculate_golden(self):
        import numpy as np

        output_h = output_w = self.out_hw
        output_c = self.out_channel

        output = np.zeros((output_h, output_w, output_c), dtype=np.int32)
        weight = self.weight_data.reshape(self.weight_data.shape[0], -1)
        stride = self.stride
        for row in range(output_h):
            for col in range(output_w):
                input = self.input_data[
                    stride * row : stride * row + self.ker_size,
                    stride * col : stride * col + self.ker_size,
                    :,
                ].reshape(-1, 1)
                golden = np.matmul(weight.astype(np.int32), input.astype(np.int32))
                output[row, col, :] = golden.reshape(-1)
        return output

    def _assert_check_input_and_weight_shape(self, input, weight):
        """
        assert input.shape is [in_hw, in_hw, in_channel]
        assert weight.shape is [out_channel, ker_size, ker_size, in_channel]
        """
        assert len(input.shape) == 3, f"{input.shape=}"
        assert (
            input.shape[0] == input.shape[1] and input.shape[0] == self.in_hw
        ), f"{input.shape=}"
        assert input.shape[2] == self.in_channel, f"{input.shape=}"

        assert len(weight.shape) == 4, f"{weight.shape=}"
        assert weight.shape[0] == self.out_channel, f"{weight.shape=}"
        assert weight.shape[3] == self.in_channel, f"{weight.shape=}"
        assert (
            weight.shape[1] == weight.shape[2] and weight.shape[2] == self.ker_size
        ), f"{weight.shape=}"

    def _apply_padding(self, input_data):
        """
        input.shape: H,W,C
        """
        import numpy as np

        input_data = np.pad(
            input_data,
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return input_data

    def _apply_im2col(self, input_data):
        import numpy as np

        in_hw, _, in_channel = input_data.shape
        ker_size = self.ker_size
        stride = self.stride
        out_hw = self.out_hw
        im2col_input = np.zeros((out_hw, in_hw, ker_size, in_channel), dtype=np.int8)
        for ow in range(out_hw):
            iw = ow * stride
            for ih in range(in_hw):
                im2col_input[ow, ih, :, :] = input_data[ih, iw : iw + ker_size, :]
        im2col_input = im2col_input.reshape(out_hw, -1)
        return im2col_input

    def get_image(self, simulator, **image_kwargs):
        assert False, "Not implemented"

    def fill_template(self, src_path, dst_path, simulator):
        assert False, "Not implemented"


class DenseConv2dTestHelper(Conv2dTestHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        self.output_bytes = 4
        self.output_dtype = np.int32
        self.im2col = False

    def _make_dense_data(self, weight, simulator):
        from cim_compiler.data_processor.dense import convert_dense_conv2d_weight

        macro_config = simulator.macro_config
        bitwidth = 8
        n_group = macro_config.n_group
        n_vcol = macro_config.n_bcol // bitwidth
        n_macro_per_group = macro_config.n_macro_per_group
        n_group_vcol = n_macro_per_group * n_vcol
        config = {
            "n_vcol": n_vcol,
            "n_group": n_group,
            "n_macro": macro_config.n_macro,
            "n_comp": macro_config.n_comp,
        }
        converted_weight, pimset_mask = convert_dense_conv2d_weight(weight, config)

        assert len(converted_weight.shape) == 5, f"{converted_weight.shape=}"
        assert (
            converted_weight.shape[2] == macro_config.n_comp
        ), f"{converted_weight.shape=}, {macro_config.n_comp=}"
        assert (
            converted_weight.shape[3] == n_group
        ), f"{converted_weight.shape=}, {n_group=}"
        assert (
            converted_weight.shape[4] == n_group_vcol
        ), f"{converted_weight.shape=}, {n_group_vcol=}"

        out_spatial_tile, out_reduce_tile, _, _, _ = converted_weight.shape
        converted_weight = converted_weight.reshape(
            out_spatial_tile, out_reduce_tile, -1
        )

        print(f"{converted_weight.shape=}, {converted_weight.dtype=}")
        # print(converted_weight)

        return converted_weight, pimset_mask

    def get_image(self, simulator, input=None, weight=None):
        import numpy as np

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
        if self.im2col:
            self.input_data_im2col = self._apply_im2col(self.input_data)

        self.converted_weight, self.pimset_mask = self._make_dense_data(
            self.weight_data, simulator
        )

        assert self.input_data.dtype == np.int8, f"{self.input_data.dtype=}"
        assert self.converted_weight.dtype == np.int8, f"{self.converted_weight.dtype=}"

        if self.im2col:
            input_data = bytearray(self.input_data_im2col)
        else:
            input_data = bytearray(self.input_data)
        converted_weight_bytes = bytearray(self.converted_weight)
        pimset_mask_bytes = bytearray(self.pimset_mask)


        # import pdb; pdb.set_trace()
        image = input_data + converted_weight_bytes + pimset_mask_bytes
        self.output_offset = len(image)

        return image

    def _make_template_config(self, simulator):
        macro_config = simulator.macro_config
        mask_config = simulator.mask_config

        # 准备模板上下文
        n_group = macro_config.n_group
        bitwidth = 8
        n_macro = macro_config.n_macro
        n_macro_per_group = macro_config.n_macro_per_group
        n_vcol = macro_config.n_vcol(bitwidth)
        n_group_vcol = n_macro_per_group * n_vcol
        n_row = macro_config.n_row
        n_comp = macro_config.n_comp
        n_macro_reduce = n_row * n_comp
        mask_base = simulator.memory_space.get_base_of("pim_mask_data_reg_buffer")
        in_hw_padding = self.in_hw + 2 * self.padding

        context = {
            "OUTPUT_CHANNEL": self.out_channel,
            "INPUT_ROW": in_hw_padding,
            "INPUT_COL": in_hw_padding,
            "INPUT_CHANNEL": self.in_channel,
            "OUTPUT_ROW": self.out_hw,
            "OUTPUT_COL": self.out_hw,
            "KERNEL_SIZE": self.ker_size,
            "PADDING": 0,
            "STRIDE": self.stride,
            "N_MACRO": macro_config.n_macro,
            "N_MACRO_PER_GROUP": n_macro_per_group,
            "N_VCOL": n_vcol,
            "N_GROUP_VCOL": n_group_vcol,
            "N_GROUP": n_group,
            "N_ROW": n_row,
            "N_COMP": n_comp,
            "N_MACRO_REDUCE": n_macro_reduce,
            "INPUT_BUFFER_SIZE_PER_GROUP": self.input_buffer_size_per_group,
            "OUT_SPATIAL_TILE": self.converted_weight.shape[0],
            "OUT_REDUCE_TILE": self.converted_weight.shape[1],
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


class BitSparseConv2dTestHelper(Conv2dTestHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        self.output_bytes = 4
        self.output_dtype = np.int32
        self.im2col = False

    def _get_mock_weight(self):
        from cim_compiler.utils.bit_sparse_weight_transform import generate_valid_weight

        # weight = generate_valid_weight([self.out_channel, self.ker_size, self.ker_size, self.in_channel], 2)
        weight = generate_valid_weight(
            [self.out_channel, self.ker_size, self.ker_size, self.in_channel], 2
        )
        return weight

    def _make_bit_sparse_data(self, weight, simulator):
        from types import SimpleNamespace

        import numpy as np

        from cim_compiler.utils.bit_sparse_weight_transform import (
            outsum_mask_to_transfer_mask,
            parse_out_begin_channel,
            parse_out_mask,
            parse_out_mask_and_transfer_mask,
            weight_transform_group,
        )
        from cim_compiler.utils.df_layout import tensor_bits_to_int8

        macro_config = simulator.macro_config
        bitwidth = 8
        n_group = macro_config.n_group
        n_comp = macro_config.n_comp
        n_bcol = macro_config.n_bcol
        n_macro_per_group = macro_config.n_macro_per_group
        n_group_bcol = n_macro_per_group * n_bcol
        op_cfg = SimpleNamespace(
            out_channel=weight.shape[0],
        )
        cim_cfg = SimpleNamespace(
            bits_column=n_bcol,
            n_macro=simulator.macro_config.n_macro // n_group,  # n_macro_per_group
            n_group=n_group,
            n_comp=simulator.macro_config.n_comp,
        )
        bit_sparse_weight, meta, fold = weight_transform_group(
            weight, cim_cfg, op_cfg, "OHWI"
        )
        padded_ele_in_filter = bit_sparse_weight.shape[1]
        assert padded_ele_in_filter % n_comp == 0, f"{padded_ele_in_filter=}"
        bit_sparse_weight = bit_sparse_weight.reshape(
            bit_sparse_weight.shape[0],
            padded_ele_in_filter // n_comp,
            n_comp,
            bit_sparse_weight.shape[2],
            bit_sparse_weight.shape[3],
        )
        outsum_mask, transfer_mask, pimset_mask = parse_out_mask_and_transfer_mask(
            fold, n_group_bcol
        )
        out_begin_channel = parse_out_begin_channel(fold)

        assert len(bit_sparse_weight.shape) == 5, f"{bit_sparse_weight.shape=}"
        assert (
            bit_sparse_weight.shape[2] == macro_config.n_comp
        ), f"{bit_sparse_weight.shape=}, {macro_config.n_comp=}"
        assert (
            bit_sparse_weight.shape[3] == n_group
        ), f"{bit_sparse_weight.shape=}, {n_group=}"
        assert bit_sparse_weight.shape[4] == (
            n_group_bcol // 8
        ), f"{bit_sparse_weight.shape=}, {(n_group_bcol//8)=}"
        assert outsum_mask.shape[1] == (
            n_group_bcol // 8
        ), f"{outsum_mask.shape=}, {(n_group_bcol//8)=}"
        assert transfer_mask.shape[1] == (
            n_group_bcol // 8
        ), f"{outsum_mask.shape=}, {(n_group_bcol//8)=}"
        assert pimset_mask.shape[1] == (
            n_group_bcol // 8
        ), f"{pimset_mask.shape=}, {(n_group_bcol//8)=}"
        assert (
            outsum_mask.shape[0] == bit_sparse_weight.shape[0]
        ), f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"
        assert (
            transfer_mask.shape[0] == bit_sparse_weight.shape[0]
        ), f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"
        assert (
            pimset_mask.shape[0] == bit_sparse_weight.shape[0]
        ), f"{pimset_mask.shape=}, {bit_sparse_weight.shape=}"

        assert bit_sparse_weight.dtype == np.int8, f"{bit_sparse_weight.dtype=}"
        assert outsum_mask.dtype == np.int8, f"{outsum_mask.dtype=}"
        assert transfer_mask.dtype == np.int8, f"{transfer_mask.dtype=}"
        assert pimset_mask.dtype == np.int8, f"{pimset_mask.dtype=}"
        assert meta.dtype == np.int8, f"{meta.dtype=}"

        assert len(out_begin_channel.shape) == 1
        assert (
            out_begin_channel.shape[0] == bit_sparse_weight.shape[0] + 1
        ), f"{out_begin_channel.shape=}, {bit_sparse_weight.shape=}"
        assert (
            out_begin_channel[-1] == self.out_channel
        ), f"{out_begin_channel[-1]=}, {self.out_channel=}"
        assert out_begin_channel.dtype == np.int32, f"{out_begin_channel.dtype=}"

        return (
            bit_sparse_weight,
            meta,
            outsum_mask,
            transfer_mask,
            pimset_mask,
            out_begin_channel,
        )

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
        if self.im2col:
            self.input_data_im2col = self._apply_im2col(self.input_data)

        (
            bit_sparse_weight,
            meta,
            outsum_mask,
            transfer_mask,
            pimset_mask,
            out_begin_channel,
        ) = self._make_bit_sparse_data(self.weight_data, simulator)

        self.converted_weight = bit_sparse_weight
        self.meta = meta
        self.outsum_mask = outsum_mask
        self.transfer_mask = transfer_mask
        self.pimset_mask = pimset_mask
        self.out_begin_channel = out_begin_channel

        if self.im2col:
            input_data = bytearray(self.input_data_im2col)
        else:
            input_data = bytearray(self.input_data)
        converted_weight_bytes = bytearray(self.converted_weight)
        meta_bytes = bytearray(self.meta)
        outsum_offset_bytes = bytearray(self.outsum_mask)
        transfer_offset_bytes = bytearray(self.transfer_mask)
        pimset_mask_bytes = bytearray(self.pimset_mask)
        out_begin_channel_bytes = bytearray(self.out_begin_channel)


        # import pdb; pdb.set_trace()
        image = (
            input_data
            + converted_weight_bytes
            + meta_bytes
            + outsum_offset_bytes
            + transfer_offset_bytes
            + pimset_mask_bytes
            + out_begin_channel_bytes
        )

        self.output_offset = len(image)

        return image

    def _make_template_config(self, simulator):
        macro_config = simulator.macro_config
        mask_config = simulator.mask_config

        # 准备模板上下文
        n_group = macro_config.n_group
        bitwidth = 8
        n_macro = macro_config.n_macro
        n_macro_per_group = macro_config.n_macro_per_group
        n_vcol = macro_config.n_vcol(bitwidth)
        n_bcol = macro_config.n_bcol
        n_group_vcol = n_macro_per_group * n_vcol
        n_group_bcol = n_macro_per_group * n_bcol
        n_row = macro_config.n_row
        n_comp = macro_config.n_comp
        n_macro_reduce = n_row * n_comp
        meta_base = simulator.memory_space.get_base_of("pim_meta_data_reg_buffer")
        in_hw_padding = self.in_hw + 2 * self.padding
        max_out_channel_use = (
            self.out_begin_channel[1:] - self.out_begin_channel[:-1]
        ).max()
        context = {
            "OUTPUT_CHANNEL": self.out_channel,
            "INPUT_ROW": in_hw_padding,
            "INPUT_COL": in_hw_padding,
            "INPUT_CHANNEL": self.in_channel,
            "OUTPUT_ROW": self.out_hw,
            "OUTPUT_COL": self.out_hw,
            "KERNEL_SIZE": self.ker_size,
            "PADDING": 0,
            "STRIDE": self.stride,
            "N_MACRO": macro_config.n_macro,
            "N_MACRO_PER_GROUP": n_macro_per_group,
            "N_VCOL": n_vcol,
            "N_GROUP_VCOL": n_group_vcol,
            "N_GROUP_BCOL": n_group_bcol,
            "N_GROUP": n_group,
            "N_ROW": n_row,
            "N_COMP": n_comp,
            "N_MACRO_REDUCE": n_macro_reduce,
            "INPUT_BUFFER_SIZE_PER_GROUP": self.input_buffer_size_per_group,
            "OUT_SPATIAL_TILE": self.converted_weight.shape[0],
            "OUT_REDUCE_TILE": self.converted_weight.shape[1],
            "BIT_SPARSE_META_BASE_ADDR": meta_base,
            "OUT_CHANNEL_BEGIN_LEN": self.out_begin_channel.shape[0],
            "MAX_OUTER_SPATIEL_TILE_SIZE": max_out_channel_use,
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


class ValueSparseConv2dTestHelper(Conv2dTestHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        self.output_bytes = 4
        self.output_dtype = np.int32

    def _get_mock_weight(self):
        from cim_compiler.utils.bit_sparse_weight_transform import generate_valid_weight

        weight = np.random.randint(
            -100,
            100,
            size=[self.out_channel, self.ker_size, self.ker_size, self.in_channel],
            dtype=np.int8,
        )
        mask1 = np.random.randint(
            0,
            2,
            size=[self.out_channel, self.ker_size, self.ker_size, self.in_channel],
            dtype=np.int8,
        )
        mask2 = np.random.randint(
            0,
            2,
            size=[self.out_channel, self.ker_size, self.ker_size, self.in_channel],
            dtype=np.int8,
        )
        mask3 = np.random.randint(
            0,
            2,
            size=[self.out_channel, self.ker_size, self.ker_size, self.in_channel],
            dtype=np.int8,
        )
        weight = weight * mask1 * mask2 * mask3
        return weight

    def _make_value_sparse_data(self, weight, simulator):
        from cim_compiler.data_processor.dense import convert_value_sparse_conv2d_weight

        macro_config = simulator.macro_config
        n_from = simulator.mask_config.n_from
        n_to = simulator.mask_config.n_to
        bitwidth = 8
        n_group = macro_config.n_group
        config = {
            "n_row": macro_config.n_row,
            "n_vcol": macro_config.n_bcol // bitwidth,
            "n_group": n_group,
            "n_macro": macro_config.n_macro,
            "n_comp": macro_config.n_comp,
            "n_value_sparse_from": n_from,
            "n_value_sparse_to": n_to,
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

    def get_image(self, simulator, input=None, weight=None):
        import numpy as np

        from cim_compiler.utils.df_layout import tensor_bits_to_int8

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
        if self.im2col:
            self.input_data_im2col = self._apply_im2col(self.input_data)

        result = self._make_value_sparse_data(self.weight_data, simulator)
        self.converted_weight = result["converted_weight"]
        self.mask = result["mask"]
        self.mapping_reduce_to_macro = result["mapping_reduce_to_macro"]
        self.mapping_macro_to_from = result["mapping_macro_to_from"]
        self.mapping_from_to_row = result["mapping_from_to_row"]
        self.mapping_macro_to_row = result["mapping_macro_to_row"]
        self.pimset_mask = result["pimset_mask"]


        assert self.input_data.dtype == np.int8, f"{self.input_data.dtype=}"
        assert self.converted_weight.dtype == np.int8, f"{self.converted_weight.dtype=}"
        assert self.mask.dtype == np.int8, f"{self.mask.dtype=}"
        assert (
            self.mapping_reduce_to_macro.dtype == np.int32
        ), f"{self.mapping_reduce_to_macro.dtype=}"
        assert (
            self.mapping_macro_to_from.dtype == np.int32
        ), f"{self.mapping_macro_to_from.dtype=}"
        assert (
            self.mapping_from_to_row.dtype == np.int32
        ), f"{self.mapping_from_to_row.dtype=}"
        assert (
            self.mapping_macro_to_row.dtype == np.int32
        ), f"{self.mapping_macro_to_row.dtype=}"

        assert (
            self.mask.sum(axis=2) <= simulator.mask_config.n_to
        ).all(), f"{self.mask.sum(axis=2)=}, {simulator.mask_config.n_to=}"

        assert self.mask.shape[-1] % 8 == 0, f"{self.mask.shape=}"
        mask_bits = self.mask.reshape(*self.mask.shape[:2], self.mask.shape[-1] // 8, 8)
        mask_bits = tensor_bits_to_int8(mask_bits)

        if self.im2col:
            input_data = bytearray(self.input_data_im2col)
        else:
            input_data = bytearray(self.input_data)
        converted_weight_bytes = bytearray(self.converted_weight)
        mask_bytes = bytearray(mask_bits)
        mapping_reduce_to_macro_bytes = bytearray(self.mapping_reduce_to_macro)
        mapping_macro_to_from_bytes = bytearray(self.mapping_macro_to_from)
        mapping_from_to_row_bytes = bytearray(self.mapping_from_to_row)
        mapping_macro_to_row_bytes = bytearray(self.mapping_macro_to_row)
        pimset_mask_bytes = bytearray(self.pimset_mask)


        # import pdb; pdb.set_trace()
        image = (
            input_data
            + converted_weight_bytes
            + mask_bytes
            + mapping_reduce_to_macro_bytes
            + mapping_macro_to_from_bytes
            + mapping_macro_to_row_bytes
            + mapping_from_to_row_bytes
            + pimset_mask_bytes
        )
        self.output_offset = len(image)
        # image = input_data + index_bytes + tile_list_bytes
        return image

    def _make_template_config(self, simulator):
        macro_config = simulator.macro_config
        mask_config = simulator.mask_config

        # 准备模板上下文
        n_group = macro_config.n_group
        bitwidth = 8
        n_macro = macro_config.n_macro
        n_macro_per_group = macro_config.n_macro_per_group
        n_vcol = macro_config.n_vcol(bitwidth)
        n_group_vcol = n_macro_per_group * n_vcol
        n_row = macro_config.n_row
        n_comp = macro_config.n_comp
        n_macro_reduce = n_row * n_comp
        mask_base = simulator.memory_space.get_base_of("pim_mask_data_reg_buffer")

        in_hw_padding = self.in_hw + 2 * self.padding
        context = {
            "OUTPUT_CHANNEL": self.out_channel,
            "INPUT_ROW": in_hw_padding,
            "INPUT_COL": in_hw_padding,
            "INPUT_CHANNEL": self.in_channel,
            "OUTPUT_ROW": self.out_hw,
            "OUTPUT_COL": self.out_hw,
            "KERNEL_SIZE": self.ker_size,
            "PADDING": 0,
            "STRIDE": self.stride,
            "N_MACRO": macro_config.n_macro,
            "N_MACRO_PER_GROUP": n_macro_per_group,
            "N_VCOL": n_vcol,
            "N_GROUP_VCOL": n_group_vcol,
            "N_GROUP": n_group,
            "N_ROW": n_row,
            "N_COMP": n_comp,
            "N_MACRO_REDUCE": n_macro_reduce,
            "INPUT_BUFFER_SIZE_PER_GROUP": self.input_buffer_size_per_group,
            "TIME": self.converted_weight.shape[0],
            "N_FROM": mask_config.n_from,
            "N_TO": mask_config.n_to,
            # 'FROM_OVER_TO': mask_config.n_from // mask_config.n_to,
            "VALUE_SPARSE_MASK_BASE_ADDR": mask_base,
            "MAPPING_REDUCE_TO_MACRO_LENGTH": self.mapping_reduce_to_macro.shape[0],
            "MAPPING_MACRO_TO_FROM_LENGTH": self.mapping_macro_to_from.shape[0],
            "MAPPING_MACRO_TO_ROW_LENGTH": self.mapping_macro_to_row.shape[0],
            "MAPPING_FROM_TO_ROW_LENGTH": self.mapping_from_to_row.shape[0],
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


class ValueBitSparseConv2dTestHelper(Conv2dTestHelper):
    def __init__(self, op_config):
        super().__init__(op_config)
        self.output_bytes = 4
        self.output_dtype = np.int32

    def _get_mock_weight(self):
        import numpy as np

        """
        weight: 32 * 32 * 3 * 3
        input: 32 * 8 * 8
        """

        import random

        from cim_compiler.utils.bit_sparse_weight_transform import generate_valid_weight

        np.random.seed(4)
        random.seed(4)

        # weight = generate_valid_weight([self.out_channel, self.ker_size, self.ker_size, self.in_channel], threshold=2)
        # weight = np.zeros((self.out_channel, self.ker_size, self.ker_size, self.in_channel), dtype=np.int8) + 3
        weight = generate_valid_weight(
            [self.out_channel, self.ker_size, self.ker_size, self.in_channel],
            threshold=2,
        )
        # for filter_begin in range(0, self.out_channel, 8):
        #     filter_end = min(filter_begin+8, self.out_channel)
        #     mask = np.random.randint(0, 2, size=weight.shape[1:], dtype=np.int8)
        #     weight[filter_begin:filter_end, :, :, :] = weight[filter_begin:filter_end, :, :, :] * mask
        #     mask = np.random.randint(0, 2, size=weight.shape[1:], dtype=np.int8)
        #     weight[filter_begin:filter_end, :, :, :] = weight[filter_begin:filter_end, :, :, :] * mask

        return weight

    def _make_value_bit_sparse_data(self, weight, simulator):
        from cim_compiler.utils.bit_value_sparse_weight_transform import (
            convert_value_bit_sparse_conv2d_weight,
        )

        macro_config = simulator.macro_config
        n_from = simulator.mask_config.n_from
        n_to = simulator.mask_config.n_to
        bitwidth = 8
        n_group = macro_config.n_group
        config = {
            "n_row": macro_config.n_row,
            "n_bcol": macro_config.n_bcol,
            "n_group": n_group,
            "n_macro": macro_config.n_macro,
            "n_comp": macro_config.n_comp,
            "n_value_sparse_from": n_from,
            "n_value_sparse_to": n_to,
        }
        result = convert_value_bit_sparse_conv2d_weight(weight, config)

        return result

    def get_image(self, simulator, input=None, weight=None):
        import numpy as np

        from cim_compiler.utils.df_layout import tensor_bits_to_int8

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
        if self.im2col:
            self.input_data_im2col = self._apply_im2col(self.input_data)

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
        pimset_mask = bit_sparse_result["pimset_mask"]


        assert self.input_data.dtype == np.int8, f"{self.input_data.dtype=}"
        assert converted_weight.dtype == np.int8, f"{converted_weight.dtype=}"
        assert mask.dtype == np.int8, f"{mask.dtype=}"
        assert (
            self.mapping_reduce_to_macro.dtype == np.int32
        ), f"{self.mapping_reduce_to_macro.dtype=}"
        assert (
            self.mapping_macro_to_from.dtype == np.int32
        ), f"{self.mapping_macro_to_from.dtype=}"
        assert (
            self.mapping_from_to_row.dtype == np.int32
        ), f"{self.mapping_from_to_row.dtype=}"
        assert (
            self.mapping_macro_to_row.dtype == np.int32
        ), f"{self.mapping_macro_to_row.dtype=}"
        assert (
            mask.sum(axis=2) <= simulator.mask_config.n_to
        ).all(), f"{mask.sum(axis=2)=}, {simulator.mask_config.n_to=}"

        macro_config = simulator.macro_config
        bitwidth = 8
        n_group = macro_config.n_group
        n_comp = macro_config.n_comp
        n_bcol = macro_config.n_bcol
        n_macro_per_group = macro_config.n_macro_per_group
        n_group_bcol = n_macro_per_group * n_bcol

        assert outsum_mask.shape[1] == (
            n_group_bcol
        ), f"{outsum_mask.shape=}, {(n_group_bcol)=}"
        assert transfer_mask.shape[1] == (
            n_group_bcol
        ), f"{outsum_mask.shape=}, {(n_group_bcol)=}"
        assert pimset_mask.shape[1] == (
            n_group_bcol
        ), f"{pimset_mask.shape=}, {(n_group_bcol)=}"
        assert (
            outsum_mask.shape[0] == self.mapping_reduce_to_macro.shape[0]
        ), f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"
        assert (
            transfer_mask.shape[0] == self.mapping_reduce_to_macro.shape[0]
        ), f"{outsum_mask.shape=}, {bit_sparse_weight.shape=}"
        assert (
            pimset_mask.shape[0] == self.mapping_reduce_to_macro.shape[0]
        ), f"{pimset_mask.shape=}, {bit_sparse_weight.shape=}"

        assert mask.shape[-1] % 8 == 0, f"{mask.shape=}"
        mask_bits = mask.reshape(*mask.shape[:-1], mask.shape[-1] // 8, 8)
        mask_bits = tensor_bits_to_int8(mask_bits)
        assert outsum_mask.shape[-1] % 8 == 0, f"{outsum_mask.shape=}"
        outsum_mask_bits = outsum_mask.reshape(
            *outsum_mask.shape[:-1], outsum_mask.shape[-1] // 8, 8
        )
        outsum_mask_bits = tensor_bits_to_int8(outsum_mask_bits)
        assert transfer_mask.shape[-1] % 8 == 0, f"{transfer_mask.shape=}"
        transfer_mask_bits = transfer_mask.reshape(
            *transfer_mask.shape[:-1], transfer_mask.shape[-1] // 8, 8
        )
        transfer_mask_bits = tensor_bits_to_int8(transfer_mask_bits)
        assert pimset_mask.shape[-1] % 8 == 0, f"{pimset_mask.shape=}"
        pimset_mask_bits = pimset_mask.reshape(
            *pimset_mask.shape[:-1], pimset_mask.shape[-1] // 8, 8
        )
        pimset_mask_bits = tensor_bits_to_int8(pimset_mask_bits)

        if self.im2col:
            input_data = bytearray(self.input_data_im2col)
        else:
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
        pimset_offset_bytes = bytearray(pimset_mask_bits)

        data_for_test = np.array([12345], dtype=np.int32)
        data_for_test_bytes = bytearray(data_for_test)
        image = (
            input_data
            + converted_weight_bytes
            + mask_bytes
            + mapping_reduce_to_macro_bytes
            + mapping_macro_to_from_bytes
            + mapping_macro_to_row_bytes
            + mapping_from_to_row_bytes
            + meta_bytes
            + outsum_offset_bytes
            + transfer_offset_bytes
            + pimset_offset_bytes
        )
        self.output_offset = len(image)
        # image = input_data + index_bytes + tile_list_bytes
        return image

    def _make_template_config(self, simulator):
        macro_config = simulator.macro_config
        mask_config = simulator.mask_config

        # 准备模板上下文
        n_group = macro_config.n_group
        bitwidth = 8
        n_macro = macro_config.n_macro
        n_macro_per_group = macro_config.n_macro_per_group
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
            "OUTPUT_CHANNEL": self.out_channel,
            "INPUT_ROW": in_hw_padding,
            "INPUT_COL": in_hw_padding,
            "INPUT_CHANNEL": self.in_channel,
            "OUTPUT_ROW": self.out_hw,
            "OUTPUT_COL": self.out_hw,
            "KERNEL_SIZE": self.ker_size,
            "PADDING": 0,
            "STRIDE": self.stride,
            "N_MACRO": macro_config.n_macro,
            "N_MACRO_PER_GROUP": n_macro_per_group,
            "N_VCOL": n_vcol,
            "N_GROUP_VCOL": n_group_vcol,
            "N_GROUP_BCOL": n_group_bcol,
            "N_GROUP": n_group,
            "N_ROW": n_row,
            "N_COMP": n_comp,
            "N_MACRO_REDUCE": n_macro_reduce,
            "INPUT_BUFFER_SIZE_PER_GROUP": self.input_buffer_size_per_group,
            "TIME": self.converted_weight.shape[0],
            "N_FROM": mask_config.n_from,
            "N_TO": mask_config.n_to,
            # 'FROM_OVER_TO': mask_config.n_from // mask_config.n_to,
            "VALUE_SPARSE_MASK_BASE_ADDR": mask_base,
            "MAPPING_REDUCE_TO_MACRO_LENGTH": self.mapping_reduce_to_macro.shape[0],
            "MAPPING_MACRO_TO_FROM_LENGTH": self.mapping_macro_to_from.shape[0],
            "MAPPING_MACRO_TO_ROW_LENGTH": self.mapping_macro_to_row.shape[0],
            "MAPPING_FROM_TO_ROW_LENGTH": self.mapping_from_to_row.shape[0],
            "BIT_SPARSE_META_BASE_ADDR": meta_base,
            "OUT_SPATIAL_TILE": self.mapping_reduce_to_macro.shape[0],
            "N_FILTER_PER_GROUP": 8 * n_macro_per_group,
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


class QuantizeHelper:
    def __init__(self):
        pass

    def _get_mock_bias(self):
        import numpy as np

        bias = np.random.randint(-4, 5, size=(self.out_channel), dtype=np.int32)
        # bias = np.zeros((self.out_channel,), dtype=np.int32)
        return bias

    def _get_mock_scale(self):
        import numpy as np

        scale = np.random.rand(self.out_channel).astype(np.float32)
        # scale = np.ones((self.out_channel,)).astype(np.float32) * 0.5
        return scale

    def _get_mock_out_zp(self):
        import numpy as np

        out_zp = np.zeros((1,), dtype=np.int32) + 3
        return out_zp

    def _calculate_golden_quantize(self):
        import numpy as np

        from cim_compiler.utils.round import banker_round

        output_h = output_w = self.out_hw
        output_c = self.out_channel

        output = np.zeros((output_h, output_w, output_c), dtype=np.int32)
        weight = self.weight_data.reshape(self.weight_data.shape[0], -1)
        stride = self.stride
        for row in range(output_h):
            for col in range(output_w):
                input = self.input_data[
                    stride * row : stride * row + self.ker_size,
                    stride * col : stride * col + self.ker_size,
                    :,
                ].reshape(-1, 1)
                golden = np.matmul(weight.astype(np.int32), input.astype(np.int32))
                output[row, col, :] = golden.reshape(-1)

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

    def get_image_quantify(
        self, simulator, bias=None, scale=None, out_zp=None, relu=False
    ):
        import numpy as np

        from cim_compiler.utils.bias_scale_fuse import bias_scale_fuse

        if bias is None:
            bias = self._get_mock_bias()

        if scale is None:
            scale = self._get_mock_scale()

        if out_zp is None:
            out_zp = self._get_mock_out_zp()

        self.bias = bias
        self.scale = scale
        self.out_zp = out_zp
        self.relu = relu

        assert self.bias.dtype == np.int32, f"{self.bias.dtype=}"
        assert (
            self.bias.size == self.out_channel
        ), f"{self.bias.size=}, {self.out_channel=}"
        assert self.scale.dtype == np.float32, f"{self.scale.dtype=}"
        assert (
            self.scale.size == self.out_channel
        ), f"{self.scale.size=}, {self.out_channel=}"
        assert self.out_zp.dtype == np.int32, f"{self.out_zp.dtype=}"
        assert self.out_zp.size == 1, f"{self.out_zp.size=}"

        bias_scale_data = bias_scale_fuse(self.bias, self.scale)
        out_zp_data = bytearray(self.out_zp)
        return bias_scale_data + out_zp_data
