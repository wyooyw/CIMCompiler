import pytest
from utils.predict_pimcompute_count import predict_pimcompute_count_for_conv2d_dense
from test.op.base import TestBase

class TestConv(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            "conv2d/vs/normal",
            "conv2d/dense/normal",
            "conv2d/bs/normal",
            "conv2d/vs_bs/normal",
            # quantify
            'conv2d/dense/quantize' ,
            'conv2d/bs/quantize',
            'conv2d/vs/quantize' ,
            'conv2d/vs_bs/quantize'
        ],
    )
    @pytest.mark.parametrize(
        "op_config",
        [
            {"out_channel": 4, "in_channel": 3, "ker_size": 3, "in_hw": 8, "out_hw": 6},
            # *[{
            #     "out_channel": ch_out,
            #     "in_channel": ch_in,
            #     "ker_size": 3,
            #     "in_hw": 8,
            #     "out_hw": 6,
            # } for (ch_out, ch_in) in [
            #     (16, 3), (32, 16), (64, 16), (384, 16), (256, 96)
            # ]],
            # *[{
            #     "out_channel": ch_out,
            #     "in_channel": ch_in,
            #     "ker_size": 3,
            #     "in_hw": 4,
            #     "out_hw": 2,
            # } for (ch_out, ch_in) in [
            #     (16, 384), (256, 384)
            # ]],

            # # stride=2 配置
            # *[{
            #     "out_channel": ch_out,
            #     "in_channel": ch_in,
            #     "ker_size": 3,
            #     "in_hw": in_hw,
            #     "out_hw": out_hw,
            #     "padding": 1,
            #     "stride": 2,
            # } for (ch_out, ch_in, in_hw, out_hw) in [
            #     (32, 16, 8, 4), (64, 16, 8, 4),
            #     (16, 384, 4, 2), (384, 16, 8, 4)
            # ]],
            
            # # 1x1 卷积配置
            # *[{
            #     "out_channel": ch_out,
            #     "in_channel": ch_in,
            #     "ker_size": 1,
            #     "in_hw": 1,
            #     "out_hw": 1,
            # } for (ch_out, ch_in) in [
            #     (32, 16), (64, 16), (16, 384), (384, 16)
            # ]],
        ],
    )
    def test_pim_compute(self, casename, op_config):
        pimcompute_count = predict_pimcompute_count_for_conv2d_dense(
            self.simulator.macro_config, op_config, group_size=16
        )
        self.run_op_test(casename, op_config, pimcompute_count)

if __name__ == "__main__":
    TestPIMComputeValueSparse.setup_class()
    tester = TestPIMComputeValueSparse()
    tester.setup_method()
    tester.test_pim_compute(
        "conv2d/dense/quantize",
        {
            "out_channel": 128,
            "in_channel": 32,
            "ker_size": 3,
            "in_hw": 8,
            "out_hw": 8,
            "padding": 1,
        },
    )
