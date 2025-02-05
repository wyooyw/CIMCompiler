import pytest
from test.op.base import TestBase

class TestDWConvSIMD(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            'dwconv2d/simd'
        ],
    )
    @pytest.mark.parametrize(
        "op_config",
        [
            {"out_channel": 4, "in_channel": 4, "ker_size": 3, "in_hw": 8, "out_hw": 8, "padding": 1},
            {"out_channel": 8, "in_channel": 8, "ker_size": 3, "in_hw": 8, "out_hw": 8, "padding": 1},
            {"out_channel": 16, "in_channel": 16, "ker_size": 3, "in_hw": 8, "out_hw": 8, "padding": 1},
            {"out_channel": 32, "in_channel": 32, "ker_size": 3, "in_hw": 8, "out_hw": 8, "padding": 1},
            {"out_channel": 128, "in_channel": 128, "ker_size": 3, "in_hw": 8, "out_hw": 8, "padding": 1},
            {"out_channel": 192, "in_channel": 192, "ker_size": 3, "in_hw": 32, "out_hw": 16, "padding": 1, "stride": 2 }
        ],
    )
    def test_pim_compute(self, casename, op_config):
        return self.run_op_test(casename, op_config)