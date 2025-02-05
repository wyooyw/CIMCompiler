import pytest
from test.op.base import TestBase

class TestPooling(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            'max_pooling',
            'avg_pooling',
            'relu'
        ],
    )
    @pytest.mark.parametrize(
        "op_config",
        [
            {"out_channel": 4, "in_channel": 4, "ker_size": 2, "in_hw": 4, "out_hw": 2},
            {"out_channel": 512, "in_channel": 512, "ker_size": 2, "in_hw": 32, "out_hw": 16},
            {"out_channel": 1, "in_channel": 1, "ker_size": 2, "in_hw": 16, "out_hw": 8},
            {"out_channel": 4, "in_channel": 4, "ker_size": 4, "in_hw": 16, "out_hw": 4},
            {"out_channel": 4, "in_channel": 4, "ker_size": 4, "in_hw": 4, "out_hw": 1},
           
        ],
    )
    def test_pim_compute(self, casename, op_config):
        self.run_op_test(casename, op_config)