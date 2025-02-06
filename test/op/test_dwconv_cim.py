import pytest
from test.op.base import TestBase

class TestDWConvCIM(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            # quantify
            "dwconv2d/mvm",
        ],
    )
    @pytest.mark.parametrize(
        "op_config",
        [
            *[{
                "out_channel": channel,
                "in_channel": channel,
                "ker_size": 3,
                "in_hw": 8,
                "out_hw": 6,
            } for channel in [
                16, # 32, 64, 384
            ]],
            # *[{
            #     "out_channel": channel,
            #     "in_channel": channel,
            #     "ker_size": 3,
            #     "in_hw": 8,
            #     "out_hw": 4,
            #     "padding": 1,
            #     "stride": 2,
            # } for channel in [
            #     16, 32, 64, 384
            # ]],
        ],
    )
    def test_pim_compute(self, casename, op_config):
        self.run_op_test(casename, op_config)