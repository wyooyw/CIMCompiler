import pytest
from test.op.base import TestBase

class TestLinear(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            # normal
            "linear/dense/normal",
            "linear/bs/normal",
            # quantify
            'linear/dense/quantize',
            'linear/bs/quantize',
        ],
    )
    @pytest.mark.parametrize(
        "op_config",
        [
            {"out_channel": 32, "in_channel": 16},
            # {"out_channel": 16, "in_channel": 32},
            # {"out_channel": 512, "in_channel": 16},
            # {"out_channel": 16, "in_channel": 512},
        ],
    )
    def test_pim_compute(self, casename, op_config):
        op_config["ker_size"] = 1
        op_config["in_hw"] = 1
        op_config["out_hw"] = 1
        op_config["padding"] = 0

        self.run_op_test(casename, op_config)