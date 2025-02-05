import pytest
from test.op.base import TestBase

class TestLinear(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            # 'linear/dense',
            "linear/dense_onegroup",
            "linear/bit_sparse",
            # quantify
            # 'linear/dense_quantify',
            'linear/bit_sparse_quantify',
            'linear/dense_quantify_onegroup'
        ],
    )
    @pytest.mark.parametrize(
        "op_config",
        [
            {"out_channel": 32, "in_channel": 16},
            {"out_channel": 16, "in_channel": 32},
            {"out_channel": 512, "in_channel": 16},
            {"out_channel": 16, "in_channel": 512},
        ],
    )
    def test_pim_compute(self, casename, op_config):
        op_config["ker_size"] = 1
        op_config["in_hw"] = 1
        op_config["out_hw"] = 1
        op_config["padding"] = 0

        self.run_op_test(casename, op_config)