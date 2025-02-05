import pytest
from test.compiler.pimcompute.base import TestBase

class TestResAddQuantize(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            'resadd_quantize'
        ],
    )
    @pytest.mark.parametrize(
        "op_config",
        [
            {"in_channel": 4, "in_hw": 8},
            {"in_channel": 32, "in_hw": 32},
        ],
    )
    def test_pim_compute(self, casename, op_config):
        self.run_op_test(casename, op_config)