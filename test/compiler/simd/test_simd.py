import pytest
from test.compiler.base import TestBase

class TestSIMD(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            "vsadd"
        ],
    )
    def test_control_flow(self, casename):
        casename = f"simd/{casename}"
        self.run_test(casename)