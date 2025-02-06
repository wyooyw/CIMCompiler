import pytest
from test.compiler.base import TestBase

class TestIf(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            "if"
        ],
    )
    def test_control_flow(self, casename):
        casename = f"control_flow/if/{casename}"
        self.run_test(casename)