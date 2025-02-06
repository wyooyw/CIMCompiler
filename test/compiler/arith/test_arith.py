import pytest
from test.compiler.base import TestBase

class TestArith(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            "number",
            # "cond",
        ],
    )
    def test_arith(self, casename):
        casename = f"arith/{casename}"
        self.run_test(casename)