import pytest
from test.compiler.base import TestBase

class TestForLoop(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            "print_in_loop",
            "count_in_loop",
            "accumulate_in_loop",
            "print_in_double_loop",
            "count_in_double_loop",
            "accumulate_in_double_loop",
            "fibonacci",
            "twin_loop",
            
        ],
    )
    def test_control_flow(self, casename):
        casename = f"control_flow/for_loop/{casename}"
        self.run_test(casename)