import pytest
from test.compiler.base import TestBase

class TestSIMD(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            "vsadd",
            "vvadd"
        ],
    )
    def test_control_flow(self, casename):
        casename = f"simd/{casename}"
        self.run_test(casename)

if __name__ == "__main__":
    TestSIMD.setup_class()
    tester = TestSIMD()
    tester.setup_method()
    tester.test_control_flow(
        "vsadd"
    )
