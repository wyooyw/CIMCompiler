import pytest
from test.compiler.base import TestBase

class TestMemory(TestBase):

    @pytest.mark.parametrize(
        "casename",
        [
            "load_save",
            "load_save_var_index",
            "load_save_in_loop",
            "slice",
            "slice_var_index",
            "slice_chain",
            "trans",
        ],
    )
    def test_memory_with_print(self, casename):
        casename = f"memory/print/{casename}"
        self.run_test(casename)
    
    @pytest.mark.parametrize("casename", ["trans", "trans_across_memory"])
    def test_memory_with_image(self, casename):
        casename = f"memory/image/{casename}"
        self.run_test_with_image(casename)