import pytest
from test.compiler.base import TestBase

class TestMemory(TestBase):

    # @pytest.mark.parametrize(
    #     "casename",
    #     [
    #         "load_save",
    #         "load_save_var_index",
    #         "load_save_in_loop",
    #         "slice",
    #         "slice_var_index",
    #         "slice_chain",
    #         "trans",
    #     ],
    # )
    # def test_memory_with_print(self, casename):
    #     casename = f"memory/print/{casename}"
    #     self.run_test(casename)
    
    @pytest.mark.parametrize("casename", ["trans", "trans_across_memory"])
    def test_memory_with_image(self, casename):
        casename = f"memory/image/{casename}"
        self.run_test_with_image(casename)
    #     case_dir = os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)), "image", casename
    #     )
    #     assert os.path.exists(case_dir), f"{case_dir} not exists"
    #     assert os.path.isdir(case_dir), f"{case_dir} is not a directory"

    #     # Prepare path
    #     input_path = os.path.join(case_dir, "code.cim")
    #     test_helper_path = os.path.join(case_dir, "helper.py")
    #     assert os.path.exists(input_path), f"{input_path} not exists"
    #     assert os.path.exists(test_helper_path), f"{test_helper_path} not exists"

    #     output_folder = os.path.join(case_dir, ".result")
    #     os.makedirs(output_folder, exist_ok=True)

    #     # Get helper
    #     with open(test_helper_path, "r") as file:
    #         code = file.read()
    #     local_namespace = {}
    #     exec(code, {}, local_namespace)
    #     Helper = local_namespace["TestHelper"]
    #     helper = Helper()

    #     # load image
    #     image = helper.get_image()
    #     global_memory = self.simulator.memory_space.get_memory_by_name("global")
    #     global_memory.write(image, global_memory.offset, len(image))

    #     # run compiler
    #     cmd = f"bash compile.sh isa {input_path} {output_folder} {self.config_path}"
    #     result = subprocess.run(cmd.split(" "), capture_output=True, text=True)
    #     print("输出:", result.stdout)
    #     print("错误:", result.stderr)
    #     assert result.returncode == 0

    #     # get output code
    #     output_path = os.path.join(output_folder, "final_code.json")
    #     with open(output_path, "r") as f:
    #         code = json.load(f)

    #     # run code in simulator
    #     status = self.simulator.run_code(code)
    #     assert status == self.simulator.FINISH

    #     # check result
    #     print_record = self.simulator.print_record
    #     helper.check_image(global_memory.read_all())


if __name__ == "__main__":
    TestMemory.setup_class()
    test_for_loop = TestMemory()
    test_for_loop.setup_method()
    # test_for_loop.test_memory_with_image("trans_across_memory")
    test_for_loop.test_memory_with_print("load_save")
