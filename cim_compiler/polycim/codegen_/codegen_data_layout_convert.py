import os
import subprocess
import tempfile
import time

import islpy as isl
import numpy as np

import polycim.utils.utils as utils
from polycim.codegen_.codegen import CodeStmt
from polycim.codegen_.codegen_c import CCodeGenerator
from polycim.utils.logger import get_logger

logger = get_logger(__name__)


class DataLayoutConvertCodegen(CCodeGenerator):
    def __init__(self, accrel_lhs, accrel_rhs):
        super().__init__()
        self.domain = accrel_lhs.domain().intersect(accrel_rhs.domain())
        self.accrel_lhs = accrel_lhs
        self.accrel_rhs = accrel_rhs
        self.shape_lhs = utils.get_box_hull_shape(accrel_lhs.range())
        self.shape_rhs = utils.get_box_hull_shape(accrel_rhs.range())
        self.name_lhs = self.accrel_lhs.get_tuple_name(isl.dim_type.out)
        self.name_rhs = self.accrel_rhs.get_tuple_name(isl.dim_type.out)

    def codegen_buffer_define(self, depth):
        code_list = []
        shape_lhs_str = ",".join([str(s) for s in self.shape_lhs])
        shape_rhs_str = ",".join([str(s) for s in self.shape_rhs])
        lhs_name = self.accrel_lhs.get_tuple_name(isl.dim_type.out)
        rhs_name = self.accrel_rhs.get_tuple_name(isl.dim_type.out)
        code_lhs = CodeStmt(
            code=f"Eigen::Tensor<int, {len(self.shape_lhs)}, Eigen::RowMajor> {lhs_name}({shape_lhs_str});",
            depth=depth,
        )
        code_rhs = CodeStmt(
            code=f"Eigen::Tensor<int, {len(self.shape_rhs)}, Eigen::RowMajor> {rhs_name}({shape_rhs_str});",
            depth=depth,
        )
        code_list.append(code_lhs)
        code_list.append(code_rhs)

        return code_list

    def codegen_call_fn(self, call_name, call_args, depth):
        if call_name == "DATA_MOVE":
            return self.codegen_data_move(call_args, depth)
        raise NotImplementedError

    def codegen_data_move(self, call_args, depth):
        ast_accrel_lhs = self.get_access_from_pw_multi_aff(
            self.accrel_lhs.as_pw_multi_aff(), call_args
        )
        ast_accrel_rhs = self.get_access_from_pw_multi_aff(
            self.accrel_rhs.as_pw_multi_aff(), call_args
        )
        lhs_access_code, lhs_access_vars = self.codegen_access_indices(
            ast_accrel_lhs, depth
        )
        rhs_access_code, rhs_access_vars = self.codegen_access_indices(
            ast_accrel_rhs, depth
        )
        lhs_name = self.accrel_lhs.get_tuple_name(isl.dim_type.out)
        rhs_name = self.accrel_rhs.get_tuple_name(isl.dim_type.out)
        trans_code = CodeStmt(
            code=f"{lhs_name}({','.join(lhs_access_vars)}) = {rhs_name}({','.join(rhs_access_vars)});",
            depth=depth,
        )
        return [*lhs_access_code, *rhs_access_code, trans_code]

    def codegen_access_indices(self, access_op, depth):
        assert (
            access_op.get_type() == isl._isl.ast_expr_type.op
        ), f"{access_op.get_type()=}"
        name = access_op.get_op_arg(0).id_get_id().get_name()

        index_code_list = []
        index_var_list = []
        for arg_id in range(1, access_op.get_op_n_arg()):
            index_expr = access_op.get_op_arg(arg_id)
            index_code, index_var = self.codegen_expression(index_expr, depth)
            index_code_list.extend(index_code)
            index_var_list.append(index_var)

        return index_code_list, index_var_list

    def codegen_includes(self, depth):
        code_includes = [
            CodeStmt(code="#include <unsupported/Eigen/CXX11/Tensor>", depth=depth),
            CodeStmt(code="#include <fstream>", depth=depth),
            CodeStmt(code="#include <algorithm>", depth=depth),
            CodeStmt(code="#include <iostream>", depth=depth),
        ]
        return code_includes

    def codegen_function_load_and_save_from_file(self, depth):
        lhs_dim = len(self.shape_lhs)
        rhs_dim = len(self.shape_rhs)
        code_str = """
void load_from_file(Eigen::Tensor<int, %d, Eigen::RowMajor> &tensor, const std::string &file_path) {
    std::ifstream file(file_path);
    int value;
    for (int i = 0; i < tensor.size(); ++i) {
        file >> value;
        tensor.data()[i] = value;
    }
}

void save_to_file(Eigen::Tensor<int, %d, Eigen::RowMajor> &tensor, const std::string &file_path) {
    std::ofstream file(file_path);
    for (int i = 0; i < tensor.size(); ++i) {
        file << tensor.data()[i] << "\\n";
    }
}
""" % (
            rhs_dim,
            lhs_dim,
        )
        codegen_load_and_save = CodeStmt(code=code_str, depth=depth)
        return [codegen_load_and_save]

    def codegen_load_from_file(self, tensor_name, depth):
        code_str = f"load_from_file({tensor_name}, argv[1]);"
        codegen_load_from_file = CodeStmt(code=code_str, depth=depth)
        return [codegen_load_from_file]

    def codegen_save_to_file(self, tensor_name, depth):
        code_str = f"save_to_file({tensor_name}, argv[2]);"
        codegen_save_to_file = CodeStmt(code=code_str, depth=depth)
        return [codegen_save_to_file]

    def codegen_str(self, node, indent_unit=4):
        code_includes = self.codegen_includes(0)
        code_helper_functions = self.codegen_helper_functions(0)
        code_function_load_and_save_from_file = (
            self.codegen_function_load_and_save_from_file(0)
        )
        code_load_from_file = self.codegen_load_from_file(self.name_rhs, 1)
        code_save_to_file = self.codegen_save_to_file(self.name_lhs, 1)
        # special_reg_settings = self.codegen_special_settings(1)
        main_begin, main_end = self.codegen_main_and_end(0)
        argument_check = self.codegen_argument_check(
            1, ["<input_file>", "<output_file>"]
        )
        buffer_define_code_list = self.codegen_buffer_define(1)
        execute_code_list = self.codegen(node, 1)
        code_str = ""
        for code_stmt in (
            code_includes
            + code_helper_functions
            + code_function_load_and_save_from_file
            + main_begin
            + argument_check
            + buffer_define_code_list
            + code_load_from_file
            + execute_code_list
            + code_save_to_file
            + main_end
        ):
            assert type(code_stmt) == CodeStmt, f"{type(code_stmt)=}, {code_stmt=}"
            code = code_stmt.code
            depth = code_stmt.depth
            code = " " * (indent_unit * depth) + code + "\n"
            code_str = code_str + code
        return code_str


def data_layout_convert_codegen_to_file(accrel_lhs, accrel_rhs, save_path):
    code = data_layout_convert_codegen(accrel_lhs, accrel_rhs)
    with open(save_path, "w") as f:
        f.write(code)
    return code


def data_layout_convert_codegen(accrel_lhs, accrel_rhs):
    domain_lhs = accrel_lhs.domain()
    domain_rhs = accrel_rhs.domain()

    ndim_lhs = domain_lhs.dim(isl.dim_type.set)
    ndim_rhs = domain_rhs.dim(isl.dim_type.set)
    assert ndim_lhs == ndim_rhs, f"{ndim_lhs=}, {ndim_rhs=}"
    n_dim = ndim_lhs

    stmt_name = "DATA_MOVE"
    domain = domain_lhs.intersect(domain_rhs)
    compute_schedule = utils.identity_map_from_set(domain)
    compute_schedule = compute_schedule.set_tuple_name(isl.dim_type.in_, stmt_name)
    compute_domain = domain.set_tuple_name(stmt_name)

    ast = utils.gen_ast(compute_domain, compute_schedule, None)
    # utils.print_code(compute_domain, compute_schedule, None)

    code_generator = DataLayoutConvertCodegen(accrel_lhs, accrel_rhs)
    code = code_generator.codegen_str(ast, 4)
    return code


def simplify_access(access):
    access_domain = access.domain()
    access_domain = access_domain.compute_divs().coalesce().remove_redundancies()

    access = access.intersect_domain(access_domain)
    access = access.make_disjoint().compute_divs().coalesce().remove_redundancies()
    return access


def gcc_compile_data_layout_convert_code(code_path, exe_path):

    cmd = ["g++", code_path, "-O3", "-I", "/usr/include/eigen3/", "-o", exe_path]
    logger.info(f"Begin to compile using:\n{cmd}")

    begin_time = time.time()
    subprocess.run(cmd, check=True)
    end_time = time.time()
    logger.info(f"Compile finished. Use time: {(end_time - begin_time):.3f}s")


def run_data_layout_convert_executable(exe_path, input_path, output_path):
    cmd = [exe_path, input_path, output_path]
    logger.info(f"Begin to run using:\n{cmd}")

    begin_time = time.time()
    subprocess.run(cmd, check=True)
    end_time = time.time()

    logger.info(f"Run finished. Use time: {(end_time - begin_time):.3f}s")


def data_layout_convert(accrel_input, accrel_output, input_data):
    accrel_input = simplify_access(accrel_input)
    accrel_output = simplify_access(accrel_output)

    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name

    output_name = accrel_output.get_tuple_name(isl.dim_type.out)
    input_name = accrel_input.get_tuple_name(isl.dim_type.out)

    output_path = os.path.join(temp_dir_path, f"{output_name}.txt")
    input_path = os.path.join(temp_dir_path, f"{input_name}.txt")

    output_shape = utils.get_box_hull_shape(accrel_output.range())
    input_shape = utils.get_box_hull_shape(accrel_input.range())
    assert tuple(input_data.shape) == tuple(
        input_shape
    ), f"{input_data.shape=}, {input_shape=}"

    try:
        code_path = os.path.join(temp_dir_path, "codegen_test.cpp")
        data_layout_convert_codegen_to_file(accrel_output, accrel_input, code_path)

        exe_path = os.path.join(temp_dir_path, "codegen_test.out")
        gcc_compile_data_layout_convert_code(code_path, exe_path)

        # prepare the input file
        np.savetxt(input_path, input_data.reshape(-1), fmt="%d")

        run_data_layout_convert_executable(exe_path, input_path, output_path)

        # check the output
        output_data = np.loadtxt(output_path, dtype=input_data.dtype).reshape(
            output_shape
        )
    finally:
        temp_dir.cleanup()  # Uncomment this line if you want to delete the directory manually later

    return output_data


if __name__ == "__main__":
    accrel_lhs = isl.Map("{ [i, j] -> A[i, j] : 0 <= i < 4 and 0 <= j < 4 }")
    accrel_rhs = isl.Map("{ [i, j] -> B[j, i] : 0 <= i < 4 and 0 <= j < 4 }")
    input_data = np.arange(16).astype(np.int32).reshape(4, 4)
    output_data = data_layout_convert(accrel_lhs, accrel_rhs, input_data)
    print(f"{output_data=}")
