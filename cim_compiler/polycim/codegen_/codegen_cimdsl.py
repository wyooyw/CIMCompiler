import math
import os

import islpy as isl

import polycim.utils.utils as utils
from polycim.codegen_.codegen import CodeStmt, alloc_unique_stmt, alloc_unique_var
from polycim.config import get_config
from polycim.op.base_operator import (
    DataMovement,
    DataMovementOperator,
    PartialSumDataMovement,
    TensorAccessRelation,
)
from polycim.op.buffer_manager import BufferManager
from polycim.utils.logger import get_logger, level_tqdm

logger = get_logger(__name__)


class CodeGenerator:
    def __init__(self, op, name_to_op):
        self.op = op
        self.name_to_op = name_to_op
        self.buffer_manager = BufferManager()

    def codegen_includes(self, depth):
        CIMCOMPILER_HOME = os.environ["CIMCOMPILER_HOME"]
        include_list = [
            os.path.join(
                CIMCOMPILER_HOME, "cim_compiler/op/common/def_special_regs.cim"
            ),
            os.path.join(CIMCOMPILER_HOME, "cim_compiler/op/common/simd.cim"),
        ]
        include_code_list = []
        for include_path in include_list:
            with open(include_path, "r") as f:
                include_code = CodeStmt(code=f.read(), depth=depth)
            include_code_list.append(include_code)
        return include_code_list

    def codegen_special_settings(self, depth):
        use_group = self.op.attr["n_use_group"]
        use_comp = self.op.attr["n_use_comp"]
        cim_cfg = get_config()
        assert (
            use_group > cim_cfg.n_group // 2 and use_group <= cim_cfg.n_group
        ), f"{use_group=}, {cim_cfg.n_group=}"
        special_regs_setting = [
            CodeStmt(
                code="SpecialRegSet(SPECIAL_REG_INPUT_BIT_WIDTH, 8);", depth=depth
            ),
            CodeStmt(
                code="SpecialRegSet(SPECIAL_REG_WEIGHT_BIT_WIDTH, 8);", depth=depth
            ),
            CodeStmt(
                code="SpecialRegSet(SPECIAL_REG_OUTPUT_BIT_WIDTH, 32);", depth=depth
            ),
            CodeStmt(
                code=f"SpecialRegSet(SPECIAL_REG_GROUP_SIZE, {cim_cfg.n_macro_per_group});",
                depth=depth,
            ),
            CodeStmt(
                code=f"SpecialRegSet(SPECIAL_REG_ACTIVATION_GROUP_NUM, {use_group});",
                depth=depth,
            ),
            CodeStmt(
                code=f"SpecialRegSet(SPECIAL_REG_ACTIVATION_ELEMENT_COL_NUM, {cim_cfg.n_group_vcol});",
                depth=depth,
            ),
            CodeStmt(
                code=f"SpecialRegSet(SPECIAL_REG_GROUP_INPUT_STEP, {use_comp});",
                depth=depth,
            ),
            CodeStmt(
                code="SpecialRegSet(SPECIAL_REG_SIMD_INPUT_1_BIT_WIDTH, 32);",
                depth=depth,
            ),
            CodeStmt(
                code="SpecialRegSet(SPECIAL_REG_SIMD_INPUT_2_BIT_WIDTH, 32);",
                depth=depth,
            ),
            CodeStmt(
                code="SpecialRegSet(SPECIAL_REG_SIMD_OUTPUT_BIT_WIDTH, 32);",
                depth=depth,
            ),
        ]
        return special_regs_setting

    def codegen_buffer_define(self, depth):
        self.buffer_manager.add_buffers_from_op(self.op)
        buffer_name_to_info = self.buffer_manager.get_buffer_name_to_info()
        code_list = []

        global_buffer_info = {
            name[0]: info
            for name, info in buffer_name_to_info.items()
            if info.memory_name == "global"
        }
        assert len(global_buffer_info) == 3, f"{global_buffer_info=}"
        for t in ["I", "W", "O"]:
            buf_info = global_buffer_info[t]
            shape_str = ",".join([str(s) for s in buf_info.shape])
            memory_name_big = "__" + buf_info.memory_name.upper() + "__"
            dtype = "int32" if t == "O" else "int8"
            code = CodeStmt(
                code=f"{buf_info.name} = Buffer(<{shape_str}>, {dtype}, {memory_name_big});",
                depth=depth,
            )
            code_list.append(code)

        for name, info in buffer_name_to_info.items():
            if info.memory_name == "global":
                continue
            if name[0] not in ["I", "W", "O"]:
                continue
            shape_str = ",".join([str(s) for s in info.shape])
            memory_name_big = "__" + info.memory_name.upper() + "__"
            dtype = "int32" if name[0] == "O" else "int8"
            code = CodeStmt(
                code=f"{name} = Buffer(<{shape_str}>, {dtype}, {memory_name_big});",
                depth=depth,
            )
            code_list.append(code)

        return code_list

    def codegen_const_buffer_define(self, depth):
        zero_scalar_buffer_name = "zero_scalar_buffer"
        code = CodeStmt(
            code=f"{zero_scalar_buffer_name} = Buffer(<1>, int32, __INPUT_MEMORY__);",
            depth=depth,
        )
        self.buffer_manager.add_buffer(zero_scalar_buffer_name, [1], "input_memory")
        return [code]

    def codegen_cimset(self, depth):
        cim_cfg = get_config()
        mask_num = int(math.ceil(cim_cfg.n_group_vcol / 8)) * 8

        self.buffer_manager.add_buffer("cim_mask_global", [mask_num], "global")
        self.buffer_manager.add_buffer("cim_mask_local", [mask_num], "input_memory")
        code_list = [
            CodeStmt(
                code=f"cim_mask_global = Buffer(<{mask_num}>, int1, __GLOBAL__);",
                depth=depth,
            ),
            CodeStmt(
                code=f"cim_mask_local = Buffer(<{mask_num}>, int1, __INPUT_MEMORY__);",
                depth=depth,
            ),
            CodeStmt(
                code=f"Trans(cim_mask_global, cim_mask_local);",
                depth=depth,
            ),
            CodeStmt(
                code="CIMSet(cim_mask_local);",
                depth=depth,
            ),
        ]
        return code_list

    def codegen_aff(self, aff, arg_list, depth):
        n_domain_iter = aff.dim(isl.dim_type.in_)
        domain_iter_names = [
            aff.get_dim_name(isl.dim_type.in_, i) for i in range(n_domain_iter)
        ]
        coef_list = []
        for i in range(aff.dim(isl.dim_type.in_)):
            coef = aff.get_coefficient_val(isl.dim_type.in_, i)
            coef_list.append(coef)

        code_list = []
        mul_var_list = []
        for coef, var in zip(coef_list, arg_list):
            code, new_var = self._codegen_expr_mul((coef, var), depth)
            code_list.append(code)
            mul_var_list.append(var)

        sum_var = mul_var_list[0]
        for var in mul_var_list[1:]:
            sum_code, sum_var = self._codegen_expr_add((var, sum_var), depth)
            code_list.append(sum_code)

        return code_list, sum_var

    def _codegen_expr_max(self, arg_list, depth):
        assert len(arg_list) == 2, f"{len(arg_list)=}"
        new_var = alloc_unique_var()
        code = CodeStmt(
            code=f"{new_var} = Max({arg_list[0]}, {arg_list[1]});", depth=depth
        )
        return code, new_var

    def _codegen_expr_min(self, arg_list, depth):
        if len(arg_list) > 2:
            pre_code, pre_var = self._codegen_expr_min(arg_list[:-1], depth)
            arg_list = [pre_var, arg_list[-1]]
        else:
            pre_code = []

        assert len(arg_list) == 2, f"{len(arg_list)=}"
        new_var = alloc_unique_var()
        code = CodeStmt(
            code=f"{new_var} = Min({arg_list[0]}, {arg_list[1]});", depth=depth
        )
        return [*pre_code, code], new_var

    def _codegen_expr_add(self, arg_list, depth):
        return self.codegen_expr_binary("+", arg_list, depth)

    def _codegen_expr_sub(self, arg_list, depth):
        return self.codegen_expr_binary("-", arg_list, depth)

    def _codegen_expr_mul(self, arg_list, depth):
        return self.codegen_expr_binary("*", arg_list, depth)

    def _codegen_expr_div(self, arg_list, depth):
        return self.codegen_expr_binary("/", arg_list, depth)

    def _codegen_expr_rem(self, arg_list, depth):
        return self.codegen_expr_binary("%", arg_list, depth)

    def _codegen_expr_minus(self, arg_list, depth):
        assert len(arg_list) == 1, f"{len(arg_list)=}"
        new_var = alloc_unique_var()
        code = CodeStmt(code=f"{new_var} = 0 - {arg_list[0]};", depth=depth)
        return code, new_var

    def _codegen_expr_select(self, expr, depth):
        assert expr.get_op_n_arg() == 3, f"{expr.get_op_n_arg()=}"
        cond = expr.get_op_arg(0)
        true_expr = expr.get_op_arg(1)
        false_expr = expr.get_op_arg(2)

        cond_code_list, cond_var = self.codegen_expression(cond, depth)
        true_code_list, true_var = self.codegen_expression(true_expr, depth)
        false_code_list, false_var = self.codegen_expression(false_expr, depth)

        new_var = alloc_unique_var()
        # code = CodeStmt(
        #     code=f"{new_var} = Select({cond_var}, {true_var}, {false_var});",
        #     depth=depth
        # )
        code = CodeStmt(
            code=f"{new_var} = 0; if ({cond_var}) carry ({new_var}) {{ {new_var} = {true_var}; }} else {{ {new_var} = {false_var}; }};",
            depth=depth,
        )
        return [*cond_code_list, *true_code_list, *false_code_list, code], new_var

    def codegen_expr_ge(self, arg_list, depth):
        return self.codegen_expr_binary(">=", arg_list, depth)

    def codegen_expr_eq(self, arg_list, depth):
        return self.codegen_expr_binary("==", arg_list, depth)

    def codegen_expr_le(self, arg_list, depth):
        return self.codegen_expr_binary("<=", arg_list, depth)

    def codegen_expr_and(self, arg_list, depth):
        return self.codegen_expr_binary("&&", arg_list, depth)

    def codegen_expr_or(self, arg_list, depth):
        return self.codegen_expr_binary("||", arg_list, depth)

    def codegen_expr_binary(self, binary_op, arg_list, depth):
        assert len(arg_list) == 2, f"{len(arg_list)=}"

        new_var = alloc_unique_var()
        code = CodeStmt(
            code=f"{new_var} = {arg_list[0]} {binary_op} {arg_list[1]};", depth=depth
        )
        return [code], new_var

    def codegen_expression_op(self, expr, depth):
        code_list = []
        var_list = []
        for i in range(expr.get_op_n_arg()):
            code, var = self.codegen_expression(expr.get_op_arg(i), depth)
            code_list.extend(code)
            var_list.append(var)

        if expr.get_op_type() == isl._isl.ast_expr_op_type.max:
            code, new_var = self._codegen_expr_max(var_list, depth)
            code_list.append(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.min:
            code, new_var = self._codegen_expr_min(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.add:
            code, new_var = self._codegen_expr_add(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.sub:
            code, new_var = self._codegen_expr_sub(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.mul:
            code, new_var = self._codegen_expr_mul(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.minus:
            code, new_var = self._codegen_expr_minus(var_list, depth)
            code_list.append(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.pdiv_q:
            code, new_var = self._codegen_expr_div(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.pdiv_r:
            code, new_var = self._codegen_expr_rem(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.select:
            code, new_var = self._codegen_expr_select(expr, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.ge:
            code, new_var = self.codegen_expr_ge(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.eq:
            code, new_var = self.codegen_expr_eq(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.le:
            code, new_var = self.codegen_expr_le(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.and_:
            code, new_var = self.codegen_expr_and(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.or_:
            code, new_var = self.codegen_expr_or(var_list, depth)
            code_list.extend(code)
        else:
            print(expr)
            assert False, f"{expr.get_op_type()}"

        return code_list, new_var

    def codegen_expression_id(self, expr, depth):
        # new_var = alloc_unique_var()
        old_var = expr.id_get_id().get_name()
        # code = CodeStmt(code=f"{new_var} = {old_var};", depth=depth)
        # return [code], new_var
        return [], old_var

    def codegen_expression_int(self, expr, depth):
        # new_var = alloc_unique_var()
        int_val = expr.int_get_val()
        # code = CodeStmt(code=f"{new_var} = {int_val};", depth=depth)
        # return [code], new_var
        return [], str(int_val)

    def codegen_expression(self, expr, depth):
        assert isinstance(expr, isl._isl.AstExpr), f"{type(expr)=}"
        if expr.get_type() == isl._isl.ast_expr_type.op:
            return self.codegen_expression_op(expr, depth)
        elif expr.get_type() == isl._isl.ast_expr_type.id:
            return self.codegen_expression_id(expr, depth)
        elif expr.get_type() == isl._isl.ast_expr_type.int:
            return self.codegen_expression_int(expr, depth)
        else:
            print(f"{expr.get_type()=}")
            assert False

    def codegen_cond_to_upperbound_expr(self, cond, depth):
        assert cond.get_type() == isl._isl.ast_expr_type.op, f"{cond.get_type()=}"
        assert (
            cond.get_op_type() == isl._isl.ast_expr_op_type.le
        ), f"{cond.get_op_type()=}"
        assert cond.get_op_n_arg() == 2, f"{cond.get_op_n_arg()=}"
        ub_codes, ub_var = self.codegen_expression(cond.get_op_arg(1), depth)
        add_codes, add_var = self._codegen_expr_add([ub_var, 1], depth)

        return ub_codes + add_codes, add_var

    def codegen_for(self, node, depth):
        body = node.for_get_body()
        cond = node.for_get_cond()
        inc = node.for_get_inc()
        init = node.for_get_init()
        iterator = node.for_get_iterator()

        init_codes, init_var = self.codegen_expression(init, depth)
        ub_codes, ub_var = self.codegen_cond_to_upperbound_expr(cond, depth)

        assert (
            iterator.get_type() == isl._isl.ast_expr_type.id
        ), f"{iterator.get_type()=}"
        iter_name = iterator.id_get_id().get_name()

        assert inc.get_type() == isl._isl.ast_expr_type.int, f"{inc.get_type()=}"
        inc_val = inc.int_get_val()

        for_code = CodeStmt(
            code=f"for {iter_name} in range({init_var}, {ub_var}, {inc_val}) carry (null) {{",
            depth=depth,
        )

        for_code_close = CodeStmt(code="};", depth=depth)
        body_code_list = self.codegen(body, depth + 1)

        total_code_list = [
            *init_codes,
            *ub_codes,
            for_code,
            *body_code_list,
            for_code_close,
        ]
        return total_code_list

    def codegen_block(self, node, depth):
        children = node.block_get_children()
        n_ast_node = children.n_ast_node()
        # print(f"block {n_ast_node=}")
        total_code_list = []
        for i in range(children.n_ast_node()):
            child = children.get_at(i)
            code_list = self.codegen(child, depth)
            total_code_list.extend(code_list)
        return total_code_list

    def codegen_call(self, expr, depth):
        call_name = expr.get_op_arg(0).id_get_id().get_name()

        # print("codegen_call")
        call_args = []
        for arg_id in range(1, expr.get_op_n_arg()):
            arg = expr.get_op_arg(arg_id)
            if arg.get_type() == isl._isl.ast_expr_type.id:
                call_args.append(arg.id_get_id().get_name())
            elif arg.get_type() == isl._isl.ast_expr_type.int:
                call_args.append(str(arg.int_get_val()))

        op = self.name_to_op[call_name]

        if type(op) == DataMovement:

            call_code = self.codegen_call_data_movement(op, call_args, depth)

        elif type(op) == DataMovementOperator:

            call_code = self.codegen_call_cim_compute(op, call_args, depth)

        elif type(op) == tuple:
            op, name = op

            if name == "PartialSum.clear":
                call_code = self.codegen_call_partial_sum_clear(op, call_args, depth)
            elif name == "PartialSum.sum":
                call_code = self.codegen_call_partial_sum_sum(op, call_args, depth)
            elif name == "CIMOutput":
                call_code = self.codegen_call_cim_output(op, call_args, depth)
            else:
                raise NotImplementedError(f"{name=}")

        else:

            call_args_str = ",".join(call_args)
            call_code = CodeStmt(code=f"{call_name}({call_args_str})", depth=depth)
            call_code = [call_code]

        return call_code

    def codegen_call_partial_sum_clear(self, op, call_args, depth):
        assert type(op) == PartialSumDataMovement

        code_list_O, slice_var_O = self.codegen_tensor_access_from_pw_multi_aff(
            op.access_O, call_args, depth
        )
        zero_scalar_var = "zero_scalar_buffer"
        clear_code = CodeStmt(
            code=f"SIMD(VSET, {slice_var_O}, {zero_scalar_var}, {slice_var_O});",
            depth=depth,
        )
        return [*code_list_O, clear_code]

    def codegen_call_partial_sum_sum(self, op, call_args, depth):
        assert type(op) == PartialSumDataMovement

        output_call_args = call_args[: op.level] + call_args[op.level + 1 :]
        code_list_O, slice_var_O = self.codegen_tensor_access_from_pw_multi_aff(
            op.access_O, output_call_args, depth
        )
        code_list_I, slice_var_I = self.codegen_tensor_access_from_pw_multi_aff(
            op.access_I, call_args, depth
        )
        sum_code = CodeStmt(
            code=f"SIMD(VVADD, {slice_var_O}, {slice_var_I}, {slice_var_O});",
            depth=depth,
        )
        return [*code_list_O, *code_list_I, sum_code]

    def codegen_call_cim_output(self, op, call_args, depth):
        assert type(op) == DataMovement, f"{type(op)=}"

        cim_reg_out_buffer = self.buffer_manager.get_buffer_by_memory_name(
            "cim_output_reg_buffer"
        )
        cim_output_code = CodeStmt(
            code=f"CIMOutput({get_config().n_group_vcol}, 0, {cim_reg_out_buffer.name});",
            depth=depth,
        )
        return [cim_output_code]

    def codegen_call_data_movement(self, op, call_args, depth):
        assert type(op) == DataMovement

        # generate offset for each array
        code_list_I, slice_var_I = self.codegen_tensor_access_from_pw_multi_aff(
            op.access_I, call_args, depth
        )
        code_list_O, slice_var_O = self.codegen_tensor_access_from_pw_multi_aff(
            op.access_O, call_args, depth
        )
        trans_code = CodeStmt(code=f"Trans({slice_var_I}, {slice_var_O});", depth=depth)

        return [*code_list_I, *code_list_O, trans_code]

    def codegen_call_cim_compute(self, op, call_args, depth):
        assert type(op) == DataMovementOperator

        # generate offset for each array
        code_list_I, slice_var_I = self.codegen_tensor_access_from_pw_multi_aff(
            op.access_I, call_args, depth
        )
        # code_list_O, slice_var_O = self.codegen_tensor_access_from_pw_multi_aff(op.access_O, call_args, depth)
        code_list_W, slice_var_W = self.codegen_tensor_access_from_pw_multi_aff(
            op.access_W, call_args, depth
        )

        # generate compute code
        compute_code = CodeStmt(
            # code=f"CIMComputeDense({slice_var_I}, {slice_var_O}, {slice_var_W});",
            code=f"CIMComputeDense({slice_var_I}, {slice_var_W});",
            depth=depth,
        )

        # cim output
        # cim_reg_out_buffer = self.buffer_manager.get_buffer_by_memory_name("cim_output_reg_buffer")
        # cim_output_code = CodeStmt(
        #     code=f"CIMOutput({get_config().n_group_vcol}, 0, {cim_reg_out_buffer.name});",
        #     depth=depth,
        # )

        # return [*code_list_I, *code_list_O, *code_list_W, compute_code]
        return [*code_list_I, *code_list_W, compute_code]

    def codegen_tensor_access_from_pw_multi_aff(self, tensor_access, call_args, depth):
        assert type(tensor_access) == TensorAccessRelation
        offsets, sizes = tensor_access.offsets, tensor_access.sizes
        access_offset = self._get_access_from_pw_multi_aff(
            offsets.as_pw_multi_aff(), call_args
        )
        access_size = self._get_access_from_pw_multi_aff(
            sizes.as_pw_multi_aff(), call_args
        )

        code_list = []
        # parse offset
        offset_code_list, offset_var_list = self.codegen_access_indices(
            access_offset, depth
        )
        size_code_list, size_var_list = self.codegen_access_indices(access_size, depth)
        code_list.extend(offset_code_list)
        code_list.extend(size_code_list)
        # import pdb; pdb.set_trace()
        # generate slice code
        slice_var = alloc_unique_var()
        buffer_name = offsets.get_tuple_name(isl.dim_type.out)
        offsets_str = ",".join(offset_var_list)
        sizes_str = ",".join(size_var_list)
        # strides_str = ",".join(["1" for i in range(len(size_var_list))])

        # build slice
        slices = []
        buffer_name_to_info = self.buffer_manager.get_buffer_name_to_info()
        buffer_info = buffer_name_to_info[buffer_name]
        for i, (offset, size) in enumerate(zip(offset_var_list, size_var_list)):
            if size == "1":
                slice = offset
            elif offset == "0" and size == str(buffer_info.shape[i]):
                slice = ":"
            elif offset == "0":
                slice = f":{size}"
            else:
                slice = f"{offset}:{offset} + {size}"
            slices.append(slice)
        slices_str = ",".join(slices)
        
        slice_code = CodeStmt(
            # code=f"{slice_var} = Slice({buffer_name}, [{offsets_str}], [{sizes_str}], [{strides_str}]);",
            code=f"{slice_var} = {buffer_name}[{slices_str}];",
            depth=depth,
        )
        code_list.append(slice_code)

        return code_list, slice_var

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

    def _get_access_from_pw_multi_aff(self, pw_multi_aff, call_args):
        call_args_str = ",".join(call_args)
        context = isl.Set(f"{{ [{call_args_str}] : }}")
        build = isl.AstBuild.from_context(context)
        access = build.access_from_pw_multi_aff(pw_multi_aff)
        return access

    def codegen_user(self, node, depth):
        expr = node.user_get_expr()
        if expr.get_op_type() == isl._isl.ast_expr_op_type.call:
            return self.codegen_call(expr, depth)
        else:
            assert False, f"{expr.get_op_type()}"

    def codegen(self, node, depth=0):
        if node.get_type() == isl._isl.ast_node_type.for_:
            return self.codegen_for(node, depth)
        elif node.get_type() == isl._isl.ast_node_type.block:
            return self.codegen_block(node, depth)
        elif node.get_type() == isl._isl.ast_node_type.user:
            return self.codegen_user(node, depth)
        else:
            assert False, f"{node.get_type()=}"

    def codegen_main_and_end(self, depth):
        main_begin = CodeStmt(code="def main(null<int8>) {", depth=depth)
        main_end = CodeStmt(code="}", depth=depth)
        return [main_begin], [main_end]

    def codegen_str(self, node, indent_unit=4):
        includes = self.codegen_includes(0)
        special_reg_settings = self.codegen_special_settings(1)
        main_begin, main_end = self.codegen_main_and_end(0)
        # cimset_code_list = self.codegen_cimset(1)
        buffer_define_code_list = self.codegen_buffer_define(1)
        const_buffer_define_code_list = self.codegen_const_buffer_define(1)
        execute_code_list = self.codegen(node, 1)
        code_str = ""
        for code_stmt in (
            includes
            + main_begin
            + special_reg_settings
            # + cimset_code_list
            + buffer_define_code_list
            + const_buffer_define_code_list
            + execute_code_list
            + main_end
        ):
            assert type(code_stmt) == CodeStmt, f"{type(code_stmt)=}"
            code = code_stmt.code
            depth = code_stmt.depth
            code = " " * (indent_unit * depth) + code + "\n"
            code_str = code_str + code
        return code_str


def operator_to_ast(op):
    assert type(op) == Operator


def insert_const_dim_in_range(map_, pos, si):
    map_ = map_.insert_dims(isl.dim_type.out, pos, 1)
    val = isl.Val.int_from_si(map_.get_ctx(), si)
    map_ = map_.upper_bound_val(isl.dim_type.out, pos, val)
    map_ = map_.lower_bound_val(isl.dim_type.out, pos, val)
    return map_


def insert_many_const_dim_in_range(map_, pos, size, si):
    map_ = map_.insert_dims(isl.dim_type.out, pos, size)
    val = isl.Val.int_from_si(map_.get_ctx(), si)
    for i in range(size):
        map_ = map_.upper_bound_val(isl.dim_type.out, pos + i, val)
        map_ = map_.lower_bound_val(isl.dim_type.out, pos + i, val)
    return map_


def align_compute_and_assign_schedules(
    compute_schedule, assign_schedules, levels, type_list
):
    level_to_assign_schedule = dict()
    assign_schedule_to_level = dict()
    for assign_schedule, level, type_ in zip(assign_schedules, levels, type_list):
        if level in level_to_assign_schedule:
            level_to_assign_schedule[level].append([assign_schedule, type_])
        else:
            level_to_assign_schedule[level] = [[assign_schedule, type_]]

        assign_schedule_to_level[assign_schedule] = level

    sorted_levels = sorted(list(level_to_assign_schedule.keys()))
    # print(f"{sorted_levels=}")
    # import pdb; pdb.set_trace()

    # insert dims
    for idx, level in enumerate(sorted_levels):
        assign_schedules_at_level = level_to_assign_schedule[level]
        scalar_dim_idx = 0
        # insert constant dims for assign schedules at current level
        for i in range(len(assign_schedules_at_level)):
            assign_schedule, type_ = assign_schedules_at_level[i]
            if type_ in ["I", "W"]:
                assign_schedule = insert_const_dim_in_range(
                    assign_schedule, level + idx, scalar_dim_idx
                )
                assign_schedules_at_level[i] = [assign_schedule, type_]
                scalar_dim_idx += 1

        # insert dims for other schedule
        const = len(assign_schedules_at_level)
        for other_level in sorted_levels:
            if other_level <= level:
                continue
            assign_schedules_at_other_level = level_to_assign_schedule[other_level]
            for i in range(len(assign_schedules_at_other_level)):
                assign_schedules_at_other_level[i][0] = insert_const_dim_in_range(
                    assign_schedules_at_other_level[i][0], level + idx, scalar_dim_idx
                )
                # scalar_dim_idx += 1

        compute_schedule = insert_const_dim_in_range(
            compute_schedule, level + idx, scalar_dim_idx
        )
        scalar_dim_idx += 1

        # insert constant dims for assign schedules at current level
        for i in range(len(assign_schedules_at_level)):
            assign_schedule, type_ = assign_schedules_at_level[i]
            if type_ in ["O", "PartialSum.clear", "PartialSum.sum", "CIMOutput"]:
                assign_schedule = insert_const_dim_in_range(
                    assign_schedule, level + idx, scalar_dim_idx
                )
                assign_schedules_at_level[i] = [assign_schedule, type_]
                scalar_dim_idx += 1
                # print(f"{type_=}")

    # import pdb; pdb.set_trace()

    # padding schedule at end
    max_range_size = compute_schedule.dim(isl.dim_type.out)
    for level in sorted_levels:
        assign_schedules_at_level = level_to_assign_schedule[level]
        for assign_schedule, type_ in assign_schedules_at_level:
            range_size = assign_schedule.dim(isl.dim_type.out)
            max_range_size = max(max_range_size, range_size)

    cur_range_size = compute_schedule.dim(isl.dim_type.out)
    compute_schedule = insert_many_const_dim_in_range(
        compute_schedule, cur_range_size, max_range_size - cur_range_size, 0
    )
    for level in sorted_levels:
        assign_schedules_at_level = level_to_assign_schedule[level]
        for i in range(len(assign_schedules_at_level)):
            assign_schedule, type_ = assign_schedules_at_level[i]
            cur_range_size = assign_schedule.dim(isl.dim_type.out)
            assign_schedule = insert_many_const_dim_in_range(
                assign_schedule, cur_range_size, max_range_size - cur_range_size, 0
            )
            assign_schedules_at_level[i] = [assign_schedule, type_]

    union_schedule = compute_schedule
    for level in sorted_levels:
        assign_schedules_at_level = level_to_assign_schedule[level]
        for assign_schedule, _ in assign_schedules_at_level:
            union_schedule = union_schedule.add_map(assign_schedule)

    # import pdb; pdb.set_trace()

    return union_schedule


def data_movement_operator_to_dsl(op):
    assert type(op) == DataMovementOperator

    name_to_op = dict()

    comp_stmt_name = alloc_unique_stmt()
    compute_schedule = utils.identity_map_from_set(op.domain)
    compute_schedule = compute_schedule.set_tuple_name(isl.dim_type.in_, comp_stmt_name)
    compute_domain = op.domain.set_tuple_name(comp_stmt_name)
    name_to_op[comp_stmt_name] = op

    assign_domain_list = []
    assign_schedule_list = []
    level_list = []
    type_list = []
    for name, data_movement_list in op.data_movement.items():
        # import pdb; pdb.set_trace()
        if name == "O":
            data_movement_list = data_movement_list[::-1]
        for idx, data_movement in enumerate(data_movement_list):
            if name == "O" and idx == 0:  # len(data_movement_list) - 1:
                stmt_name = "CIMOutput." + alloc_unique_stmt()
                assign_domain = data_movement.domain.set_tuple_name(stmt_name)
                assign_schedule = utils.identity_map_from_set(assign_domain)

                assign_domain_list.append(assign_domain)
                assign_schedule_list.append(assign_schedule)
                level_list.append(data_movement.level)
                type_list.append("CIMOutput")
                name_to_op[stmt_name] = (data_movement, "CIMOutput")

            if isinstance(data_movement, PartialSumDataMovement):
                stmt_name = "PartialSum.clear." + alloc_unique_stmt()
                assign_domain = data_movement.domain.set_tuple_name(stmt_name)
                assign_schedule = utils.identity_map_from_set(assign_domain)

                assign_domain_list.append(assign_domain)
                assign_schedule_list.append(assign_schedule)
                level_list.append(data_movement.level)
                type_list.append("PartialSum.clear")
                name_to_op[stmt_name] = (data_movement, "PartialSum.clear")

                stmt_name = "PartialSum.sum" + alloc_unique_stmt()
                assign_domain = data_movement.domain_partial_sum.set_tuple_name(
                    stmt_name
                )
                assign_schedule = utils.identity_map_from_set(assign_domain)
                assign_domain_list.append(assign_domain)
                assign_schedule_list.append(assign_schedule)
                level_list.append(data_movement.level)
                type_list.append("PartialSum.sum")
                name_to_op[stmt_name] = (data_movement, "PartialSum.sum")
            else:
                stmt_name = alloc_unique_stmt()
                assign_domain = data_movement.domain.set_tuple_name(stmt_name)
                assign_schedule = utils.identity_map_from_set(assign_domain)

                assign_domain_list.append(assign_domain)
                assign_schedule_list.append(assign_schedule)
                level_list.append(data_movement.level)
                type_list.append(data_movement.type_)
                name_to_op[stmt_name] = data_movement

    # make union_domain
    union_domain = compute_domain
    for assign_domain in assign_domain_list:
        union_domain = union_domain.add_set(assign_domain)

    # make union_schedule
    union_schedule = align_compute_and_assign_schedules(
        compute_schedule=compute_schedule,
        assign_schedules=assign_schedule_list,
        levels=level_list,
        type_list=type_list,
    )

    ast = utils.gen_ast(union_domain, union_schedule, None)
    code_generator = CodeGenerator(op, name_to_op)
    code = code_generator.codegen_str(ast, 4)
    logger.info(code)
    buffer_manager = code_generator.buffer_manager
    return code, buffer_manager


def codegen_pass(op_list):
    new_op_list = []
    for idx, op in level_tqdm(enumerate(op_list)):
        if type(op) == DataMovementOperator:
            dsl, buffer_manager = data_movement_operator_to_dsl(op)
            op.dsl = dsl
            op.buffer_manager = buffer_manager
            new_op_list.append(op)
        else:
            assert False, f"{type(op)=}"

            # exit()
    return new_op_list


def get_aff_str(aff, domain_vector):
    n_domain_iter = aff.dim(isl.dim_type.in_)
    domain_iter_names = [
        aff.get_dim_name(isl.dim_type.in_, i) for i in range(n_domain_iter)
    ]
    coef_list = []
    for i in range(aff.dim(isl.dim_type.in_)):
        coeff = aff.get_coefficient_val(isl.dim_type.in_, i)
        coef_list.append(coeff)

    # return pw_aff.to_str()


if __name__ == "__main__":
    acc_rel = isl.Map("{ [i,j] -> A[i + j, j, 4] }")
    pw_multi_aff = acc_rel.as_pw_multi_aff()
    print(pw_multi_aff)
    multi_aff = pw_multi_aff.as_multi_aff()
    # get each aff
    aff = multi_aff.get_at(0)
    get_aff_str(aff, None)
    # aff = aff.get_div(0)
    # print(aff.to_str())
    # print(aff.get_div(0))
    # get_pw_aff_str(aff, isl.Set("{ [i,j] }"))
    # for aff in aff_list:
    #     print(aff)

    # for pw_aff in multi_pw_aff.get_pw_aff_list():
    #     print(pw_aff)
    # domain_iters = ["a0", "a1"]
    # acc_rel = acc_rel.set_dim_id(isl.dim_type.in_, 0, isl.Id(domain_iters[0]))
    # acc_rel = acc_rel.set_dim_id(isl.dim_type.in_, 1, isl.Id(domain_iters[1]))
    # # take [i0, i1] into acc_rel, and get range expression
    # # domain = isl.Set("{ [i0, i1] }")
    # # acc_rel = acc_rel.intersect_domain(domain)
    # print(acc_rel.to_str())
    # # pma = acc_rel.as_pw_multi_aff()
    # # print(pma)
    # print(isl.stat.ok)
