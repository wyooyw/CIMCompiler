import islpy as isl

from polycim.codegen_.codegen import Codegen, CodeStmt, alloc_unique_var


class CCodeGenerator(Codegen):
    def __init__(self):
        self.var_to_const = {}

    def get_var(self, var):
        if var in self.var_to_const:
            return self.var_to_const[var]
        else:
            return var

    def set_var_const(self, var, const):
        self.var_to_const[var] = const

    def codegen_buffer_define(self, depth):
        raise NotImplementedError

    def _codegen_expr_max(self, arg_list, depth):
        if len(arg_list) > 2:
            pre_code, pre_var = self._codegen_expr_max(arg_list[:-1], depth)
            arg_list = [pre_var, arg_list[-1]]
        else:
            pre_code = []

        assert len(arg_list) == 2, f"{len(arg_list)=}"
        new_var = alloc_unique_var()
        code = CodeStmt(
            code=f"int {new_var} = std::max({arg_list[0]}, {arg_list[1]});", depth=depth
        )
        return [*pre_code, code], new_var

    def _codegen_expr_min(self, arg_list, depth):
        if len(arg_list) > 2:
            pre_code, pre_var = self._codegen_expr_min(arg_list[:-1], depth)
            arg_list = [pre_var, arg_list[-1]]
        else:
            pre_code = []

        assert len(arg_list) == 2, f"{len(arg_list)=}"
        new_var = alloc_unique_var()
        code = CodeStmt(
            code=f"int {new_var} = std::min({arg_list[0]}, {arg_list[1]});", depth=depth
        )
        return [*pre_code, code], new_var

    def _codegen_expr_fdiv_q(self, arg_list, depth):
        assert len(arg_list) == 2, f"{len(arg_list)=}"
        new_var = alloc_unique_var()
        code = CodeStmt(
            code=f"int {new_var} = fdiv_q({arg_list[0]}, {arg_list[1]});", depth=depth
        )
        return [code], new_var

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
        code = CodeStmt(code=f"int {new_var} = 0 - {arg_list[0]};", depth=depth)
        return [code], new_var

    def _codegen_expr_select(self, arg_list, depth):
        assert len(arg_list) == 3, f"{len(arg_list)=}"
        cond_var = arg_list[0]
        true_var = arg_list[1]
        false_var = arg_list[2]

        new_var = alloc_unique_var()
        code = CodeStmt(
            code=f"int {new_var} = {cond_var} ? {true_var} : {false_var};",
            depth=depth,
        )
        return [code], new_var

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
            code=f"int {new_var} = {arg_list[0]} {binary_op} {arg_list[1]};",
            depth=depth,
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
            code_list.extend(code)
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
            code_list.extend(code)
        elif expr.get_op_type() in (
            isl._isl.ast_expr_op_type.pdiv_q,
            isl._isl.ast_expr_op_type.div,
        ):
            code, new_var = self._codegen_expr_div(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.fdiv_q:
            code, new_var = self._codegen_expr_fdiv_q(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() in (
            isl._isl.ast_expr_op_type.pdiv_r,
            isl._isl.ast_expr_op_type.zdiv_r,
        ):
            code, new_var = self._codegen_expr_rem(var_list, depth)
            code_list.extend(code)
        elif expr.get_op_type() == isl._isl.ast_expr_op_type.select:
            code, new_var = self._codegen_expr_select(var_list, depth)
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
        new_var = alloc_unique_var()
        old_var = expr.id_get_id().get_name()
        # code = CodeStmt(code=f"int {new_var} = {old_var};", depth=depth)
        # return [code], new_var
        self.set_var_const(new_var, old_var)
        return [], old_var

    def codegen_expression_int(self, expr, depth):
        new_var = alloc_unique_var()
        int_val = expr.int_get_val()
        # code = CodeStmt(code=f"int {new_var} = {int_val};", depth=depth)
        # return [code], new_var
        self.set_var_const(new_var, int_val)
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
        assert cond.get_op_n_arg() == 2, f"{cond.get_op_n_arg()=}"
        if cond.get_op_type() == isl._isl.ast_expr_op_type.le:
            ub_codes, ub_var = self.codegen_expression(cond.get_op_arg(1), depth)
            add_codes, add_var = self._codegen_expr_add([ub_var, 1], depth)
        elif cond.get_op_type() == isl._isl.ast_expr_op_type.lt:
            ub_codes, ub_var = self.codegen_expression(cond.get_op_arg(1), depth)
            add_codes, add_var = self._codegen_expr_add([ub_var, 0], depth)
        else:
            assert False, f"{cond.get_op_type()=}"

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

        # for_code = CodeStmt(
        #     code=f"for {iter_name} in range({init_var}, {ub_var}, {inc_val}) carry (null) {{",
        #     depth=depth,
        # )
        for_code = CodeStmt(
            code=f"for(int {iter_name} = {init_var}; {iter_name} < {ub_var}; {iter_name} += {inc_val}) {{",
            depth=depth,
        )

        for_code_close = CodeStmt(code="}", depth=depth)
        body_code_list = self.codegen(body, depth + 1)

        total_code_list = [
            *init_codes,
            *ub_codes,
            for_code,
            *body_code_list,
            for_code_close,
        ]
        return total_code_list

    def codegen_if(self, node, depth):
        cond = node.if_get_cond()
        cond_codes, cond_var = self.codegen_expression(cond, depth)

        then_body = node.if_get_then_node()
        then_body_codes = self.codegen(then_body, depth + 1)

        total_code_list = [
            *cond_codes,
            CodeStmt(code=f"if ({cond_var}) {{", depth=depth),
            *then_body_codes,
            CodeStmt(code="}", depth=depth),
        ]

        if node.if_has_else():
            else_body = node.if_get_else_node()
            else_body_codes = self.codegen(else_body, depth + 1)
            total_code_list.extend(
                [
                    CodeStmt(code="else {", depth=depth),
                    *else_body_codes,
                    CodeStmt(code="}", depth=depth),
                ]
            )

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

    def codegen_call_fn(self, call_name, call_args, depth):
        assert False, f"Not implemented"

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

        # op = self.name_to_op[call_name]

        # try:
        call_code = self.codegen_call_fn(call_name, call_args, depth)
        # except:
        #     call_args_str = ",".join(call_args)
        #     call_code = CodeStmt(code=f"{call_name}({call_args_str})", depth=depth)
        #     call_code = [call_code]

        return call_code

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
        strides_str = ",".join(["1" for i in range(len(size_var_list))])
        slice_code = CodeStmt(
            code=f"{slice_var} = Slice({buffer_name}, [{offsets_str}], [{sizes_str}], [{strides_str}]);",
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
        elif node.get_type() == isl._isl.ast_node_type.if_:
            return self.codegen_if(node, depth)
        else:
            assert False, f"{node.get_type()=}"

    def codegen_main_and_end(self, depth):
        main_begin = CodeStmt(code="int main(int argc, char* argv[]) {", depth=depth)
        main_end = CodeStmt(code="}", depth=depth)
        return [main_begin], [main_end]

    def codegen_argument_check(self, depth, option_names):
        code_str = [
            f"if (argc != {len(option_names) + 1}) {{",
            f"std::cerr << \"Usage: \" << argv[0] << \" {' '.join(option_names)}\" << std::endl;",
            "return 1;",
            "}",
        ]
        argument_check = [
            CodeStmt(code=code_str[0], depth=depth),
            CodeStmt(code=code_str[1], depth=depth + 1),
            CodeStmt(code=code_str[2], depth=depth + 1),
            CodeStmt(code=code_str[3], depth=depth),
        ]
        return argument_check

    def codegen_fdiv_q(self, depth):
        code_str = """
// fdiv_q is not supported in C, so we need to use floor to implement it
int fdiv_q(int dividend, int divisor) {
    int quotient = dividend / divisor;
    if ((dividend % divisor != 0) && ((dividend < 0) != (divisor < 0))) {
        quotient--;
    }
    return quotient;
}"""
        code_fdiv_q = CodeStmt(code=code_str, depth=depth)
        return [code_fdiv_q]

    def codegen_helper_functions(self, depth):
        code_fdiv_q = self.codegen_fdiv_q(depth)
        return code_fdiv_q

    def codegen_str(self, node, indent_unit=4):
        # special_reg_defs = self.codegen_special_defs(0)
        # special_reg_settings = self.codegen_special_settings(1)
        main_begin, main_end = self.codegen_main_and_end(0)
        buffer_define_code_list = self.codegen_buffer_define(1)
        execute_code_list = self.codegen(node, 1)
        code_str = ""
        for code_stmt in (
            # special_reg_defs
            main_begin
            # special_reg_settings
            + buffer_define_code_list
            + execute_code_list
            + main_end
        ):
            assert type(code_stmt) == CodeStmt, f"{type(code_stmt)=}"
            code = code_stmt.code
            depth = code_stmt.depth
            code = " " * (indent_unit * depth) + code + "\n"
            code_str = code_str + code
        return code_str
