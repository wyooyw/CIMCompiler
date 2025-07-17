from dataclasses import dataclass

import islpy as isl

unique_name_idx = 0
char_set = "abcdefghijklmnopqrstuvwxyz"


def alloc_unique_var():
    global unique_name_idx

    idx = unique_name_idx
    char_set_len = len(char_set)
    name = "_"
    while True:
        char = idx % char_set_len
        idx = idx // char_set_len
        name += char_set[char]
        if idx == 0:
            break

    unique_name_idx += 1
    return name


@dataclass
class CodeStmt:
    code: str
    depth: int


def _codegen_expr_max(arg_list, depth):
    assert len(arg_list) == 2, f"{len(code_list)=}, {len(arg_list)=}"
    new_var = alloc_unique_var()
    code = CodeStmt(code=f"{new_var} = max({arg_list[0]}, {arg_list[1]});", depth=depth)
    return code, new_var


def _codegen_expr_min(arg_list, depth):
    assert len(arg_list) == 2, f"{len(code_list)=}, {len(arg_list)=}"
    new_var = alloc_unique_var()
    code = CodeStmt(code=f"{new_var} = min({arg_list[0]}, {arg_list[1]});", depth=depth)
    return code, new_var


def _codegen_expr_add(arg_list, depth):
    assert len(arg_list) == 2, f"{len(code_list)=}, {len(arg_list)=}"
    new_var = alloc_unique_var()
    code = CodeStmt(code=f"{new_var} = {arg_list[0]} + {arg_list[1]};", depth=depth)
    return code, new_var


def _codegen_expr_sub(arg_list, depth):
    assert len(arg_list) == 2, f"{len(code_list)=}, {len(arg_list)=}"
    new_var = alloc_unique_var()
    code = CodeStmt(code=f"{new_var} = {arg_list[0]} - {arg_list[1]};", depth=depth)
    return code, new_var


def _codegen_expr_mul(arg_list, depth):
    assert len(arg_list) == 2, f"{len(code_list)=}, {len(arg_list)=}"
    new_var = alloc_unique_var()
    code = CodeStmt(code=f"{new_var} = {arg_list[0]} - {arg_list[1]};", depth=depth)
    return code, new_var


def _codegen_expr_minus(arg_list, depth):
    assert len(arg_list) == 1, f"{len(code_list)=}, {len(arg_list)=}"
    new_var = alloc_unique_var()
    code = CodeStmt(code=f"{new_var} = 0 - {arg_list[0]};", depth=depth)
    return code, new_var


def codegen_expression_op(expr, depth):
    code_list = []
    var_list = []
    for i in range(expr.get_op_n_arg()):
        code, var = codegen_expression(expr.get_op_arg(i), depth)
        code_list.extend(code)
        var_list.append(var)

    if expr.get_op_type() == isl._isl.ast_expr_op_type.max:
        code, new_var = _codegen_expr_max(var_list, depth)
        code_list.append(code)
    elif expr.get_op_type() == isl._isl.ast_expr_op_type.min:
        code, new_var = _codegen_expr_min(var_list, depth)
        code_list.append(code)
    elif expr.get_op_type() == isl._isl.ast_expr_op_type.add:
        code, new_var = _codegen_expr_add(var_list, depth)
        code_list.append(code)
    elif expr.get_op_type() == isl._isl.ast_expr_op_type.sub:
        code, new_var = _codegen_expr_sub(var_list, depth)
        code_list.append(code)
    elif expr.get_op_type() == isl._isl.ast_expr_op_type.mul:
        code, new_var = _codegen_expr_mul(var_list, depth)
        code_list.append(code)
    elif expr.get_op_type() == isl._isl.ast_expr_op_type.minus:
        code, new_var = _codegen_expr_minus(var_list, depth)
        code_list.append(code)
    else:
        print(expr)
        assert False, f"{expr.get_op_type()}"

    return code_list, new_var


def codegen_expression_id(expr, depth):
    new_var = alloc_unique_var()
    old_var = expr.id_get_id().get_name()
    code = CodeStmt(code=f"{new_var} = {old_var};", depth=depth)
    return [code], new_var


def codegen_expression_int(expr, depth):
    new_var = alloc_unique_var()
    int_val = expr.int_get_val()
    code = CodeStmt(code=f"{new_var} = {int_val};", depth=depth)
    return [code], new_var


def codegen_expression(expr, depth):
    assert isinstance(expr, isl._isl.AstExpr), f"{type(expr)=}"
    if expr.get_type() == isl._isl.ast_expr_type.op:
        return codegen_expression_op(expr, depth)
    elif expr.get_type() == isl._isl.ast_expr_type.id:
        return codegen_expression_id(expr, depth)
    elif expr.get_type() == isl._isl.ast_expr_type.int:
        return codegen_expression_int(expr, depth)


def codegen_cond_to_upperbound_expr(cond, depth):
    assert cond.get_type() == isl._isl.ast_expr_type.op, f"{cond.get_type()=}"
    assert cond.get_op_type() == isl._isl.ast_expr_op_type.le, f"{cond.get_op_type()=}"
    assert cond.get_op_n_arg() == 2, f"{cond.get_op_n_arg()=}"
    return codegen_expression(cond.get_op_arg(1), depth)


def codegen_for(node, depth):
    body = node.for_get_body()
    cond = node.for_get_cond()
    inc = node.for_get_inc()
    init = node.for_get_init()
    iterator = node.for_get_iterator()

    init_codes, init_var = codegen_expression(init, depth)
    ub_codes, ub_var = codegen_cond_to_upperbound_expr(cond, depth)

    assert iterator.get_type() == isl._isl.ast_expr_type.id, f"{iterator.get_type()=}"
    iter_name = iterator.id_get_id().get_name()

    assert inc.get_type() == isl._isl.ast_expr_type.int, f"{inc.get_type()=}"
    inc_val = inc.int_get_val()

    for_code = CodeStmt(
        code=f"for {iter_name} in range({init_var}, {ub_var}, {inc_val}) carry (null) {{",
        depth=depth,
    )

    for_code_close = CodeStmt(code="};", depth=depth)
    body_code_list = codegen(body, depth + 1)

    total_code_list = [
        *init_codes,
        *ub_codes,
        for_code,
        *body_code_list,
        for_code_close,
    ]
    return total_code_list


def codegen_block(node, depth):
    children = node.block_get_children()
    n_ast_node = children.n_ast_node()
    print(f"block {n_ast_node=}")
    total_code_list = []
    for i in range(children.n_ast_node()):
        child = children.get_at(i)
        code_list = codegen(child, depth)
        total_code_list.extend(code_list)
    return total_code_list


def codegen_call(expr, depth):
    call_name = expr.get_op_arg(0).id_get_id().get_name()
    call_args = []
    for arg_id in range(1, expr.get_op_n_arg()):
        arg = expr.get_op_arg(arg_id)
        if arg.get_type() == isl._isl.ast_expr_type.id:
            call_args.append(arg.id_get_id().get_name())
        elif arg.get_type() == isl._isl.ast_expr_type.int:
            call_args.append(str(arg.int_get_val()))
    call_args = ",".join(call_args)
    call_code = CodeStmt(code=f"{call_name}({call_args})", depth=depth)
    return [call_code]


def codegen_user(node, depth):
    expr = node.user_get_expr()
    if expr.get_op_type() == isl._isl.ast_expr_op_type.call:
        return codegen_call(expr, depth)
    else:
        assert False, f"{expr.get_op_type()}"


def codegen(node, depth=0):
    if node.get_type() == isl._isl.ast_node_type.for_:
        return codegen_for(node, depth)
    elif node.get_type() == isl._isl.ast_node_type.block:
        return codegen_block(node, depth)
    elif node.get_type() == isl._isl.ast_node_type.user:
        return codegen_user(node, depth)
    else:
        assert False, f"{node.get_type()=}"


def codegen_str(node, indent_unit=4):
    code_list = codegen(node, 0)
    code_str = ""
    for code_stmt in code_list:
        code = code_stmt.code
        depth = code_stmt.depth
        code = " " * (indent_unit * depth) + code + "\n"
        code_str = code_str + code
    return code_str


if __name__ == "__main__":
    # acc_rel = isl.Map("{ [i,j] -> A[i + j, j, 4] }")
    # domain_iters = ["a0", "a1"]
    # acc_rel = acc_rel.set_dim_id(isl.dim_type.in_, 0, isl.Id(domain_iters[0]))
    # acc_rel = acc_rel.set_dim_id(isl.dim_type.in_, 1, isl.Id(domain_iters[1]))
    # # take [i0, i1] into acc_rel, and get range expression
    # # domain = isl.Set("{ [i0, i1] }")
    # # acc_rel = acc_rel.intersect_domain(domain)
    # print(acc_rel.to_str())
    # # pma = acc_rel.as_pw_multi_aff()
    # # print(pma)
    print(isl.stat.ok)
