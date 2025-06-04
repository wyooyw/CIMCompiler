from dataclasses import dataclass

import islpy as isl


@dataclass
class CodeStmt:
    code: str
    depth: int


unique_name_idx = 0
char_set = "abcdefghijklmnopqrstuvwxyz"


def alloc_unique_var():
    global unique_name_idx

    idx = unique_name_idx
    char_set_len = len(char_set)
    name = ""
    while True:
        char = idx % char_set_len
        idx = idx // char_set_len
        name += char_set[char]
        if idx == 0:
            break

    unique_name_idx += 1
    return name + "_"


unique_stmt_idx = 0
char_stmt_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def alloc_unique_stmt():
    global unique_stmt_idx

    idx = unique_stmt_idx
    char_set_len = len(char_stmt_set)
    name = ""
    while True:
        char = idx % char_set_len
        idx = idx // char_set_len
        name += char_stmt_set[char]
        if idx == 0:
            break

    unique_stmt_idx += 1
    return name + "_"


class Codegen:
    def __init__(self):
        pass

    def get_access_from_pw_multi_aff(self, pw_multi_aff, call_args):
        call_args_str = ",".join(call_args)
        context = isl.Set(f"{{ [{call_args_str}] : }}")
        build = isl.AstBuild.from_context(context)
        access = build.access_from_pw_multi_aff(pw_multi_aff)
        return access
