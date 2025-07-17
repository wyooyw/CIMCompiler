import re


def get_code_list(file_path):
    code_list = []
    with open(file_path, "r") as file:
        for line in file.readlines():
            line = line.strip()
            code_list.append(line)
    return code_list


def get_all_vars(code_list):
    """
    Pattern match:
    int kc_ = xxx;
    """
    all_vars = []
    pattern = r"int (\w+)_ = "
    for i, code in enumerate(code_list):
        print(f"{code=}")
        match = re.search(pattern, code)
        if match:
            var_name = match.group(1) + "_"
            all_vars.append((i, var_name))

    return all_vars


def find_whole_word(word, text):
    # 使用 \b 来匹配单词边界
    pattern = r"\b" + re.escape(word) + r"\b"
    matches = re.findall(pattern, text)
    return matches


def detect_unuse_var(code_list, all_vars):
    """ """
    unuse_vars = []
    for def_line_id, var in all_vars:
        is_used = False

        for line_id, code in enumerate(code_list):
            if line_id == def_line_id:
                continue
            if find_whole_word(var, code):
                is_used = True
                break
        if not is_used:
            unuse_vars.append(var)
    return unuse_vars


def count_constant_var(code_list):
    """ """
    constant_vars = []
    pattern = r"int (\w+)_ = \d+;"
    for code in code_list:
        match = re.search(pattern, code)
        if match:
            constant_vars.append(match.group(1))
    return constant_vars


if __name__ == "__main__":
    file_path = ".temp/codegen_test.cpp"
    code_list = get_code_list(file_path)
    print(f"{len(code_list)=}")
    all_vars = get_all_vars(code_list)
    print(f"{len(all_vars)=}")
    unuse_vars = detect_unuse_var(code_list, all_vars)
    print(f"{unuse_vars=}")
    # constant_vars = count_constant_var(code_list)
    # print(f"{constant_vars=}")
    # print(f"{len(constant_vars)=}")
