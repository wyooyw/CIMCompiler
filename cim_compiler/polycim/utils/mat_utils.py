import sympy
from sympy import Matrix, lcm


def scale_to_integer_per_row(A_inv):
    # 对每一行分别求最小公倍数和缩放
    A_inv_int = Matrix([[]])
    for row_idx in range(A_inv.rows):
        row = A_inv[row_idx, :]

        # 提取行中每个元素的分母
        denominators = [elem.as_numer_denom()[1] for elem in row if elem.is_Rational]

        # 计算最小公倍数
        if denominators:  # 确保分母列表不为空
            lcm_denominator = lcm(denominators)
        else:
            lcm_denominator = 1
        # print(f"{lcm_denominator=}")
        # 将行乘以最小公倍数以得到整数行
        int_row = row * lcm_denominator

        # 将整数行添加到结果矩阵中
        A_inv_int = A_inv_int.col_join(Matrix(int_row))

    return A_inv_int


def scale_to_integer(A_inv):
    # 提取逆矩阵中每个元素的分母
    denominators = [elem.as_numer_denom()[1] for elem in A_inv if elem.is_Rational]
    # print(f"{denominators=}")
    # 计算最小公倍数
    if denominators:  # 确保分母列表不为空
        lcm_denominator = lcm(denominators)
    else:
        lcm_denominator = 1
    # print(f"{lcm_denominator=}")
    # 将逆矩阵乘以最小公倍数以得到整数矩阵
    A_inv_int = A_inv * lcm_denominator
    return A_inv_int


def inv_scale_to_integer(A, scale_per_row=False):
    # print(f"{A=}")
    # 检查矩阵A是否是方阵
    assert A.shape[0] == A.shape[1], "矩阵A不是方阵"

    # 检查矩阵A的每个元素是否都是整数且是常数
    assert all(
        isinstance(entry, sympy.core.numbers.Integer) for entry in A
    ), "矩阵A的元素不全是整数"

    # 计算逆矩阵
    A_inv = A.inv()
    # print(f"{A_inv=}")
    if scale_per_row:
        A_inv_int = scale_to_integer_per_row(A_inv)
    else:
        A_inv_int = scale_to_integer(A_inv)
    # print(f"{A_inv_int=}")
    # 输出结果
    return A_inv_int


def find_independent_columns(A):
    _, inds = A.rref()  # to check the rows you need to transpose!
    A = A[:, inds]
    return A


def find_independent_rows(A):
    At = A.transpose()
    _, inds = At.rref()  # to check the rows you need to transpose!
    At = At[:, inds]
    return At.transpose()


if __name__ == "__main__":
    A = Matrix([[1, 2], [3, 4]])
    # for entry in A:
    #     is_integer = isinstance(entry, sympy.core.numbers.Integer)
    #     print(type(entry), entry, is_integer)

    # exit()
    A_inv_int = inv_scale_to_integer(A, scale_per_row=True)
    # print(A_inv_int)
