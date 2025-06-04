#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

int main() {
    // 创建一个3x3x3的张量
    Eigen::Tensor<int, 3> tensor(3, 3, 3);

    // 初始化张量
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                tensor(i, j, k) = i + j + k;
            }
        }
    }

    // 打印张量
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                std::cout << "tensor(" << i << ", " << j << ", " << k << ") = " << tensor(i, j, k) << std::endl;
            }
        }
    }

    return 0;
}