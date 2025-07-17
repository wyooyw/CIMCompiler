#include <unsupported/Eigen/CXX11/Tensor>

void load_from_file(Eigen::Tensor<int, Eigen::Dynamic> &tensor, const std::string &file_path) {
    std::ifstream file(file_path);
    int value;
    for (int i = 0; i < tensor.size(); ++i) {
        file >> value;
        tensor.data()[i] = value;
    }
}

void save_to_file(Eigen::Tensor<int, Eigen::Dynamic> &tensor, const std::string &file_path) {
    std::ofstream file(file_path);
    for (int i = 0; i < tensor.size(); ++i) {
        file << tensor.data()[i] << "\n";
    }
}

int main() {
    Eigen::Tensor<int, 2> A(10,10);
    Eigen::Tensor<int, 2> B(10,10);
    load_from_file(B, "B.txt");
    int a_ = 0;
    int b_ = 9;
    int c_ = b_ + 1;
    for(int c0 = a_; c0 < c_; c0 += 1) {
        int d_ = 0;
        int e_ = 9;
        int f_ = e_ + 1;
        for(int c1 = d_; c1 < f_; c1 += 1) {
            int g_ = c0;
            int h_ = c1;
            int i_ = c1;
            int j_ = c0;
            A(g_,h_) = B(i_,j_);
        }
    }
    save_to_file(A, "A.txt");
}
