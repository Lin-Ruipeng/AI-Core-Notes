// cholesky分解
// 编译需要包含头文件目录 g++ main.cpp -O3 -march=native -I /usr/include/eigen3
#include <Eigen/Dense> // Eigen库
#include <iostream>

auto main() -> int {
  const size_t N = 10;

  // 希尔伯特矩阵H 1.0 / ( i + j - 1),但是得注意到编程语言从0开始索引!
  Eigen::MatrixXd mat_H(N, N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      mat_H(i, j) = 1.0 / (i + j + 1);
    }
  }

  // 真实解X_true 全是 1.0
  Eigen::VectorXd x_true = Eigen::VectorXd::Ones(N);

  // b向量
  Eigen::VectorXd b = mat_H * x_true;

  // 方法A: 求逆矩阵求解 H^{-1} * b
  Eigen::VectorXd x_a = mat_H.inverse() * b;

  // 方法B: 用Cholesky分解 HPC最佳实践
  Eigen::VectorXd x_b = mat_H.llt().solve(b);

  // 方法C: 暴力破解 LU分解(不是正定矩阵就得用这个)
  Eigen::VectorXd x_c = mat_H.lu().solve(b);

  // 报告误差范数
  std::cout << "解线性方程组的误差报告:\n"
            << "\n求逆矩阵法的误差范数: " << (x_a - x_true).norm()
            << "\nCholesky分解误差范数: " << (x_b - x_true).norm()
            << "\nLU分解法的误差范数  : " << (x_c - x_true).norm() << std::endl;
}
