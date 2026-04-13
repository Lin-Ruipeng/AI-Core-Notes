// SVD 奇异值分解
// 编译需要包含头文件目录 g++ main.cpp -O3 -march=native -I /usr/include/eigen3
#include <Eigen/Dense>
#include <iostream>

auto main() -> int {
  const size_t ROWS = 3; // 行含义：三次测量
  const size_t COLS = 2; // 列含义：确定两轴

  // IMU观测矩阵
  Eigen::MatrixXd matix_A(ROWS, COLS);
  matix_A << 1, 2.0001, 2, 4.0001, 3, 6.0001;

  // 理论解
  Eigen::VectorXd x_true(COLS);
  x_true << 1.0, 1.0;

  // 向量b
  Eigen::VectorXd b = matix_A * x_true;

  // 方法 A（自杀式求解）
  Eigen::MatrixXd matix_ATA = matix_A.transpose() * matix_A;
  Eigen::LLT<Eigen::MatrixXd> llt(matix_ATA);

  if (llt.info() != Eigen::Success) {
    std::cout << "LLT 分解失败，矩阵不正定\n";
  } else {
    // A乘以了A^T, b也要乘!
    Eigen::VectorXd x_a = llt.solve(matix_A.transpose() * b);
    std::cout << "方法A的解算结果为: " << x_a.transpose() << "\n";
    std::cout << "方法A的误差范数为: " << (x_a - x_true).norm() << "\n";
  }

  // 方法 B（黑盒伪逆）
  Eigen::VectorXd x_b =
      matix_A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
  std::cout << "方法B的解算结果为: " << x_b.transpose() << "\n";
  std::cout << "方法B的误差范数为: " << (x_b - x_true).norm() << "\n";

  // 方法 C（HPC 手动截断 SVD）
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matix_A, Eigen::ComputeThinU |
                                                     Eigen::ComputeThinV);
  Eigen::VectorXd S = svd.singularValues(); // 获取奇异值
  std::cout << "方法C的奇异值向量: " << S.transpose() << "\n";
  auto tolerance = 1e-10 * matix_A.rows() * S(0); // 手动设定阈值
  Eigen::VectorXd S_inv = S;                      // 先继承原始值
  for (int i = 0; i < S.size(); ++i) {
    if (S(i) > tolerance) {
      S_inv(i) = 1.0 / S(i); // 大于阈值存倒数
    } else {
      S_inv(i) = 0.0; // 小于阈值存0
    }
  }
  // $x = V \Sigma^{-1}_{truncated} U^T b$
  Eigen::VectorXd x_c =
      svd.matrixV() * S_inv.asDiagonal() * svd.matrixU().transpose() * b;
  // S_inv取过倒数了!

  std::cout << "方法C的解算结果为: " << x_c.transpose() << "\n";
  std::cout << "方法C的误差范数为: " << (x_c - x_true).norm() << "\n";
}
