#include <Eigen/Dense> // Eigen库 -I /usr/include/eigen3
#include <cmath>
#include <iostream>
#include <omp.h> // -fopenmp
#include <random>
#include <vector>

// 原始方法计算: 输入z_i 输出这个点的概率密度
auto calc_direct_pdf(const Eigen::Vector3d &z, const Eigen::Vector3d &u,
                     const Eigen::MatrixXd &Sigma) -> double {
  // 公式: P(z) = \frac{1}{\sqrt{(2\pi)^D |\Sigma|}}
  // \exp\left( -\frac{1}{2} (z-\mu)^T \Sigma^{-1} (z-\mu) \right)
  constexpr int D = 3;            // 三维
  Eigen::Vector3d diff = (z - u); // 向量差值

  // 计算马氏距离平方
  double mahal_dist_sq = (diff.transpose() * Sigma.inverse() * diff).value();

  // 归一化系数
  double det = Sigma.determinant();
  double norm = 1.0 / std::sqrt(std::pow(2.0 * M_PI, D) * det);

  return norm * std::exp(-0.5 * mahal_dist_sq);
  // return 1.0 / (std::sqrt(2.0 * M_PI) * 2.0 * M_PI * Sigma.determinant()) *
  //        std::exp(-0.5 * (z - u).transpose() * Sigma.inverse() * (z - u));
}

// 取对数之后公式
// \log P(z) = -\frac{1}{2} \left[ \underbrace{D\log(2\pi)}_{\text{常数项}} +
// \underbrace{\log|\Sigma|}_{\text{行列式项}} + \underbrace{(z-\mu)^T
// \Sigma^{-1} (z-\mu)}_{\text{马氏距离项}} \right]

// 取对数方式计算: 输入z_i 输出这个点的概率密度

// 本函数计算取log之后的常数项
auto calc_log_pdf_const(const Eigen::Vector3d &z, const Eigen::Vector3d &u,
                        const Eigen::MatrixXd &L) -> double {
  // 取对数之后的常数项: -1/2 * (D log (2*pi) + log |Sigma|)
  constexpr int D = 3; // 三维
  // (优化点1) LLT分解之后 Sigma 的行列式就是 L对角线元素乘积之后的平方
  // 所以 log |Sigma| = log |L|^2 = 2 log |L|
  // L 又是对角阵, 只需要对角元素相乘就是行列式值
  double log_det = 2.0 * log(L.diagonal().prod());
  // 解释一下-1.5是因为本来要求根号下 2pi的三次方的倒数,
  // 所以三次方和倒数被提出log, 外层本来就要除以2
  // 全部放一起就是 - 3 / 2 = -1.5的效果
  return -1.5 * log(2.0 * M_PI) - 0.5 * log_det;
}

// 本函数计算马氏距离 (z-u)^T * Sigme^-1 * (z-u)
auto calc_log_pdf_mahal(const Eigen::Vector3d &z, const Eigen::Vector3d &u,
                        const Eigen::MatrixXd &L) -> double {
  constexpr int D = 3;            // 三维
  Eigen::Vector3d diff = (z - u); // 向量差值

  // (优化点2) 正定对称矩阵不要直接求逆! 要用Cholesky分解

  // 回想一下, 前面说了 x = A^-1 * b
  // 不要这样写: x = A.inverse() * b;
  // 而要这样写: x = A.llt().solve(b);

  // 本函数需要返回 (z-u)^T * Sigma^-1 * (z-u)
  // 注意到后面两个矩阵的相乘和前面的求解是一致的!
  // return diff.transpose() * Sigma.llt().solve(diff);

  // 但是还有更优的优化!
  // 仔细看: (z-u)^T * L * L^T * (z-u) 作于互为转置!
  // 所以只需要算出 L^T (z-u) 的范数的平方!
  // return (L.solve(diff)).squaredNorm();

  // 因为参数类型不能表答处L是一个下三角阵, 所以不能直接.solve
  Eigen::Vector3d y = L.triangularView<Eigen::Lower>().solve(diff);
  return y.squaredNorm();
}

auto main() -> int {
  // 1. 产生数据
  // 协方差矩阵 Sigma (对角线为0.01)
  Eigen::MatrixXd Sigma(3, 3);
  Sigma << 100.0, 0, 0, 0, 100.0, 0, 0, 0, 100.0;
  // 均值向量 u (0)
  Eigen::Vector3d u;
  u << 0, 0, 0;
  // 1000个带噪声数据点
  const int N = 1000;
  std::vector<Eigen::Vector3d> z_i;
  z_i.resize(N);     // 提前初始化好
#pragma omp parallel // 并行生成
  {
    // 每个线程一个生成器
    int tid = omp_get_thread_num();
    std::mt19937 gen(std::random_device{}() + tid);
    std::normal_distribution<> dist(0, 10.0); // 标准差10.0

    // 这里的标准差要和 Eigen::MatrixXd Sigma(3, 3); 存的方差对应上!

#pragma omp for // 并行生成
    for (int i = 0; i < z_i.size(); ++i) {
      z_i[i] << dist(gen), dist(gen), dist(gen);
    }
  }

  // 验证一下确实是(0,0.1)
  double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
#pragma omp parallel for reduction(+ : sum_x, sum_y, sum_z)
  for (int i = 0; i < z_i.size(); ++i) {
    sum_x += z_i[i](0);
    sum_y += z_i[i](1);
    sum_z += z_i[i](2);
  }
  double mean_x = sum_x / z_i.size();
  double mean_y = sum_y / z_i.size();
  double mean_z = sum_z / z_i.size();

  double sq_sum_x = 0.0, sq_sum_y = 0.0, sq_sum_z = 0.0;
#pragma omp parallel for reduction(+ : sq_sum_x, sq_sum_y, sq_sum_z)
  for (int i = 0; i < z_i.size(); ++i) {
    sq_sum_x += (z_i[i](0) - mean_x) * (z_i[i](0) - mean_x);
    sq_sum_y += (z_i[i](1) - mean_y) * (z_i[i](1) - mean_y);
    sq_sum_z += (z_i[i](2) - mean_z) * (z_i[i](2) - mean_z);
  }

  double stddev_x = std::sqrt(sq_sum_x / z_i.size());
  double stddev_y = std::sqrt(sq_sum_y / z_i.size());
  double stddev_z = std::sqrt(sq_sum_z / z_i.size());

  std::cout << "x: mean = " << mean_x << ", std_dev = " << stddev_x << "\n";
  std::cout << "y: mean = " << mean_y << ", std_dev = " << stddev_y << "\n";
  std::cout << "z: mean = " << mean_z << ", std_dev = " << stddev_z << "\n";

  // 调用一般方法计算
  double likelihood = 1.0;
#pragma omp parallel for reduction(* : likelihood)
  for (int i = 0; i < z_i.size(); ++i) {
    likelihood *= calc_direct_pdf(z_i[i], u, Sigma);
  }
  std::cout << "调用一般方法计算得到的最大似然估计值为: " << likelihood << "\n";

  // 调用log方法计算
  double log_likelihood = 0.0;
  Eigen::MatrixXd L = Sigma.llt().matrixL(); // LLT分解取出L矩阵
#pragma omp parallel for reduction(+ : log_likelihood)
  for (int i = 0; i < z_i.size(); ++i) {
    log_likelihood += calc_log_pdf_const(z_i[i], u, L) -
                      0.5 * calc_log_pdf_mahal(z_i[i], u, L);
  }
  std::cout << "调用对数方法计算得到的最大似然估计值为: " << exp(log_likelihood)
            << "\n";
  std::cout << "调用对数方法计算得到的log_likelihood: " << log_likelihood
            << "\n";
}