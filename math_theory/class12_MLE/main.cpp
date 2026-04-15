// 梯度下降算法
#include <iostream>
#include <omp.h> // -fopenmp
#include <random>
#include <vector>

// 代价函数
auto ComputeCost(const std::vector<double> &z, double alpha) -> double {
  double cost = 0.0;
// 代价函数公式: \sum _{k=2} ^N (z_k - a * z_{k-1})^2 可以并行加速!
#pragma omp parallel for reduction(+ : cost)
  for (size_t i = 1; i < z.size(); ++i) {
    cost += (z[i] - alpha * z[i - 1]) * (z[i] - alpha * z[i - 1]);
  }
  return cost;
}

// 中心求导 h 是求导时的步长
template <typename Func>
auto DerivativeFun(Func f, const std::vector<double> &z, double alpha,
                   double h) -> double {
  // 中心求导就是: [f(x+h) - f(x-h)] / (2 * h)
  return (f(z, alpha + h) - f(z, alpha - h)) / (2.0 * h);
}

auto main() -> int {
  // 仿真数据
  double a_true = 0.98; // 真值
  double x0 = 10.0;     // 初始值

  std::mt19937 gen(std::random_device{}());           // 随机白噪声
  std::normal_distribution<> dist(0, std::sqrt(0.5)); // 方差0.5

  const size_t N = 100000; // 生成点数

  std::vector<double> z;
  z.resize(N);
  z[0] = x0;
  // 准备数据总是依赖于上一个数据, 不能omp加速!
  for (size_t i = 1; i < N; ++i) {
    // IMU的零偏的数学模型: x_k = a*x_{k-1} + w_k
    z[i] = a_true * z[i - 1] + dist(gen);
  }

  // 梯度下降法
  double lr = 3.3e-8;  // 学习率
  double alpha = 0.5;  // 初始值
  double h = 1.0e-5;   // 求导步进
  size_t epochs = 100; // 轮次

  for (size_t i = 0; i < epochs; ++i) {
    if ((i + 1) % 10 == 0) {
      std::cout << "epochs: " << i + 1 << ", alpha: " << alpha
                << ", cost: " << ComputeCost(z, alpha) << "\n";
    } // 梯度下降算法 x -= lr * f`(x)
    double grad = DerivativeFun(ComputeCost, z, alpha, h);
    alpha -= lr * grad;
  }
}
