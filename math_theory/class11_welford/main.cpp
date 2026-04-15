// Welford 算法
#include <iostream>
#include <random>

auto main() -> int {
  // 数据准备
  const size_t N = 100000;
  std::mt19937 gen(std::random_device{}());
  std::normal_distribution<> dist(1.0e10, 1.0); // 均值10亿,标准差1

  // 方法1数据
  double sum_x = 0.0, sum_x_sq = 0.0;
  // 方法2数据
  double M = 0.0, S = 0.0;

  // 计算
  for (size_t i = 0; i < N; ++i) {
    double sensor_output = dist(gen);

    // 方法1: 传统方法 (error)
    sum_x += sensor_output;
    sum_x_sq += sensor_output * sensor_output;

    // 方法2: Welford算法
    size_t k = i + 1; // 对应数学公式里的第 k 个数, 编程是0下标开始
    // 1. 计算差值 \delta = x_k - M_{k-1}
    double delta = sensor_output - M;
    // 2. 更新均值 M_k = M_{k-1} + \frac{\delta}{k}
    M += delta / static_cast<double>(k);
    // 3. 更新平方和 S_k = S_{k-1} + \delta \times (x_k - M_k)
    S += delta * (sensor_output - M);
  }

  // 方法1: 方差计算公式: 平方和的平均 - 平均值的平方
  double mean = sum_x / N;
  double var = (sum_x_sq / N) - mean * mean;

  // 方法2: Welford
  // 4. 方差 Var = \frac{S_k}{k - 1}
  double var_welford = S / static_cast<double>(N - 1);

  // 输出
  std::cout << "采用一般方法计算得到方差: " << var
            << "\n采用Welford算法计算得到方差: " << var_welford << "\n";
}
