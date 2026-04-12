// 测试矩阵乘法的速度, 对比缓存命中与miss
// g++ main.cpp -O3 -march=native 关闭优化对比更明显
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

auto main() -> int {
  int N = 1024;

  // 提前分配内存
  std::vector<double> matrix_A(N * N);
  std::vector<double> matrix_B(N * N);
  std::vector<double> matrix_C1(N * N);
  std::vector<double> matrix_C2(N * N);

  for (size_t i = 0; i < N * N; ++i) {
    matrix_A[i] = (i * 7 + 3) % 100;
    matrix_B[i] = (i * 7 + 3) % 100;
  }

  // 额外说明!
  // 注意到矩阵乘法是需要累加的!
  // 所以需要考虑大数吃小数!
  // 甚至需要考虑符号相反时灾难性相消!

  auto start1 = std::chrono::high_resolution_clock::now();
  // 方式1: 一般顺序
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t k = 0; k < N; ++k) {
        matrix_C1[i * N + j] += matrix_A[i * N + k] * matrix_B[k * N + j];
      }
    }
  }
  auto end1 = std::chrono::high_resolution_clock::now();

  auto start2 = std::chrono::high_resolution_clock::now();
  // 方式2: 交换顺序
  for (size_t i = 0; i < N; ++i) {
    for (size_t k = 0; k < N; ++k) {
      double temp = matrix_A[i * N + k];
      for (size_t j = 0; j < N; ++j) {
        matrix_C2[i * N + j] += temp * matrix_B[k * N + j];
      }
    }
  }
  auto end2 = std::chrono::high_resolution_clock::now();

  // 比对结果
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      // 浮点数比较不能用 == 和 != , 极易出错!
      if (std::abs(matrix_C1[i * N + j] - matrix_C2[i * N + j]) > 1e-9) {
        std::cout << "ERROR: 结果比对显示两种运算结果超越了容差范围(1e-9)\n";
        return 1;
      }
    }
  }
  std::cout << "两种计算方式的计算结果在容差范围(1e-9)内一致!\n";

  auto duration1 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);
  auto duration2 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);

  std::cout << "报告计算时间: \n"
            << "一般顺序(ijk)计算用时: " << duration1.count() << "ns\n"
            << "交换顺序(ikj)计算用时: " << duration2.count() << "ns\n";
}
