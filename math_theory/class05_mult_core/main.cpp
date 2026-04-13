// open MP 并行计算提高性能
// 需要使用扩展库! g++ main.cpp -O3 -march=native -fopenmp
#include <chrono>
#include <iostream>
#include <omp.h> // 引入open MP
#include <random>
#include <vector>

auto main() -> int {
  const size_t NUMS_SIZE = 100000000;
  std::vector<double> nums;
  nums.reserve(NUMS_SIZE);

  // 创建随机数
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> ddis(0.0, 100.0);

  for (size_t i = 0; i < NUMS_SIZE; ++i) {
    nums.push_back(ddis(gen));
  }

  double result1 = 0.0;
  double result2 = 0.0;
  double result3 = 0.0;

  auto start1 = std::chrono::high_resolution_clock::now();
  // 1. 单线运算
  for (size_t i = 0; i < NUMS_SIZE; ++i) {
    result1 += nums[i] * nums[i];
  }
  result1 = std::sqrt(result1 / NUMS_SIZE);
  auto end1 = std::chrono::high_resolution_clock::now();
  auto duration1 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);

  auto start2 = std::chrono::high_resolution_clock::now();
// 2. 并行不归约 注意 for (const auto & n : nums) 可能无法被omp优化!
#pragma omp parallel for
  for (size_t i = 0; i < NUMS_SIZE; ++i) {
    result2 += nums[i] * nums[i];
  }
  result2 = std::sqrt(result2 / NUMS_SIZE);
  auto end2 = std::chrono::high_resolution_clock::now();
  auto duration2 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);

  auto start3 = std::chrono::high_resolution_clock::now();
  // 3. 并行归约 (对result3进行归约, 乘法独立跨线程不共享不需要归约)
#pragma omp parallel for reduction(+ : result3)
  for (size_t i = 0; i < NUMS_SIZE; ++i) {
    result3 += nums[i] * nums[i];
  }
  result3 = std::sqrt(result3 / NUMS_SIZE);
  auto end3 = std::chrono::high_resolution_clock::now();
  auto duration3 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3);

  // 4. 报告
  std::cout << "omp最大线程数: " << omp_get_max_threads();
  std::cout << "\n结果报告: " << "\n单线计算结果  : " << result1
            << "\n并行不归约结果: " << result2
            << "\n并行归约结果  : " << result3 << "\n耗时报告: \n"
            << "单线计算耗时  : " << duration1.count() << "ns\n"
            << "并行不归约耗时: " << duration2.count() << "ns\n"
            << "并行归约耗时  : " << duration3.count() << "ns\n";
}
