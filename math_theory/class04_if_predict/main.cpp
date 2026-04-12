// 分支预测性能对比
// g++ main.cpp -O3 -march=native
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

auto main() -> int {
  const size_t NUMS_SIZE = 10000000;
  // 生成随机数
  // 1. 创建随机数引擎（种子源）
  std::random_device rd;  // 硬件随机数（如果支持）
  std::mt19937 gen(rd()); // Mersenne Twister算法， seeded with rd

  // 2. 创建分布器（定义随机数范围和分布）
  std::uniform_real_distribution<float> ddis(-30.0f, 30.0f); // 浮点数

  // 3. 生成随机数
  std::vector<float> frames_1;
  std::vector<float> frames_2;
  frames_1.reserve(NUMS_SIZE);
  frames_2.reserve(NUMS_SIZE);
  for (size_t i = 0; i < NUMS_SIZE; ++i) {
    float random = ddis(gen);
    frames_1.push_back(random);
    frames_2.push_back(random);
  }

  auto start1 = std::chrono::high_resolution_clock::now();
  // 毁掉分支预测
  for (auto &acc : frames_1) {
    volatile float v = acc; // 阻止寄存器优化
    if (v > 16.0f) {        // 限制最大不超过16.0
      acc = 16.0f;
    } else if (v < -16.0f) { // 限制最小不超过-16.0
      acc = -16.0f;
    }
  }
  auto end1 = std::chrono::high_resolution_clock::now();

  auto start2 = std::chrono::high_resolution_clock::now();
  // 无分支(高性能)
  for (auto &acc : frames_2) {
    acc = std::min(acc, 16.0f);  // 限制最大不超过16.0
    acc = std::max(acc, -16.0f); // 限制最小不超过-16.0
  }
  auto end2 = std::chrono::high_resolution_clock::now();

  // 验证
  for (size_t i = 0; i < NUMS_SIZE; ++i) {
    if (std::abs(frames_1[i] - frames_2[i]) > 1e-9) {
      std::cout << "ERROR: 两种计算方式得到的结果误差不在容许范围内!\n";
      return 1;
    }
  }
  std::cout << "在容许误差内, 可以认为两种计算方式结果相同!\n";

  // 性能比对
  auto duration1 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);
  auto duration2 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);

  std::cout << "性能比对: \n"
            << "无法分支预测时的处理耗时: " << duration1.count() << "ns\n"
            << "没有分支预测失败处理耗时: " << duration2.count() << "ns\n";
}