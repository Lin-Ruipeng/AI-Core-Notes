#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <omp.h> // -fopenmp
#include <random>
#include <vector>

struct Stats {
  double sum{0};    // 和
  double mean{0};   // 均值
  double stddev{0}; // 标准差
};

auto Statistics(std::vector<double> &v) -> Stats {
  // 1. 求和
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < v.size(); ++i) {
    sum += v[i];
  }
  // 2. 均值
  double mean = sum / v.size();
  // 3. 方差
  double sq_sum = 0.0;
#pragma omp parallel for reduction(+ : sq_sum)
  for (int i = 0; i < v.size(); ++i) {
    sq_sum += (mean - v[i]) * (mean - v[i]);
  }
  double stddev = std::sqrt(sq_sum / v.size());

  return Stats{.sum = sum, .mean = mean, .stddev = stddev};
}

auto main() -> int {
  // 1. 参数设定
  const int time = 1000;
  const int sample_rate = 1000;
  const int N = time * sample_rate;

  std::vector<double> v1;
  v1.reserve(N);
  std::vector<double> v2;
  v2.resize(N); // 因为不用push_back而是用[]访问
  std::vector<double> v3;
  v3.resize(N); // 因为不用push_back而是用[]访问

  // 2. 单线程基准
  std::random_device rd;
  std::mt19937 gen(rd());
  // 生成随机数
  std::normal_distribution<> dist(0.0, 1.0); // N~(0,1)
  auto start1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    v1.push_back(dist(gen));
  }
  auto end1 = std::chrono::high_resolution_clock::now();
  // 计算耗时
  auto duration1 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);

  // 3. 多线程但错误
  auto start2 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    v2[i] = dist(gen); // 灾难!(因为gen依赖上次结果,多线程访问会数据竞争)
  }
  auto end2 = std::chrono::high_resolution_clock::now();
  // 计算耗时
  auto duration2 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);

  // 4. 多线程正确
  auto start3 = std::chrono::high_resolution_clock::now();
#pragma omp parallel
  {
    int tid = omp_get_thread_num(); // 获取线程ID
    std::mt19937 gen(std::random_device{}() + tid);
    // 每个线程独立创建生成器并且种子不一样!
    std::normal_distribution<> dist(0.0, 1.0);

#pragma omp for
    for (int i = 0; i < N; ++i) {
      v3[i] = dist(gen); // push_back是线程不安全的!
    }
  }
  auto end3 = std::chrono::high_resolution_clock::now();
  // 计算耗时
  auto duration3 =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3);

  // 5. 报告
  Stats s1 = Statistics(v1);
  Stats s2 = Statistics(v2);
  Stats s3 = Statistics(v3);
  std::cout << "方法1: 单线程生成数据 耗时:" << duration1.count()
            << "ns mean = " << s1.mean << ", stddev = " << s1.stddev << "\n";
  std::cout << "方法2: 多线程错误生成 耗时:" << duration2.count()
            << "ns mean = " << s2.mean << ", stddev = " << s2.stddev << "\n";
  std::cout << "方法3: 多线程正确生成 耗时:" << duration3.count()
            << "ns mean = " << s3.mean << ", stddev = " << s3.stddev << "\n";
}
