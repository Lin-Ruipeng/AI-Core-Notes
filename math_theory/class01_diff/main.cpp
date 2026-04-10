#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// 测试函数原函数
auto original_function(double x) -> double {
  return std::exp(2.0 * x) * std::sin(x);
}

// 测试函数的导函数
auto derivative_function(double x) -> double {
  return std::exp(2.0 * x) * (2.0 * std::sin(x) + std::cos(x));
}

// 前向差分求导
template <typename Func>
auto forward_difference(Func f, double x, double h) -> double {
  return (f(x + h) - f(x)) / h;
}

// 中心差分求导
template <typename Func>
auto central_difference(Func f, double x, double h) -> double {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

// 计算相对误差的百分值
auto relative_error(double exact, double estimated) -> double {
  return 100.0 * std::abs((estimated - exact) / exact);
}

auto main() -> int {
  // 1. 参数准备
  double x = 1.0;                        // 要求导的位置
  double exact = derivative_function(x); // 准确的导数值
  std::vector<double> h_list{
      std::pow(10.0, -2), std::pow(10.0, -3),  std::pow(10.0, -4),
      std::pow(10.0, -5), std::pow(10.0, -6),  std::pow(10.0, -7),
      std::pow(10.0, -8), std::pow(10.0, -10), std::pow(10.0, -15),
  }; // 测试步长列表
  std::vector<double> forward_relative_error_list; // 前向求导相对误差
  forward_relative_error_list.reserve(h_list.size()); // 提前保留内存
  std::vector<double> central_relative_error_list; // 中心求导相对误差
  central_relative_error_list.reserve(h_list.size()); // 提前保留内存

  // 2. 计算
  for (const auto &h : h_list) {
    // 前向求导
    forward_relative_error_list.push_back(
        relative_error(exact, forward_difference(original_function, x, h)));
    // 中心求导
    central_relative_error_list.push_back(
        relative_error(exact, central_difference(original_function, x, h)));
  }

  // 3. 输出
  int print_width = 20;
  int num_point = 4;
  // 开启科学计数法
  std::cout << std::scientific;
  // 始终显示小数点（1.000e+00 而非 1e+00）
  std::cout << std::showpoint;
  // 设置小数点后显示几位
  std::cout << std::setprecision(num_point);

  std::cout << "\n前向求导和中心求导在不同的步长h下的误差对比表格\n";
  std::cout << std::left          // 左侧对齐
            << std::setfill('-'); // 填充字符设为 '-'
  std::cout << std::setw(print_width) << "stpe width: h"
            << std::setw(print_width) << "forward_error(%)"
            << std::setw(print_width) << "central_error(%)" << "\n";
  std::cout << std::setfill(' '); // 恢复空格填充
  for (std::size_t i = 0; i < h_list.size(); ++i) {
    std::cout << std::setw(print_width) << h_list[i] << std::setw(print_width)
              << forward_relative_error_list[i] << std::setw(print_width)
              << central_relative_error_list[i] << "\n";
  }
}