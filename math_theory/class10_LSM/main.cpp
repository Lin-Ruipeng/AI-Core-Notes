// 最小二乘法
#include <Eigen/Dense> // Eigen库 -I /usr/include/eigen3
#include <iostream>
#include <vector>

struct Radar {
  Eigen::Vector2d posi; // 雷达位置
  double dist{0};       // 测距结果
};

// X是猜测的坐标, 函数计算其与测量值的损失
auto ComputeCost(const Eigen::Vector2d &X, const Radar &r) -> double {
  // 预测距离: h_sq = (X-x)^2 + (Y - y)^2 注意故意不开方!
  double h_sq = (X - r.posi).squaredNorm();
  // 观测距离(一起平方)
  double z_sq = r.dist * r.dist;
  return h_sq - z_sq; // 返回残差cost
}

auto main() -> int {
  // 数据准备
  Radar ra = {{0.0, 0.0}, 5.0};
  Radar rb = {{10.0, 0.0}, 8.0};

  std::vector<Eigen::Vector2d> X_list = {{8.0, 8.0}, {3.2, 4.1}, {3.0, 4.0}};

  for (const auto &X : X_list) {
    Eigen::Vector2d cost(ComputeCost(X, ra), ComputeCost(X, rb));
    double total_cost = cost.dot(cost); // 和 cost^T * cost 效果一样

    std::cout << "预测位置为: (" << X.transpose()
              << ") 计算损失为: " << total_cost << "\n";
  }
}
