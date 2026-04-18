// kalman 滤波
#include <Eigen/Dense> // -I /usr/include/eigen3
#include <cmath>
#include <iostream>
#include <random>

// 对象名{初始值} 方式初始化
struct KalmanPara {
  // ===================== 固定滤波参数 (const 保护，禁止修改)
  // 状态转移矩阵 F: 2x2 (dt=1.0)，静态矩阵(推荐)
  const Eigen::Matrix2d F{{1, 1}, {0, 1}};
  // 观测矩阵 H: 1x2
  const Eigen::RowVector2d H{1.0, 0.0};
  // 过程噪声 Q：单位阵*0.01 → 纯列表初始化
  const Eigen::Matrix2d Q{{0.01, 0}, {0, 0.01}};
  // 观测噪声 R：1x1矩阵，矩阵就是严格双{}
  const Eigen::Matrix<double, 1, 1> R{{1.0}};
  // 2x2单位矩阵 I：纯列表初始化
  const Eigen::Matrix2d I{{1, 0}, {0, 1}};
  // ===================== 动态状态变量 (全部 {} 初始化) =====================
  // 最优估计 x_hat：位置+速度
  Eigen::Vector2d x_hat{0.0, 0.0};
  // 估计协方差 P：初始大不确定性，纯列表初始化
  Eigen::Matrix2d P{{10, 0}, {0, 10}};
  // 观测值 z (1维)
  Eigen::Matrix<double, 1, 1> z{0.0};
  // 残差 y
  Eigen::Matrix<double, 1, 1> y{0.0};
  // 总不确定度 S
  Eigen::Matrix<double, 1, 1> S{0.0};
  // 卡尔曼增益 K
  Eigen::Matrix<double, 2, 1> K{0.0, 0.0};
};

// 传入观测值 + 滤波参数，完成一次卡尔曼迭代（预测+更新）
void calc_kalman(const double observation, KalmanPara &para, bool use_joseph) {
  // 1. 把外部观测值写入结构体
  para.z(0, 0) = observation;

  // ------------------- 卡尔曼滤波 -------------------
  // 1. 预测步骤
  // x_hat = F * x_hat;  // 1. 状态预测
  para.x_hat = para.F * para.x_hat;

  // P = F * P * F^T + Q; // 2. 协方差预测
  para.P = para.F * para.P * para.F.transpose() + para.Q; // 2. 协方差预测

  // 2. 更新步骤
  // y = z - H * x_hat; // 3. 残差
  para.y = para.z - para.H * para.x_hat;

  // S = H * P * H^T + R; // 4.1 卡尔曼增益
  para.S = para.H * para.P * para.H.transpose() + para.R;
  // K = P * H^T * S^-1; // 4.2 卡尔曼增益
  // 看到矩阵求逆就要想起来别直接求逆!
  // 这里因为S_k恰好是1x1大小的,所以取倒数就完事了
  para.K = para.P * para.H.transpose() * (1.0 / para.S(0, 0));

  // x_hat = x_hat + K * y; // 5. 状态更新
  para.x_hat = para.x_hat + para.K * para.y;

  // P = (I - K * H) * P;  // 6. 协方差更新
  // Joseph 写法! 防止分解失败! P = (I-K*H)*P^-1*(I-K*H)^T+K*R*K^T
  Eigen::Matrix2d A = para.I - para.K * para.H;

  if (use_joseph) {
    Eigen::Matrix2d P_joseph;
    P_joseph.noalias() =
        A * para.P * A.transpose() + para.K * para.R * para.K.transpose();
    para.P = P_joseph;
  } else { // 不使用Joseph方法 朴素写法(实际工程中不要用!)
    para.P = A * para.P;
  }
}

auto main() -> int {
  const int N = 100; // 循环次数

  KalmanPara kp_naive;  // 原始朴素方法
  KalmanPara kp_joseph; // joseph方法
  // 故意引入微小误差, 不严格对称! 更新P时如果用朴素写法可能会出问题
  kp_naive.P(0, 1) = 0.1;
  kp_naive.P(1, 0) = 0.101;
  kp_joseph.P(0, 1) = 0.1;
  kp_joseph.P(1, 0) = 0.101;

  // 观测误差
  std::mt19937 gen(std::random_device{}());
  // 观测噪声（标量 R，比如设为 1.0）记得要和卡尔曼滤波器里对应上
  std::normal_distribution noise_r(0.0, std::sqrt(1.0));

  // 过程噪声（假设 Q 是对角阵，比如 Q = diag(0.01, 0.01)）
  std::normal_distribution<> noise_q_pos(0.0, std::sqrt(0.01));
  std::normal_distribution<> noise_q_vel(0.0, std::sqrt(0.01));

  const Eigen::Matrix2d F{{1, 1}, {0, 1}};
  // 2. 生成上帝视角的真实轨迹
  Eigen::Vector2d x_true(0.0, 1.0); // 初始位置0，初始速度1
  for (int i = 1; i <= N; ++i) {

    // 真实世界演化：必须加上过程噪声！
    x_true = F * x_true + Eigen::Vector2d(noise_q_pos(gen), noise_q_vel(gen));

    // 传感器观测真实世界：加上观测噪声
    double z = x_true(0) + noise_r(gen);

    // 接下来把 z 喂给卡尔曼滤波器
    calc_kalman(z, kp_naive, false);
    calc_kalman(z, kp_joseph, true);

    if (i % 10 == 0) {
      std::cout << "x true: " << x_true.transpose() << ",i = " << i << "\n";
      std::cout << "naive : x_hat = " << kp_naive.x_hat.transpose()
                << " ,P error = " << kp_naive.P(0, 1) - kp_naive.P(1, 0)
                << "\n";
      std::cout << "joseph: x_hat = " << kp_joseph.x_hat.transpose()
                << " ,P error = " << kp_joseph.P(0, 1) - kp_joseph.P(1, 0)
                << "\n";
    }
  }
}
