#include <iostream>
#include <numeric> // 数学计算库

// 普通叠加
auto normal_summation(float v, float a, float dt) -> float {
  return v + a * dt;
}

// kahan补偿方法, 参数c是补偿累加器, 需要对应好!
auto kahan_summation(float v, float a, float dt, float &c) -> float {
  // 1. 小数补上丢失的精度: y = y - c
  float y = a * dt - c; // a * dt 是本次相加的小数, 补偿量也一定是小数
  // 2. 常规加法: t = sum + y
  float t = v + y; // v 是本次相加的大数, 相当于sum
  // 3. 反向计算精度损失: c = (t - sum) - y
  // 因为没有精度损失的话, 应该有 t - sum == y
  c = (t - v) - y; // (求和后大数 - 求和前大数) - 小数
  // 4. 更新总和: sum = t
  return t; // 返回给外部
}

// 1. 定义一个零成本的伪装迭代器
struct DummyIterator {
  std::size_t count;
  // 必须提供的迭代器类型别名（让 std::accumulate 能编译通过）
  using iterator_category = std::forward_iterator_tag;
  using value_type = float;
  using difference_type = std::ptrdiff_t;
  using pointer = float *;
  using reference = float &;

  // 构造函数
  explicit DummyIterator(std::size_t c) : count(c) {}

  // 解引用操作：返回一个无意义的值，反正 Lambda 会忽略它
  value_type operator*() const { return 0.0; }

  // 递增操作：只增加计数，不移动任何内存指针
  DummyIterator &operator++() {
    ++count;
    return *this;
  }
  DummyIterator operator++(int) {
    DummyIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  // 比较操作：控制循环什么时候停止
  bool operator==(const DummyIterator &rhs) const { return count == rhs.count; }
  bool operator!=(const DummyIterator &rhs) const { return count != rhs.count; }
};

auto main() -> int {
  const float initial_velocity = 10000.0f; // 初速度
  const float acceleration = 0.0001f;      // 加速度
  const float total_time = 10000.0f;       // 总时间
  const float sampling_rate = 10.0f;       // 采样率
  const float delta_time = 1.0f / sampling_rate;

  // 0. 计算标准终值速度: v = v0 + a * t
  const float final_speed = initial_velocity + acceleration * total_time;

  std::size_t iterate_count = total_time * sampling_rate; // 迭代次数

  // 2. 普通方法计算
  float normal_speed = initial_velocity;
  for (size_t i = 0; i < iterate_count; ++i) {
    normal_speed = normal_summation(normal_speed, acceleration, delta_time);
  }

  // 2. 手动kahan补偿
  float kahan_speed = initial_velocity;
  float c = 0.0f;
  for (size_t i = 0; i < iterate_count; ++i) {
    kahan_speed = kahan_summation(kahan_speed, acceleration, delta_time, c);
  }

  // 3. 调用库
  float dv = acceleration * delta_time;
  std::cout << "dv = " << dv << "\n";
  float stl_speed = std::accumulate(
      DummyIterator(0), DummyIterator(iterate_count), initial_velocity,
      [dv](float acc, float) { return acc + dv; });

  // 打印结果
  std::cout << "终值速度(公式算法): \t" << final_speed << "\n";
  std::cout << "终值速度(正常累加): \t" << normal_speed << "\n";
  std::cout << "终值速度(kahan累加): \t" << kahan_speed << "\n";
  std::cout << "终值速度(STL库): \t" << stl_speed << "\n";
}