import numpy as np
import matplotlib.pylab as plt


# 均方根误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 交叉熵损失(适用于独热编码且mini-batch)
def cross_entopy_error(y, t):
    if y.ndim == 1:  # y 的维度为1才进来, 这里是单个样本的预测
        # 对于单样本, 增加一个维度, 从 [10] 到 [1, 10] 的维度
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 处理多维度情况, 第一个维度肯定是batch size
    batch_size = y.shape[0]
    #  $ L = -\frac{1}{N}\sum_{n=1}^N \sum_{k} t_{nk} \log y_{nk} $
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    # 这里把所有batch的交叉熵损失都求出来,并且取了平均值!


# 中心差分
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


# 一元函数
def function_1(x):
    return 0.01 * x**2 + 0.1 * x


def draw1():
    x = np.arange(0.0, 20.0, 0.1)  # 以0.1为单位，从0到20的数组x
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()


# 二元函数
def function_2(x):
    return np.sum(x**2)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # 算f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # 算f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        # 中心差分
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
    return grad


def comput():
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    print(numerical_gradient(function_2, np.array([3.0, 0.0])))


# 梯度下降法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)  # 求取梯度
        x -= lr * grad  # 梯度下降

    return x  # x这个参数就被更新到最佳了


def function_22(x):
    return x[0] ** 2 + x[1] ** 2


# 用梯度法求最小值
def comput2():
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_22, init_x=init_x, lr=0.1, step_num=100))


if __name__ == "__main__":
    # draw1()
    # comput()
    comput2()
