import numpy as np
import matplotlib.pylab as plt


# 阶跃函数
def step_function(x):
    # 运用了广播技巧!
    y = x > 0
    return np.int_(y)


def draw_setp():
    x = np.arange(-5, 5, 0.1)
    y = step_function(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # 而且支持广播!


def draw_sigmoid():
    x = np.arange(-15, 15, 0.1)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.show()


def ReLU(x):
    return np.maximum(0, x)
    # 使用np提供的函数, 以此支持广播! 为了并行计算!


def draw_ReLU():
    x = np.arange(-5, 5, 0.1)
    y = ReLU(x)

    plt.plot(x, y)
    plt.show()


def narray():
    A = np.array([1, 2, 3, 4])
    print(A)
    print(np.ndim(A))  # 维度
    print(A.shape)
    print(A.shape[0])

    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(B)
    print(np.ndim(B))
    print(B.shape)


def calc_matrix():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print(np.dot(A, B))  # 点积

    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[1, 2], [3, 4], [5, 6]])

    print(np.dot(A, B))  # 点积


def identity_function(x):
    return x


def liner():
    X = np.array([1.0, 0.5])  # 1x2
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 2x3
    B1 = np.array([0.1, 0.2, 0.3])  # 3x1

    # 实现X*W + B 数据维度 (1x2 * 2x3) + 1x3 = 1x3
    A1 = np.dot(X, W1) + B1

    # 激活函数
    Z1 = sigmoid(A1)

    print(A1)
    print(Z1)

    # 第二层线性层
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])

    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)

    print(A2)
    print(Z2)

    # 第三层

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])

    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)

    print(A3)
    print(Y)


def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])
    return network


def forward(network, x):
    # 取出参数
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    # 前向计算三层
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


def run_network():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)


def softmax(a):
    C = np.max(a)  # 取出最大值
    exp_a = np.exp(a - C)  # 避免溢出
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


if __name__ == "__main__":
    # draw_setp()
    # draw_sigmoid()
    # draw_ReLU()
    # narray()
    # calc_matrix()
    # liner()
    run_network()
