import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 这里实现动量的公式
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

            # 公式实现(除数不能为0, +1e-7)
            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


def draw():
    import matplotlib.pyplot as plt

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x = np.random.randn(1000, 100)  # 1000个数据
    node_num = 100  # 各隐藏层的节点（神经元）数
    hidden_layer_size = 5  # 隐藏层有5层
    activations = {}  # 激活值的结果保存在这里

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i - 1]

        # 这一行是随机数权重
        # w = np.random.randn(node_num, node_num) * 1
        # w = np.random.randn(node_num, node_num) * 0.01
        w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

        z = np.dot(x, w)
        a = sigmoid(z)  # sigmoid函数
        activations[i] = a

    # 绘制直方图
    for i, a in activations.items():
        plt.subplot(1, len(activations), i + 1)
        plt.title(str(i + 1) + "-layer")
        plt.hist(a.flatten(), 30, range=(0, 1))
    plt.show()


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.random(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


if __name__ == "__main__":
    draw()
