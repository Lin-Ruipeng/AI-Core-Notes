import numpy as np


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 避免溢出
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 高斯分布初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


def test_net():
    net = simpleNet()
    print(net.W)  # 查看权重参数
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))
    t = np.array([0, 0, 1])
    print(net.loss(x, t))  # loss

    # def f(W):
    #     z = np.dot(x, W)  # x 是外部的 (2,), W是 (2,3)
    #     y = softmax(z)  # softmax可以处理 1-d 或 batch
    #     loss = cross_entropy_error(y, t)
    #     return loss

    f = lambda w: net.loss(x, t)

    # 求梯度
    dW = numerical_gradient(f, net.W)
    print(dW)


if __name__ == "__main__":
    test_net()
