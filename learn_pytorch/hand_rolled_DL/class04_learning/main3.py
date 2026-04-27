import sys, os
import numpy as np

sys.path.append("..")

from common.functions import sigmoid, softmax, cross_entropy_error
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist


class TwoLayerNet:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        weight_init_std=0.01,
    ):
        # 初始化权重
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accu = np.sum(y == t)
        return accu

    # def numerical_gradient_c(self, x, t):
    #     def loss_W(W):
    #         self.loss(x, t)

    #     grads = {}
    #     grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
    #     grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
    #     grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
    #     grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

    #     return grads
    def numerical_gradient_c(self, x, t):
        grads = {}
        for key in ["W1", "b1", "W2", "b2"]:
            # 使用默认参数 key=key 捕获当前循环的值，避免闭包延迟绑定问题
            def f(p, key=key):
                # 保存原始值
                original = self.params[key]
                # 临时替换为扰动后的值
                self.params[key] = p
                # 计算损失
                loss_val = self.loss(x, t)
                # 恢复原始值
                self.params[key] = original
                return loss_val

            grads[key] = numerical_gradient(f, self.params[key])
        return grads


def test1():
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params["W1"].shape)
    print(net.params["b1"].shape)
    print(net.params["W2"].shape)
    print(net.params["b2"].shape)

    x = np.random.rand(100, 784)  # 模拟生成100张图
    y = net.predict(x)
    t = np.random.rand(100, 10)

    print("开始求导")
    grads = net.numerical_gradient_c(x, t)  # 求取梯度


def test_train():

    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True
    )

    train_loss_list = []

    # 超参数
    iters_num = 10
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        print("运行第", i + 1, "轮")
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        # grad = network.gradient(x_batch, t_batch)  # 高速版!
        grad = network.numerical_gradient_c(x_batch, t_batch)

        # 更新参数
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]

        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        print("本轮求取出的损失值", loss)


def test_train_pro():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True
    )

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 超参数
    iters_num = 10
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    # 平均每个epoch的重复次数
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        # 获取mini batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        grad = network.numerical_gradient_c(x_batch, t_batch)

        # 更新参数
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        print("第", i + 1, "轮, loss=", loss)

        # 计算每个epoch的精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


if __name__ == "__main__":
    print("工作路径: ", os.getcwd())
    # test1()
    # test_train()
    test_train_pro()
