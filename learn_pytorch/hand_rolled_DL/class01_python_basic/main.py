import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


def nparray():
    print("创建一维数组")
    x = np.array([1.0, 2.0, 3.0])
    print(x)

    print("基础运算")
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)

    print("广播机制")
    print(x / 2.0)


def npnarray():
    print("创建多维数组")
    A = np.array([[1, 2], [3, 4]])
    print(A)
    print(A.shape)  # 形状
    print(A.dtype)  # 元素的数据类型

    print("广播机制")
    print(A * 10)

    print("广播机制2")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([10, 20])
    print(A * B)


def read():
    print("访问元素")
    X = np.array([[51, 55], [14, 19], [0, 4]])
    print(X[0])  # 0行
    print(X[0][1])  # (0,1) -> (1,2)
    for row in X:
        print(row)  # 按行遍历

    print("展平操作")
    X = X.flatten()
    print(X)

    print("不连续索引(抽取)")
    print(X[np.array([0, 2, 4])])

    print("元素筛选")
    print(X > 15)
    print(X[X > 15])


def plot():
    # x是0到6,且步长为0.1的数组
    x = np.arange(0, 6, 0.1)
    # y = np.sin(x)

    # # 绘制图像
    # plt.plot(x, y)
    # plt.show()

    # 再建立一个新图像
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, label="cos", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sin and cos")
    plt.legend()
    plt.show()


# 显示图片
def show_img():
    img = imread("./p1.png")
    plt.imshow(img)

    plt.show()


if __name__ == "__main__":
    # nparray()
    # npnarray()
    # read()
    # plot()
    show_img()
