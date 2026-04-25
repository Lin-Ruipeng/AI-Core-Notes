import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 定义模型 继承 nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def forward():
    # 加速设备
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device.")

    # 创建模型 并且加速
    model = NeuralNetwork().to(device)
    print(model)  # 可以打印出模型的结构!

    # 调用模型进行推理
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")


# 看每一层网络都在干什么
def layers():
    #
    input_image = torch.rand(3, 28, 28)
    print(input_image.size())

    # 展平层
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())

    # FCN
    layer1 = nn.Linear(in_features=28 * 28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1)

    # ReLU
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")

    # 有序容器
    seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10),
    )
    input_image = torch.rand(3, 28, 28)
    logits = seq_modules(input_image)

    # softmax
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)
    print(pred_probab)


# 看参数
def para():
    model = NeuralNetwork()

    print(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")


if __name__ == "__main__":
    # forward()
    # layers()
    para()
