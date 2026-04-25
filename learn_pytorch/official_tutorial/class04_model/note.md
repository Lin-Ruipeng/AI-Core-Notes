# Build Model

神经网络由对数据执行操作的层/模块组成。

torch. nn命名空间提供了构建自己的神经网络所需的所有构建块。

## 调用加速设备

```python
import torch
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device.")
```

## 定义模型

就是写一个类继承`nn.Model`然后实现类里的`__init__`和`forword(self, x)`这两个方法
`__init__`里需要定义好网络的各个层
`forward()`里将各个层拼接成一条完整的运算流水线（计算图）


运行模型时会自动调用forward()以及一些后台操作。不要直接显式调用model.forward()！

## 看模型的结构和参数

```python
model = NeuralNetwork()

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
```
