# Tensors

## 基础概念

这是一种数据结构，形似数组或者矩阵。
其和Numpy的narray共享相同的底层内存，可以做到零开销拷贝，
并且Tensors可以存在各种硬件加速器上，比如GPU。
最后，Tensors类型优化了自动差分（也就是求导）。

## 初始化Tensor

```python
import torch
tensor = torch.tensor(data) # 这里的data是各种数据类型的矩阵
```

## 转移到加速单元

通过 `.to` 方法实现!
```python
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator()) 
```

## 一些计算

```python
# 对于张量 tensor

y = tensor @ tensor.T # 矩阵乘法
z = tensor * tensor # 矩阵点乘
s = tensor.sum() # 求和
num = s.item() # 单个元素的tensor转成python内置类型
n = tensor.numpy() # 转换成numpy类型
# 把np类型转换成tensor
n = np.ones(5)
t = torch.from_numpy(n)
```

注意numpy和tensor在底层共享内存! 也就是你修改n或者t都会影响到另外一个! 
