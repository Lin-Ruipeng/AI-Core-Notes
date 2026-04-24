import torch
import numpy as np


def init_tensor():
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(x_data)

    np_array = np.array(data)
    x_np = torch.tensor(np_array)
    print(x_np)

    # retains the properties of x_data
    x_ones = torch.ones_like(x_data)
    print(f"Ones Tensor: \n {x_ones} \n")

    # overrides the datatype of x_data
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f"Random Tensor: \n {x_rand} \n")

    shape = (2, 3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")


def attributes_tensor():
    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")


def operation_tensor():
    tensor = torch.rand(3, 4)
    # We move our tensor to the current accelerator if available
    if torch.accelerator.is_available():
        print("加速单元可用! 加速单元是: ", torch.accelerator.current_accelerator())
        tensor.to(torch.accelerator.current_accelerator())

    tensor = torch.ones(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:, 1] = 0  # 第二行整行置0
    print(tensor)

    # 串联 concatenate (需要指定在哪个维度上!)
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    # dim = 1就是4x12的矩阵, dim = 0就是12x4的矩阵
    print(t1)


def operation_tensor2():
    tensor = torch.rand(3, 4)

    # 矩阵乘法
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)

    print("y1 = ", y1)
    print("y2 = ", y2)

    y3 = torch.rand_like(y1)
    torch.matmul(tensor, tensor.T, out=y3)

    print("y3 = ", y3)

    # 矩阵点乘
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(z1)
    torch.mul(tensor, tensor, out=z3)

    print("z1 = ", z1)
    print("z2 = ", z2)
    print("z3 = ", z3)

    # 单元素的tensor, 转回为python数据类型
    agg = tensor.sum()
    agg_item = agg.item()
    print("Before: data:", agg, "type:", type(agg))
    print("After : data:", agg_item, "type:", type(agg_item))

    # in-place 就地操作(下划线标识), 也就是不创建临时变量进行计算
    # 比如就地全部加法: .add_()
    tensor = torch.ones(3, 4)
    tensor.add_(5)  # 全元素就地 +5
    print(tensor)
    # 虽然省内存, 但是会丢失历史记录, 就会导致求导出问题, 所以不鼓励使用!


def tensor_with_numpy():
    # tensor->numpy
    t = torch.ones(5)
    print(f"t: {t}")
    n = t.numpy()  # 转换成numpy
    print(f"n: {n}")

    # 注意这是绑定关系! 你修改tensor也会影响到numpy!
    t.add_(5)
    print(f"t: {t}")
    print(f"n: {n}")

    # numpy->tensor
    n = np.ones(4)
    t = torch.from_numpy(n)
    print(f"t: {t}")
    print(f"n: {n}")
    # 一样是绑定关系!
    np.add(n, 3, out=n)
    print(f"t: {t}")
    print(f"n: {n}")


if __name__ == "__main__":
    # init_tensor()
    # attributes_tensor()
    # operation_tensor()
    # operation_tensor2()
    tensor_with_numpy()
