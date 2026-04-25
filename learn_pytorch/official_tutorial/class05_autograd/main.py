import torch


def graph1():
    # 手动构建一个计算图, 线性: y = w*x + b
    # 1.构建输入输出
    x = torch.ones(5)  # 输入tensor
    y = torch.zeros(3)  # 预期输出, 也就是真实标签
    # 2.构建权重, 这个需要能够自动求导!
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    # 3.构建计算图
    # 前向传播
    z = torch.matmul(x, w) + b  # 实现y=w*x+b
    # 求取损失
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")

    loss.backward()  # 反向传播!
    print(w.grad)
    print(b.grad)


def no_auto_grad():
    x = torch.ones(5)  # 输入tensor
    y = torch.zeros(3)  # 预期输出, 也就是真实标签
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    z = torch.matmul(x, w) + b  # 实现y=w*x+b
    print(z.requires_grad)

    # 关闭自动求导
    with torch.no_grad():
        z = torch.matmul(x, w) + b
    print(z.requires_grad)


def no_auto_grad2():
    x = torch.ones(5)  # 输入tensor
    y = torch.zeros(3)  # 预期输出, 也就是真实标签
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    z = torch.matmul(x, w) + b  # 实现y=w*x+b
    print(z.requires_grad)

    # 关闭自动求导
    z_det = z.detach()
    print(z_det.requires_grad)


def jacobian():
    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp + 1).pow(2).t()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"First call\n{inp.grad}")
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nSecond call\n{inp.grad}")
    inp.grad.zero_()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{inp.grad}")


if __name__ == "__main__":
    # graph1()
    # no_auto_grad()
    # no_auto_grad2()
    jacobian()
