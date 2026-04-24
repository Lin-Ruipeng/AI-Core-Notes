import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # 转换成为tensor类型, 并且归一化到 [0, 1]
    transform=ToTensor(),
    # 这里对标签进行了 独热编码, 而且用到了Lambda来包裹用户的匿名函数
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0,  # 10个元素的默认值为1
            torch.tensor(y),  # y 作为索引
            value=1,  # y 下标设定值为1
        )
    ),
)
