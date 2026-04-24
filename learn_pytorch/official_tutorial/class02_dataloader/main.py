import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import decode_image


def download_dataset():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # 标签映射
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]

        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])  # 图片标题
        plt.axis("off")  # 关闭当前子图的坐标轴
        # squeeze() 移除维度为 1 的通道维度，得到形状 [28, 28] 的二维数组
        plt.imshow(img.squeeze(), cmap="gray")  # 显示灰度图片
    plt.show()


# 自定义类, 自己创建的数据集类
class CustomImageDataset(Dataset):  # 继承Dataset
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform=None,
        target_transform=None,
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(
            self.img_dir,
            self.img_labels.iloc[index, 0],
        )  # 取出图片文件
        image = decode_image(img_path)
        label = self.img_labels.iloc[index, 1]  # 取出标签
        # 转换处理
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def create_my_dataset():
    # data = CustomImageDataset()
    pass


def use_dataloader():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # 载入数据集到 dataloader 里
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # 用 dataloader 迭代, 用 next() 取出一个元素来展示
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


if __name__ == "__main__":
    # download_dataset()
    create_my_dataset()
    use_dataloader()
