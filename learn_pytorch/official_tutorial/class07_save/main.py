import torch
import torchvision.models as models


def save_model():
    model = models.vgg16(weights="IMAGENET1K_V1")
    # 保存
    torch.save(model.state_dict(), "model_weights.pth")
    print(model)


# 载入模型之前需要先创建好一模一样的模型结构, 然后把参数载入!
def load_model():
    model = models.vgg16()  # 注意这里没有指定weights所以只是一个有结构的模型,没有参数
    # 载入
    model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
    model.eval()
    print(model)


def model_file_with_shape():
    model = models.vgg16()
    print(model)
    torch.save(model, "model.pth")  # 带形状保存
    model2 = torch.load("model.pth", weights_only=False)  # 带形状载入
    print(model2)


if __name__ == "__main__":
    # save_model()
    # load_model()
    model_file_with_shape()
