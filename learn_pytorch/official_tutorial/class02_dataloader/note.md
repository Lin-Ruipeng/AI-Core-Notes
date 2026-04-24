# DataLoader and Dataset

既可以用一些预装的数据集, 也可以自己导入数据集

## 自己导入数据集

必须手动实现三个函数: 
1. __init__
2. __len__
3. __getitem__

## Dataloader

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

注意, 这里的 batch_size 是 minibatch 的大小!
这里的 shuffle 不是只打乱一次, 而是每个 epoch 都会打乱一次! 以防止过拟合!
