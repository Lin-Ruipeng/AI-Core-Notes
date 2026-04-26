# Save and Load the Model

## 注意

保存和载入模型时，只对权重到文件的存取，这是最佳实践！
模型的结构自行用python代码实现！

## 保存

```python
model = Model() # 用一个类构造一模一样的结构
torch.save(model.state_dict(), "model_weights.pth")
```

## 载入

```python
model = Model() # 用一个类构造一模一样的结构
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
```