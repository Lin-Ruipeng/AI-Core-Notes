import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 32  # 批次大小：并行处理多少条独立的序列
block_size = 8  # 最大的预测的上下文长度
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(r"tiny_skills/Hand_rolled_GPT/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 这里是这段文本中出现的所有不重复字符
chars = sorted(set(text))
vocab_size = len(chars)  # 单词表大小 (实际上是字母+符号表)

# 创建一个映射 (字符, 整数) 这里就是 Tokenizer
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # 编码器: 字符串 -> 整数列表
decode = lambda l: "".join([itos[i] for i in l])  # 解码器：整数列表 -> 字符串

# 训练和测试 数据集切分
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 前面 90% 认为是训练集
train_data = data[:n]
val_data = data[n:]


# 数据载入
def get_batch(split):
    # 生成一个小批次的数据 (x, y) 即输入输出对
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 这里稍微解释下，就是预测的是下一个token，所以先确定一个下标ix那么输入 data[ix] 预测输出 data[ix+1]
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# 一个极简的 bigram 模型
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 每个 token 直接从查阅表中读出下一个 token 的 logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # 训练和推理共用这个函数，所以这里会需要计算一个 loss
    def forward(self, idx, targets=None):
        # idx 和 targets 都是 (B, T) 这个形状的整数 tensor
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是当前的上下文的 (B, T) 数组的索引
        for _ in range(max_new_tokens):
            # 先预测
            logits, loss = self(idx)
            # 只看最后一个时间步
            logits = logits[:, -1, :]  # 变成了 (B, C)
            # 使用 softmax 获取概率
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # 从分布里采样 (一定要注意这里是按照概率采样!)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 添加采样的下标到运行的序列中
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # 评估损失
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
    # 采样一个批次
    xb, yb = get_batch("train")
    # 评估损失
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 从模型中生成
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
