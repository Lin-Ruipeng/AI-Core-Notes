import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 64  # 批处理大小，并行训练的序列个数
block_size = 256  # 预测的最大的上下文长度
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384  # 嵌入维度/向量维度
n_head = 6  # 多头注意力的头数量
n_layer = 6
dropout = 0.2

"""
这里解释一下 n_embd 这个嵌入维度贯穿整个模型
- token 嵌入表: vocab_size * n_embd
-   位置 嵌入表: block_size * n_embd
- 每个注意力头的 Q/K/V 投影维度
- FFN 中间层的维度
- 最终输出 logits 前的隐藏状态
"""

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


class Head(nn.Module):
    """单头的自注意力"""

    def __init__(self, head_size):
        super().__init__()
        # 一个注意力头的 KQV 向量
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入: (batch, time-step, channels)
        # 输出: (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs) 用一个token的编码转换到一个K向量
        q = self.query(x)  # (B, T, hs) 用一个token的编码转换到一个Q向量
        # 计算注意力分数 ("affinities")
        # 先计算出 \frac{Q \dot K ^ T}{ \sqrt{K_d} } 也就是注意力机制公式的圆括号内的部分
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # 这里是在做掩码，注意这里构建一个对角线和下三角为0,但是上三角是负无穷的矩阵
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        # 如果将这个矩阵的每行做softmax那么就可以得到用来求均值的矩阵
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # 在掩码矩阵这里还引入dropout
        # 对这些值进行加权聚合,也就是注意力公式的最后一个v乘上来
        v = self.value(x)  # 就算是v也要查表得到
        out = wei @ v  # （B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out  # 返回带有掩码的自注意力机制计算结果,完美和论文公式符合


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # 多头不就是多个单头么，直接for就完事了
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 但是要注意，本来是希望输入c维度经过注意力机制输出c维度的，现在用多头的话，就用a个头，每个头输出b
        # 最后你需要把每个头的输出拼接到一起，最后再加一个全连接网络融合一下不同的头的信息，最后还是输出c维
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        # 这里的num_heads*head_size就是多个头的输出拼接的维度，n_embd就是输出维度
        self.dropout = nn.Dropout(dropout)  # 这里也加一个 dropout

    def forward(self, x):
        # 生硬的拼接多头注意力的输出
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 用全连接层融合输出并进行 dropout
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """一个简单的前馈网络层 (FFN)"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 输入是n_embd维的，先升维4倍
            nn.ReLU(),  # 激活函数
            nn.Linear(4 * n_embd, n_embd),  # 降维回到嵌入维度
            nn.Dropout(dropout),  # 再来一个 dropout
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, n_embd, n_head):
        # n_embd 是嵌入维度； n_head 是多头的头数
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # 多头自注意力层
        self.ffwd = FeedFoward(n_embd)  # 前馈神经网络层
        self.ln1 = nn.LayerNorm(n_embd)  # 层归一化!
        self.ln2 = nn.LayerNorm(n_embd)  # 层归一化!

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # 残差连接自注意力层!
        x = x + self.ffwd(self.ln2(x))  # 残差连接前向反馈层!
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 每个 token 直接从查找表中读取下一个 token 的 logits
        # 这里的 logits 意思就是 softmax回归 在做softmax之前的原始分数
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )  # 创建一个词嵌入表 (词语个数, 嵌入维度)
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )  # 创建一个位置嵌入表 (位置个数, 嵌入维度)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # 最后一层的标准化，有均值和标准差两个训练参数
        self.lm_head = nn.Linear(
            n_embd, vocab_size
        )  # 最后一层线性层，嵌入维度到词表数，用于多分类逻辑回归 (给每个预测词打分)
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx 以及 targets 都是 (B, T) 的整数 tensor
        tok_emb = self.token_embedding_table(idx)  # (B, T, C) 词向量嵌入，编码
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C) 位置编码
        x = tok_emb + pos_emb  # 直接数值相加，维度不变
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, Tb vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是当前上下文的下标 (B, T)
        for _ in range(max_new_tokens):
            # 取最后一个 block_size 的 tokens 下标
            idx_cond = idx[:, -block_size:]
            # 获得预测结果
            logits, loss = self(idx_cond)
            # 只看最后的时间步
            logits = logits[:, -1, :]  # 变成 (B, C)
            # 获取概率
            probs = F.softmax(logits, dim=-1)
            # 按照概率采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 把预测的结果追加到正在运行的序列的最后
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPTLanguageModel()
m = model.to(device)
# 打印参数量
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameterss")

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
