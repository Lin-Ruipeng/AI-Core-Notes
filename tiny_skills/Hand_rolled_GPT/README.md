# 手搓 GPT

本课参考 [Let's build GPT](https://www.bilibili.com/video/BV1ZvgQ6iEtN/)

不使用现成的模块，只使用 torch 的基本组件，构建了一个简单的 GPT 模型，并且使用莎士比亚的文本数据集训练了一个可以模仿莎士比亚风格的文字的 GPT 模型。本模型采用 decoder-only 的架构。

本课的两个文件分别是：
1. [简单的 bigram 模型](./bigram.py)
2. [简单的 transfromer 模型](./gpt.py)

## 快速开始

```bash
# 1. 同步环境: 推荐在 Linux / WSL 环境下，并且有 cuda 13 驱动
uv sync
# 2. 运行
uv run tiny_skills/Hand_rolled_GPT/bigram.py
uv run tiny_skills/Hand_rolled_GPT/gpt.py
```
