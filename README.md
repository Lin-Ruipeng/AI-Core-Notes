# AI-Core-Notes

AI 核心知识库：从数学推导到端侧部署的全链路学习笔记，涵盖 HPC 数值计算、PyTorch 深度学习、AI 工具开发与 ONNX 模型部署。

## 项目总览

| 模块 | 目录 | 核心内容 | 语言 |
|------|------|----------|------|
| HPC 数学理论 | `math_theory/` | 数值微积分、线性代数、概率统计的 HPC 实现 | C++17 |
| PyTorch 深度学习 | `learn_pytorch/` | 从零实现神经网络 + PyTorch 官方教程 | Python |
| 小型 AI 工具 | `tiny_skills/` | LLM API 封装、手写 Agent、MCP 原型 | Python |
| ONNX 端侧部署 | `ONNX/` | 模型导出与端侧推理（待完善） | Python |

---

## 一、HPC 数学理论（`math_theory/`）

从高性能计算与端侧部署的视角，系统复习微积分→线性代数→概率论→数理统计，每节课包含 `note.md`（理论笔记）与 `main.cpp`（数值实现）。

| 课号 | 主题 | 核心知识点 | HPC 要点 |
|------|------|-----------|----------|
| class01 | 数值微分与中心差分法 | 泰勒展开、截断误差与舍入误差的权衡 | 灾难性相消、最优步长 h ≈ 1e-5 |
| class02 | 数值积分与累积误差控制 | 梯形法/Simpson 法、浮点累加误差 | 大数吃小数、Kahan 求和算法 |
| class03 | 矩阵乘法的内存博弈 | 缓存行、行主序/列主序、循环交换 | ijk→ikj 循环顺序性能差 10~50 倍 |
| class04 | 分支预测与无分支编程 | CPU 流水线、条件分支惩罚 | 用 `fmin`/`fmax`、三元运算符替代分支 |
| class05 | 多核并行与 OpenMP | Fork-Join 模型、共享内存并行 | `#pragma omp parallel for reduction(+:sum)` |
| class06 | Cholesky 分解 | 对称正定矩阵分解、求解线性方程组 | 禁止显式求逆，使用 `A.llt().solve(b)` |
| class07 | SVD 与截断伪逆 | 奇异值分解、Moore-Penrose 伪逆 | 秩亏检测、病态矩阵、截断阈值 1e-10 |
| class08 | 高斯分布与噪声建模 | 正态分布、Box-Muller 变换 | 禁止共享随机数引擎，每个线程独立引擎 |
| class09 | 对数概率空间 | 概率乘法下溢出、log-sum-exp 技巧 | 连乘变连加，使用 `log1p` 提升精度 |
| class10 | 最小二乘法 | MLE → LSM 推导、QR 分解 | 永不开方，避免数值不稳定的平方根计算 |
| class11 | Welford 算法 | 在线方差计算、单遍扫描递推 | 数值稳定性远优于 `E[X²]−E[X]²` 公式 |
| class12 | MLE 与梯度下降 | 似然函数、梯度上升、凸优化 | OpenMP 并行计算代价函数 + 串行参数更新 |
| class13 | 卡尔曼滤波 | 预测-更新迭代、状态协方差矩阵 | Joseph 形式协方差更新，避免协方差崩溃 |

---

## 二、PyTorch 深度学习（`learn_pytorch/`）

### 2.1 手撸深度学习（`hand_rolled_DL/`）

基于《深度学习入门——基于 Python 的理论与实现》（鱼书），从零实现神经网络，**仅依赖 NumPy**。

| 课号 | 主题 |
|------|------|
| class01 | Python 基础与 NumPy 入门 |
| class02 | 感知机与逻辑门 |
| class03 | 神经网络与激活函数 |
| class04 | 学习算法与梯度下降 |
| class05 | 误差反向传播法 |
| class06 | 训练技巧与优化方法 |
| class07 | 卷积神经网络（CNN） |
| class08 | 深度学习实战 |

公用模块见于 `common/`（激活函数、梯度、层实现、优化器、训练器），`dataset/` 包含 MNIST 数据集加载。

### 2.2 官方教程（`official_tutorial/`）

基于 PyTorch 官方教程的实战笔记，覆盖张量操作到模型保存全流程。

| 课号 | 主题 |
|------|------|
| class01 | 张量基础与 GPU 加速 |
| class02 | DataLoader 与数据集 |
| class03 | Transforms 数据变换 |
| class04 | 模型构建（nn.Module） |
| class05 | 自动求导（Autograd） |
| class06 | 训练循环与评估 |
| class07 | 模型保存与加载 |

---

## 三、小型 AI 工具（`tiny_skills/`）

| 项目 | 说明 | 状态 |
|------|------|------|
| `APIuseLLM/` | 智谱 GLM API 封装，多轮对话与工具调用 | ✅ 可用 |
| `DeepSeek_doc/` | DeepSeek API 文档示例与代码 | ✅ 可用 |
| `Hand_rolled_OpenClaw/` | 手写 Agent 框架原型（Agent + Skill 定义） | ✅ 可用 |
| `hand_rolled_MCP/` | MCP（Model Context Protocol）原型 | 🚧 待完善 |

---

## 四、ONNX 端侧部署（`ONNX/`）

模型导出（PyTorch → ONNX）及端侧推理部署的学习笔记。

> 🚧 目录结构已建立，内容待完善。

---

## 开发环境配置

### Python 环境（uv 管理）

```bash
# 安装 Python 3.12+
uv sync                    # 创建虚拟环境并安装依赖
source .venv/bin/activate  # 激活虚拟环境
```

核心依赖：`torch>=2.11.0`、`torchvision>=0.26.0`（CUDA 13.0）、`onnxscript>=0.7.0`、`openai>=2.35.1`、`zai-sdk>=0.2.2`、`matplotlib>=3.10.9`、`pandas>=3.0.2`、`pytest>=9.0.3`

### C++ 环境（Eigen + OpenMP）

```bash
# Ubuntu / WSL
sudo apt install libeigen3-dev libomp-dev cmake g++
g++ -std=c++17 -O3 -march=native -fopenmp -I/usr/include/eigen3 main.cpp -o main
```

C++ 代码采用 C++17 标准，使用 Eigen 3 进行线性代数运算，OpenMP 实现多核并行加速。

---

## 项目目录结构

```
AI-Core-Notes/
├── math_theory/                  # HPC 数学理论（13 节课）
│   ├── class01_diff/             #   数值微分
│   ├── class02_integral/         #   数值积分
│   ├── class03_cache_miss/       #   矩阵乘法与缓存
│   ├── class04_if_predict/       #   分支预测
│   ├── class05_mult_core/        #   OpenMP 多核并行
│   ├── class06_cholesky/         #   Cholesky 分解
│   ├── class07_SVD/              #   SVD 分解
│   ├── class08_gauss/            #   高斯分布
│   ├── class09_log/              #   对数概率空间
│   ├── class10_LSM/              #   最小二乘法
│   ├── class11_welford/          #   Welford 在线方差
│   ├── class12_MLE/              #   MLE 与梯度下降
│   ├── class13_kalman/           #   卡尔曼滤波
│   └── prompt.md                 #   HPC 教学 Prompt
├── learn_pytorch/                # PyTorch 深度学习
│   ├── hand_rolled_DL/           #   从零实现（鱼书）
│   │   ├── class01_python_basic/ #     Python 基础
│   │   ├── class02_perceptron/   #     感知机
│   │   ├── class03_Neural_network/ #   神经网络
│   │   ├── class04_learning/     #     学习算法
│   │   ├── class05_bp/           #     反向传播
│   │   ├── class06_tips/         #     训练技巧
│   │   ├── class07_CNN/          #     卷积神经网络
│   │   ├── class08_deep_learn/   #     深度学习
│   │   ├── common/               #     公用模块
│   │   └── dataset/              #     MNIST 数据集
│   └── official_tutorial/        #   PyTorch 官方教程
│       ├── class01_tensors/      #     张量基础
│       ├── class02_dataloader/   #     数据加载
│       ├── class03_transforms/   #     数据变换
│       ├── class04_model/        #     模型构建
│       ├── class05_autograd/     #     自动求导
│       ├── class06_train/        #     训练循环
│       └── class07_save/         #     模型保存
├── tiny_skills/                  # 小型 AI 工具
│   ├── APIuseLLM/                #   智谱 API 封装
│   ├── DeepSeek_doc/             #   DeepSeek API 示例
│   ├── Hand_rolled_OpenClaw/     #   手写 Agent
│   └── hand_rolled_MCP/          #   MCP 原型（待完善）
├── ONNX/                         # ONNX 端侧部署（待完善）
├── data/                         # 数据集目录（已 gitignore）
├── .venv/                        # Python 虚拟环境
├── .clang-format                 # C++ 代码格式化配置
├── .cmake-format.yaml            # CMake 格式化配置
├── .editorconfig                 # 编辑器统一配置
├── .python-version               # Python 版本锁定
├── pyproject.toml                # Python 项目配置
└── README.md                     # 本文件
```

---

## 学习路径建议

```
数学基础（HPC 数值计算） → 深度学习理论（从零实现） → PyTorch 工程实践 → 小型工具开发 → ONNX 端侧部署
```

1. **入门**：从 `math_theory/class01~class05` 开始，掌握数值计算基础与 HPC 思维
2. **进阶**：学习 `math_theory/class06~class13`，深入线性代数与概率统计的数值实现
3. **深度学习理论**：跟随 `learn_pytorch/hand_rolled_DL/` 从零搭建神经网络
4. **框架实践**：通过 `learn_pytorch/official_tutorial/` 熟悉 PyTorch 生态
5. **工具链**：探索 `tiny_skills/` 中的 API 调用与 Agent 开发
6. **部署**：待 ONNX 模块完善后，学习模型导出与端侧推理

---

## 许可

MIT License — Copyright © 2026 Fox Lin（林惢朋）
