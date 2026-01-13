# 4.2 循环神经网络基础 (RNN)

> RNN 核心原理、从零实现与梯度问题分析

---

## 为什么学习 RNN？

**序列数据无处不在**：
- 文本：每个词依赖前面的上下文
- 股票：今天的价格受历史影响
- 语音：每个音素与前后相关
- 视频：帧与帧之间有连续性

**前馈网络的局限**：
- 固定长度输入，无法处理变长序列
- 无法捕获时间依赖关系
- 参数量随序列长度爆炸

**RNN 的解决方案**：
- 循环结构处理任意长度序列
- 隐状态作为"记忆"传递信息
- 参数共享，高效处理时序数据

---

## 学习目标

完成本模块后，你将能够：

- [ ] 理解 RNN 的核心思想和数学原理
- [ ] 用 NumPy 从零实现基础 RNN
- [ ] 深入理解 BPTT 反向传播算法
- [ ] 分析梯度消失/爆炸问题的成因
- [ ] 为学习 LSTM/GRU 打下坚实基础

---

## 前置知识

请确保已完成以下学习：

- ✅ **阶段 3**: 神经网络与深度学习基础
  - 反向传播算法（链式法则）
  - 梯度下降优化
  - PyTorch 基础
- ✅ **4.1 CNN**: 卷积神经网络（推荐但非必需）

---

## 模块结构

```
4.2_rnn/
│
├── 📖 RNN 基础 (01-04)
│   ├── 01_why_rnn.ipynb              # 为什么需要 RNN
│   ├── 02_rnn_from_scratch.ipynb     # RNN 从零实现
│   ├── 03_bptt_algorithm.ipynb       # BPTT 反向传播
│   └── 04_gradient_problems.ipynb    # 梯度消失/爆炸问题
│
└── 📝 exercises/                      # 练习文件
    └── exercise_01_rnn.py            # RNN 从零实现练习
```

---

## 学习路径（推荐顺序）

```
01 为什么需要 RNN
    ↓ 建立直觉：序列数据的特点和挑战
02 RNN 从零实现
    ↓ 掌握核心：前向传播、隐状态更新
03 BPTT 反向传播
    ↓ 理解梯度：时间维度的链式法则
04 梯度消失/爆炸
    ↓ 发现问题：为什么需要 LSTM/GRU

→→→ 继续学习：4.3_lstm/（LSTM/GRU 及序列建模）
```

---

## 核心概念速览

| 概念 | 英文 | 说明 |
|------|------|------|
| 循环神经网络 | RNN | 处理序列数据的基础架构 |
| 隐状态 | Hidden State | RNN 的"记忆"，传递时序信息 |
| BPTT | Backprop Through Time | RNN 的反向传播算法 |
| 梯度消失 | Vanishing Gradient | 长序列训练困难的原因 |
| 梯度爆炸 | Exploding Gradient | 梯度过大导致数值不稳定 |
| 梯度裁剪 | Gradient Clipping | 防止梯度爆炸的常用技巧 |

---

## 符号约定

| 符号 | 含义 | 维度 |
|------|------|------|
| $x_t$ | 时间步 t 的输入 | (input_size,) |
| $h_t$ | 时间步 t 的隐状态 | (hidden_size,) |
| $y_t$ | 时间步 t 的输出 | (output_size,) |
| $W_{xh}$ | 输入到隐层的权重 | (hidden_size, input_size) |
| $W_{hh}$ | 隐层到隐层的权重 | (hidden_size, hidden_size) |
| $W_{hy}$ | 隐层到输出的权重 | (output_size, hidden_size) |
| $T$ | 序列长度 | 标量 |
| $\tanh$ | 双曲正切激活函数 | - |

---

## RNN 核心公式

### 前向传播

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

### BPTT 反向传播（简化）

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial h_t} \cdot \prod_{k=t}^{1} \frac{\partial h_k}{\partial h_{k-1}}$$

**梯度消失原因**：当 $\|W_{hh}\| < 1$ 时，连乘项 → 0

**梯度爆炸原因**：当 $\|W_{hh}\| > 1$ 时，连乘项 → ∞

---

## 常见问题 (FAQ)

### Q1: 为什么 RNN 不能直接用于长序列？

由于 BPTT 中的连乘效应：
- 梯度消失：长距离依赖信息无法有效传递
- 梯度爆炸：数值不稳定，训练发散

**解决方案**：使用 LSTM/GRU（详见 4.3_lstm）

### Q2: 梯度裁剪如何使用？

```python
# PyTorch 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### Q3: RNN 的输入输出形状？

```python
# PyTorch RNN
# 输入: (seq_len, batch, input_size)
# 输出: (seq_len, batch, hidden_size)
# 隐状态: (num_layers, batch, hidden_size)
```

---

## 下一步学习

完成本模块后，继续学习：

📚 **4.3_lstm/** - LSTM/GRU 及序列建模
- LSTM 门控机制深入解析
- GRU 简化设计
- 双向 RNN 和多层堆叠
- 语言模型与 Seq2Seq
- 注意力机制
- 时间序列预测

---

## 参考资源

### 论文
- [RNN 原论文](https://www.cs.toronto.edu/~hinton/absps/pdp8.pdf) - Rumelhart et al., 1986

### 教程
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Christopher Olah
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy

---

## 开始学习

准备好了吗？从第一章开始你的 RNN 学习之旅！

👉 [01_why_rnn.ipynb](./01_why_rnn.ipynb) - 为什么需要 RNN

---

*最后更新: 2025-01*
