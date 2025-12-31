# 阶段4：深度学习核心架构

> 从 MLP 到 CNN、RNN、Transformer、生成模型的完整学习路径

---

## 前置知识

在学习本阶段前，你应该已经掌握：

- [x] 阶段3：神经网络与深度学习基础（MLP、反向传播、优化器等）
- [x] PyTorch 或 TensorFlow 基础使用
- [x] 损失函数、正则化、Batch Normalization 等训练技巧

---

## 核心概念：神经网络家族

```
神经网络 (Neural Network)
    │
    ├── MLP (多层感知机) ← 阶段3 ✅ 已学
    │   └── 全连接层，适合表格数据
    │
    ├── CNN (卷积神经网络) ← 4.1
    │   └── 卷积 + 池化，适合图像/视觉
    │
    ├── RNN (循环神经网络) ← 4.2
    │   └── 有记忆，适合序列/时序数据
    │
    ├── Transformer ← 4.3 ⭐最重要
    │   └── 注意力机制，现在最主流
    │
    └── 生成模型 ← 4.4
        ├── VAE (变分自编码器)
        ├── GAN (生成对抗网络)
        └── Diffusion (扩散模型)
```

---

## 模块总览

| 模块 | 名称 | 重要程度 | 核心内容 | 应用场景 |
|------|------|----------|----------|----------|
| 4.1 | CNN | ⭐⭐⭐ | 卷积、池化、ResNet | 图像分类、目标检测 |
| 4.2 | RNN | ⭐⭐⭐ | LSTM、GRU、Seq2Seq | 文本、时间序列 |
| 4.3 | Transformer | ⭐⭐⭐⭐⭐ | 注意力、BERT、GPT | NLP、CV、生成 |
| 4.4 | 生成模型 | ⭐⭐⭐⭐ | VAE、GAN、Diffusion | 图像生成、创意AI |
| 4.5 | GNN | ⭐⭐ | 图卷积、节点嵌入 | 社交网络、分子 |
| 4.6 | 深度RL | ⭐⭐ | DQN、PPO | 游戏AI、机器人 |

---

## 学习路径

### 必学核心路径（推荐顺序）

```
4.1 CNN ──→ 4.2 RNN ──→ 4.3 Transformer ──→ 4.4 生成模型
   │           │              │                   │
   ↓           ↓              ↓                   ↓
 图像处理   序列建模      现代NLP/CV         创意生成
```

### 选学扩展路径

- **4.5 GNN**：对图数据（社交网络、分子结构）感兴趣时学习
- **4.6 深度RL**：对游戏AI、机器人控制感兴趣时学习

---

## 各模块详细内容

### 4.1 卷积神经网络 (CNN)

```
4.1_cnn/
├── 01_convolution_basics.ipynb    # 卷积操作从零实现
├── 02_pooling_and_architecture.ipynb  # 池化层与网络架构
├── 03_classic_cnns.ipynb          # LeNet → AlexNet → VGG → ResNet
├── 04_transfer_learning.ipynb     # 迁移学习实战
├── 05_feature_visualization.ipynb # 特征可视化
└── project_image_classifier.py    # 项目：图像分类器
```

**核心概念：**
- 卷积操作：特征提取、参数共享
- 池化层：下采样、平移不变性
- 感受野：每个神经元"看到"的区域
- 经典架构演进：LeNet → AlexNet → VGG → ResNet → EfficientNet

### 4.2 循环神经网络 (RNN)

```
4.2_rnn/
├── 01_rnn_basics.ipynb            # RNN基础与序列建模
├── 02_lstm_gru.ipynb              # LSTM与GRU详解
├── 03_bidirectional_multilayer.ipynb  # 双向/多层RNN
├── 04_seq2seq.ipynb               # Seq2Seq架构
└── project_text_generator.py      # 项目：文本生成器
```

**核心概念：**
- 隐状态：网络的"记忆"
- 梯度消失/爆炸：长序列训练困难
- LSTM门控：遗忘门、输入门、输出门
- GRU：LSTM的简化版本

### 4.3 Transformer架构 ⭐

```
4.3_transformer/
├── 01_attention_mechanism.ipynb   # 注意力机制详解
├── 02_transformer_from_scratch.ipynb  # 从零实现Transformer
├── 03_bert_intro.ipynb            # BERT原理与使用
├── 04_gpt_intro.ipynb             # GPT原理与使用
├── 05_huggingface_practice.ipynb  # HuggingFace实战
└── project_text_classifier.py     # 项目：BERT微调
```

**核心概念：**
- 自注意力（Self-Attention）：Q、K、V
- 多头注意力：不同的"关注角度"
- 位置编码：告诉模型顺序信息
- BERT：双向编码器，理解任务
- GPT：自回归解码器，生成任务

### 4.4 生成模型

```
4.4_generative_models/
├── 01_autoencoder.ipynb           # 自编码器（AE）
├── 02_vae.ipynb                   # 变分自编码器（VAE）
├── 03_gan_basics.ipynb            # GAN基础
├── 04_dcgan_wgan.ipynb            # DCGAN与WGAN
├── 05_diffusion_intro.ipynb       # 扩散模型入门
└── project_image_generator.py     # 项目：图像生成
```

**核心概念：**
- 自编码器：编码-解码结构
- VAE：隐空间连续化，可采样生成
- GAN：生成器 vs 判别器的对抗训练
- 扩散模型：加噪-去噪，DALL-E/Stable Diffusion的基础

### 4.5 图神经网络 (GNN) [选学]

```
4.5_gnn/
├── 01_graph_basics.ipynb          # 图的基础概念
├── 02_gcn_graphsage.ipynb         # GCN与GraphSAGE
└── project_node_classification.py # 项目：节点分类
```

### 4.6 深度强化学习 [选学]

```
4.6_deep_rl/
├── 01_rl_basics.ipynb             # 强化学习基础
├── 02_dqn.ipynb                   # DQN详解
├── 03_policy_gradient.ipynb       # PPO等策略梯度
└── project_game_ai.py             # 项目：游戏AI
```

---

## 与阶段3的联系

| 阶段3 知识点 | 在阶段4中的应用 |
|-------------|----------------|
| 反向传播 | CNN/RNN 中的梯度计算 |
| 激活函数 | ReLU 在 CNN 中广泛使用 |
| Batch Normalization | CNN 标配，Transformer 用 Layer Norm |
| 残差连接 | ResNet 核心，Transformer 标配 |
| Dropout | 各类网络的正则化 |
| Adam 优化器 | 深度网络训练首选 |

---

## 学习建议

1. **动手实现**：每个概念都从零用 NumPy 实现一遍
2. **理解再用库**：理解原理后再使用 PyTorch/TensorFlow
3. **可视化**：画出网络结构、特征图、注意力分布
4. **做项目**：每个模块都有配套项目，务必完成

---

## 后续方向

完成阶段4后，可以选择：

- **阶段5.1 计算机视觉 (CV)**：目标检测、图像分割、3D视觉
- **阶段5.2 自然语言处理 (NLP)**：大语言模型、RAG、Agent
- **阶段5.3 生成式AI**：Stable Diffusion、视频生成

---

**开始学习吧！** 从 `4.1_cnn/01_convolution_basics.ipynb` 开始！
