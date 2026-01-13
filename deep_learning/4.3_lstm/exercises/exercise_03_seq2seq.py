"""
练习 4: Seq2Seq 模型

目标：
1. 实现 Encoder-Decoder 架构
2. 理解 Teacher Forcing
3. 实现简单的序列转换任务（如反转字符串）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================
# 练习 4.1: 实现编码器
# ============================================

class Encoder(nn.Module):
    """
    Seq2Seq 编码器

    TODO: 完成实现
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, src):
        """
        src: (batch, src_len)
        返回: hidden, cell
        """
        # TODO: 实现编码器前向传播
        embedded = None  # 词嵌入
        outputs, (hidden, cell) = None, (None, None)  # LSTM

        return hidden, cell


# ============================================
# 练习 4.2: 实现解码器
# ============================================

class Decoder(nn.Module):
    """
    Seq2Seq 解码器

    TODO: 完成实现
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden, cell):
        """
        input: (batch, 1) 当前时间步输入
        返回: prediction, hidden, cell
        """
        # TODO: 实现解码器单步前向传播
        prediction = None

        return prediction, hidden, cell


# ============================================
# 练习 4.3: 组合成完整的 Seq2Seq
# ============================================

class Seq2Seq(nn.Module):
    """
    完整的 Seq2Seq 模型

    TODO: 完成 forward 方法
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # TODO: 编码
        hidden, cell = None, None

        # TODO: 解码（使用 teacher forcing）
        input = trg[:, 0:1]  # 第一个输入是 <sos>

        for t in range(1, trg_len):
            # TODO: 解码器单步
            # TODO: 存储输出
            # TODO: 决定是否使用 teacher forcing
            pass

        return outputs


# ============================================
# 练习 4.4: 字符串反转任务
# ============================================

def create_reverse_dataset(n_samples=1000, max_len=10):
    """
    创建字符串反转数据集

    例如: "abc" → "cba"
    """
    chars = list("abcdefghij")
    vocab = {c: i + 2 for i, c in enumerate(chars)}
    vocab['<pad>'] = 0
    vocab['<sos>'] = 1

    src_data = []
    trg_data = []

    for _ in range(n_samples):
        length = np.random.randint(3, max_len + 1)
        seq = np.random.choice(chars, length)

        src = [vocab[c] for c in seq]
        trg = [vocab['<sos>']] + [vocab[c] for c in reversed(seq)]

        # 填充
        src = src + [0] * (max_len - len(src))
        trg = trg + [0] * (max_len + 1 - len(trg))

        src_data.append(src)
        trg_data.append(trg)

    return (torch.LongTensor(src_data),
            torch.LongTensor(trg_data),
            vocab)


# ============================================
# 测试
# ============================================

def test_seq2seq():
    device = torch.device('cpu')

    # 创建数据
    src, trg, vocab = create_reverse_dataset(100)
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {trg.shape}")
    print(f"词表: {vocab}")

    # 创建模型
    vocab_size = len(vocab)
    encoder = Encoder(vocab_size, embed_dim=32, hidden_size=64)
    decoder = Decoder(vocab_size, embed_dim=32, hidden_size=64)
    model = Seq2Seq(encoder, decoder, device)

    # 测试前向传播
    outputs = model(src[:4], trg[:4])
    if outputs is not None and outputs.sum() != 0:
        print(f"✓ Seq2Seq 输出形状: {outputs.shape}")
    else:
        print("✗ Seq2Seq 未完全实现")

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_seq2seq()
