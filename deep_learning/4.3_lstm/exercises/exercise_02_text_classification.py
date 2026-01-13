"""
练习 3: 文本分类

目标：
1. 构建文本分类数据集
2. 实现 LSTM 分类器
3. 训练和评估模型

使用 PyTorch 完成下面的 TODO 部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ============================================
# 练习 3.1: 实现数据集类
# ============================================

class TextDataset(Dataset):
    """
    简单的文本分类数据集

    TODO: 完成 __getitem__ 方法
    """

    def __init__(self, texts, labels, vocab, max_len=50):
        """
        参数:
            texts: 文本列表
            labels: 标签列表
            vocab: 词表字典 {word: index}
            max_len: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        返回:
            x: 词索引张量, shape: (max_len,)
            y: 标签张量
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # TODO: 将文本转为词索引
        # 提示:
        # 1. 分词: words = text.lower().split()
        # 2. 查词表: vocab.get(word, 0)  # 0 是 <unk> 的索引
        # 3. 填充/截断到 max_len
        # 4. 转为 tensor

        x = None  # 替换这行
        y = None  # 替换这行

        return x, y


# ============================================
# 练习 3.2: 实现 LSTM 分类器
# ============================================

class LSTMClassifier(nn.Module):
    """
    基于 LSTM 的文本分类器

    TODO: 完成 forward 方法
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # 双向时维度翻倍
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        参数:
            x: (batch, seq_len) 词索引

        返回:
            logits: (batch, num_classes)
        """
        # TODO: 实现前向传播
        # 步骤:
        # 1. 词嵌入
        # 2. LSTM
        # 3. 取最后一个时间步（或池化）
        # 4. Dropout + 全连接

        logits = None  # 替换这行

        return logits


# ============================================
# 练习 3.3: 训练循环
# ============================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个 epoch

    TODO: 完成训练循环
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        # TODO: 前向传播
        # TODO: 计算损失
        # TODO: 反向传播
        # TODO: 更新参数
        # TODO: 统计准确率

        pass  # 替换这行

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """
    评估模型

    TODO: 完成评估循环
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # TODO: 前向传播并计算准确率

            pass  # 替换这行

    return total_loss / len(dataloader), correct / total


# ============================================
# 测试
# ============================================

def test_components():
    # 模拟数据
    texts = [
        "I love this movie it is great",
        "This film is terrible I hate it",
        "What a wonderful experience",
        "Worst movie ever do not watch"
    ]
    labels = [1, 0, 1, 0]  # 1: 正面, 0: 负面

    # 构建简单词表
    all_words = set()
    for t in texts:
        all_words.update(t.lower().split())
    vocab = {w: i + 1 for i, w in enumerate(sorted(all_words))}
    vocab['<pad>'] = 0

    print(f"词表大小: {len(vocab)}")

    # 测试数据集
    dataset = TextDataset(texts, labels, vocab, max_len=10)
    x, y = dataset[0]
    if x is not None:
        print(f"✓ 数据集 - x shape: {x.shape}, y: {y}")
    else:
        print("✗ 数据集未实现")

    # 测试模型
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=32,
        hidden_size=64,
        num_classes=2
    )

    x_batch = torch.randint(0, len(vocab), (4, 10))
    logits = model(x_batch)
    if logits is not None:
        print(f"✓ 模型 - logits shape: {logits.shape}")
    else:
        print("✗ 模型未实现")


if __name__ == "__main__":
    test_components()
