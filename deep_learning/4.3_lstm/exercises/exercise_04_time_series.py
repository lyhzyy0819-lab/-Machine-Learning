"""
练习 5: 时间序列预测

目标：
1. 准备时间序列数据
2. 实现 LSTM 预测模型
3. 评估预测性能
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============================================
# 练习 5.1: 创建滑动窗口数据集
# ============================================

def create_sequences(data, seq_len, pred_len=1):
    """
    创建滑动窗口序列

    参数:
        data: 一维时间序列 (numpy array)
        seq_len: 输入序列长度
        pred_len: 预测长度

    返回:
        X: 输入序列, shape: (n_samples, seq_len)
        y: 目标值, shape: (n_samples, pred_len)
    """
    # TODO: 实现滑动窗口
    X, y = [], []

    for i in range(len(data) - seq_len - pred_len + 1):
        # TODO: 提取输入和目标
        pass

    return np.array(X), np.array(y)


# ============================================
# 练习 5.2: LSTM 预测模型
# ============================================

class LSTMPredictor(nn.Module):
    """
    LSTM 时间序列预测模型

    TODO: 完成实现
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        # TODO: 定义 LSTM 和全连接层
        self.lstm = None
        self.fc = None

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        返回: (batch, output_size)
        """
        # TODO: 前向传播
        # 提示: 只取 LSTM 最后一个时间步的输出
        return None


# ============================================
# 练习 5.3: 训练和评估
# ============================================

def train_and_evaluate():
    """
    完整的训练和评估流程
    """
    # 生成合成数据
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    data = np.sin(t * 0.1) + 0.5 * np.sin(t * 0.3) + np.random.randn(1000) * 0.1

    # 标准化
    mean, std = data.mean(), data.std()
    data_norm = (data - mean) / std

    # 创建序列
    seq_len = 30
    X, y = create_sequences(data_norm, seq_len)

    if X is None or len(X) == 0:
        print("✗ create_sequences 未实现")
        return

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # 划分数据集
    split = int(len(X) * 0.8)
    X_train = torch.FloatTensor(X[:split]).unsqueeze(-1)
    y_train = torch.FloatTensor(y[:split])
    X_test = torch.FloatTensor(X[split:]).unsqueeze(-1)
    y_test = torch.FloatTensor(y[split:])

    # 创建模型
    model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2)

    # 检查模型
    pred = model(X_train[:4])
    if pred is None:
        print("✗ LSTMPredictor 未实现")
        return

    print(f"✓ 模型输出形状: {pred.shape}")

    # TODO: 实现训练循环
    # 提示:
    # 1. 定义 optimizer 和 criterion
    # 2. 训练多个 epoch
    # 3. 记录损失
    # 4. 在测试集上评估

    print("\n继续实现训练循环...")


# ============================================
# 可视化辅助函数
# ============================================

def plot_predictions(y_true, y_pred, title="预测结果"):
    """
    绘制预测结果对比图
    """
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:100], label='真实值', linewidth=2)
    plt.plot(y_pred[:100], label='预测值', linewidth=2, linestyle='--')
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    train_and_evaluate()
