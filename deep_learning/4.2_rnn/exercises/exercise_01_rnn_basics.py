"""
练习 1: RNN 基础

目标：
1. 理解 RNN 的基本结构
2. 手动实现 RNN 前向传播
3. 观察隐状态的变化

完成下面标记为 TODO 的部分
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================
# 练习 1.1: 实现 RNN 单步前向传播
# ============================================

def rnn_step(x_t, h_prev, W_xh, W_hh, b_h):
    """
    RNN 单步前向传播

    公式: h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

    参数:
        x_t: 当前输入, shape: (input_size,)
        h_prev: 上一时刻隐状态, shape: (hidden_size,)
        W_xh: 输入权重, shape: (hidden_size, input_size)
        W_hh: 隐层权重, shape: (hidden_size, hidden_size)
        b_h: 偏置, shape: (hidden_size,)

    返回:
        h_t: 当前隐状态, shape: (hidden_size,)
    """
    # TODO: 实现 RNN 单步前向传播
    # 提示: 使用 np.tanh() 激活函数

    h_t = None  # 替换这行

    return h_t


# ============================================
# 练习 1.2: 处理整个序列
# ============================================

def rnn_forward(X, h_0, W_xh, W_hh, b_h):
    """
    RNN 前向传播（处理整个序列）

    参数:
        X: 输入序列, shape: (seq_len, input_size)
        h_0: 初始隐状态, shape: (hidden_size,)
        W_xh, W_hh, b_h: 权重和偏置

    返回:
        h_all: 所有隐状态, shape: (seq_len, hidden_size)
        h_last: 最后一个隐状态
    """
    seq_len = X.shape[0]
    hidden_size = h_0.shape[0]

    # TODO: 初始化存储所有隐状态的数组
    h_all = None  # 替换这行

    h_t = h_0

    # TODO: 遍历每个时间步，调用 rnn_step
    for t in range(seq_len):
        pass  # 替换这行

    return h_all, h_t


# ============================================
# 测试你的实现
# ============================================

def test_implementation():
    np.random.seed(42)

    input_size = 4
    hidden_size = 8
    seq_len = 10

    # 初始化参数
    scale = np.sqrt(2.0 / (input_size + hidden_size))
    W_xh = np.random.randn(hidden_size, input_size) * scale
    W_hh = np.random.randn(hidden_size, hidden_size) * scale
    b_h = np.zeros(hidden_size)

    # 输入序列
    X = np.random.randn(seq_len, input_size)
    h_0 = np.zeros(hidden_size)

    # 测试 rnn_step
    h_1 = rnn_step(X[0], h_0, W_xh, W_hh, b_h)
    if h_1 is not None:
        print(f"✓ rnn_step 输出形状: {h_1.shape}")
        assert h_1.shape == (hidden_size,), "形状错误！"
        print(f"  h_1[:3] = {h_1[:3]}")
    else:
        print("✗ rnn_step 未实现")

    # 测试 rnn_forward
    h_all, h_last = rnn_forward(X, h_0, W_xh, W_hh, b_h)
    if h_all is not None:
        print(f"\n✓ rnn_forward 输出形状: {h_all.shape}")
        assert h_all.shape == (seq_len, hidden_size), "形状错误！"
        print(f"  最后隐状态: {h_last[:3]}")

        # 可视化隐状态演化
        plt.figure(figsize=(12, 4))
        plt.imshow(h_all.T, aspect='auto', cmap='RdBu')
        plt.colorbar()
        plt.xlabel('时间步')
        plt.ylabel('隐状态维度')
        plt.title('RNN 隐状态演化')
        plt.show()
    else:
        print("✗ rnn_forward 未实现")


if __name__ == "__main__":
    test_implementation()
