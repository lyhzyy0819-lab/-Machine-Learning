"""
练习 2: LSTM 实现

目标：
1. 理解 LSTM 的门控机制
2. 实现 LSTM Cell
3. 对比 LSTM 和 RNN 的记忆能力

完成下面标记为 TODO 的部分
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """数值稳定的 Sigmoid"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


# ============================================
# 练习 2.1: 实现 LSTM Cell
# ============================================

def lstm_step(x_t, h_prev, c_prev, W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o):
    """
    LSTM 单步前向传播

    公式:
        f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)  # 遗忘门
        i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)  # 输入门
        c_tilde = tanh(W_c @ [h_{t-1}, x_t] + b_c) # 候选细胞状态
        c_t = f_t * c_{t-1} + i_t * c_tilde        # 更新细胞状态
        o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)  # 输出门
        h_t = o_t * tanh(c_t)                       # 更新隐状态

    参数:
        x_t: 当前输入, shape: (input_size,)
        h_prev: 上一时刻隐状态, shape: (hidden_size,)
        c_prev: 上一时刻细胞状态, shape: (hidden_size,)
        W_f, W_i, W_c, W_o: 各门的权重, shape: (hidden_size, hidden_size + input_size)
        b_f, b_i, b_c, b_o: 各门的偏置, shape: (hidden_size,)

    返回:
        h_t: 当前隐状态
        c_t: 当前细胞状态
        gates: 字典，包含各门的值（用于分析）
    """
    # TODO: 拼接 h_prev 和 x_t
    concat = None  # 替换这行

    # TODO: 计算遗忘门
    f_t = None  # 替换这行

    # TODO: 计算输入门
    i_t = None  # 替换这行

    # TODO: 计算候选细胞状态
    c_tilde = None  # 替换这行

    # TODO: 更新细胞状态
    c_t = None  # 替换这行

    # TODO: 计算输出门
    o_t = None  # 替换这行

    # TODO: 更新隐状态
    h_t = None  # 替换这行

    gates = {
        'forget': f_t,
        'input': i_t,
        'candidate': c_tilde,
        'output': o_t
    }

    return h_t, c_t, gates


# ============================================
# 练习 2.2: 分析门的行为
# ============================================

def analyze_gates(gates_history):
    """
    分析各门在序列处理过程中的行为

    参数:
        gates_history: 列表，每个元素是一个时间步的 gates 字典

    TODO: 绘制每个门的平均值随时间变化的曲线
    """
    seq_len = len(gates_history)

    # TODO: 提取每个门的平均值
    forget_means = []
    input_means = []
    output_means = []

    for gates in gates_history:
        # TODO: 计算每个门的平均值
        pass

    # TODO: 绘制图表
    plt.figure(figsize=(12, 4))
    # ... 你的绘图代码
    plt.show()


# ============================================
# 测试
# ============================================

def test_lstm():
    np.random.seed(42)

    input_size = 4
    hidden_size = 8
    seq_len = 20

    # 初始化参数
    concat_size = hidden_size + input_size
    scale = np.sqrt(2.0 / (concat_size + hidden_size))

    W_f = np.random.randn(hidden_size, concat_size) * scale
    W_i = np.random.randn(hidden_size, concat_size) * scale
    W_c = np.random.randn(hidden_size, concat_size) * scale
    W_o = np.random.randn(hidden_size, concat_size) * scale

    b_f = np.ones(hidden_size)  # 遗忘门偏置初始化为 1
    b_i = np.zeros(hidden_size)
    b_c = np.zeros(hidden_size)
    b_o = np.zeros(hidden_size)

    # 输入序列
    X = np.random.randn(seq_len, input_size)
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)

    # 运行 LSTM
    gates_history = []
    h_history = []
    c_history = []

    for t in range(seq_len):
        h, c, gates = lstm_step(X[t], h, c, W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o)
        if h is not None:
            gates_history.append(gates)
            h_history.append(h.copy())
            c_history.append(c.copy())

    if h is not None:
        print(f"✓ LSTM 实现成功！")
        print(f"  最终隐状态: {h[:3]}")
        print(f"  最终细胞状态: {c[:3]}")

        # 可视化
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        axes[0].imshow(np.array(h_history).T, aspect='auto', cmap='RdBu')
        axes[0].set_title('隐状态 h 的演化')
        axes[0].set_xlabel('时间步')
        axes[0].set_ylabel('维度')

        axes[1].imshow(np.array(c_history).T, aspect='auto', cmap='RdBu')
        axes[1].set_title('细胞状态 c 的演化')
        axes[1].set_xlabel('时间步')
        axes[1].set_ylabel('维度')

        plt.tight_layout()
        plt.show()
    else:
        print("✗ LSTM 未实现")


if __name__ == "__main__":
    test_lstm()
