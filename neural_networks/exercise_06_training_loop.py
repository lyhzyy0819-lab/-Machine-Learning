"""
练习1：完整训练循环

本练习演示如何使用 training_utils 模块构建一个包含所有训练技巧的完整训练循环：
- Cosine Annealing 学习率调度
- 梯度裁剪 (max_norm=1.0)
- 早停 (patience=10)
- 每10个epoch进行训练诊断

使用 sklearn.datasets.load_digits 数据集（8x8 手写数字图像，10个类别）

运行方式:
    python exercise_06_training_loop.py

作者: Machine Learning 学习项目
日期: 2024年12月
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入自定义训练工具
from training_utils import (
    TrainingMonitor,
    CosineAnnealingScheduler,
    diagnose_training
)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)


# =============================================================================
#                           简单神经网络
# =============================================================================

class SimpleNeuralNetwork:
    """
    简单的两层神经网络（用于演示）

    结构: Input -> Hidden (ReLU) -> Output (Softmax)

    参数:
    -----
    input_size : int
        输入特征数
    hidden_size : int
        隐藏层神经元数
    output_size : int
        输出类别数
    l2_reg : float
        L2 正则化系数，默认 0.01
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 l2_reg: float = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l2_reg = l2_reg

        # He 初始化（适用于 ReLU）
        # W1: (input_size, hidden_size)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        # W2: (hidden_size, output_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # 缓存（用于反向传播）
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
        -----
        X : np.ndarray, shape (batch_size, input_size)
            输入数据

        返回:
        -----
        np.ndarray, shape (batch_size, output_size)
            预测概率
        """
        # 隐藏层: Z1 = X @ W1 + b1
        Z1 = X @ self.W1 + self.b1

        # ReLU 激活: A1 = max(0, Z1)
        A1 = np.maximum(0, Z1)

        # 输出层: Z2 = A1 @ W2 + b2
        Z2 = A1 @ self.W2 + self.b2

        # Softmax: 将输出转换为概率
        # 为了数值稳定性，减去最大值
        exp_Z2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        A2 = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)

        # 缓存中间结果
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

        return A2

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算交叉熵损失 + L2 正则化

        参数:
        -----
        y_pred : np.ndarray, shape (batch_size, output_size)
            预测概率
        y_true : np.ndarray, shape (batch_size,)
            真实标签

        返回:
        -----
        float
            总损失
        """
        batch_size = y_true.shape[0]

        # 交叉熵损失: -1/N * Σ log(p_correct)
        # 添加小值防止 log(0)
        log_probs = -np.log(y_pred[range(batch_size), y_true] + 1e-8)
        ce_loss = np.mean(log_probs)

        # L2 正则化: λ/2 * (||W1||^2 + ||W2||^2)
        l2_loss = 0.5 * self.l2_reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))

        return ce_loss + l2_loss

    def backward(self, y_true: np.ndarray) -> list:
        """
        反向传播，计算梯度

        参数:
        -----
        y_true : np.ndarray, shape (batch_size,)
            真实标签

        返回:
        -----
        list
            梯度列表 [dW1, db1, dW2, db2]
        """
        batch_size = y_true.shape[0]
        X, Z1, A1, Z2, A2 = (self.cache['X'], self.cache['Z1'],
                             self.cache['A1'], self.cache['Z2'], self.cache['A2'])

        # ===== 输出层梯度 =====
        # dZ2 = A2 - one_hot(y_true)
        dZ2 = A2.copy()
        dZ2[range(batch_size), y_true] -= 1
        dZ2 /= batch_size

        # dW2 = A1.T @ dZ2 + L2正则项
        dW2 = A1.T @ dZ2 + self.l2_reg * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # ===== 隐藏层梯度 =====
        # dA1 = dZ2 @ W2.T
        dA1 = dZ2 @ self.W2.T

        # ReLU 的导数: dZ1 = dA1 * (Z1 > 0)
        dZ1 = dA1 * (Z1 > 0)

        # dW1 = X.T @ dZ1 + L2正则项
        dW1 = X.T @ dZ1 + self.l2_reg * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return [dW1, db1, dW2, db2]

    def update_parameters(self, gradients: list, learning_rate: float) -> None:
        """
        使用梯度下降更新参数

        参数:
        -----
        gradients : list
            梯度列表 [dW1, db1, dW2, db2]
        learning_rate : float
            学习率
        """
        dW1, db1, dW2, db2 = gradients
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        preds = self.predict(X)
        return np.mean(preds == y)


# =============================================================================
#                           数据准备
# =============================================================================

def load_and_prepare_data():
    """
    加载并准备 digits 数据集

    返回:
    -----
    Tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("=" * 60)
    print("  数据准备")
    print("=" * 60)

    # 加载数据
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"\n数据集信息:")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]} (8x8 图像)")
    print(f"  类别数: {len(np.unique(y))} (数字 0-9)")

    # 划分数据集: 60% 训练, 20% 验证, 20% 测试
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\n数据划分:")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\n数据已标准化")
    print("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
#                           训练函数
# =============================================================================

def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    创建小批量数据迭代器

    参数:
    -----
    X : np.ndarray
        特征矩阵
    y : np.ndarray
        标签向量
    batch_size : int
        批量大小
    shuffle : bool
        是否打乱数据

    Yields:
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_batch, y_batch)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def train_with_monitor(model: SimpleNeuralNetwork,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       monitor: TrainingMonitor,
                       max_epochs: int = 100,
                       batch_size: int = 32) -> dict:
    """
    使用 TrainingMonitor 进行完整训练

    参数:
    -----
    model : SimpleNeuralNetwork
        神经网络模型
    X_train, y_train : np.ndarray
        训练数据
    X_val, y_val : np.ndarray
        验证数据
    monitor : TrainingMonitor
        训练监控器
    max_epochs : int
        最大 epoch 数
    batch_size : int
        批量大小

    返回:
    -----
    dict
        训练历史
    """
    print("\n" + "=" * 60)
    print("  开始训练")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  最大 Epochs: {max_epochs}")
    print(f"  批量大小: {batch_size}")
    print(f"  学习率调度: Cosine Annealing")
    print(f"  梯度裁剪: max_norm=1.0")
    print(f"  早停: patience=10")
    print(f"  诊断间隔: 每 10 个 epoch")

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(max_epochs):
        # ===== Epoch 开始 =====
        monitor.on_epoch_start(epoch)

        # ===== 训练阶段 =====
        train_losses = []
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            # 前向传播
            y_pred = model.forward(X_batch)

            # 计算损失
            loss = model.compute_loss(y_pred, y_batch)
            train_losses.append(loss)

            # 反向传播
            gradients = model.backward(y_batch)

            # 梯度裁剪（通过 monitor）
            gradients = monitor.on_batch_end(gradients)

            # 参数更新
            learning_rate = monitor.get_current_lr()
            model.update_parameters(gradients, learning_rate)

        # 计算 epoch 级别的指标
        train_loss = np.mean(train_losses)
        train_acc = model.accuracy(X_train, y_train)

        # 验证
        val_pred = model.forward(X_val)
        val_loss = model.compute_loss(val_pred, y_val)
        val_acc = model.accuracy(X_val, y_val)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # ===== Epoch 结束 =====
        # 传入训练和验证 loss，检查是否需要早停
        if monitor.on_epoch_end(train_loss, val_loss, model):
            print(f"\n训练在 Epoch {epoch + 1} 提前停止")
            break

    return history


# =============================================================================
#                           可视化
# =============================================================================

def visualize_results(history: dict, save_path: str = None):
    """
    可视化训练结果

    参数:
    -----
    history : dict
        训练历史
    save_path : str, optional
        保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss 曲线
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='训练 Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='验证 Loss', linewidth=2)

    best_epoch = np.argmin(history['val_loss']) + 1
    best_loss = min(history['val_loss'])
    ax.scatter([best_epoch], [best_loss], c='green', s=100, zorder=5,
               label=f'最佳 (Epoch {best_epoch})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss 曲线', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 准确率曲线
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)

    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    ax.scatter([best_epoch], [best_acc], c='green', s=100, zorder=5,
               label=f'最佳 (Epoch {best_epoch})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('准确率曲线', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图像已保存到: {save_path}")

    plt.show()


# =============================================================================
#                           主程序
# =============================================================================

def main():
    """主程序入口"""
    print("\n" + "=" * 60)
    print("  练习1：完整训练循环")
    print("  使用 training_utils 模块实现所有训练技巧")
    print("=" * 60)

    # 1. 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()

    # 2. 创建模型
    print("\n创建神经网络...")
    model = SimpleNeuralNetwork(
        input_size=64,      # 8x8 = 64 特征
        hidden_size=128,    # 隐藏层神经元
        output_size=10,     # 10 个数字类别
        l2_reg=0.001        # L2 正则化
    )
    print(f"  输入层: {model.input_size}")
    print(f"  隐藏层: {model.hidden_size}")
    print(f"  输出层: {model.output_size}")

    # 3. 创建训练监控器
    print("\n创建训练监控器...")
    monitor = TrainingMonitor(
        # 学习率调度: Cosine Annealing
        lr_scheduler=CosineAnnealingScheduler(
            initial_lr=0.1,     # 起始学习率
            min_lr=1e-4,        # 最小学习率
            T_max=100           # 周期长度
        ),
        # 梯度裁剪
        gradient_clip_norm=1.0,
        # 早停
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        # 定期诊断
        diagnose_every=10,
        verbose=True
    )

    # 4. 训练
    history = train_with_monitor(
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        monitor=monitor,
        max_epochs=100,
        batch_size=32
    )

    # 5. 训练总结
    monitor.summary()

    # 6. 在测试集上评估
    print("\n" + "=" * 60)
    print("  测试集评估")
    print("=" * 60)

    test_acc = model.accuracy(X_test, y_test)
    print(f"\n测试集准确率: {test_acc:.2%}")

    # 7. 可视化
    visualize_results(history, save_path='exercise_06_training_results.png')

    # 8. 使用监控器的可视化
    monitor.plot_history(save_path='exercise_06_monitor_history.png')

    print("\n" + "=" * 60)
    print("  练习1 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
