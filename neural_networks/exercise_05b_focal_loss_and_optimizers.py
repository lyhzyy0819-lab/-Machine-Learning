"""
05b 练习2和练习3：Focal Loss对比 + 优化器对比

练习2：在不平衡数据集上（正负样本比1:10），比较交叉熵和Focal Loss的训练效果
练习3：在MNIST上比较SGD、SGD+Momentum、Adam、AdamW的收敛速度和准确率
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 优化器实现
# =============================================================================

class SGD:
    """
    随机梯度下降优化器

    最基础的优化器，直接沿梯度方向更新参数
    公式: θ = θ - lr * ∇L
    """
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        """执行一步SGD更新"""
        return params - self.lr * grads


class SGDMomentum:
    """
    带动量的SGD优化器

    动量可以加速收敛并减少震荡
    公式:
        v = β * v + (1 - β) * ∇L
        θ = θ - lr * v
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = None  # 速度（动量）

    def update(self, params, grads):
        """执行一步带动量的SGD更新"""
        if self.v is None:
            self.v = np.zeros_like(params)

        # 更新动量
        self.v = self.momentum * self.v + (1 - self.momentum) * grads

        # 更新参数
        return params - self.lr * self.v


class Adam:
    """
    Adam优化器 - Adaptive Moment Estimation

    结合了动量和RMSprop的优点：
    - 一阶矩(m): 梯度的指数移动平均（类似动量）
    - 二阶矩(v): 梯度平方的指数移动平均（自适应学习率）
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1  # 一阶矩衰减率
        self.beta2 = beta2  # 二阶矩衰减率
        self.epsilon = epsilon
        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.t = 0     # 时间步

    def update(self, params, grads):
        """执行一步Adam更新"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # 更新一阶矩和二阶矩
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # 偏差修正（训练初期m和v偏向0，需要修正）
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 更新参数
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class AdamW:
    """
    AdamW优化器 - Adam with decoupled Weight decay

    与Adam的区别：权重衰减直接作用于参数，不通过自适应学习率
    这在实践中通常能获得更好的泛化性能
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        """执行一步AdamW更新"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # 更新一阶矩和二阶矩（与Adam相同）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # 参数更新（关键区别！）
        # 第一部分：Adam更新
        adam_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        # 第二部分：解耦的权重衰减（直接作用于参数，不受自适应学习率影响）
        weight_decay_update = self.lr * self.weight_decay * params

        return params - adam_update - weight_decay_update


# =============================================================================
# 练习2：Focal Loss vs 交叉熵 在不平衡数据集上的对比
# =============================================================================

def sigmoid(x):
    """Sigmoid激活函数，带数值稳定性处理"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def cross_entropy_loss(y_true, y_pred):
    """
    二分类交叉熵损失

    公式: L = -[y*log(p) + (1-y)*log(1-p)]
    """
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal Loss - 解决类别不平衡问题

    公式: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    核心思想：
    - (1 - p_t)^γ 是调制因子
    - 当样本易分类(p_t大)时，调制因子小，损失被抑制
    - 当样本难分类(p_t小)时，调制因子大，损失保持

    参数:
        gamma: 聚焦参数，控制对简单样本的抑制程度
        alpha: 平衡因子，正样本的权重
    """
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 计算 p_t（对正确类别的预测概率）
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)

    # 计算 alpha_t
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

    # Focal Loss
    focal_weight = (1 - p_t) ** gamma  # 调制因子
    loss = -alpha_t * focal_weight * np.log(p_t)

    return np.mean(loss)


class BinaryClassifier:
    """
    简单的二分类神经网络（单隐层）

    结构: 输入 -> 隐层(ReLU) -> 输出(Sigmoid)
    """
    def __init__(self, input_dim, hidden_dim=32):
        # Xavier初始化
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

    def forward(self, X):
        """前向传播"""
        # 隐层
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU

        # 输出层
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2.flatten()

    def backward(self, X, y_true, y_pred, loss_type='ce', gamma=2.0, alpha=0.25):
        """
        反向传播计算梯度

        参数:
            loss_type: 'ce'为交叉熵, 'focal'为Focal Loss
        """
        m = X.shape[0]
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        if loss_type == 'ce':
            # 交叉熵的梯度: dL/dz = p - y
            dz2 = y_pred - y_true
        else:
            # Focal Loss的梯度（推导较复杂，这里给出结果）
            p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
            alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

            # dFL/dp = α_t * (1-p_t)^(γ-1) * [γ*p_t*log(p_t) + p_t - 1]
            focal_weight = (1 - p_t) ** gamma
            log_p_t = np.log(p_t + epsilon)

            # 对于二分类，需要考虑正负样本的情况
            dL_dp = alpha_t * ((1 - p_t) ** (gamma - 1)) * (
                gamma * p_t * log_p_t + p_t - 1
            ) / p_t

            # dp/dz = p * (1 - p)（sigmoid导数）
            dp_dz = y_pred * (1 - y_pred)

            # 正负样本的符号调整
            sign = np.where(y_true == 1, 1, -1)
            dz2 = dL_dp * dp_dz * sign

        # 反向传播到隐层
        dW2 = self.a1.T @ dz2 / m
        db2 = np.mean(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)  # ReLU导数

        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def get_params(self):
        """获取所有参数（展平为一维向量）"""
        return np.concatenate([
            self.W1.flatten(), self.b1.flatten(),
            self.W2.flatten(), self.b2.flatten()
        ])

    def set_params(self, params):
        """设置所有参数"""
        idx = 0

        size = self.W1.size
        self.W1 = params[idx:idx+size].reshape(self.W1.shape)
        idx += size

        size = self.b1.size
        self.b1 = params[idx:idx+size].reshape(self.b1.shape)
        idx += size

        size = self.W2.size
        self.W2 = params[idx:idx+size].reshape(self.W2.shape)
        idx += size

        size = self.b2.size
        self.b2 = params[idx:idx+size].reshape(self.b2.shape)

    def get_grads(self, grads_dict):
        """将梯度字典展平为一维向量"""
        return np.concatenate([
            grads_dict['W1'].flatten(), grads_dict['b1'].flatten(),
            grads_dict['W2'].flatten(), grads_dict['b2'].flatten()
        ])


def exercise_2_focal_loss_comparison():
    """
    练习2：在不平衡数据集上比较交叉熵和Focal Loss

    - 正负样本比例 1:10
    - 比较两种损失函数的训练效果
    - 重点关注少数类的召回率
    """
    print("=" * 70)
    print("练习2：Focal Loss vs 交叉熵 在不平衡数据集上的对比")
    print("=" * 70)

    # 1. 生成不平衡数据集（正负样本比1:10）
    np.random.seed(42)
    n_samples = 1100
    n_positive = 100   # 正样本（少数类）
    n_negative = 1000  # 负样本（多数类）

    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[n_negative/n_samples, n_positive/n_samples],
        flip_y=0.01,
        random_state=42
    )

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n数据集信息:")
    print(f"  训练集: {len(y_train)} 样本 (正:{sum(y_train)}, 负:{len(y_train)-sum(y_train)})")
    print(f"  测试集: {len(y_test)} 样本 (正:{sum(y_test)}, 负:{len(y_test)-sum(y_test)})")
    print(f"  正负样本比: 1:{(len(y_train)-sum(y_train))//sum(y_train)}")

    # 2. 训练两个模型
    input_dim = X_train.shape[1]
    n_epochs = 200
    learning_rate = 0.1

    results = {}

    for loss_type, loss_name in [('ce', '交叉熵'), ('focal', 'Focal Loss')]:
        print(f"\n训练模型: {loss_name}")

        # 初始化模型和优化器（使用相同的随机种子确保公平比较）
        np.random.seed(123)
        model = BinaryClassifier(input_dim, hidden_dim=32)
        optimizer = Adam(learning_rate=learning_rate)

        history = {'loss': [], 'accuracy': [], 'recall': []}

        for epoch in range(n_epochs):
            # 前向传播
            y_pred = model.forward(X_train)

            # 计算损失
            if loss_type == 'ce':
                loss = cross_entropy_loss(y_train, y_pred)
            else:
                loss = focal_loss(y_train, y_pred, gamma=2.0, alpha=0.25)

            # 反向传播
            grads = model.backward(X_train, y_train, y_pred, loss_type)

            # 更新参数
            params = model.get_params()
            grads_flat = model.get_grads(grads)
            new_params = optimizer.update(params, grads_flat)
            model.set_params(new_params)

            # 记录指标
            y_pred_label = (y_pred > 0.5).astype(int)
            acc = accuracy_score(y_train, y_pred_label)
            recall = recall_score(y_train, y_pred_label, zero_division=0)

            history['loss'].append(loss)
            history['accuracy'].append(acc)
            history['recall'].append(recall)

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}, Recall={recall:.4f}")

        # 测试集评估
        y_test_pred = model.forward(X_test)
        y_test_label = (y_test_pred > 0.5).astype(int)

        results[loss_name] = {
            'history': history,
            'test_accuracy': accuracy_score(y_test, y_test_label),
            'test_precision': precision_score(y_test, y_test_label, zero_division=0),
            'test_recall': recall_score(y_test, y_test_label, zero_division=0),
            'test_f1': f1_score(y_test, y_test_label, zero_division=0)
        }

    # 3. 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 损失曲线
    ax = axes[0, 0]
    for loss_name, data in results.items():
        ax.plot(data['history']['loss'], label=loss_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练损失曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 准确率曲线
    ax = axes[0, 1]
    for loss_name, data in results.items():
        ax.plot(data['history']['accuracy'], label=loss_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('训练准确率曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 召回率曲线（少数类）
    ax = axes[1, 0]
    for loss_name, data in results.items():
        ax.plot(data['history']['recall'], label=loss_name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall (少数类)')
    ax.set_title('少数类召回率曲线 (关键指标!)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 测试集指标对比
    ax = axes[1, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(metrics))
    width = 0.35

    ce_values = [results['交叉熵']['test_accuracy'], results['交叉熵']['test_precision'],
                 results['交叉熵']['test_recall'], results['交叉熵']['test_f1']]
    fl_values = [results['Focal Loss']['test_accuracy'], results['Focal Loss']['test_precision'],
                 results['Focal Loss']['test_recall'], results['Focal Loss']['test_f1']]

    bars1 = ax.bar(x - width/2, ce_values, width, label='交叉熵', color='steelblue')
    bars2 = ax.bar(x + width/2, fl_values, width, label='Focal Loss', color='coral')

    ax.set_ylabel('Score')
    ax.set_title('测试集性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # 添加数值标签
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('exercise_05b_focal_loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印总结
    print("\n" + "=" * 70)
    print("测试集结果对比:")
    print("=" * 70)
    print(f"{'指标':<12} {'交叉熵':>12} {'Focal Loss':>12} {'差异':>12}")
    print("-" * 50)
    for metric, key in [('Accuracy', 'test_accuracy'), ('Precision', 'test_precision'),
                        ('Recall', 'test_recall'), ('F1', 'test_f1')]:
        ce = results['交叉熵'][key]
        fl = results['Focal Loss'][key]
        diff = fl - ce
        print(f"{metric:<12} {ce:>12.4f} {fl:>12.4f} {diff:>+12.4f}")

    print("\n结论:")
    print("  - Focal Loss通过降低简单样本的权重，让模型更关注困难样本")
    print("  - 在不平衡数据集上，Focal Loss通常能提高少数类的召回率")
    print("  - gamma参数控制对简单样本的抑制程度，gamma越大抑制越强")


# =============================================================================
# 练习3：MNIST上优化器对比
# =============================================================================

def softmax(x):
    """Softmax函数，带数值稳定性处理"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_multiclass(y_true, y_pred):
    """多分类交叉熵损失"""
    epsilon = 1e-10
    n_samples = y_true.shape[0]
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / n_samples


class MLP:
    """
    多层感知机（用于MNIST分类）

    结构: 784 -> 128 -> 64 -> 10
    """
    def __init__(self):
        # He初始化
        self.W1 = np.random.randn(784, 128) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros(64)
        self.W3 = np.random.randn(64, 10) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros(10)

    def forward(self, X):
        """前向传播"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU

        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = softmax(self.z3)

        return self.a3

    def backward(self, X, y_true, y_pred):
        """反向传播"""
        m = X.shape[0]

        # 输出层梯度（softmax + 交叉熵的导数简化为 p - y）
        dz3 = (y_pred - y_true) / m
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        # 隐层2梯度
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (self.z2 > 0)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # 隐层1梯度
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3
        }

    def get_params(self):
        """获取所有参数"""
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3
        ])

    def set_params(self, params):
        """设置所有参数"""
        idx = 0

        size = 784 * 128
        self.W1 = params[idx:idx+size].reshape(784, 128)
        idx += size
        self.b1 = params[idx:idx+128]
        idx += 128

        size = 128 * 64
        self.W2 = params[idx:idx+size].reshape(128, 64)
        idx += size
        self.b2 = params[idx:idx+64]
        idx += 64

        size = 64 * 10
        self.W3 = params[idx:idx+size].reshape(64, 10)
        idx += size
        self.b3 = params[idx:idx+10]

    def get_grads(self, grads_dict):
        """将梯度字典展平"""
        return np.concatenate([
            grads_dict['W1'].flatten(), grads_dict['b1'],
            grads_dict['W2'].flatten(), grads_dict['b2'],
            grads_dict['W3'].flatten(), grads_dict['b3']
        ])

    def copy_params_from(self, other):
        """从另一个模型复制参数"""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


def exercise_3_optimizer_comparison():
    """
    练习3：在MNIST上比较优化器

    比较 SGD、SGD+Momentum、Adam、AdamW 的收敛速度和最终准确率
    """
    print("\n" + "=" * 70)
    print("练习3：MNIST上优化器对比")
    print("=" * 70)

    # 1. 加载MNIST数据
    print("\n加载MNIST数据...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
    except Exception as e:
        print(f"加载MNIST失败: {e}")
        print("使用随机生成的数据代替...")
        np.random.seed(42)
        X = np.random.randn(10000, 784)
        y = np.random.randint(0, 10, 10000)

    # 使用部分数据加速训练
    n_samples = 10000
    X = X[:n_samples] / 255.0  # 归一化到[0,1]
    y = y[:n_samples]

    # One-hot编码
    y_onehot = np.zeros((len(y), 10))
    y_onehot[np.arange(len(y)), y] = 1

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
        X, y_onehot, y, test_size=0.2, random_state=42
    )

    print(f"训练集: {len(y_train)} 样本")
    print(f"测试集: {len(y_test)} 样本")

    # 2. 定义优化器配置
    optimizers_config = {
        'SGD': lambda: SGD(learning_rate=0.5),
        'SGD+Momentum': lambda: SGDMomentum(learning_rate=0.5, momentum=0.9),
        'Adam': lambda: Adam(learning_rate=0.001),
        'AdamW': lambda: AdamW(learning_rate=0.001, weight_decay=0.01)
    }

    # 3. 训练所有模型
    n_epochs = 30
    batch_size = 128
    n_batches = len(X_train) // batch_size

    # 创建一个初始模型，所有优化器使用相同的初始权重
    np.random.seed(42)
    init_model = MLP()

    results = {}

    for opt_name, opt_factory in optimizers_config.items():
        print(f"\n训练: {opt_name}")

        # 创建模型并复制初始权重
        model = MLP()
        model.copy_params_from(init_model)
        optimizer = opt_factory()

        history = {'loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(n_epochs):
            # 打乱数据
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # 前向传播
                y_pred = model.forward(X_batch)
                loss = cross_entropy_multiclass(y_batch, y_pred)
                epoch_loss += loss

                # 反向传播
                grads = model.backward(X_batch, y_batch, y_pred)

                # 更新参数
                params = model.get_params()
                grads_flat = model.get_grads(grads)
                new_params = optimizer.update(params, grads_flat)
                model.set_params(new_params)

            # 记录指标
            avg_loss = epoch_loss / n_batches

            # 训练准确率
            train_pred = model.forward(X_train)
            train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train_labels)

            # 测试准确率
            test_pred = model.forward(X_test)
            test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test_labels)

            history['loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

        results[opt_name] = history

    # 4. 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 损失曲线
    ax = axes[0]
    for i, (opt_name, history) in enumerate(results.items()):
        ax.plot(history['loss'], label=opt_name, linewidth=2, color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练损失曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 训练准确率
    ax = axes[1]
    for i, (opt_name, history) in enumerate(results.items()):
        ax.plot(history['train_acc'], label=opt_name, linewidth=2, color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('训练准确率曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 最终测试准确率条形图
    ax = axes[2]
    final_accs = [results[name]['test_acc'][-1] for name in optimizers_config.keys()]
    bars = ax.bar(optimizers_config.keys(), final_accs, color=colors)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('最终测试准确率对比')
    ax.set_ylim(0.8, 1.0)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('exercise_05b_optimizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印总结
    print("\n" + "=" * 70)
    print("优化器对比总结:")
    print("=" * 70)
    print(f"{'优化器':<15} {'最终Loss':>12} {'训练Acc':>12} {'测试Acc':>12}")
    print("-" * 55)
    for opt_name, history in results.items():
        print(f"{opt_name:<15} {history['loss'][-1]:>12.4f} "
              f"{history['train_acc'][-1]:>12.4f} {history['test_acc'][-1]:>12.4f}")

    print("\n观察结论:")
    print("  1. SGD收敛最慢，需要更多epoch才能达到较好效果")
    print("  2. SGD+Momentum通过累积梯度方向，加速了收敛")
    print("  3. Adam自适应调整学习率，收敛快且稳定")
    print("  4. AdamW加入权重衰减，可能有更好的泛化能力")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("05b 高级优化器练习")
    print("=" * 70)

    # 运行练习2
    exercise_2_focal_loss_comparison()

    # 运行练习3
    exercise_3_optimizer_comparison()

    print("\n" + "=" * 70)
    print("所有练习完成！")
    print("生成的图片:")
    print("  - exercise_05b_focal_loss_comparison.png")
    print("  - exercise_05b_optimizer_comparison.png")
    print("=" * 70)
