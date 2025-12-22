"""
============================================================================
练习1：Elastic Net正则化 - 从零到一完整实现
============================================================================

📚 问题背景：
    L1和L2正则化各有优缺点：
    - L1：产生稀疏解，但梯度不连续
    - L2：梯度平滑，但不产生稀疏性

    Elastic Net结合两者优点！

🎯 学习目标：
    1. 理解Elastic Net的数学原理
    2. 从零实现Elastic Net正则化
    3. 对比L1、L2、Elastic Net的效果
    4. 分析α参数对结果的影响

============================================================================
"""

# ============================================================================
# 第1部分：导入库和环境配置
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("练习1：Elastic Net正则化 - 从零到一完整实现")
print("=" * 70)


# ============================================================================
# 第2部分：数学原理详解
# ============================================================================
"""
📖 Elastic Net正则化公式：

    L_total = L_CE + α * λ * ||W||₁ + (1-α)/2 * λ * ||W||₂²

其中：
    - L_CE: 交叉熵损失（Cross Entropy Loss）
    - λ (lambda): 正则化强度，控制惩罚力度
    - α (alpha): L1/L2比例系数，α ∈ [0, 1]
        - α = 1: 纯L1正则化（Lasso）
        - α = 0: 纯L2正则化（Ridge）
        - α = 0.5: L1和L2各占一半

📐 梯度计算：

    ∂L_total/∂W = ∂L_CE/∂W + α * λ * sign(W) + (1-α) * λ * W
                  ─────────   ───────────────   ─────────────
                  原始梯度      L1项梯度           L2项梯度

💡 为什么Elastic Net更好？
    1. 继承L1的稀疏性：可以进行特征选择
    2. 继承L2的稳定性：在高度相关特征时更稳定
    3. 可调节α平衡两者
"""


# ============================================================================
# 第3部分：激活函数定义
# ============================================================================

def relu(z):
    """
    ReLU激活函数 (Rectified Linear Unit)

    数学公式：
        ReLU(z) = max(0, z)

    特点：
        - z > 0 时，输出 z
        - z <= 0 时，输出 0

    参数:
        z: 输入值，可以是标量或数组

    返回:
        与输入形状相同的数组
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    ReLU的导数

    数学公式：
        ReLU'(z) = { 1, if z > 0
                   { 0, if z <= 0

    注意：在z=0处，导数理论上未定义，但实践中通常取0

    参数:
        z: 输入值

    返回:
        导数值（0或1）
    """
    return (z > 0).astype(float)


def sigmoid(z):
    """
    Sigmoid激活函数

    数学公式：
        σ(z) = 1 / (1 + e^(-z))

    特点：
        - 输出范围 (0, 1)，适合二分类
        - 将任意实数映射到概率值

    参数:
        z: 输入值

    返回:
        概率值，范围 (0, 1)

    实现细节：
        np.clip(z, -500, 500) 防止数值溢出
        - 当z很大时，exp(-z) ≈ 0，sigmoid ≈ 1
        - 当z很小时，exp(-z) 可能溢出，所以限制范围
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# ============================================================================
# 第4部分：Elastic Net神经网络实现
# ============================================================================

class ElasticNetNetwork:
    """
    带Elastic Net正则化的神经网络

    网络结构：
        输入层 (2) → 隐藏层1 (64) → 隐藏层2 (64) → 隐藏层3 (32) → 输出层 (1)

    Elastic Net公式：
        L_total = L_CE + α * λ * ||W||₁ + (1-α)/2 * λ * ||W||₂²

    参数:
        lambda_reg: float, 正则化强度 λ
        alpha: float, L1/L2比例 α ∈ [0, 1]
            - alpha=1: 纯L1
            - alpha=0: 纯L2
            - alpha=0.5: 混合
    """

    def __init__(self, lambda_reg=0.01, alpha=0.5):
        """
        初始化网络参数

        参数:
            lambda_reg: 正则化强度，典型值 0.001 ~ 0.1
            alpha: L1比例，0表示纯L2，1表示纯L1
        """
        # =====================================
        # 存储正则化超参数
        # =====================================
        self.lambda_reg = lambda_reg  # λ: 正则化强度
        self.alpha = alpha            # α: L1/L2比例

        # =====================================
        # 初始化权重和偏置
        # =====================================
        # 使用较大的初始化值，容易观察过拟合现象
        # 实际应用中应使用He初始化或Xavier初始化

        # 第1层: 2 → 64
        # W1形状: (64, 2), 每行是一个神经元的权重
        self.W1 = np.random.randn(64, 2) * 0.5
        self.b1 = np.zeros(64)  # 偏置初始化为0

        # 第2层: 64 → 64
        self.W2 = np.random.randn(64, 64) * 0.5
        self.b2 = np.zeros(64)

        # 第3层: 64 → 32
        self.W3 = np.random.randn(32, 64) * 0.5
        self.b3 = np.zeros(32)

        # 输出层: 32 → 1
        self.W4 = np.random.randn(1, 32) * 0.5
        self.b4 = np.zeros(1)

        print(f"网络初始化完成:")
        print(f"  正则化强度 λ = {lambda_reg}")
        print(f"  L1/L2比例 α = {alpha}")
        print(f"  网络结构: [2, 64, 64, 32, 1]")
        print(f"  总参数量: {self._count_params()}")

    def _count_params(self):
        """计算网络总参数量"""
        total = 0
        for W, b in [(self.W1, self.b1), (self.W2, self.b2),
                     (self.W3, self.b3), (self.W4, self.b4)]:
            total += W.size + b.size
        return total

    def forward(self, X):
        """
        前向传播

        数据流动过程：
            X → [W1, b1] → ReLU → [W2, b2] → ReLU → [W3, b3] → ReLU → [W4, b4] → Sigmoid → 输出

        数学公式：
            z^(l) = a^(l-1) @ W^(l).T + b^(l)  # 线性变换
            a^(l) = activation(z^(l))          # 非线性激活

        参数:
            X: 输入数据, shape (n_samples, 2)

        返回:
            output: 预测概率, shape (n_samples, 1)
        """
        # =====================================
        # 第1层：输入层 → 隐藏层1
        # =====================================
        # 线性变换: z1 = X @ W1.T + b1
        # X: (n_samples, 2), W1.T: (2, 64) → z1: (n_samples, 64)
        self.z1 = X @ self.W1.T + self.b1
        # 激活函数
        self.a1 = relu(self.z1)

        # =====================================
        # 第2层：隐藏层1 → 隐藏层2
        # =====================================
        self.z2 = self.a1 @ self.W2.T + self.b2
        self.a2 = relu(self.z2)

        # =====================================
        # 第3层：隐藏层2 → 隐藏层3
        # =====================================
        self.z3 = self.a2 @ self.W3.T + self.b3
        self.a3 = relu(self.z3)

        # =====================================
        # 输出层：隐藏层3 → 输出
        # =====================================
        # 使用Sigmoid激活，输出概率值
        self.z4 = self.a3 @ self.W4.T + self.b4
        self.a4 = sigmoid(self.z4)

        return self.a4

    def backward(self, X, y_true):
        """
        反向传播 - 包含Elastic Net正则化

        核心公式：
            ∂L/∂W = ∂L_CE/∂W + α*λ*sign(W) + (1-α)*λ*W
                    ─────────   ───────────   ─────────
                    原始梯度      L1梯度         L2梯度

        反向传播流程：
            1. 计算输出层误差 δ4 = a4 - y_true
            2. 逐层反向传播误差
            3. 计算每层的梯度
            4. 添加正则化项

        参数:
            X: 输入数据, shape (n_samples, 2)
            y_true: 真实标签, shape (n_samples,)

        返回:
            grads: 所有参数的梯度列表
        """
        m = X.shape[0]  # 样本数量

        # =====================================
        # 输出层梯度
        # =====================================
        # 对于二分类交叉熵 + Sigmoid，梯度简化为：
        # δ4 = a4 - y_true
        delta4 = (self.a4 - y_true.reshape(-1, 1)) / m

        # 权重梯度 = 原始梯度 + L1项 + L2项
        # grad_W4 = δ4.T @ a3 + α*λ*sign(W4) + (1-α)*λ*W4
        grad_W4 = (delta4.T @ self.a3 +
                   self.alpha * self.lambda_reg * np.sign(self.W4) +  # L1项
                   (1 - self.alpha) * self.lambda_reg * self.W4)      # L2项

        # 偏置梯度（偏置不正则化！）
        grad_b4 = np.sum(delta4, axis=0)

        # =====================================
        # 第3层梯度
        # =====================================
        # 误差反向传播：δ3 = (δ4 @ W4) * ReLU'(z3)
        delta3 = (delta4 @ self.W4) * relu_derivative(self.z3)

        grad_W3 = (delta3.T @ self.a2 +
                   self.alpha * self.lambda_reg * np.sign(self.W3) +
                   (1 - self.alpha) * self.lambda_reg * self.W3)
        grad_b3 = np.sum(delta3, axis=0)

        # =====================================
        # 第2层梯度
        # =====================================
        delta2 = (delta3 @ self.W3) * relu_derivative(self.z2)

        grad_W2 = (delta2.T @ self.a1 +
                   self.alpha * self.lambda_reg * np.sign(self.W2) +
                   (1 - self.alpha) * self.lambda_reg * self.W2)
        grad_b2 = np.sum(delta2, axis=0)

        # =====================================
        # 第1层梯度
        # =====================================
        delta1 = (delta2 @ self.W2) * relu_derivative(self.z1)

        grad_W1 = (delta1.T @ X +
                   self.alpha * self.lambda_reg * np.sign(self.W1) +
                   (1 - self.alpha) * self.lambda_reg * self.W1)
        grad_b1 = np.sum(delta1, axis=0)

        # 返回所有梯度
        return [grad_W1, grad_b1, grad_W2, grad_b2,
                grad_W3, grad_b3, grad_W4, grad_b4]

    def get_params(self):
        """获取所有参数"""
        return [self.W1, self.b1, self.W2, self.b2,
                self.W3, self.b3, self.W4, self.b4]

    def set_params(self, params):
        """设置所有参数"""
        self.W1, self.b1, self.W2, self.b2, \
        self.W3, self.b3, self.W4, self.b4 = params

    def compute_loss(self, X, y_true):
        """
        计算总损失（包含Elastic Net正则化项）

        公式：
            L_total = L_CE + α*λ*||W||₁ + (1-α)/2*λ*||W||₂²

        参数:
            X: 输入数据
            y_true: 真实标签

        返回:
            total_loss: 总损失值
        """
        # =====================================
        # 1. 计算交叉熵损失
        # =====================================
        y_pred = self.forward(X)
        epsilon = 1e-15  # 防止log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # 二分类交叉熵: L = -mean(y*log(p) + (1-y)*log(1-p))
        ce_loss = -np.mean(
            y_true * np.log(y_pred.flatten()) +
            (1 - y_true) * np.log(1 - y_pred.flatten())
        )

        # =====================================
        # 2. 计算L1正则化项: α * λ * Σ|W|
        # =====================================
        l1_penalty = self.alpha * self.lambda_reg * (
            np.sum(np.abs(self.W1)) +
            np.sum(np.abs(self.W2)) +
            np.sum(np.abs(self.W3)) +
            np.sum(np.abs(self.W4))
        )

        # =====================================
        # 3. 计算L2正则化项: (1-α)/2 * λ * Σ(W²)
        # =====================================
        l2_penalty = (1 - self.alpha) * self.lambda_reg / 2 * (
            np.sum(self.W1 ** 2) +
            np.sum(self.W2 ** 2) +
            np.sum(self.W3 ** 2) +
            np.sum(self.W4 ** 2)
        )

        return ce_loss + l1_penalty + l2_penalty

    def compute_accuracy(self, X, y_true):
        """
        计算分类准确率

        参数:
            X: 输入数据
            y_true: 真实标签

        返回:
            accuracy: 准确率 (0~1)
        """
        y_pred = self.forward(X)
        # 概率 >= 0.5 预测为类别1，否则为类别0
        predictions = (y_pred >= 0.5).astype(int).flatten()
        return np.mean(predictions == y_true)

    def count_zero_weights(self, threshold=1e-3):
        """
        统计接近0的权重数量（稀疏性指标）

        参数:
            threshold: 判断为0的阈值

        返回:
            (zero_count, total_count): 接近0的权重数和总权重数
        """
        total = self.W1.size + self.W2.size + self.W3.size + self.W4.size
        zero_count = (
            np.sum(np.abs(self.W1) < threshold) +
            np.sum(np.abs(self.W2) < threshold) +
            np.sum(np.abs(self.W3) < threshold) +
            np.sum(np.abs(self.W4) < threshold)
        )
        return zero_count, total


# ============================================================================
# 第5部分：训练函数
# ============================================================================

def train_elastic_net(X_train, y_train, X_test, y_test,
                      lambda_reg=0.01, alpha=0.5,
                      n_epochs=500, learning_rate=0.01,
                      verbose=True):
    """
    训练带Elastic Net正则化的神经网络

    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        lambda_reg: 正则化强度
        alpha: L1比例 (0~1)
        n_epochs: 训练轮数
        learning_rate: 学习率
        verbose: 是否打印训练信息

    返回:
        model: 训练好的模型
        history: 训练历史（损失和准确率）
    """
    # 创建模型
    model = ElasticNetNetwork(lambda_reg=lambda_reg, alpha=alpha)

    # 记录训练历史
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'sparsity': []
    }

    # =====================================
    # 训练循环
    # =====================================
    for epoch in range(n_epochs):
        # ----- 前向传播 -----
        model.forward(X_train)

        # ----- 反向传播 -----
        grads = model.backward(X_train, y_train)

        # ----- 参数更新（梯度下降） -----
        # W_new = W_old - learning_rate * gradient
        params = model.get_params()
        updated_params = [p - learning_rate * g for p, g in zip(params, grads)]
        model.set_params(updated_params)

        # ----- 记录指标 -----
        if epoch % 10 == 0:
            train_loss = model.compute_loss(X_train, y_train)
            test_loss = model.compute_loss(X_test, y_test)
            train_acc = model.compute_accuracy(X_train, y_train)
            test_acc = model.compute_accuracy(X_test, y_test)
            zero_w, total_w = model.count_zero_weights()
            sparsity = zero_w / total_w * 100

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['sparsity'].append(sparsity)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d}: "
                      f"Train Acc={train_acc:.4f}, "
                      f"Test Acc={test_acc:.4f}, "
                      f"Sparsity={sparsity:.1f}%")

    return model, history


# ============================================================================
# 第6部分：实验对比
# ============================================================================

if __name__ == "__main__":

    # =====================================
    # 1. 准备数据
    # =====================================
    print("\n" + "=" * 70)
    print("第1步：准备数据")
    print("=" * 70)

    # 生成月牙形数据集（经典的非线性分类问题）
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 数据标准化（重要！使训练更稳定）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # =====================================
    # 2. 对比不同的α值
    # =====================================
    print("\n" + "=" * 70)
    print("第2步：对比不同的α值")
    print("=" * 70)

    # 测试的α值：
    # α = 0.0: 纯L2正则化
    # α = 0.3: 70% L2 + 30% L1
    # α = 0.5: 50% L2 + 50% L1
    # α = 0.7: 30% L2 + 70% L1
    # α = 1.0: 纯L1正则化
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = {}

    for alpha in alphas:
        print(f"\n{'─' * 50}")
        print(f"训练模型: α = {alpha}")
        print(f"{'─' * 50}")

        model, history = train_elastic_net(
            X_train, y_train, X_test, y_test,
            lambda_reg=0.01,
            alpha=alpha,
            n_epochs=500,
            learning_rate=0.01,
            verbose=True
        )

        results[alpha] = {
            'model': model,
            'history': history,
            'final_test_acc': history['test_acc'][-1],
            'final_sparsity': history['sparsity'][-1]
        }

        print(f"最终测试准确率: {history['test_acc'][-1]:.4f}")
        print(f"最终稀疏性: {history['sparsity'][-1]:.1f}%")

    # =====================================
    # 3. 可视化结果
    # =====================================
    print("\n" + "=" * 70)
    print("第3步：可视化结果")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_plot = np.arange(0, 50) * 10

    # ----- 图1：测试准确率对比 -----
    ax1 = axes[0, 0]
    for alpha in alphas:
        ax1.plot(epochs_plot, results[alpha]['history']['test_acc'],
                 label=f'α={alpha}', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Test Accuracy', fontsize=11)
    ax1.set_title('测试准确率对比（不同α值）', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ----- 图2：稀疏性对比 -----
    ax2 = axes[0, 1]
    for alpha in alphas:
        ax2.plot(epochs_plot, results[alpha]['history']['sparsity'],
                 label=f'α={alpha}', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Sparsity (%)', fontsize=11)
    ax2.set_title('权重稀疏性对比（不同α值）', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ----- 图3：最终结果柱状图 -----
    ax3 = axes[1, 0]
    x_pos = np.arange(len(alphas))
    width = 0.35

    final_accs = [results[a]['final_test_acc'] for a in alphas]
    final_sparsities = [results[a]['final_sparsity'] / 100 for a in alphas]

    bars1 = ax3.bar(x_pos - width/2, final_accs, width,
                    label='测试准确率', alpha=0.8, color='#3498db')
    bars2 = ax3.bar(x_pos + width/2, final_sparsities, width,
                    label='稀疏性比例', alpha=0.8, color='#e74c3c')

    ax3.set_xlabel('α (L1比例)', fontsize=11)
    ax3.set_ylabel('Value', fontsize=11)
    ax3.set_title('最终结果对比', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{a}' for a in alphas])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # ----- 图4：结论总结 -----
    ax4 = axes[1, 1]
    summary_text = """
╔══════════════════════════════════════════════════════════╗
║           Elastic Net正则化实验总结                       ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  📐 公式:                                                ║
║     L = L_CE + α·λ·||W||₁ + (1-α)/2·λ·||W||₂²           ║
║                                                          ║
║  🔬 实验结论:                                            ║
║                                                          ║
║  α = 0 (纯L2):                                          ║
║    • 稀疏性最低                                          ║
║    • 权重均匀缩小                                        ║
║                                                          ║
║  α = 1 (纯L1):                                          ║
║    • 稀疏性最高                                          ║
║    • 很多权重变为0                                       ║
║                                                          ║
║  α ∈ (0,1) (Elastic Net):                               ║
║    • 结合两者优点                                        ║
║    • 适度稀疏 + 稳定训练                                 ║
║                                                          ║
║  💡 推荐: α = 0.5 是很好的起点                          ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    ax4.text(0.02, 0.5, summary_text, fontsize=9, verticalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/elastic_net_results.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # =====================================
    # 4. 打印最终总结
    # =====================================
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    print(f"\n{'α值':<8} {'测试准确率':<12} {'稀疏性':<10} {'说明':<20}")
    print("-" * 60)

    descriptions = {
        0.0: "纯L2正则化",
        0.3: "偏L2的Elastic Net",
        0.5: "均衡Elastic Net",
        0.7: "偏L1的Elastic Net",
        1.0: "纯L1正则化"
    }

    for alpha in alphas:
        acc = results[alpha]['final_test_acc']
        sparsity = results[alpha]['final_sparsity']
        desc = descriptions[alpha]
        print(f"{alpha:<8} {acc:<12.4f} {sparsity:<10.1f}% {desc:<20}")

    print("\n" + "=" * 70)
    print("✅ 练习1完成！")
    print("=" * 70)
    print("""
📚 学习要点:
    1. Elastic Net = L1 + L2正则化的结合
    2. α参数控制L1和L2的比例
    3. α越大，稀疏性越强
    4. 实际应用中，α=0.5是很好的起点
    5. 可以通过交叉验证选择最佳的λ和α
""")
