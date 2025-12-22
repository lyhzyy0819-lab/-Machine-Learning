"""
============================================================================
练习3：最佳λ搜索 - 使用验证集选择最优正则化强度
============================================================================

📚 问题背景：
    L2正则化的强度参数λ是一个超参数，需要通过实验来确定最佳值。
    λ过小：正则化效果弱，可能过拟合
    λ过大：正则化效果过强，可能欠拟合

🎯 学习目标：
    1. 理解超参数调优的重要性
    2. 掌握使用验证集选择超参数的方法
    3. 绘制验证性能 vs λ的曲线
    4. 找到最佳的λ值
    5. 理解偏差-方差权衡

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
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("练习3：最佳λ搜索 - L2正则化超参数调优")
print("=" * 70)


# ============================================================================
# 第2部分：超参数调优原理
# ============================================================================
"""
📖 为什么需要验证集？

    数据划分：
        原始数据 → 训练集 + 验证集 + 测试集

    各集合的作用：
        训练集：用于训练模型（更新参数）
        验证集：用于选择超参数（如λ）
        测试集：最终评估模型性能（只用一次！）

    ⚠️ 重要原则：
        绝对不能用测试集来选择超参数！
        否则测试集的信息会"泄露"到模型中，导致过于乐观的性能估计。

📐 λ搜索的数学理解：

    L2正则化损失：
        L(θ) = L_CE(θ) + λ/2 * ||θ||²

    λ的影响：
        λ → 0: 无正则化，模型复杂度高，可能过拟合
        λ → ∞: 强正则化，θ → 0，模型过于简单，欠拟合

    最佳λ：在偏差和方差之间取得平衡

💡 搜索策略：

    1. 网格搜索（Grid Search）：在预定义的候选值中搜索
       常用候选值：[0.0001, 0.001, 0.01, 0.1, 1.0]

    2. 对数尺度搜索：因为λ的最佳值通常在多个数量级中变化
       使用 10^(-4), 10^(-3), 10^(-2), 10^(-1), 10^0

    3. 随机搜索：在某个范围内随机采样
"""


# ============================================================================
# 第3部分：激活函数和网络组件
# ============================================================================

def relu(z):
    """ReLU激活函数: max(0, z)"""
    return np.maximum(0, z)


def relu_derivative(z):
    """ReLU导数: 1 if z > 0 else 0"""
    return (z > 0).astype(float)


def sigmoid(z):
    """Sigmoid激活函数: 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# ============================================================================
# 第4部分：神经网络实现
# ============================================================================

class L2RegularizedNetwork:
    """
    带L2正则化的神经网络

    用于λ搜索实验的网络实现

    网络结构：2 → 32 → 16 → 1
    （比之前稍小，适合小数据集）
    """

    def __init__(self, lambda_reg=0.01):
        """
        初始化网络

        参数:
            lambda_reg: L2正则化强度λ
        """
        self.lambda_reg = lambda_reg

        # Xavier初始化
        np.random.seed(42)  # 保证每次初始化相同，便于对比

        self.W1 = np.random.randn(32, 2) * np.sqrt(1.0 / 2)
        self.b1 = np.zeros(32)

        self.W2 = np.random.randn(16, 32) * np.sqrt(1.0 / 32)
        self.b2 = np.zeros(16)

        self.W3 = np.random.randn(1, 16) * np.sqrt(1.0 / 16)
        self.b3 = np.zeros(1)

    def forward(self, X):
        """前向传播"""
        self.z1 = X @ self.W1.T + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.W2.T + self.b2
        self.a2 = relu(self.z2)

        self.z3 = self.a2 @ self.W3.T + self.b3
        self.a3 = sigmoid(self.z3)

        return self.a3

    def backward(self, X, y_true):
        """反向传播（带L2正则化）"""
        m = X.shape[0]

        # 输出层
        delta3 = (self.a3 - y_true.reshape(-1, 1)) / m
        grad_W3 = delta3.T @ self.a2 + self.lambda_reg * self.W3
        grad_b3 = np.sum(delta3, axis=0)

        # 第2层
        delta2 = (delta3 @ self.W3) * relu_derivative(self.z2)
        grad_W2 = delta2.T @ self.a1 + self.lambda_reg * self.W2
        grad_b2 = np.sum(delta2, axis=0)

        # 第1层
        delta1 = (delta2 @ self.W2) * relu_derivative(self.z1)
        grad_W1 = delta1.T @ X + self.lambda_reg * self.W1
        grad_b1 = np.sum(delta1, axis=0)

        return [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = params

    def compute_loss(self, X, y_true):
        """计算总损失（包含L2正则化项）"""
        y_pred = self.forward(X)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # 交叉熵损失
        ce_loss = -np.mean(
            y_true * np.log(y_pred.flatten()) +
            (1 - y_true) * np.log(1 - y_pred.flatten())
        )

        # L2正则化项
        l2_penalty = self.lambda_reg / 2 * (
            np.sum(self.W1 ** 2) +
            np.sum(self.W2 ** 2) +
            np.sum(self.W3 ** 2)
        )

        return ce_loss + l2_penalty

    def compute_accuracy(self, X, y_true):
        """计算准确率"""
        y_pred = self.forward(X)
        predictions = (y_pred >= 0.5).astype(int).flatten()
        return np.mean(predictions == y_true)

    def get_weight_norm(self):
        """计算权重的L2范数（用于分析）"""
        return np.sqrt(
            np.sum(self.W1 ** 2) +
            np.sum(self.W2 ** 2) +
            np.sum(self.W3 ** 2)
        )


# ============================================================================
# 第5部分：训练和评估函数
# ============================================================================

def train_and_evaluate(X_train, y_train, X_val, y_val, lambda_reg,
                       n_epochs=300, learning_rate=0.05, verbose=False):
    """
    训练模型并返回验证性能

    参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        lambda_reg: L2正则化强度
        n_epochs: 训练轮数
        learning_rate: 学习率
        verbose: 是否打印详细信息

    返回:
        final_train_acc: 最终训练准确率
        final_val_acc: 最终验证准确率
        final_weight_norm: 最终权重范数
        history: 训练历史
    """
    model = L2RegularizedNetwork(lambda_reg=lambda_reg)

    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(n_epochs):
        # 前向传播
        model.forward(X_train)

        # 反向传播
        grads = model.backward(X_train, y_train)

        # 参数更新
        params = model.get_params()
        updated_params = [p - learning_rate * g for p, g in zip(params, grads)]
        model.set_params(updated_params)

        # 记录指标
        if epoch % 10 == 0:
            train_acc = model.compute_accuracy(X_train, y_train)
            val_acc = model.compute_accuracy(X_val, y_val)
            train_loss = model.compute_loss(X_train, y_train)
            val_loss = model.compute_loss(X_val, y_val)

            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch}: Train={train_acc:.4f}, Val={val_acc:.4f}")

    final_train_acc = model.compute_accuracy(X_train, y_train)
    final_val_acc = model.compute_accuracy(X_val, y_val)
    final_weight_norm = model.get_weight_norm()

    return final_train_acc, final_val_acc, final_weight_norm, history, model


# ============================================================================
# 第6部分：λ搜索实验
# ============================================================================

if __name__ == "__main__":

    # =====================================
    # 1. 准备数据
    # =====================================
    print("\n" + "=" * 70)
    print("第1步：准备数据")
    print("=" * 70)

    # 生成月牙形数据
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

    # 划分：训练集 60%，验证集 20%，测试集 20%
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
    )

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # =====================================
    # 2. 定义候选λ值
    # =====================================
    print("\n" + "=" * 70)
    print("第2步：定义候选λ值")
    print("=" * 70)

    # 使用对数尺度的候选值
    # 从10^-4到10^1，共6个数量级
    lambda_candidates = [0.0001, 0.001, 0.01, 0.1, 1.0]

    print("候选λ值:")
    for lam in lambda_candidates:
        print(f"  λ = {lam}")

    # =====================================
    # 3. 网格搜索
    # =====================================
    print("\n" + "=" * 70)
    print("第3步：网格搜索最佳λ")
    print("=" * 70)

    results = {}

    for lam in lambda_candidates:
        print(f"\n{'─' * 50}")
        print(f"测试 λ = {lam}")
        print(f"{'─' * 50}")

        train_acc, val_acc, weight_norm, history, model = train_and_evaluate(
            X_train, y_train, X_val, y_val,
            lambda_reg=lam,
            n_epochs=300,
            learning_rate=0.05,
            verbose=True
        )

        results[lam] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'weight_norm': weight_norm,
            'history': history,
            'model': model
        }

        print(f"最终结果: Train={train_acc:.4f}, Val={val_acc:.4f}, ||W||={weight_norm:.4f}")

    # =====================================
    # 4. 找出最佳λ
    # =====================================
    print("\n" + "=" * 70)
    print("第4步：确定最佳λ")
    print("=" * 70)

    best_lambda = max(results, key=lambda x: results[x]['val_acc'])
    best_val_acc = results[best_lambda]['val_acc']

    print(f"\n🏆 最佳λ值: {best_lambda}")
    print(f"   验证集准确率: {best_val_acc:.4f}")

    # =====================================
    # 5. 在测试集上评估最佳模型
    # =====================================
    print("\n" + "=" * 70)
    print("第5步：在测试集上评估")
    print("=" * 70)

    best_model = results[best_lambda]['model']
    test_acc = best_model.compute_accuracy(X_test, y_test)

    print(f"\n使用最佳λ = {best_lambda}:")
    print(f"  训练准确率: {results[best_lambda]['train_acc']:.4f}")
    print(f"  验证准确率: {results[best_lambda]['val_acc']:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")

    # =====================================
    # 6. 可视化结果
    # =====================================
    print("\n" + "=" * 70)
    print("第6步：可视化结果")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ----- 图1：验证准确率 vs λ -----
    ax1 = axes[0, 0]
    train_accs = [results[lam]['train_acc'] for lam in lambda_candidates]
    val_accs = [results[lam]['val_acc'] for lam in lambda_candidates]

    ax1.semilogx(lambda_candidates, train_accs, 'o-', linewidth=2, markersize=8,
                 label='训练准确率', color='#3498db')
    ax1.semilogx(lambda_candidates, val_accs, 's-', linewidth=2, markersize=8,
                 label='验证准确率', color='#e74c3c')
    ax1.axvline(x=best_lambda, color='green', linestyle='--', linewidth=2,
                label=f'最佳λ={best_lambda}')

    ax1.set_xlabel('λ (对数尺度)', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('准确率 vs 正则化强度λ', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ----- 图2：权重范数 vs λ -----
    ax2 = axes[0, 1]
    weight_norms = [results[lam]['weight_norm'] for lam in lambda_candidates]

    ax2.semilogx(lambda_candidates, weight_norms, 'D-', linewidth=2, markersize=8,
                 color='#9b59b6')
    ax2.axvline(x=best_lambda, color='green', linestyle='--', linewidth=2)

    ax2.set_xlabel('λ (对数尺度)', fontsize=11)
    ax2.set_ylabel('||W|| (权重L2范数)', fontsize=11)
    ax2.set_title('权重范数 vs 正则化强度λ', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 添加注释
    ax2.annotate('λ增大\n权重缩小',
                 xy=(0.5, weight_norms[3]),
                 xytext=(0.2, weight_norms[3] + 1),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='gray'))

    # ----- 图3：不同λ的学习曲线 -----
    ax3 = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_candidates)))

    for i, lam in enumerate(lambda_candidates):
        epochs = np.arange(len(results[lam]['history']['val_acc'])) * 10
        ax3.plot(epochs, results[lam]['history']['val_acc'],
                 linewidth=2, color=colors[i], label=f'λ={lam}')

    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('验证准确率', fontsize=11)
    ax3.set_title('不同λ的验证准确率曲线', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ----- 图4：结果汇总表 -----
    ax4 = axes[1, 1]

    summary_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    λ搜索实验结果汇总                               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📊 搜索结果:                                                    ║
║                                                                  ║
║  {'λ值':<12} {'训练准确率':<12} {'验证准确率':<12} {'权重范数':<12}          ║
║  {'─'*52}   ║
"""
    for lam in lambda_candidates:
        star = ' ⭐' if lam == best_lambda else ''
        summary_text += f"""║  {lam:<12} {results[lam]['train_acc']:<12.4f} {results[lam]['val_acc']:<12.4f} {results[lam]['weight_norm']:<12.4f}{star}║
"""

    summary_text += f"""║                                                                  ║
║  🏆 最佳λ值: {best_lambda}                                                ║
║                                                                  ║
║  📈 最终测试集准确率: {test_acc:.4f}                                      ║
║                                                                  ║
║  💡 观察结论:                                                    ║
║    • λ过小: 过拟合 (训练高，验证低)                               ║
║    • λ过大: 欠拟合 (训练和验证都低)                               ║
║    • 最佳λ: 偏差-方差平衡点                                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

    ax4.text(0.02, 0.5, summary_text, fontsize=8.5, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/lambda_search_results.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # =====================================
    # 7. 偏差-方差分析图
    # =====================================
    print("\n" + "=" * 70)
    print("第7步：偏差-方差权衡分析")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 计算偏差和方差的近似指标
    # 偏差 ≈ 1 - 训练准确率（训练误差）
    # 方差 ≈ 训练准确率 - 验证准确率（泛化差距）
    biases = [1 - results[lam]['train_acc'] for lam in lambda_candidates]
    variances = [results[lam]['train_acc'] - results[lam]['val_acc']
                 for lam in lambda_candidates]
    total_errors = [1 - results[lam]['val_acc'] for lam in lambda_candidates]

    ax.semilogx(lambda_candidates, biases, 'o-', linewidth=2, markersize=8,
                label='偏差 (训练误差)', color='#3498db')
    ax.semilogx(lambda_candidates, variances, 's-', linewidth=2, markersize=8,
                label='方差 (泛化差距)', color='#e74c3c')
    ax.semilogx(lambda_candidates, total_errors, '^-', linewidth=2, markersize=8,
                label='总误差 (验证误差)', color='#2ecc71')
    ax.axvline(x=best_lambda, color='purple', linestyle='--', linewidth=2,
               label=f'最佳λ={best_lambda}')

    ax.set_xlabel('λ (正则化强度)', fontsize=12)
    ax.set_ylabel('误差', fontsize=12)
    ax.set_title('偏差-方差权衡 (Bias-Variance Tradeoff)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加区域标注
    ax.annotate('欠拟合区域\n(高偏差)',
                xy=(lambda_candidates[-1], biases[-1]),
                xytext=(lambda_candidates[-1] * 2, biases[-1] + 0.05),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.annotate('过拟合区域\n(高方差)',
                xy=(lambda_candidates[0], variances[0]),
                xytext=(lambda_candidates[0] * 0.1, variances[0] + 0.02),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/bias_variance_tradeoff.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # =====================================
    # 8. 打印最终总结
    # =====================================
    print("\n" + "=" * 70)
    print("实验完成！最终总结")
    print("=" * 70)

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                        λ搜索实验完成！                                  ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  📋 实验设置:                                                         ║
║    • 数据集: make_moons (300样本, noise=0.25)                         ║
║    • 划分: 训练60%, 验证20%, 测试20%                                  ║
║    • 候选λ: [0.0001, 0.001, 0.01, 0.1, 1.0]                          ║
║                                                                       ║
║  🏆 最佳结果:                                                         ║
║    • 最佳λ: {best_lambda}                                                    ║
║    • 验证准确率: {best_val_acc:.4f}                                           ║
║    • 测试准确率: {test_acc:.4f}                                           ║
║                                                                       ║
║  📚 学习要点:                                                         ║
║                                                                       ║
║    1. 验证集选择超参数                                                ║
║       • 训练集: 训练模型                                              ║
║       • 验证集: 选择超参数                                            ║
║       • 测试集: 最终评估（只用一次！）                                ║
║                                                                       ║
║    2. 对数尺度搜索                                                    ║
║       • λ的最佳值可能跨越多个数量级                                   ║
║       • 使用 10^(-4), 10^(-3), ... 更高效                            ║
║                                                                       ║
║    3. 偏差-方差权衡                                                   ║
║       • λ小: 低偏差，高方差（过拟合）                                 ║
║       • λ大: 高偏差，低方差（欠拟合）                                 ║
║       • 最佳λ: 在两者之间取得平衡                                     ║
║                                                                       ║
║    4. 权重范数随λ变化                                                 ║
║       • λ增大 → 权重范数减小                                         ║
║       • 这就是"权重衰减"名称的由来                                    ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    print("✅ 练习3完成！")
