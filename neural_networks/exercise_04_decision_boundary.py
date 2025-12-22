"""
============================================================================
练习4：过拟合可视化 - 决策边界对比分析
============================================================================

📚 问题背景：
    决策边界（Decision Boundary）是分类模型的可视化表示，
    它直观展示了模型如何将特征空间划分为不同的类别区域。

    过拟合的模型往往有非常复杂、扭曲的决策边界，
    而正则化可以使决策边界变得更加平滑。

🎯 学习目标：
    1. 理解决策边界的概念和可视化方法
    2. 观察过拟合模型的决策边界特征
    3. 对比无正则化 vs L2正则化的决策边界差异
    4. 直观理解正则化如何防止过拟合

============================================================================
"""

# ============================================================================
# 第1部分：导入库和环境配置
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("练习4：过拟合可视化 - 决策边界对比分析")
print("=" * 70)


# ============================================================================
# 第2部分：决策边界原理详解
# ============================================================================
"""
📖 什么是决策边界？

    决策边界是分类模型在特征空间中的分界线（或分界面）。

    对于二分类问题：
        - 决策边界是满足 P(y=1|x) = 0.5 的点的集合
        - 边界一侧预测为类别0，另一侧预测为类别1

    对于神经网络：
        - 决策边界可以是任意复杂的曲线
        - 网络越复杂，边界可以越扭曲
        - 这既是优点（可以拟合复杂模式）也是缺点（容易过拟合）

📐 绘制决策边界的方法：

    1. 创建网格点覆盖整个特征空间
    2. 对每个网格点进行预测
    3. 根据预测结果着色
    4. 预测类别变化的地方就是决策边界

💡 过拟合在决策边界上的表现：

    过拟合模型的特征：
        - 边界非常扭曲、复杂
        - 试图完美分开所有训练点
        - 对噪声点也"特别关照"
        - 形成"孤岛"区域

    正常模型的特征：
        - 边界相对平滑
        - 容忍一些训练误差
        - 更好地泛化到新数据
"""


# ============================================================================
# 第3部分：激活函数
# ============================================================================

def relu(z):
    """ReLU激活函数"""
    return np.maximum(0, z)


def relu_derivative(z):
    """ReLU导数"""
    return (z > 0).astype(float)


def sigmoid(z):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# ============================================================================
# 第4部分：神经网络实现
# ============================================================================

class OverfittingNetwork:
    """
    容易过拟合的大型网络

    网络结构：2 → 128 → 64 → 32 → 1
    （参数很多，容量大，容易记住训练数据）

    参数:
        lambda_reg: L2正则化强度，0表示无正则化
    """

    def __init__(self, lambda_reg=0.0):
        """
        初始化网络

        参数:
            lambda_reg: L2正则化强度
                - 0: 无正则化（容易过拟合）
                - >0: 有正则化（防止过拟合）
        """
        self.lambda_reg = lambda_reg

        # 使用相对较大的初始化值，容易观察过拟合
        np.random.seed(42)

        # 第1层：2 → 128
        self.W1 = np.random.randn(128, 2) * 0.5
        self.b1 = np.zeros(128)

        # 第2层：128 → 64
        self.W2 = np.random.randn(64, 128) * 0.5
        self.b2 = np.zeros(64)

        # 第3层：64 → 32
        self.W3 = np.random.randn(32, 64) * 0.5
        self.b3 = np.zeros(32)

        # 输出层：32 → 1
        self.W4 = np.random.randn(1, 32) * 0.5
        self.b4 = np.zeros(1)

        total_params = (self.W1.size + self.b1.size +
                        self.W2.size + self.b2.size +
                        self.W3.size + self.b3.size +
                        self.W4.size + self.b4.size)
        print(f"网络总参数量: {total_params}")
        print(f"L2正则化强度: {lambda_reg}")

    def forward(self, X):
        """
        前向传播

        X → [W1] → ReLU → [W2] → ReLU → [W3] → ReLU → [W4] → Sigmoid → 输出

        参数:
            X: 输入特征, shape (n_samples, 2)

        返回:
            预测概率, shape (n_samples, 1)
        """
        # 第1层
        self.z1 = X @ self.W1.T + self.b1
        self.a1 = relu(self.z1)

        # 第2层
        self.z2 = self.a1 @ self.W2.T + self.b2
        self.a2 = relu(self.z2)

        # 第3层
        self.z3 = self.a2 @ self.W3.T + self.b3
        self.a3 = relu(self.z3)

        # 输出层
        self.z4 = self.a3 @ self.W4.T + self.b4
        self.a4 = sigmoid(self.z4)

        return self.a4

    def backward(self, X, y_true):
        """
        反向传播

        计算损失函数对所有参数的梯度，包含L2正则化项

        参数:
            X: 输入数据
            y_true: 真实标签

        返回:
            所有参数的梯度列表
        """
        m = X.shape[0]

        # 输出层梯度
        delta4 = (self.a4 - y_true.reshape(-1, 1)) / m
        grad_W4 = delta4.T @ self.a3 + self.lambda_reg * self.W4
        grad_b4 = np.sum(delta4, axis=0)

        # 第3层梯度
        delta3 = (delta4 @ self.W4) * relu_derivative(self.z3)
        grad_W3 = delta3.T @ self.a2 + self.lambda_reg * self.W3
        grad_b3 = np.sum(delta3, axis=0)

        # 第2层梯度
        delta2 = (delta3 @ self.W3) * relu_derivative(self.z2)
        grad_W2 = delta2.T @ self.a1 + self.lambda_reg * self.W2
        grad_b2 = np.sum(delta2, axis=0)

        # 第1层梯度
        delta1 = (delta2 @ self.W2) * relu_derivative(self.z1)
        grad_W1 = delta1.T @ X + self.lambda_reg * self.W1
        grad_b1 = np.sum(delta1, axis=0)

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

    def compute_accuracy(self, X, y_true):
        """计算准确率"""
        y_pred = self.forward(X)
        predictions = (y_pred >= 0.5).astype(int).flatten()
        return np.mean(predictions == y_true)

    def predict_proba(self, X):
        """预测概率（用于绘制决策边界）"""
        return self.forward(X).flatten()


# ============================================================================
# 第5部分：训练函数
# ============================================================================

def train_network(X_train, y_train, lambda_reg=0.0,
                  n_epochs=1000, learning_rate=0.01, verbose=True):
    """
    训练神经网络

    参数:
        X_train, y_train: 训练数据
        lambda_reg: L2正则化强度
        n_epochs: 训练轮数
        learning_rate: 学习率
        verbose: 是否打印信息

    返回:
        model: 训练好的模型
        history: 训练历史
    """
    model = OverfittingNetwork(lambda_reg=lambda_reg)

    history = {'train_acc': []}

    for epoch in range(n_epochs):
        # 前向传播
        model.forward(X_train)

        # 反向传播
        grads = model.backward(X_train, y_train)

        # 参数更新
        params = model.get_params()
        updated_params = [p - learning_rate * g for p, g in zip(params, grads)]
        model.set_params(updated_params)

        # 记录准确率
        if epoch % 50 == 0:
            train_acc = model.compute_accuracy(X_train, y_train)
            history['train_acc'].append(train_acc)

            if verbose and epoch % 200 == 0:
                print(f"Epoch {epoch:4d}: Train Acc = {train_acc:.4f}")

    return model, history


# ============================================================================
# 第6部分：决策边界绘制函数
# ============================================================================

def plot_decision_boundary(model, X, y, ax, title, resolution=200):
    """
    绘制决策边界

    实现步骤：
        1. 确定绘图范围（根据数据范围扩展一点边界）
        2. 创建网格点
        3. 对每个网格点预测
        4. 根据预测结果着色
        5. 叠加原始数据点

    参数:
        model: 训练好的模型
        X: 输入数据
        y: 标签
        ax: matplotlib的axes对象
        title: 图表标题
        resolution: 网格分辨率

    返回:
        None（直接绑定到ax上）
    """
    # =====================================
    # 1. 确定绘图范围
    # =====================================
    # 在数据范围基础上向外扩展0.5
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # =====================================
    # 2. 创建网格
    # =====================================
    # np.meshgrid创建网格坐标
    # xx, yy是两个2D数组，每个点(xx[i,j], yy[i,j])是网格上的一个点
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # =====================================
    # 3. 对网格点进行预测
    # =====================================
    # 将网格点展平为(n_points, 2)的形状
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # 预测每个点的类别概率
    Z = model.predict_proba(grid_points)

    # 重塑为网格形状
    Z = Z.reshape(xx.shape)

    # =====================================
    # 4. 绘制等高线填充
    # =====================================
    # 使用两种颜色表示两个类别区域
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])  # 淡蓝和淡红
    cmap_bold = ['#3498db', '#e74c3c']  # 深蓝和深红

    # contourf: 填充等高线
    # levels=[0, 0.5, 1] 表示：0-0.5一种颜色，0.5-1另一种颜色
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=cmap_light, alpha=0.7)

    # contour: 绘制决策边界线（Z=0.5的等高线）
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    # =====================================
    # 5. 绘制训练数据点
    # =====================================
    # 类别0：蓝色
    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               c=cmap_bold[0], edgecolors='white',
               s=60, label='类别0', alpha=0.9)
    # 类别1：红色
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               c=cmap_bold[1], edgecolors='white',
               s=60, label='类别1', alpha=0.9)

    # =====================================
    # 6. 设置图表属性
    # =====================================
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('特征1', fontsize=11)
    ax.set_ylabel('特征2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


# ============================================================================
# 第7部分：主实验
# ============================================================================

if __name__ == "__main__":

    # =====================================
    # 1. 准备数据
    # =====================================
    print("\n" + "=" * 70)
    print("第1步：准备数据")
    print("=" * 70)

    # 生成月牙形数据集（较少样本，容易过拟合）
    X, y = make_moons(n_samples=100, noise=0.25, random_state=42)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 为了绘图，也需要对原始数据进行标准化
    X_scaled = scaler.transform(X)

    print(f"训练集大小: {X_train_scaled.shape[0]}")
    print(f"测试集大小: {X_test_scaled.shape[0]}")

    # =====================================
    # 2. 训练无正则化模型（容易过拟合）
    # =====================================
    print("\n" + "=" * 70)
    print("第2步：训练无正则化模型（λ=0）")
    print("=" * 70)

    model_no_reg, history_no_reg = train_network(
        X_train_scaled, y_train,
        lambda_reg=0.0,  # 无正则化
        n_epochs=1000,
        learning_rate=0.05,
        verbose=True
    )

    train_acc_no_reg = model_no_reg.compute_accuracy(X_train_scaled, y_train)
    test_acc_no_reg = model_no_reg.compute_accuracy(X_test_scaled, y_test)
    print(f"\n无正则化模型:")
    print(f"  训练准确率: {train_acc_no_reg:.4f}")
    print(f"  测试准确率: {test_acc_no_reg:.4f}")
    print(f"  过拟合程度: {train_acc_no_reg - test_acc_no_reg:.4f}")

    # =====================================
    # 3. 训练L2正则化模型
    # =====================================
    print("\n" + "=" * 70)
    print("第3步：训练L2正则化模型（λ=0.01）")
    print("=" * 70)

    model_l2_reg, history_l2_reg = train_network(
        X_train_scaled, y_train,
        lambda_reg=0.01,  # L2正则化
        n_epochs=1000,
        learning_rate=0.05,
        verbose=True
    )

    train_acc_l2_reg = model_l2_reg.compute_accuracy(X_train_scaled, y_train)
    test_acc_l2_reg = model_l2_reg.compute_accuracy(X_test_scaled, y_test)
    print(f"\nL2正则化模型:")
    print(f"  训练准确率: {train_acc_l2_reg:.4f}")
    print(f"  测试准确率: {test_acc_l2_reg:.4f}")
    print(f"  过拟合程度: {train_acc_l2_reg - test_acc_l2_reg:.4f}")

    # =====================================
    # 4. 训练强正则化模型
    # =====================================
    print("\n" + "=" * 70)
    print("第4步：训练强正则化模型（λ=0.1）")
    print("=" * 70)

    model_strong_reg, history_strong_reg = train_network(
        X_train_scaled, y_train,
        lambda_reg=0.1,  # 强L2正则化
        n_epochs=1000,
        learning_rate=0.05,
        verbose=True
    )

    train_acc_strong = model_strong_reg.compute_accuracy(X_train_scaled, y_train)
    test_acc_strong = model_strong_reg.compute_accuracy(X_test_scaled, y_test)
    print(f"\n强正则化模型:")
    print(f"  训练准确率: {train_acc_strong:.4f}")
    print(f"  测试准确率: {test_acc_strong:.4f}")
    print(f"  过拟合程度: {train_acc_strong - test_acc_strong:.4f}")

    # =====================================
    # 5. 绘制决策边界对比图
    # =====================================
    print("\n" + "=" * 70)
    print("第5步：绘制决策边界对比图")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ----- 图1：无正则化（过拟合） -----
    plot_decision_boundary(
        model_no_reg, X_train_scaled, y_train, axes[0, 0],
        f'无正则化 (λ=0)\n训练:{train_acc_no_reg:.2%}, 测试:{test_acc_no_reg:.2%}'
    )

    # ----- 图2：适度正则化 -----
    plot_decision_boundary(
        model_l2_reg, X_train_scaled, y_train, axes[0, 1],
        f'L2正则化 (λ=0.01)\n训练:{train_acc_l2_reg:.2%}, 测试:{test_acc_l2_reg:.2%}'
    )

    # ----- 图3：强正则化 -----
    plot_decision_boundary(
        model_strong_reg, X_train_scaled, y_train, axes[1, 0],
        f'强正则化 (λ=0.1)\n训练:{train_acc_strong:.2%}, 测试:{test_acc_strong:.2%}'
    )

    # ----- 图4：结论总结 -----
    ax4 = axes[1, 1]
    summary_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║              决策边界对比分析总结                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📊 实验结果:                                                    ║
║                                                                  ║
║  无正则化 (λ=0):                                                ║
║    • 决策边界非常扭曲、复杂                                       ║
║    • 试图完美分开所有训练点                                       ║
║    • 训练准确率高但测试准确率低                                   ║
║    → 严重过拟合                                                  ║
║                                                                  ║
║  适度正则化 (λ=0.01):                                           ║
║    • 决策边界相对平滑                                            ║
║    • 容忍一些训练误差                                            ║
║    • 训练和测试准确率接近                                        ║
║    → 良好泛化                                                    ║
║                                                                  ║
║  强正则化 (λ=0.1):                                              ║
║    • 决策边界过于简单                                            ║
║    • 可能无法捕捉数据真实模式                                     ║
║    → 可能欠拟合                                                  ║
║                                                                  ║
║  💡 关键洞察:                                                    ║
║    决策边界的复杂度反映了模型的"记忆"程度                         ║
║    过拟合 = 记住了训练数据的噪声                                  ║
║    正则化 = 限制模型的记忆能力                                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    ax4.text(0.02, 0.5, summary_text, fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/decision_boundary_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # =====================================
    # 6. 绘制训练曲线对比
    # =====================================
    print("\n" + "=" * 70)
    print("第6步：绘制训练曲线对比")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = np.arange(len(history_no_reg['train_acc'])) * 50

    ax.plot(epochs, history_no_reg['train_acc'],
            label='无正则化 (λ=0)', linewidth=2, color='#e74c3c')
    ax.plot(epochs, history_l2_reg['train_acc'],
            label='L2正则化 (λ=0.01)', linewidth=2, color='#3498db')
    ax.plot(epochs, history_strong_reg['train_acc'],
            label='强正则化 (λ=0.1)', linewidth=2, color='#2ecc71')

    ax.axhline(y=test_acc_no_reg, color='#e74c3c', linestyle='--', alpha=0.5)
    ax.axhline(y=test_acc_l2_reg, color='#3498db', linestyle='--', alpha=0.5)
    ax.axhline(y=test_acc_strong, color='#2ecc71', linestyle='--', alpha=0.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('训练准确率', fontsize=12)
    ax.set_title('不同正则化强度的训练曲线对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加测试准确率标注
    ax.annotate(f'测试: {test_acc_no_reg:.2%}',
                xy=(epochs[-1], test_acc_no_reg),
                xytext=(epochs[-1] + 50, test_acc_no_reg),
                fontsize=9, color='#e74c3c')
    ax.annotate(f'测试: {test_acc_l2_reg:.2%}',
                xy=(epochs[-1], test_acc_l2_reg),
                xytext=(epochs[-1] + 50, test_acc_l2_reg),
                fontsize=9, color='#3498db')

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/training_curves_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # =====================================
    # 7. 打印最终总结
    # =====================================
    print("\n" + "=" * 70)
    print("实验完成！最终总结")
    print("=" * 70)

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    决策边界可视化实验完成！                             ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  📊 三种模型对比:                                                     ║
║                                                                       ║
║  模型              λ值     训练准确率   测试准确率   过拟合程度         ║
║  ─────────────────────────────────────────────────────────────────   ║
║  无正则化          0       {train_acc_no_reg:.4f}       {test_acc_no_reg:.4f}       {train_acc_no_reg - test_acc_no_reg:.4f}           ║
║  适度正则化        0.01    {train_acc_l2_reg:.4f}       {test_acc_l2_reg:.4f}       {train_acc_l2_reg - test_acc_l2_reg:.4f}           ║
║  强正则化          0.1     {train_acc_strong:.4f}       {test_acc_strong:.4f}       {train_acc_strong - test_acc_strong:.4f}           ║
║                                                                       ║
║  🔍 决策边界特征分析:                                                 ║
║                                                                       ║
║  过拟合模型 (λ=0):                                                    ║
║    • 边界高度扭曲，试图绕过每一个训练点                                ║
║    • 形成不规则的"孤岛"区域                                            ║
║    • 对噪声过度敏感                                                   ║
║                                                                       ║
║  正则化模型 (λ>0):                                                    ║
║    • 边界更加平滑                                                     ║
║    • 不强求完美分类所有训练点                                          ║
║    • 更好地捕捉数据的真实模式                                          ║
║                                                                       ║
║  💡 核心洞察:                                                         ║
║                                                                       ║
║    1. 决策边界复杂度 ∝ 模型过拟合程度                                  ║
║                                                                       ║
║    2. 正则化通过惩罚大权重来限制边界复杂度                             ║
║                                                                       ║
║    3. 好的模型应该学习"信号"而非"噪声"                                ║
║                                                                       ║
║    4. 平滑的决策边界往往泛化性能更好                                   ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    print("✅ 练习4完成！所有4个练习都已完成。")
    print("\n生成的文件：")
    print("  - exercise_01_elastic_net.py")
    print("  - exercise_02_data_augmentation.py")
    print("  - exercise_03_lambda_search.py")
    print("  - exercise_04_decision_boundary.py")
