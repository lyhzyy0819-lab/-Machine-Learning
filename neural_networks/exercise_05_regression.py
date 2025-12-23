"""
============================================================================
练习5：神经网络回归实战
============================================================================

📚 问题背景：
    在实际应用中，回归问题和分类问题同样重要。
    本练习将帮助你深入理解神经网络回归的各个方面。

🎯 学习目标：
    1. 实现 Huber Loss 及其梯度
    2. 完成房价预测完整流程
    3. 理解特征重要性分析
    4. 掌握超参数对回归性能的影响

============================================================================
"""

# ============================================================================
# 第1部分：导入库和环境配置
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("练习5：神经网络回归实战")
print("=" * 70)


# ============================================================================
# 第2部分：练习1 - 实现 Huber Loss
# ============================================================================
"""
📖 Huber Loss 原理

Huber Loss 是 MSE 和 MAE 的结合：
- 当 |error| <= delta: L = 0.5 * error^2 (像 MSE)
- 当 |error| > delta:  L = delta * (|error| - 0.5 * delta) (像 MAE)

优点：
- 小误差时：光滑、可微、收敛快
- 大误差时：对异常值鲁棒

梯度：
- 当 |error| <= delta: dL/d_pred = -error (像 MSE)
- 当 |error| > delta:  dL/d_pred = -delta * sign(error) (像 MAE)
"""

print("\n" + "="*70)
print("练习1：实现 Huber Loss 及其梯度")
print("="*70)


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss 损失函数

    📐 数学公式：
        当 |y - ŷ| <= δ: L = 0.5 * (y - ŷ)²
        当 |y - ŷ| > δ:  L = δ * (|y - ŷ| - 0.5 * δ)

    参数:
        y_true: 真实值, shape (n_samples,) 或 (n_samples, 1)
        y_pred: 预测值, shape (n_samples,) 或 (n_samples, 1)
        delta: 切换阈值，控制 MSE 和 MAE 的切换点
               默认 1.0

    返回:
        loss: 标量，Huber 损失值

    💡 提示：
        - 使用 np.where() 根据条件选择不同的计算方式
        - 注意处理 y_true 和 y_pred 的形状
    """
    # ===== 你的代码 =====
    # 第1步：计算预测误差
    error = y_true.flatten() - y_pred.flatten()

    # 第2步：计算绝对误差
    abs_error = np.abs(error)

    # 第3步：根据误差大小选择计算方式
    # 小误差（|e| <= delta）：使用 MSE 形式 0.5 * e^2
    # 大误差（|e| > delta）：使用线性形式 delta * (|e| - 0.5 * delta)
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)

    # 第4步：根据条件选择
    loss = np.where(abs_error <= delta, quadratic, linear)

    # 第5步：返回平均损失
    return np.mean(loss)
    # ===== 代码结束 =====


def huber_loss_gradient(y_true, y_pred, delta=1.0):
    """
    Huber Loss 对预测值的梯度

    📐 梯度公式：
        当 |y - ŷ| <= δ: ∂L/∂ŷ = -(y - ŷ) = (ŷ - y)
        当 |y - ŷ| > δ:  ∂L/∂ŷ = -δ * sign(y - ŷ) = δ * sign(ŷ - y)

    参数:
        y_true: 真实值
        y_pred: 预测值
        delta: 切换阈值

    返回:
        gradient: 梯度, shape 与 y_pred 相同

    💡 提示：
        - 梯度表示损失对预测值的导数
        - 注意 sign 函数：np.sign()
    """
    # ===== 你的代码 =====
    # 第1步：计算误差（注意方向：pred - true）
    error = y_pred.flatten() - y_true.flatten()

    # 第2步：计算绝对误差
    abs_error = np.abs(error)

    # 第3步：根据误差大小计算梯度
    # 小误差：梯度 = error（MSE 的梯度）
    # 大误差：梯度 = delta * sign(error)（MAE 的梯度，被 delta 截断）
    grad = np.where(abs_error <= delta, error, delta * np.sign(error))

    # 第4步：返回平均梯度
    return grad.reshape(y_pred.shape) / len(y_true)
    # ===== 代码结束 =====


# 测试 Huber Loss 实现
print("\n测试 Huber Loss 实现:")
y_true_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred_test = np.array([1.1, 2.5, 2.5, 4.2, 10.0])  # 最后一个是异常预测

print(f"真实值: {y_true_test}")
print(f"预测值: {y_pred_test}")
print(f"误差:   {y_true_test - y_pred_test}")

mse = np.mean((y_true_test - y_pred_test) ** 2)
mae = np.mean(np.abs(y_true_test - y_pred_test))
huber = huber_loss(y_true_test, y_pred_test, delta=1.0)

print(f"\nMSE Loss:   {mse:.4f}")
print(f"MAE Loss:   {mae:.4f}")
print(f"Huber Loss: {huber:.4f}")
print("\n✓ 如果 Huber 介于 MSE 和 MAE 之间（但更接近 MAE），则实现正确！")


# ============================================================================
# 第3部分：练习2 - 完整的 MLP 回归
# ============================================================================

print("\n" + "="*70)
print("练习2：完整的房价预测 MLP")
print("="*70)


class MLPRegressorWithHuber:
    """
    支持多种损失函数的 MLP 回归器

    💡 与分类网络的区别：
    1. 输出层无激活函数（线性输出）
    2. 支持 MSE、MAE、Huber 三种损失
    3. 使用 R²、RMSE 评估
    """

    def __init__(self, layer_sizes, loss_type='mse', huber_delta=1.0):
        """
        初始化网络

        参数:
            layer_sizes: 各层神经元数量列表，如 [8, 64, 32, 1]
            loss_type: 损失函数类型 'mse', 'mae', 或 'huber'
            huber_delta: Huber 损失的 delta 参数
        """
        self.layer_sizes = layer_sizes
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.n_layers = len(layer_sizes)

        # 初始化权重和偏置
        self.weights = []
        self.biases = []

        # 使用 He 初始化
        for i in range(self.n_layers - 1):
            # 权重 shape: (当前层, 下一层)
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        # 记录训练历史
        self.train_losses = []
        self.val_losses = []

        print(f"网络结构: {' -> '.join(map(str, layer_sizes))}")
        print(f"损失函数: {loss_type.upper()}")

    def relu(self, z):
        """ReLU 激活函数"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU 导数"""
        return (z > 0).astype(float)

    def forward(self, X):
        """
        前向传播

        参数:
            X: 输入, shape (n_samples, n_features)

        返回:
            output: 预测值, shape (n_samples, 1)
        """
        self.activations = [X]  # 保存各层激活值
        self.z_values = []       # 保存各层线性输出

        current = X

        # 隐藏层：使用 ReLU
        for i in range(self.n_layers - 2):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)
            current = a

        # 输出层：无激活函数（线性）
        z_out = current @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z_out)
        self.activations.append(z_out)  # 输出层的激活就是线性输出

        return z_out

    def compute_loss(self, y_true, y_pred):
        """
        计算损失

        根据 self.loss_type 选择不同的损失函数
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        if self.loss_type == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.loss_type == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif self.loss_type == 'huber':
            return huber_loss(y_true, y_pred, self.huber_delta)
        else:
            raise ValueError(f"未知的损失类型: {self.loss_type}")

    def compute_output_gradient(self, y_true, y_pred):
        """
        计算输出层梯度 dL/dz_out

        不同损失函数有不同的梯度：
        - MSE: (y_pred - y_true) / n
        - MAE: sign(y_pred - y_true) / n
        - Huber: 结合 MSE 和 MAE
        """
        n = len(y_true)
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        if self.loss_type == 'mse':
            # MSE 梯度: 2 * (pred - true) / n，简化为 (pred - true) / n
            grad = (y_pred - y_true) / n
        elif self.loss_type == 'mae':
            # MAE 梯度: sign(pred - true) / n
            grad = np.sign(y_pred - y_true) / n
        elif self.loss_type == 'huber':
            # Huber 梯度
            grad = huber_loss_gradient(y_true, y_pred, self.huber_delta).flatten()

        return grad.reshape(-1, 1)

    def backward(self, y_true, learning_rate=0.01):
        """
        反向传播 + 参数更新
        """
        n_samples = y_true.shape[0]

        # 输出层梯度
        dz = self.compute_output_gradient(y_true, self.activations[-1])

        # 从后向前计算梯度
        for i in range(self.n_layers - 2, -1, -1):
            # 计算权重和偏置梯度
            dW = self.activations[i].T @ dz
            db = np.sum(dz, axis=0, keepdims=True)

            # 计算传递给前一层的梯度
            if i > 0:
                da = dz @ self.weights[i].T
                dz = da * self.relu_derivative(self.z_values[i-1])

            # 更新参数
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, learning_rate=0.01, batch_size=32, verbose=True):
        """
        训练模型
        """
        n_samples = X_train.shape[0]
        y_train = y_train.reshape(-1, 1)
        if y_val is not None:
            y_val = y_val.reshape(-1, 1)

        self.train_losses = []
        self.val_losses = []

        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # 小批量训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # 前向 + 反向
                self.forward(X_batch)
                self.backward(y_batch, learning_rate)

            # 记录损失
            train_pred = self.forward(X_train)
            train_loss = self.compute_loss(y_train, train_pred)
            self.train_losses.append(train_loss)

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 50 == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}"
                if X_val is not None:
                    msg += f" - Val Loss: {val_loss:.4f}"
                print(msg)

    def predict(self, X):
        """预测"""
        return self.forward(X)

    def score(self, X, y_true):
        """计算 R² 分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y_true.flatten() - y_pred.flatten()) ** 2)
        ss_tot = np.sum((y_true.flatten() - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


# 加载数据
print("\n加载加州房价数据集...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 数据划分
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# 标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")


# ============================================================================
# 第4部分：对比不同损失函数
# ============================================================================

print("\n" + "="*70)
print("对比三种损失函数的效果")
print("="*70)

results = {}

for loss_type in ['mse', 'mae', 'huber']:
    print(f"\n--- 训练 {loss_type.upper()} 模型 ---")

    model = MLPRegressorWithHuber(
        layer_sizes=[8, 64, 32, 1],
        loss_type=loss_type,
        huber_delta=1.0
    )

    model.fit(
        X_train_scaled, y_train_scaled,
        X_val=X_val_scaled, y_val=y_val_scaled,
        epochs=200,
        learning_rate=0.01,
        batch_size=64,
        verbose=True
    )

    # 评估
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results[loss_type] = {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'train_losses': model.train_losses,
        'val_losses': model.val_losses
    }

    print(f"R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")


# ============================================================================
# 第5部分：可视化结果
# ============================================================================

print("\n" + "="*70)
print("可视化结果")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ----- 图1：学习曲线对比 -----
ax1 = axes[0, 0]
colors = {'mse': 'blue', 'mae': 'red', 'huber': 'green'}
for loss_type, data in results.items():
    epochs = range(1, len(data['train_losses']) + 1)
    ax1.plot(epochs, data['val_losses'], color=colors[loss_type],
             linewidth=2, label=f'{loss_type.upper()} 验证损失')

ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('验证损失学习曲线对比', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ----- 图2：评估指标对比 -----
ax2 = axes[0, 1]
metrics = ['R²', 'RMSE', 'MAE']
x_pos = np.arange(len(metrics))
width = 0.25

for i, (loss_type, data) in enumerate(results.items()):
    values = [data['r2'], data['rmse'], data['mae']]
    ax2.bar(x_pos + i*width, values, width, label=loss_type.upper(), color=colors[loss_type])

ax2.set_xticks(x_pos + width)
ax2.set_xticklabels(metrics)
ax2.set_ylabel('数值', fontsize=11)
ax2.set_title('评估指标对比', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# ----- 图3：预测值 vs 真实值（最佳模型）-----
ax3 = axes[1, 0]
best_loss = max(results.keys(), key=lambda k: results[k]['r2'])
best_model = results[best_loss]['model']
y_pred_scaled = best_model.predict(X_test_scaled)
y_pred_best = scaler_y.inverse_transform(y_pred_scaled)

ax3.scatter(y_test, y_pred_best, alpha=0.3, s=10, c='blue')
min_val, max_val = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
ax3.set_xlabel('真实房价', fontsize=11)
ax3.set_ylabel('预测房价', fontsize=11)
ax3.set_title(f'最佳模型 ({best_loss.upper()}) 预测 vs 真实', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ----- 图4：残差分布 -----
ax4 = axes[1, 1]
residuals = y_test - y_pred_best.flatten()
ax4.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差')
ax4.axvline(x=np.mean(residuals), color='green', linestyle='-', linewidth=2,
            label=f'均值={np.mean(residuals):.3f}')
ax4.set_xlabel('残差 (真实 - 预测)', fontsize=11)
ax4.set_ylabel('频数', fontsize=11)
ax4.set_title('残差分布', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/regression_results.png',
            dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# 第6部分：练习3 - 特征重要性分析
# ============================================================================

print("\n" + "="*70)
print("练习3：特征重要性分析")
print("="*70)

"""
📖 特征重要性分析方法

方法1：基于权重的分析
    - 查看第一层权重的绝对值大小
    - 权重越大，特征影响越大

方法2：置换重要性（Permutation Importance）
    - 打乱某个特征的值
    - 观察模型性能下降程度
    - 下降越多，特征越重要
"""

# 使用置换重要性
print("\n使用置换重要性分析特征...")

feature_names = housing.feature_names
importance_scores = []

# 基准性能
baseline_pred = best_model.predict(X_test_scaled)
baseline_r2 = r2_score(y_test, scaler_y.inverse_transform(baseline_pred))

print(f"基准 R²: {baseline_r2:.4f}\n")

for i, name in enumerate(feature_names):
    # 复制测试数据
    X_permuted = X_test_scaled.copy()

    # 打乱第 i 个特征
    np.random.shuffle(X_permuted[:, i])

    # 计算打乱后的性能
    permuted_pred = best_model.predict(X_permuted)
    permuted_r2 = r2_score(y_test, scaler_y.inverse_transform(permuted_pred))

    # 重要性 = 性能下降程度
    importance = baseline_r2 - permuted_r2
    importance_scores.append(importance)

    print(f"  {name:<15}: 打乱后 R² = {permuted_r2:.4f}, 下降 = {importance:.4f}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(importance_scores)
plt.barh(range(len(feature_names)), np.array(importance_scores)[sorted_idx], color='steelblue')
plt.yticks(range(len(feature_names)), np.array(feature_names)[sorted_idx])
plt.xlabel('重要性 (R² 下降)', fontsize=11)
plt.title('特征重要性分析 (置换重要性)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/feature_importance.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\n💡 特征重要性解读:")
most_important = feature_names[np.argmax(importance_scores)]
print(f"   最重要的特征: {most_important}")
print("   这通常与房价有很强的相关性！")


# ============================================================================
# 第7部分：总结
# ============================================================================

print("\n" + "="*70)
print("练习总结")
print("="*70)

print("""
✅ 完成的内容:

1. Huber Loss 实现
   - 理解了 Huber Loss 的数学原理
   - 实现了损失函数和梯度计算

2. 完整的 MLP 回归
   - 支持 MSE、MAE、Huber 三种损失
   - 使用加州房价数据集训练

3. 损失函数对比
   - MSE: 收敛快，但对异常值敏感
   - MAE: 对异常值鲁棒，但可能收敛不稳定
   - Huber: 结合两者优点

4. 特征重要性分析
   - 使用置换重要性方法
   - 识别了最重要的特征

📊 最终结果:
""")

for loss_type, data in results.items():
    print(f"   {loss_type.upper()}: R² = {data['r2']:.4f}, RMSE = {data['rmse']:.4f}")

print(f"\n🏆 最佳模型: {best_loss.upper()} (R² = {results[best_loss]['r2']:.4f})")
print("\n" + "="*70)
