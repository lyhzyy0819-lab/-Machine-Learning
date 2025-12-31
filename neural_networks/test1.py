import numpy as np
import matplotlib.pyplot as plt


class Mixup:
    """
    Mixup数据增强 - 从零实现

    论文: "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)

    核心思想:
        - 通过线性插值混合两个样本及其标签
        - 创建虚拟的训练样本，扩展训练分布
        - 产生更平滑的决策边界，提高泛化能力

    优点:
        - 实现简单
        - 无额外计算开销（训练时混合）
        - 对各种模型和任务都有效
    """

    def __init__(self, alpha=0.2):
        """
        初始化Mixup

        参数:
            alpha: float, Beta分布的参数
                   alpha=0: 不混合（退化为标准训练）
                   alpha=1: 均匀分布，强混合
                   推荐值: 0.2 - 0.4
        """
        self.alpha = alpha

    def sample_lambda(self):
        """
        从Beta分布采样混合系数λ

        返回:
            float: λ ∈ [0, 1]

        Beta分布的性质:
            - α = β 时分布对称
            - α < 1 时分布呈U形，倾向于取接近0或1的值
            - α > 1 时分布呈倒U形，倾向于取接近0.5的值
        """
        if self.alpha > 0:
            # 从Beta(α, α)采样
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            # alpha=0时不混合
            lam = 1.0
        return lam

    def mix_samples(self, x1, y1, x2, y2):
        """
        混合两个样本

        参数:
            x1, x2: numpy数组, 输入图像
            y1, y2: numpy数组, one-hot编码的标签

        返回:
            mixed_x: 混合后的图像
            mixed_y: 混合后的标签（软标签）
            lam: 使用的混合系数

        公式:
            x̃ = λ * x1 + (1-λ) * x2
            ỹ = λ * y1 + (1-λ) * y2
        """
        lam = self.sample_lambda()

        # 混合图像
        mixed_x = lam * x1 + (1 - lam) * x2

        # 混合标签（软标签）
        mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_x, mixed_y, lam

    def mix_batch(self, batch_x, batch_y):
        """
        对一个batch进行Mixup

        参数:
            batch_x: numpy数组, shape: (batch_size, H, W, C)
            batch_y: numpy数组, shape: (batch_size, n_classes), one-hot编码

        返回:
            mixed_x: 混合后的batch
            mixed_y: 混合后的标签
            lam: 使用的混合系数

        策略:
            将batch打乱，然后与原batch混合
            这样每个样本都与另一个随机样本混合
        """
        batch_size = len(batch_x)
        lam = self.sample_lambda()

        # 随机打乱索引
        shuffle_indices = np.random.permutation(batch_size)

        # 混合
        mixed_x = lam * batch_x + (1 - lam) * batch_x[shuffle_indices]
        mixed_y = lam * batch_y + (1 - lam) * batch_y[shuffle_indices]

        return mixed_x, mixed_y, lam

def exercise_1_gridmask():
    """
    练习1: 实现GridMask
    提示:
        1. 创建一个网格掩码
        2. 将掩码应用到图像上
    """
    # 在这里填写你的代码
    pass


# =============================================================================
# 练习3: 完整实现MixUp训练循环
# =============================================================================

def exercise_3_mixup_training():
    """
    练习3（挑战）: 完整实现MixUp训练循环

    本练习将实现一个完整的使用Mixup数据增强的训练流程，包括：
    1. 数据加载与预处理
    2. 简单神经网络定义（从零实现）
    3. 支持软标签的交叉熵损失
    4. Mixup训练循环
    5. 效果对比可视化

    学习目标：
    - 理解Mixup如何与训练循环集成
    - 掌握软标签（soft labels）的损失计算
    - 观察Mixup的正则化效果
    """

    print("=" * 70)
    print("练习3: 完整实现MixUp训练循环")
    print("=" * 70)

    # =========================================================================
    # 第1部分: 数据加载与预处理
    # =========================================================================
    print("\n📦 第1部分: 数据加载与预处理")
    print("-" * 50)

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # 加载手写数字数据集
    # digits数据集: 1797个样本，每个样本是8x8=64维的灰度图像，共10个类别(0-9)
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"数据集大小: {X.shape[0]} 样本")
    print(f"特征维度: {X.shape[1]} (8x8像素)")
    print(f"类别数量: {len(np.unique(y))} (数字0-9)")

    # 数据标准化
    # 标准化公式: X_scaled = (X - μ) / σ
    # 这有助于加速训练收敛，避免某些特征主导梯度
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分数据集: 60%训练, 20%验证, 20%测试
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    # 将标签转换为one-hot编码
    # Mixup需要对标签进行混合，因此必须使用one-hot格式
    # 例如: 标签3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    def to_one_hot(y, n_classes):
        """
        将整数标签转换为one-hot编码

        参数:
            y: numpy数组, shape (n_samples,), 整数标签
            n_classes: int, 类别总数

        返回:
            one_hot: numpy数组, shape (n_samples, n_classes)

        示例:
            y = [0, 2, 1]
            n_classes = 3
            返回: [[1,0,0], [0,0,1], [0,1,0]]
        """
        n_samples = len(y)
        one_hot = np.zeros((n_samples, n_classes))
        # 使用高级索引: one_hot[行索引, 列索引] = 1
        one_hot[np.arange(n_samples), y] = 1
        return one_hot

    n_classes = 10
    y_train_onehot = to_one_hot(y_train, n_classes)
    y_val_onehot = to_one_hot(y_val, n_classes)
    y_test_onehot = to_one_hot(y_test, n_classes)

    print(f"标签形状（one-hot）: {y_train_onehot.shape}")

    # =========================================================================
    # 第2部分: 简单神经网络定义（从零实现）
    # =========================================================================
    print("\n🧠 第2部分: 神经网络定义")
    print("-" * 50)

    class SimpleMLP:
        """
        简单的多层感知机（从零实现）

        网络结构: 输入层(64) → 隐藏层(128, ReLU) → 输出层(10, Softmax)

        关键特性:
        - 支持软标签训练（用于Mixup）
        - He初始化（适用于ReLU激活）
        - 完整的前向传播和反向传播实现

        参数:
            input_size: int, 输入特征维度
            hidden_size: int, 隐藏层神经元数量
            output_size: int, 输出类别数量
            learning_rate: float, 学习率
        """

        def __init__(self, input_size=64, hidden_size=128, output_size=10, learning_rate=0.01):
            """
            初始化网络参数

            使用He初始化:
                W ~ N(0, sqrt(2/n_in))

            这种初始化方式特别适合ReLU激活函数，
            可以避免梯度消失/爆炸问题
            """
            self.lr = learning_rate

            # 隐藏层参数
            # W1 shape: (input_size, hidden_size) = (64, 128)
            # He初始化: 标准差 = sqrt(2 / fan_in)
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.b1 = np.zeros(hidden_size)

            # 输出层参数
            # W2 shape: (hidden_size, output_size) = (128, 10)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
            self.b2 = np.zeros(output_size)

            # 缓存（用于反向传播）
            self.cache = {}

        def relu(self, z):
            """
            ReLU激活函数: f(z) = max(0, z)

            优点:
            - 计算简单
            - 缓解梯度消失问题
            - 稀疏激活（部分神经元输出为0）
            """
            return np.maximum(0, z)

        def relu_derivative(self, z):
            """
            ReLU的导数: f'(z) = 1 if z > 0 else 0

            注意: 在z=0处导数未定义，这里取0
            """
            return (z > 0).astype(float)

        def softmax(self, z):
            """
            Softmax激活函数: softmax(z_i) = exp(z_i) / Σ_j exp(z_j)

            将输出转换为概率分布（所有输出和为1）

            数值稳定性技巧:
            - 减去最大值避免exp溢出
            - exp(z - max(z)) / Σexp(z - max(z)) 数学上等价
            """
            # 减去每行的最大值，避免数值溢出
            z_shifted = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z_shifted)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

        def forward(self, X):
            """
            前向传播

            计算流程:
                z1 = X @ W1 + b1        # 隐藏层线性变换
                h1 = relu(z1)           # 隐藏层激活
                z2 = h1 @ W2 + b2       # 输出层线性变换
                y_pred = softmax(z2)    # 输出层激活（概率）

            参数:
                X: numpy数组, shape (batch_size, input_size)

            返回:
                y_pred: numpy数组, shape (batch_size, output_size), 预测概率
            """
            # 隐藏层
            # z1 shape: (batch_size, hidden_size)
            z1 = X @ self.W1 + self.b1
            h1 = self.relu(z1)

            # 输出层
            # z2 shape: (batch_size, output_size)
            z2 = h1 @ self.W2 + self.b2
            y_pred = self.softmax(z2)

            # 缓存中间结果（反向传播需要）
            self.cache = {
                'X': X,  # 输入
                'z1': z1,  # 隐藏层线性输出
                'h1': h1,  # 隐藏层激活输出
                'z2': z2,  # 输出层线性输出
                'y_pred': y_pred  # 最终预测
            }

            return y_pred

        def backward(self, y_true):
            """
            反向传播 - 计算梯度并更新参数

            关键公式（交叉熵 + Softmax的简化）:
                δ2 = y_pred - y_true              # 输出层误差
                dW2 = h1.T @ δ2 / m               # W2梯度
                db2 = mean(δ2, axis=0)            # b2梯度

                δ1 = (δ2 @ W2.T) * relu'(z1)     # 隐藏层误差
                dW1 = X.T @ δ1 / m                # W1梯度
                db1 = mean(δ1, axis=0)            # b1梯度

            参数:
                y_true: numpy数组, shape (batch_size, output_size)
                        可以是硬标签（one-hot）或软标签（概率分布）

            注意:
                - Mixup训练时，y_true是软标签，如[0.7, 0, 0.3, 0, ...]
                - 交叉熵+Softmax的梯度公式对软标签同样适用！
            """
            # 获取缓存
            X = self.cache['X']
            h1 = self.cache['h1']
            z1 = self.cache['z1']
            y_pred = self.cache['y_pred']

            m = X.shape[0]  # batch大小

            # ===== 输出层梯度 =====
            # 交叉熵损失对Softmax输出的梯度简化为: δ2 = y_pred - y_true
            # 这个优美的结果来自于对 -Σy*log(p) 的求导
            # 无论y是硬标签还是软标签，这个公式都成立！
            delta2 = y_pred - y_true  # shape: (m, output_size)

            # W2的梯度: dL/dW2 = h1.T @ delta2 / m
            # 形状: (hidden_size, m) @ (m, output_size) = (hidden_size, output_size)
            dW2 = h1.T @ delta2 / m

            # b2的梯度: 对batch取平均
            db2 = np.mean(delta2, axis=0)

            # ===== 隐藏层梯度 =====
            # 误差反向传播: delta1 = (delta2 @ W2.T) * relu'(z1)
            # 形状: (m, output_size) @ (output_size, hidden_size) = (m, hidden_size)
            delta1 = (delta2 @ self.W2.T) * self.relu_derivative(z1)

            # W1的梯度
            dW1 = X.T @ delta1 / m

            # b1的梯度
            db1 = np.mean(delta1, axis=0)

            # ===== 参数更新（梯度下降）=====
            # θ_new = θ_old - lr * gradient
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        def predict(self, X):
            """
            预测类别标签

            返回:
                预测的类别索引（取概率最大的类别）
            """
            y_pred = self.forward(X)
            return np.argmax(y_pred, axis=1)

        def copy(self):
            """
            复制模型（用于对比实验，确保两个模型初始参数相同）
            """
            new_model = SimpleMLP(
                input_size=self.W1.shape[0],
                hidden_size=self.W1.shape[1],
                output_size=self.W2.shape[1],
                learning_rate=self.lr
            )
            new_model.W1 = self.W1.copy()
            new_model.b1 = self.b1.copy()
            new_model.W2 = self.W2.copy()
            new_model.b2 = self.b2.copy()
            return new_model

    print("SimpleMLP类定义完成")
    print("网络结构: 64 → 128 (ReLU) → 10 (Softmax)")

    # =========================================================================
    # 第3部分: 损失函数定义
    # =========================================================================
    print("\n📉 第3部分: 损失函数定义")
    print("-" * 50)

    def soft_cross_entropy(y_pred, y_true):
        """
        支持软标签的交叉熵损失

        公式: L = -Σ y_true * log(y_pred)

        与硬标签的区别:
        - 硬标签: y_true是one-hot，如[0, 0, 1, 0, ...]，只有一个1
        - 软标签: y_true是概率分布，如[0.3, 0, 0.7, 0, ...]，和为1

        Mixup训练时使用软标签:
        - 如果图像A(标签猫)和图像B(标签狗)以λ=0.7混合
        - 混合后的标签为: 0.7*猫 + 0.3*狗
        - 这就是软标签！

        参数:
            y_pred: numpy数组, shape (batch_size, n_classes), 预测概率
            y_true: numpy数组, shape (batch_size, n_classes), 真实标签（可以是软标签）

        返回:
            loss: float, 平均损失值
        """
        # 数值稳定性: 避免log(0)
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # 交叉熵计算: -Σ y_true * log(y_pred)
        # 对每个样本: 计算所有类别的加权对数和
        # 然后对batch取平均
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        return loss

    def accuracy(y_pred, y_true):
        """
        计算分类准确率

        参数:
            y_pred: 预测概率或预测标签
            y_true: 真实标签（可以是one-hot或整数形式）

        返回:
            准确率 (0到1之间)
        """
        # 如果y_pred是概率，转换为类别
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            pred_labels = np.argmax(y_pred, axis=1)
        else:
            pred_labels = y_pred

        # 如果y_true是one-hot，转换为类别
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            true_labels = np.argmax(y_true, axis=1)
        else:
            true_labels = y_true

        return np.mean(pred_labels == true_labels)

    print("损失函数: soft_cross_entropy (支持软标签)")
    print("评估函数: accuracy")

    # =========================================================================
    # 第4部分: 训练函数定义
    # =========================================================================
    print("\n🔄 第4部分: 训练函数定义")
    print("-" * 50)

    def create_batches(X, y, batch_size, shuffle=True):
        """
        创建mini-batch数据生成器

        参数:
            X: 特征数据
            y: 标签数据
            batch_size: 每个batch的大小
            shuffle: 是否打乱数据

        生成:
            (batch_X, batch_y) 元组
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            yield X[batch_indices], y[batch_indices]

    def train_with_mixup(model, X_train, y_train, X_val, y_val,
                         mixup_alpha=0.2, epochs=50, batch_size=32):
        """
        使用Mixup进行训练

        Mixup训练的核心流程:
        1. 从训练数据中采样一个batch
        2. 对batch应用Mixup增强（混合图像和标签）
        3. 用混合后的数据进行前向传播
        4. 用软标签计算损失
        5. 反向传播更新参数

        参数:
            model: SimpleMLP实例
            X_train, y_train: 训练数据（y_train应为one-hot格式）
            X_val, y_val: 验证数据
            mixup_alpha: Mixup的α参数（控制混合强度）
            epochs: 训练轮数
            batch_size: 批次大小

        返回:
            history: 字典，包含训练和验证的损失/准确率历史
        """
        # 创建Mixup增强器
        mixup = Mixup(alpha=mixup_alpha)

        # 记录训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(epochs):
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            # 遍历所有batch
            for batch_X, batch_y in create_batches(X_train, y_train, batch_size):
                # ========== Mixup的核心步骤 ==========

                # 步骤1: 应用Mixup增强
                # mix_batch会:
                #   - 打乱batch内的样本顺序
                #   - 将原样本与打乱后的样本按λ混合
                #   - λ从Beta(alpha, alpha)分布采样
                mixed_X, mixed_y, lam = mixup.mix_batch(batch_X, batch_y)

                # 步骤2: 前向传播（使用混合后的图像）
                y_pred = model.forward(mixed_X)

                # 步骤3: 计算损失（使用软标签！）
                # 这里mixed_y是软标签，如[0.7, 0, 0.3, 0, ...]
                loss = soft_cross_entropy(y_pred, mixed_y)
                epoch_losses.append(loss)

                # 步骤4: 反向传播（使用软标签计算梯度）
                # 交叉熵+Softmax的梯度公式对软标签同样适用
                model.backward(mixed_y)

                # 统计准确率（用原始标签评估）
                pred_labels = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(batch_y, axis=1)  # 使用原始标签
                epoch_correct += np.sum(pred_labels == true_labels)
                epoch_total += len(batch_y)

            # 计算训练集指标
            train_loss = np.mean(epoch_losses)
            train_acc = epoch_correct / epoch_total

            # 计算验证集指标
            val_pred = model.forward(X_val)
            val_loss = soft_cross_entropy(val_pred, y_val)
            val_acc = accuracy(val_pred, y_val)

            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1:3d}/{epochs}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        return history

    def train_without_mixup(model, X_train, y_train, X_val, y_val,
                            epochs=50, batch_size=32):
        """
        标准训练（不使用Mixup）- 作为对照组

        与Mixup训练的区别:
        - 不混合图像
        - 使用硬标签（one-hot）
        - 其他完全相同

        参数和返回值与train_with_mixup相同
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(epochs):
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0

            for batch_X, batch_y in create_batches(X_train, y_train, batch_size):
                # 标准训练：直接使用原始数据
                y_pred = model.forward(batch_X)
                loss = soft_cross_entropy(y_pred, batch_y)  # batch_y是硬标签(one-hot)
                epoch_losses.append(loss)

                model.backward(batch_y)

                pred_labels = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(batch_y, axis=1)
                epoch_correct += np.sum(pred_labels == true_labels)
                epoch_total += len(batch_y)

            train_loss = np.mean(epoch_losses)
            train_acc = epoch_correct / epoch_total

            val_pred = model.forward(X_val)
            val_loss = soft_cross_entropy(val_pred, y_val)
            val_acc = accuracy(val_pred, y_val)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1:3d}/{epochs}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        return history

    print("训练函数定义完成")
    print("  - train_with_mixup: 使用Mixup增强训练")
    print("  - train_without_mixup: 标准训练（对照组）")

    # =========================================================================
    # 第5部分: 对比实验
    # =========================================================================
    print("\n🔬 第5部分: 对比实验")
    print("-" * 50)

    # 设置随机种子，确保可重复
    np.random.seed(42)

    # 创建两个相同初始化的模型
    # 这样我们可以公平地对比Mixup的效果
    model_baseline = SimpleMLP(input_size=64, hidden_size=128, output_size=10, learning_rate=0.1)
    model_mixup = model_baseline.copy()

    # 训练参数
    epochs = 100
    batch_size = 32
    mixup_alpha = 0.4  # Mixup的α参数

    print(f"\n训练配置:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: 0.1")
    print(f"  - Mixup α: {mixup_alpha}")

    # 训练标准模型（无Mixup）
    print(f"\n📊 训练标准模型（无Mixup）...")
    history_baseline = train_without_mixup(
        model_baseline, X_train, y_train_onehot, X_val, y_val_onehot,
        epochs=epochs, batch_size=batch_size
    )

    # 训练Mixup模型
    print(f"\n📊 训练Mixup模型（α={mixup_alpha}）...")
    history_mixup = train_with_mixup(
        model_mixup, X_train, y_train_onehot, X_val, y_val_onehot,
        mixup_alpha=mixup_alpha, epochs=epochs, batch_size=batch_size
    )

    # =========================================================================
    # 第6部分: 测试集评估
    # =========================================================================
    print("\n📈 第6部分: 测试集评估")
    print("-" * 50)

    # 在测试集上评估
    test_pred_baseline = model_baseline.forward(X_test)
    test_acc_baseline = accuracy(test_pred_baseline, y_test_onehot)

    test_pred_mixup = model_mixup.forward(X_test)
    test_acc_mixup = accuracy(test_pred_mixup, y_test_onehot)

    print(f"\n测试集准确率:")
    print(f"  - 标准训练: {test_acc_baseline:.4f} ({test_acc_baseline * 100:.2f}%)")
    print(f"  - Mixup训练: {test_acc_mixup:.4f} ({test_acc_mixup * 100:.2f}%)")
    print(f"  - 提升: {(test_acc_mixup - test_acc_baseline) * 100:+.2f}%")

    # =========================================================================
    # 第7部分: 可视化对比
    # =========================================================================
    print("\n📊 第7部分: 可视化对比")
    print("-" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs_range = range(1, epochs + 1)

    # 子图1: 训练损失对比
    axes[0, 0].plot(epochs_range, history_baseline['train_loss'],
                    label='标准训练', color='blue', linewidth=2)
    axes[0, 0].plot(epochs_range, history_mixup['train_loss'],
                    label='Mixup训练', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('训练损失对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 子图2: 验证损失对比
    axes[0, 1].plot(epochs_range, history_baseline['val_loss'],
                    label='标准训练', color='blue', linewidth=2)
    axes[0, 1].plot(epochs_range, history_mixup['val_loss'],
                    label='Mixup训练', color='red', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].set_title('验证损失对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 子图3: 训练准确率对比
    axes[1, 0].plot(epochs_range, history_baseline['train_acc'],
                    label='标准训练', color='blue', linewidth=2)
    axes[1, 0].plot(epochs_range, history_mixup['train_acc'],
                    label='Mixup训练', color='red', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].set_title('训练准确率对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 子图4: 验证准确率对比
    axes[1, 1].plot(epochs_range, history_baseline['val_acc'],
                    label='标准训练', color='blue', linewidth=2)
    axes[1, 1].plot(epochs_range, history_mixup['val_acc'],
                    label='Mixup训练', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].set_title('验证准确率对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Mixup数据增强效果对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 过拟合分析图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 标准训练的过拟合分析
    axes[0].plot(epochs_range, history_baseline['train_acc'],
                 label='训练准确率', color='blue', linewidth=2)
    axes[0].plot(epochs_range, history_baseline['val_acc'],
                 label='验证准确率', color='blue', linestyle='--', linewidth=2)
    axes[0].fill_between(epochs_range,
                         history_baseline['train_acc'],
                         history_baseline['val_acc'],
                         alpha=0.3, color='blue', label='过拟合差距')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('准确率')
    axes[0].set_title('标准训练 - 过拟合分析')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Mixup训练的过拟合分析
    axes[1].plot(epochs_range, history_mixup['train_acc'],
                 label='训练准确率', color='red', linewidth=2)
    axes[1].plot(epochs_range, history_mixup['val_acc'],
                 label='验证准确率', color='red', linestyle='--', linewidth=2)
    axes[1].fill_between(epochs_range,
                         history_mixup['train_acc'],
                         history_mixup['val_acc'],
                         alpha=0.3, color='red', label='过拟合差距')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('Mixup训练 - 过拟合分析')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Mixup的正则化效果分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 第8部分: 结果总结
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 实验结论")
    print("=" * 70)

    # 计算过拟合程度（训练-验证准确率差）
    overfit_baseline = history_baseline['train_acc'][-1] - history_baseline['val_acc'][-1]
    overfit_mixup = history_mixup['train_acc'][-1] - history_mixup['val_acc'][-1]

    print(f"\n1. 测试集准确率:")
    print(f"   - 标准训练: {test_acc_baseline * 100:.2f}%")
    print(f"   - Mixup训练: {test_acc_mixup * 100:.2f}%")

    print(f"\n2. 过拟合程度（训练-验证准确率差）:")
    print(f"   - 标准训练: {overfit_baseline * 100:.2f}%")
    print(f"   - Mixup训练: {overfit_mixup * 100:.2f}%")

    print(f"\n3. Mixup的正则化效果:")
    if test_acc_mixup > test_acc_baseline:
        print(f"   ✓ Mixup提升了测试准确率 {(test_acc_mixup - test_acc_baseline) * 100:.2f}%")
    if overfit_mixup < overfit_baseline:
        print(f"   ✓ Mixup减少了过拟合程度 {(overfit_baseline - overfit_mixup) * 100:.2f}%")

    print(f"\n4. 关键洞察:")
    print("   - Mixup通过混合样本创建虚拟训练数据，增加了数据多样性")
    print("   - 软标签训练鼓励模型产生更平滑的决策边界")
    print("   - Mixup是一种有效的数据增强和正则化技术")
    print("   - 适用于各种图像分类任务，实现简单，效果显著")

    print("\n" + "=" * 70)
    print("练习3完成！")
    print("=" * 70)

    return {
        'model_baseline': model_baseline,
        'model_mixup': model_mixup,
        'history_baseline': history_baseline,
        'history_mixup': history_mixup,
        'test_acc_baseline': test_acc_baseline,
        'test_acc_mixup': test_acc_mixup
    }


results = exercise_3_mixup_training()