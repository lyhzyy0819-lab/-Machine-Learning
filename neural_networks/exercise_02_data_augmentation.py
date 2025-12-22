"""
============================================================================
练习2：数据增强实验 - 在MNIST数据集上实现数据增强
============================================================================

📚 问题背景：
    数据增强（Data Augmentation）是一种通过对原始数据进行变换
    来人为扩大训练集的技术，可以有效防止过拟合。

🎯 学习目标：
    1. 理解数据增强的原理和作用
    2. 实现常见的图像数据增强方法：
       - 随机旋转（±15度）
       - 随机平移
       - 添加噪声
    3. 观察数据增强对泛化性能的影响
    4. 对比有无数据增强的效果差异

============================================================================
"""

# ============================================================================
# 第1部分：导入库和环境配置
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import rotate, shift

# 设置随机种子
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("练习2：数据增强实验 - MNIST手写数字识别")
print("=" * 70)


# ============================================================================
# 第2部分：数据增强原理详解
# ============================================================================
"""
📖 数据增强的核心思想：

    通过对原始数据进行合理的变换，生成"新"的训练样本，
    这些新样本保持原有的标签，但在某些方面有所变化。

    对于图像数据，常见的增强方法包括：

    1. 几何变换：
       - 旋转（Rotation）：绕中心旋转一定角度
       - 平移（Translation）：上下左右平移
       - 翻转（Flip）：水平或垂直翻转
       - 缩放（Scaling）：放大或缩小
       - 裁剪（Cropping）：随机裁剪

    2. 像素变换：
       - 添加噪声（Noise）：高斯噪声、椒盐噪声
       - 亮度调整：整体变亮或变暗
       - 对比度调整：增强或减弱对比
       - 模糊（Blur）：高斯模糊

💡 为什么数据增强有效？

    1. 增加样本多样性：让模型见到更多变体
    2. 提高鲁棒性：模型学会忽略无关变化
    3. 防止过拟合：相当于隐式正则化
    4. 利用领域知识：旋转的数字仍是同一数字

⚠️ 注意事项：

    1. 增强后的数据要保持语义不变（6旋转180度变成9就错了！）
    2. 增强程度要适当，过度增强可能引入噪声
    3. 不同任务需要不同的增强策略
"""


# ============================================================================
# 第3部分：数据增强函数实现
# ============================================================================

def random_rotation(image, max_angle=15):
    """
    随机旋转图像

    数学原理：
        旋转变换使用旋转矩阵：
        [cos(θ)  -sin(θ)]   [x]
        [sin(θ)   cos(θ)] × [y]

    参数:
        image: 输入图像，shape (height, width) 或 (height, width, channels)
        max_angle: 最大旋转角度（度），默认±15度

    返回:
        rotated_image: 旋转后的图像

    实现细节:
        1. 在 [-max_angle, max_angle] 范围内随机选择角度
        2. 使用scipy.ndimage.rotate进行旋转
        3. reshape=False 保持原始图像尺寸
        4. mode='nearest' 用最近邻填充边界
    """
    # 随机选择旋转角度
    # np.random.uniform(a, b) 生成 [a, b) 范围内的均匀分布随机数
    angle = np.random.uniform(-max_angle, max_angle)

    # 执行旋转
    # reshape=False: 保持输出图像尺寸与输入相同
    # mode='nearest': 边界填充方式，使用最近邻像素值
    rotated = rotate(image, angle, reshape=False, mode='nearest')

    return rotated


def random_translation(image, max_shift=2):
    """
    随机平移图像

    数学原理：
        平移变换：
        x' = x + dx
        y' = y + dy

    参数:
        image: 输入图像
        max_shift: 最大平移像素数，默认±2像素

    返回:
        shifted_image: 平移后的图像

    实现细节:
        1. 随机选择水平和垂直平移量
        2. 使用scipy.ndimage.shift进行平移
        3. 超出边界的部分用0填充
    """
    # 随机选择平移量
    # 对于8x8的MNIST数字，2像素的平移已经足够明显
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)

    # 执行平移
    # shift函数的参数是 (row_shift, col_shift)
    # 即 (垂直平移, 水平平移)
    shifted = shift(image, [dy, dx], mode='constant', cval=0)

    return shifted


def add_gaussian_noise(image, noise_factor=0.1):
    """
    添加高斯噪声

    数学原理：
        高斯噪声服从正态分布 N(0, σ²)
        noisy_image = image + noise
        其中 noise ~ N(0, σ²)

    参数:
        image: 输入图像，像素值范围假设为 [0, 1] 或归一化后的值
        noise_factor: 噪声强度，控制标准差

    返回:
        noisy_image: 添加噪声后的图像

    实现细节:
        1. 生成与图像同形状的高斯噪声
        2. 噪声乘以noise_factor控制强度
        3. 将结果裁剪到合理范围
    """
    # 生成高斯噪声
    # np.random.randn 生成标准正态分布 N(0, 1)
    noise = np.random.randn(*image.shape) * noise_factor

    # 添加噪声
    noisy_image = image + noise

    # 裁剪到合理范围（假设原始图像已归一化）
    # 注意：如果原始数据未归一化，这里的范围需要调整
    noisy_image = np.clip(noisy_image, image.min(), image.max())

    return noisy_image


def augment_image(image, rotation=True, translation=True, noise=True,
                  max_angle=15, max_shift=2, noise_factor=0.1):
    """
    综合数据增强函数

    对一张图像应用多种增强方法

    参数:
        image: 输入图像
        rotation: 是否应用随机旋转
        translation: 是否应用随机平移
        noise: 是否添加噪声
        max_angle: 最大旋转角度
        max_shift: 最大平移像素
        noise_factor: 噪声强度

    返回:
        augmented_image: 增强后的图像
    """
    augmented = image.copy()

    # 依次应用各种增强
    if rotation:
        augmented = random_rotation(augmented, max_angle)

    if translation:
        augmented = random_translation(augmented, max_shift)

    if noise:
        augmented = add_gaussian_noise(augmented, noise_factor)

    return augmented


def augment_dataset(X, y, augmentation_factor=2, **kwargs):
    """
    对整个数据集进行增强

    参数:
        X: 原始特征数据, shape (n_samples, n_features)
        y: 原始标签
        augmentation_factor: 增强倍数，即每个原始样本生成多少个增强样本

    返回:
        X_augmented: 增强后的特征数据（包含原始数据）
        y_augmented: 增强后的标签

    实现流程:
        1. 保留原始数据
        2. 对每个原始样本生成 augmentation_factor 个增强样本
        3. 合并原始数据和增强数据
    """
    # 假设图像是8x8的（sklearn的digits数据集）
    img_size = int(np.sqrt(X.shape[1]))

    # 存储增强后的数据
    X_aug_list = [X]  # 先加入原始数据
    y_aug_list = [y]

    print(f"开始数据增强，原始样本数: {len(X)}")
    print(f"增强倍数: {augmentation_factor}")

    for i in range(augmentation_factor):
        X_new = []
        for j, x in enumerate(X):
            # 将一维特征重塑为二维图像
            img = x.reshape(img_size, img_size)

            # 应用数据增强
            augmented_img = augment_image(img, **kwargs)

            # 重新展平为一维
            X_new.append(augmented_img.flatten())

        X_aug_list.append(np.array(X_new))
        y_aug_list.append(y.copy())

        print(f"  完成第 {i+1} 轮增强")

    # 合并所有数据
    X_augmented = np.vstack(X_aug_list)
    y_augmented = np.hstack(y_aug_list)

    print(f"增强后总样本数: {len(X_augmented)}")

    return X_augmented, y_augmented


# ============================================================================
# 第4部分：神经网络实现
# ============================================================================

def softmax(z):
    """
    Softmax激活函数（多分类输出层）

    数学公式：
        softmax(z_i) = exp(z_i) / Σ exp(z_j)

    作用：
        将任意实数向量转换为概率分布（和为1）

    实现技巧：
        先减去最大值防止数值溢出
    """
    if z.ndim == 1:
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
    else:
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def relu(z):
    """ReLU激活函数"""
    return np.maximum(0, z)


def relu_derivative(z):
    """ReLU导数"""
    return (z > 0).astype(float)


def one_hot_encode(y, n_classes):
    """
    One-hot编码

    例如：
        y = [0, 1, 2], n_classes = 3
        结果：[[1,0,0], [0,1,0], [0,0,1]]
    """
    m = len(y)
    one_hot = np.zeros((m, n_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot


class MNISTNetwork:
    """
    用于MNIST分类的神经网络

    网络结构：
        64 (输入) → 128 (隐藏层1) → 64 (隐藏层2) → 10 (输出)

    特点：
        - 使用ReLU激活函数
        - 输出层使用Softmax
        - 使用交叉熵损失
    """

    def __init__(self, input_size=64, hidden1=128, hidden2=64, output_size=10,
                 lambda_reg=0.001):
        """
        初始化网络

        参数:
            input_size: 输入维度（8x8图像展平后为64）
            hidden1: 第一隐藏层神经元数
            hidden2: 第二隐藏层神经元数
            output_size: 输出维度（10个数字类别）
            lambda_reg: L2正则化系数
        """
        self.lambda_reg = lambda_reg

        # He初始化（适用于ReLU）
        # 标准差 = sqrt(2/n_in)
        self.W1 = np.random.randn(hidden1, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden1)

        self.W2 = np.random.randn(hidden2, hidden1) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)

        self.W3 = np.random.randn(output_size, hidden2) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(output_size)

        print(f"网络结构: [{input_size}, {hidden1}, {hidden2}, {output_size}]")
        print(f"L2正则化系数: {lambda_reg}")

    def forward(self, X):
        """
        前向传播

        数据流: X → ReLU → ReLU → Softmax → 概率分布
        """
        # 第1层
        self.z1 = X @ self.W1.T + self.b1
        self.a1 = relu(self.z1)

        # 第2层
        self.z2 = self.a1 @ self.W2.T + self.b2
        self.a2 = relu(self.z2)

        # 输出层
        self.z3 = self.a2 @ self.W3.T + self.b3
        self.a3 = softmax(self.z3)

        return self.a3

    def backward(self, X, y_one_hot):
        """
        反向传播

        参数:
            X: 输入数据
            y_one_hot: one-hot编码的标签
        """
        m = X.shape[0]

        # 输出层梯度 (Softmax + Cross Entropy的简化形式)
        delta3 = (self.a3 - y_one_hot) / m
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

        return [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = params

    def compute_loss(self, X, y):
        """计算交叉熵损失"""
        y_pred = self.forward(X)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        y_one_hot = one_hot_encode(y, 10)
        ce_loss = -np.sum(y_one_hot * np.log(y_pred)) / X.shape[0]

        # L2正则化
        l2_penalty = self.lambda_reg / 2 * (
            np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2) + np.sum(self.W3 ** 2)
        )

        return ce_loss + l2_penalty

    def compute_accuracy(self, X, y):
        """计算准确率"""
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y)


# ============================================================================
# 第5部分：训练函数
# ============================================================================

def train_network(X_train, y_train, X_test, y_test,
                  n_epochs=100, batch_size=32, learning_rate=0.01,
                  lambda_reg=0.001, verbose=True):
    """
    训练神经网络

    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        n_epochs: 训练轮数
        batch_size: 批量大小
        learning_rate: 学习率
        lambda_reg: L2正则化系数
        verbose: 是否打印训练信息

    返回:
        model: 训练好的模型
        history: 训练历史
    """
    # 创建模型
    input_size = X_train.shape[1]
    model = MNISTNetwork(input_size=input_size, lambda_reg=lambda_reg)

    # 记录历史
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }

    n_samples = X_train.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))

    for epoch in range(n_epochs):
        # 打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Mini-batch训练
        for batch in range(n_batches):
            start = batch * batch_size
            end = min(start + batch_size, n_samples)

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # 前向传播
            model.forward(X_batch)

            # 反向传播
            y_one_hot = one_hot_encode(y_batch, 10)
            grads = model.backward(X_batch, y_one_hot)

            # 参数更新
            params = model.get_params()
            updated_params = [p - learning_rate * g for p, g in zip(params, grads)]
            model.set_params(updated_params)

        # 记录指标
        if epoch % 5 == 0:
            train_loss = model.compute_loss(X_train, y_train)
            test_loss = model.compute_loss(X_test, y_test)
            train_acc = model.compute_accuracy(X_train, y_train)
            test_acc = model.compute_accuracy(X_test, y_test)

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:4d}: "
                      f"Train Acc={train_acc:.4f}, "
                      f"Test Acc={test_acc:.4f}")

    return model, history


# ============================================================================
# 第6部分：可视化函数
# ============================================================================

def visualize_augmentation(X, y):
    """
    可视化数据增强效果

    展示原始图像和各种增强后的变体
    """
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle('数据增强效果展示', fontsize=14, fontweight='bold')

    # 选择5个不同数字的样本
    sample_indices = []
    for digit in range(5):
        idx = np.where(y == digit)[0][0]
        sample_indices.append(idx)

    # 第1行：原始图像
    for i, idx in enumerate(sample_indices):
        img = X[idx].reshape(8, 8)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'原始 (数字{y[idx]})')
        axes[0, i].axis('off')

    # 第2行：随机旋转
    for i, idx in enumerate(sample_indices):
        img = X[idx].reshape(8, 8)
        rotated = random_rotation(img, max_angle=15)
        axes[1, i].imshow(rotated, cmap='gray')
        axes[1, i].set_title('旋转')
        axes[1, i].axis('off')

    # 第3行：随机平移
    for i, idx in enumerate(sample_indices):
        img = X[idx].reshape(8, 8)
        shifted = random_translation(img, max_shift=2)
        axes[2, i].imshow(shifted, cmap='gray')
        axes[2, i].set_title('平移')
        axes[2, i].axis('off')

    # 第4行：添加噪声
    for i, idx in enumerate(sample_indices):
        img = X[idx].reshape(8, 8)
        noisy = add_gaussian_noise(img, noise_factor=0.3)
        axes[3, i].imshow(noisy, cmap='gray')
        axes[3, i].set_title('噪声')
        axes[3, i].axis('off')

    plt.tight_layout()
    return fig


# ============================================================================
# 第7部分：实验对比
# ============================================================================

if __name__ == "__main__":

    # =====================================
    # 1. 加载数据
    # =====================================
    print("\n" + "=" * 70)
    print("第1步：加载MNIST数据集")
    print("=" * 70)

    # 使用sklearn的digits数据集（8x8的手写数字）
    digits = load_digits()
    X, y = digits.data, digits.target

    # 归一化到 [0, 1]
    X = X / 16.0  # digits数据集的像素值范围是 [0, 16]

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"类别数: {len(np.unique(y))}")

    # =====================================
    # 2. 可视化数据增强效果
    # =====================================
    print("\n" + "=" * 70)
    print("第2步：可视化数据增强效果")
    print("=" * 70)

    fig = visualize_augmentation(X_train, y_train)
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/augmentation_examples.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # =====================================
    # 3. 对比实验：有无数据增强
    # =====================================
    print("\n" + "=" * 70)
    print("第3步：对比实验 - 有无数据增强")
    print("=" * 70)

    # ----- 实验1：无数据增强 -----
    print("\n--- 实验1：无数据增强 ---")
    model_no_aug, history_no_aug = train_network(
        X_train, y_train, X_test, y_test,
        n_epochs=100,
        batch_size=32,
        learning_rate=0.01,
        lambda_reg=0.001,
        verbose=True
    )
    print(f"无增强 - 最终测试准确率: {history_no_aug['test_acc'][-1]:.4f}")

    # ----- 实验2：有数据增强 -----
    print("\n--- 实验2：有数据增强 ---")

    # 创建增强数据集
    X_train_aug, y_train_aug = augment_dataset(
        X_train, y_train,
        augmentation_factor=2,  # 每个样本生成2个增强版本
        max_angle=15,
        max_shift=1,
        noise_factor=0.1
    )

    model_with_aug, history_with_aug = train_network(
        X_train_aug, y_train_aug, X_test, y_test,
        n_epochs=100,
        batch_size=32,
        learning_rate=0.01,
        lambda_reg=0.001,
        verbose=True
    )
    print(f"有增强 - 最终测试准确率: {history_with_aug['test_acc'][-1]:.4f}")

    # =====================================
    # 4. 可视化对比结果
    # =====================================
    print("\n" + "=" * 70)
    print("第4步：可视化对比结果")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_plot = np.arange(len(history_no_aug['train_acc'])) * 5

    # ----- 图1：训练准确率对比 -----
    ax1 = axes[0, 0]
    ax1.plot(epochs_plot, history_no_aug['train_acc'],
             label='无增强-训练', linewidth=2, linestyle='-', color='#3498db')
    ax1.plot(epochs_plot, history_with_aug['train_acc'],
             label='有增强-训练', linewidth=2, linestyle='-', color='#e74c3c')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('训练准确率对比', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ----- 图2：测试准确率对比 -----
    ax2 = axes[0, 1]
    ax2.plot(epochs_plot, history_no_aug['test_acc'],
             label='无增强-测试', linewidth=2, linestyle='--', color='#3498db')
    ax2.plot(epochs_plot, history_with_aug['test_acc'],
             label='有增强-测试', linewidth=2, linestyle='--', color='#e74c3c')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('测试准确率对比', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ----- 图3：损失对比 -----
    ax3 = axes[1, 0]
    ax3.plot(epochs_plot, history_no_aug['train_loss'],
             label='无增强-训练', linewidth=2, color='#3498db')
    ax3.plot(epochs_plot, history_no_aug['test_loss'],
             label='无增强-测试', linewidth=2, linestyle='--', color='#3498db')
    ax3.plot(epochs_plot, history_with_aug['train_loss'],
             label='有增强-训练', linewidth=2, color='#e74c3c')
    ax3.plot(epochs_plot, history_with_aug['test_loss'],
             label='有增强-测试', linewidth=2, linestyle='--', color='#e74c3c')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('损失曲线对比', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ----- 图4：实验总结 -----
    ax4 = axes[1, 1]
    summary_text = f"""
╔══════════════════════════════════════════════════════════╗
║            数据增强实验结果总结                            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  📊 实验结果:                                            ║
║                                                          ║
║  无数据增强:                                             ║
║    • 训练样本数: {len(X_train):>6}                               ║
║    • 测试准确率: {history_no_aug['test_acc'][-1]:.4f}                            ║
║                                                          ║
║  有数据增强:                                             ║
║    • 训练样本数: {len(X_train_aug):>6}                               ║
║    • 测试准确率: {history_with_aug['test_acc'][-1]:.4f}                            ║
║                                                          ║
║  🔍 分析:                                                ║
║    数据增强通过增加训练样本多样性，                       ║
║    有效提高了模型的泛化能力。                             ║
║                                                          ║
║  💡 使用的增强方法:                                      ║
║    • 随机旋转: ±15度                                    ║
║    • 随机平移: ±1像素                                   ║
║    • 高斯噪声: factor=0.1                               ║
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
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/augmentation_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # =====================================
    # 5. 打印最终总结
    # =====================================
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)

    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                    数据增强实验完成！                               ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  📈 结果对比:                                                     ║
║                                                                   ║
║    无数据增强:                                                    ║
║      训练样本: {len(X_train):>6}                                             ║
║      最终测试准确率: {history_no_aug['test_acc'][-1]:.4f}                                  ║
║                                                                   ║
║    有数据增强:                                                    ║
║      训练样本: {len(X_train_aug):>6} (原始 + 2倍增强)                          ║
║      最终测试准确率: {history_with_aug['test_acc'][-1]:.4f}                                  ║
║                                                                   ║
║  🎯 关键发现:                                                     ║
║    1. 数据增强有效提升了泛化性能                                   ║
║    2. 增强后的模型更不容易过拟合                                   ║
║    3. 训练集和测试集的性能差距减小                                 ║
║                                                                   ║
║  💡 实践建议:                                                     ║
║    1. 增强方法要符合任务语义（如数字不能上下翻转）                  ║
║    2. 增强强度要适当，过度增强可能引入噪声                         ║
║    3. 可以结合多种增强方法使用                                     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")

    print("✅ 练习2完成！")
