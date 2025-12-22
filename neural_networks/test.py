# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# 设置随机种子，确保结果可复现
np.random.seed(42)

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = [
        'Arial Unicode MS',  # macOS通用
        'PingFang SC',       # macOS系统字体
        'STHeiti',           # 华文黑体
        'Heiti TC',          # 黑体-繁
        'SimHei',            # 黑体
    ]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
np.random.seed(42)


def cross_entropy_loss(y_pred, y_true):
    """
    交叉熵损失函数

    参数:
        y_pred: 预测概率, shape (n_samples, n_classes)
        y_true: 真实标签（one-hot编码）, shape (n_samples, n_classes)

    返回:
        loss: 平均损失（标量）
    """
    m = y_pred.shape[0]
    # 避免log(0)
    epsilon = 1e-8
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

    # 计算交叉熵
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
    return loss


def one_hot_encode(y, n_classes):
    """
    将类别标签转换为one-hot编码

    参数:
        y: 类别标签, shape (n_samples,)
        n_classes: 类别总数

    返回:
        one_hot: shape (n_samples, n_classes)

    示例:
        y = [0, 1, 2]
        one_hot_encode(y, 3) = [[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]]
    """
    m = len(y)
    one_hot = np.zeros((m, n_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

# 激活函数
def sigmoid(z):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """
    Sigmoid的导数
    σ'(z) = σ(z) * (1 - σ(z))
    """
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    """ReLU激活函数"""
    return np.maximum(0, z)

def relu_derivative(z):
    """
    ReLU的导数
    ReLU'(z) = 1 if z > 0 else 0
    """
    return (z > 0).astype(float)

def tanh(z):
    """Tanh激活函数"""
    return np.tanh(z)

def tanh_derivative(z):
    """
    Tanh的导数
    tanh'(z) = 1 - tanh(z)^2
    """
    t = np.tanh(z)
    return 1 - t**2

def softmax(z):
    """Softmax激活函数"""
    if z.ndim == 1:
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
    else:
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class MLPWithBackprop:
    """
    多层感知机（支持训练）

    实现了完整的前向传播、反向传播和参数更新。

    参数:
        layer_sizes: list, 每层神经元数量 [n_input, n_hidden1, ..., n_output]
        activation: str, 隐藏层激活函数 ('sigmoid', 'relu', 'tanh')
        learning_rate: float, 学习率
        random_state: int, 随机种子
    """

    def __init__(self, layer_sizes, activation='relu',
                 learning_rate=0.01, random_state=42):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.random_state = random_state

        # 选择激活函数及其导数
        activations = {
            'sigmoid': (sigmoid, sigmoid_derivative),
            'relu': (relu, relu_derivative),
            'tanh': (tanh, tanh_derivative)
        }
        self.activation, self.activation_derivative = activations[activation]

        # 初始化参数
        self._initialize_parameters()

        # 存储训练历史
        self.loss_history = []
        self.accuracy_history = []

    def _initialize_parameters(self):
        """初始化权重和偏置"""
        np.random.seed(self.random_state)

        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            # He初始化（适用于ReLU）
            if self.activation_name == 'relu':
                std = np.sqrt(2.0 / n_in)
            else:
                std = np.sqrt(1.0 / n_in)

            W = np.random.randn(n_out, n_in) * std
            b = np.zeros(n_out)

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        前向传播

        参数:
            X: 输入数据, shape (n_samples, n_features)

        返回:
            output: 网络输出, shape (n_samples, n_output)
        """
        # 存储中间值（供反向传播使用）
        self.activations = [X]
        self.z_values = []

        A = X

        # 隐藏层
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i].T + self.biases[i]
            A = self.activation(Z)

            self.z_values.append(Z)
            self.activations.append(A)

        # 输出层（Softmax）
        Z = A @ self.weights[-1].T + self.biases[-1]
        output = softmax(Z)

        self.z_values.append(Z)
        self.activations.append(output)

        return output

    def backward(self, X, y_true):
        """
        反向传播

        参数:
            X: 输入数据, shape (n_samples, n_features)
            y_true: 真实标签（one-hot编码）, shape (n_samples, n_classes)

        返回:
            grad_weights: list of arrays, 每层权重的梯度
            grad_biases: list of arrays, 每层偏置的梯度
        """
        m = X.shape[0]  # 样本数量
        n_layers = len(self.weights)

        grad_weights = [None] * n_layers
        grad_biases = [None] * n_layers

        # ===== 输出层梯度 =====
        # 对于交叉熵+Softmax，梯度简化为 y_pred - y_true
        delta = self.activations[-1] - y_true  # shape: (m, n_output)

        # 输出层权重和偏置的梯度
        grad_weights[-1] = delta.T @ self.activations[-2] / m  # (n_output, n_hidden)
        grad_biases[-1] = np.mean(delta, axis=0)  # (n_output,)

        # ===== 隐藏层梯度（从后向前） =====
        for l in range(n_layers - 2, -1, -1):
            # 1. 将误差反向传播到前一层
            #    delta_prev = (delta @ W^(l+1)) ⊙ σ'(z^(l))
            delta = (delta @ self.weights[l + 1]) * self.activation_derivative(self.z_values[l])
            # shape: (m, n_l)

            # 2. 计算当前层的权重和偏置梯度
            grad_weights[l] = delta.T @ self.activations[l] / m  # (n_l, n_{l-1})
            grad_biases[l] = np.mean(delta, axis=0)  # (n_l,)

        return grad_weights, grad_biases

    def update_parameters(self, grad_weights, grad_biases):
        """
        更新参数（梯度下降）

        参数:
            grad_weights: list of arrays, 权重梯度
            grad_biases: list of arrays, 偏置梯度
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]

    def train_step(self, X, y):
        """
        一次训练步骤（前向+反向+更新）

        参数:
            X: 输入数据, shape (n_samples, n_features)
            y: 真实标签, shape (n_samples,)

        返回:
            loss: 当前损失
            accuracy: 当前准确率
        """
        # One-hot编码
        n_classes = self.layer_sizes[-1]
        y_one_hot = one_hot_encode(y, n_classes)

        # 1. 前向传播
        y_pred = self.forward(X)

        # 2. 计算损失和准确率
        loss = cross_entropy_loss(y_pred, y_one_hot)
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y)

        # 3. 反向传播
        grad_weights, grad_biases = self.backward(X, y_one_hot)

        # 4. 更新参数
        self.update_parameters(grad_weights, grad_biases)

        return loss, accuracy

    def fit(self, X, y, epochs=100, batch_size=32,
            X_val=None, y_val=None, verbose=True):
        """
        训练网络

        参数:
            X: 训练数据, shape (n_samples, n_features)
            y: 训练标签, shape (n_samples,)
            epochs: 训练轮数
            batch_size: 批量大小
            X_val: 验证数据（可选）
            y_val: 验证标签（可选）
            verbose: 是否打印训练信息
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch训练
            epoch_loss = 0
            epoch_accuracy = 0

            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                loss, accuracy = self.train_step(X_batch, y_batch)
                epoch_loss += loss
                epoch_accuracy += accuracy

            # 平均损失和准确率
            epoch_loss /= n_batches
            epoch_accuracy /= n_batches

            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)

            # 打印训练信息
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}"

                # 如果提供了验证集，计算验证集性能
                if X_val is not None and y_val is not None:
                    val_pred = self.predict(X_val)
                    val_accuracy = np.mean(val_pred == y_val)
                    msg += f", Val Acc: {val_accuracy:.4f}"

                print(msg)

    def predict(self, X):
        """
        预测类别

        参数:
            X: 输入数据, shape (n_samples, n_features)

        返回:
            predictions: 预测的类别标签, shape (n_samples,)
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)


print("="*60)
print("实战项目：训练神经网络解决月亮数据集分类")
print("="*60)

# 1. 数据准备
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\n数据集信息：")
print(f"  训练集: {X_train.shape}")
print(f"  测试集: {X_test.shape}")

# 2. 创建网络
print("\n" + "-"*60)
print("构建神经网络: [2, 16, 8, 2]")
print("-"*60)

mlp = MLPWithBackprop(
    layer_sizes=[2, 16, 8, 2],
    activation='relu',
    learning_rate=0.1,
    random_state=42
)

# 3. 训练
print("\n" + "-"*60)
print("开始训练...")
print("-"*60)

mlp.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    X_val=X_test,
    y_val=y_test,
    verbose=True
)

# 4. 评估
print("\n" + "="*60)
print("最终评估")
print("="*60)

train_pred = mlp.predict(X_train)
test_pred = mlp.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"\n训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

print("\n混淆矩阵（测试集）：")
print(confusion_matrix(y_test, test_pred))