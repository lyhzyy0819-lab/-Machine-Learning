"""
项目1：中文微博情绪识别系统（4分类）

业务场景：情感陪伴AI智能玩具 - 识别用户对话中的情绪状态
数据集：simplifyweibo_4_moods（36万条中文微博，4类情绪）

知识点覆盖：
- MLP多层感知机（前向传播、反向传播）
- 损失函数（CrossEntropy、Focal Loss处理不平衡）
- 优化器（Adam）
- 正则化（Dropout、Early Stopping）
- 权重初始化（He初始化）
- 学习率调度

数据集下载：
1. 访问 https://github.com/SophonPlus/ChineseNlpCorpus
2. 下载 simplifyweibo_4_moods.csv
3. 放到 neural_networks/data/ 目录下

如果没有数据集，程序会使用模拟数据进行演示
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# =============================================================================
# 第一部分：文本预处理工具
# =============================================================================

def simple_tokenize(text):
    """
    简单的中文分词（不依赖jieba，用于演示）

    实际生产环境应使用jieba：
        import jieba
        return list(jieba.cut(text))
    """
    # 移除特殊字符，保留中文和英文
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', str(text))
    # 简单按字符分割（实际应用中应使用jieba分词）
    tokens = list(text.replace(' ', ''))
    return tokens


def build_vocabulary(texts, max_features=5000):
    """
    构建词汇表

    参数:
        texts: 文本列表
        max_features: 词汇表大小上限

    返回:
        word_to_idx: 词到索引的映射
    """
    # 统计词频
    word_counts = Counter()
    for text in texts:
        tokens = simple_tokenize(text)
        word_counts.update(tokens)

    # 取频率最高的词
    most_common = word_counts.most_common(max_features)

    # 构建词汇表（索引从1开始，0留给未知词）
    word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}

    print(f"词汇表大小: {len(word_to_idx)}")
    print(f"示例词汇: {list(word_to_idx.keys())[:10]}")

    return word_to_idx


def text_to_tfidf(texts, word_to_idx):
    """
    将文本转换为TF-IDF特征向量（简化版本）

    TF-IDF = TF(词频) × IDF(逆文档频率)
    - TF: 词在文档中出现的次数 / 文档总词数
    - IDF: log(文档总数 / 包含该词的文档数 + 1)

    参数:
        texts: 文本列表
        word_to_idx: 词汇表

    返回:
        tfidf_matrix: shape (n_samples, vocab_size)
    """
    n_samples = len(texts)
    vocab_size = len(word_to_idx)

    # 计算每个词的文档频率（DF）
    doc_freq = np.zeros(vocab_size)

    # 先统计DF
    for text in texts:
        tokens = set(simple_tokenize(text))  # 每个文档中词只计一次
        for token in tokens:
            if token in word_to_idx:
                idx = word_to_idx[token] - 1
                doc_freq[idx] += 1

    # 计算IDF: log(N / (df + 1)) + 1，+1避免除零
    idf = np.log(n_samples / (doc_freq + 1)) + 1

    # 构建TF-IDF矩阵
    tfidf_matrix = np.zeros((n_samples, vocab_size))

    for i, text in enumerate(texts):
        tokens = simple_tokenize(text)
        token_counts = Counter(tokens)
        total_tokens = len(tokens) if tokens else 1

        for token, count in token_counts.items():
            if token in word_to_idx:
                idx = word_to_idx[token] - 1
                # TF-IDF = (词频 / 总词数) × IDF
                tf = count / total_tokens
                tfidf_matrix[i, idx] = tf * idf[idx]

    # L2归一化（每个样本向量归一化到单位长度）
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # 避免除零
    tfidf_matrix = tfidf_matrix / norms

    return tfidf_matrix


# =============================================================================
# 第二部分：神经网络模型（从零实现）
# =============================================================================

def he_init(fan_in, fan_out):
    """
    He初始化（适用于ReLU激活函数）

    公式: W ~ N(0, sqrt(2/fan_in))

    参数:
        fan_in: 输入维度
        fan_out: 输出维度

    返回:
        W: 权重矩阵, shape (fan_in, fan_out)
    """
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std


def relu(x):
    """ReLU激活函数: max(0, x)"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU导数: 1 if x > 0 else 0"""
    return (x > 0).astype(float)


def softmax(x):
    """
    Softmax函数（数值稳定版本）

    将原始分数转换为概率分布
    公式: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    """
    # 减去最大值防止数值溢出
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """
    交叉熵损失函数

    公式: L = -sum(y_true * log(y_pred))

    参数:
        y_pred: 预测概率, shape (batch_size, num_classes)
        y_true: 真实标签(one-hot), shape (batch_size, num_classes)

    返回:
        loss: 平均损失
    """
    # 裁剪防止log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    return loss


def focal_loss(y_pred, y_true, gamma=2.0, alpha=None):
    """
    Focal Loss（处理类别不平衡）

    公式: FL = -alpha * (1 - p_t)^gamma * log(p_t)

    核心思想：
    - 对于容易分类的样本（p_t接近1），(1-p_t)^gamma接近0，降低损失权重
    - 对于难分类的样本（p_t接近0），(1-p_t)^gamma接近1，保持正常损失

    参数:
        y_pred: 预测概率, shape (batch_size, num_classes)
        y_true: 真实标签(one-hot), shape (batch_size, num_classes)
        gamma: 聚焦参数，gamma越大，对难样本的关注越强
        alpha: 类别权重，可用于进一步平衡类别

    返回:
        loss: 平均Focal Loss
    """
    # 裁剪防止log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # 计算 p_t（真实类别对应的预测概率）
    p_t = np.sum(y_pred * y_true, axis=1)

    # 计算调制因子 (1 - p_t)^gamma
    modulating_factor = (1 - p_t) ** gamma

    # 计算Focal Loss
    focal = -modulating_factor * np.log(p_t)

    # 如果有类别权重，应用它
    if alpha is not None:
        # alpha是每个类别的权重数组
        class_weights = np.sum(y_true * alpha, axis=1)
        focal = focal * class_weights

    return np.mean(focal)


class EmotionMLP:
    """
    情绪识别多层感知机

    网络结构: Input -> Dense(256) -> ReLU -> Dropout -> Dense(128) -> ReLU -> Dropout -> Dense(4) -> Softmax

    涵盖知识点：
    - He权重初始化
    - ReLU激活函数
    - Dropout正则化
    - Softmax输出层
    - 交叉熵/Focal Loss损失
    """

    def __init__(self, input_dim, hidden_dims=[256, 128], num_classes=4, dropout_rate=0.3):
        """
        初始化网络

        参数:
            input_dim: 输入特征维度（TF-IDF向量维度）
            hidden_dims: 隐藏层维度列表
            num_classes: 分类数（4种情绪）
            dropout_rate: Dropout概率
        """
        self.dropout_rate = dropout_rate
        self.training = True  # 训练/推理模式标志

        # 构建层维度列表: [input_dim, 256, 128, 4]
        layer_dims = [input_dim] + hidden_dims + [num_classes]

        # 初始化权重和偏置（使用He初始化）
        self.weights = []
        self.biases = []

        print("\n网络结构:")
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]

            W = he_init(fan_in, fan_out)
            b = np.zeros((1, fan_out))

            self.weights.append(W)
            self.biases.append(b)

            print(f"  Layer {i+1}: {fan_in} -> {fan_out}")

        # 缓存（用于反向传播）
        self.cache = {}

    def forward(self, X):
        """
        前向传播

        参数:
            X: 输入特征, shape (batch_size, input_dim)

        返回:
            output: 预测概率, shape (batch_size, num_classes)
        """
        self.cache['input'] = X
        A = X

        # 隐藏层（ReLU激活 + Dropout）
        for i in range(len(self.weights) - 1):
            # 线性变换: Z = A @ W + b
            Z = A @ self.weights[i] + self.biases[i]
            self.cache[f'Z{i}'] = Z

            # ReLU激活
            A = relu(Z)
            self.cache[f'A{i}'] = A

            # Dropout（仅训练时）
            if self.training and self.dropout_rate > 0:
                # 创建Dropout掩码
                mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
                # 应用掩码并缩放（保持期望值不变）
                A = A * mask / (1 - self.dropout_rate)
                self.cache[f'dropout_mask{i}'] = mask

        # 输出层（Softmax）
        Z_out = A @ self.weights[-1] + self.biases[-1]
        self.cache['Z_out'] = Z_out
        output = softmax(Z_out)
        self.cache['output'] = output

        return output

    def backward(self, y_true, loss_type='cross_entropy', gamma=2.0):
        """
        反向传播

        参数:
            y_true: 真实标签(one-hot), shape (batch_size, num_classes)
            loss_type: 'cross_entropy' 或 'focal'
            gamma: Focal Loss的gamma参数

        返回:
            grads_w: 权重梯度列表
            grads_b: 偏置梯度列表
        """
        batch_size = y_true.shape[0]
        y_pred = self.cache['output']

        grads_w = []
        grads_b = []

        # 输出层梯度
        # 对于softmax + cross_entropy，梯度简化为: dZ = y_pred - y_true
        if loss_type == 'cross_entropy':
            dZ = (y_pred - y_true) / batch_size
        else:
            # Focal Loss梯度（更复杂）
            # dL/dp = -gamma * (1-p_t)^(gamma-1) * log(p_t) - (1-p_t)^gamma / p_t
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            p_t = np.sum(y_pred_clipped * y_true, axis=1, keepdims=True)

            # 简化版本：使用与交叉熵类似的形式，但加入调制因子
            modulating_factor = (1 - p_t) ** gamma
            dZ = modulating_factor * (y_pred - y_true) / batch_size

        # 计算输出层梯度
        A_prev = self.cache[f'A{len(self.weights)-2}'] if len(self.weights) > 1 else self.cache['input']
        dW = A_prev.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)

        grads_w.insert(0, dW)
        grads_b.insert(0, db)

        # 反向传播到输入
        dA = dZ @ self.weights[-1].T

        # 隐藏层反向传播（从后往前）
        for i in range(len(self.weights) - 2, -1, -1):
            # Dropout梯度（训练时）
            if self.training and self.dropout_rate > 0:
                mask = self.cache[f'dropout_mask{i}']
                dA = dA * mask / (1 - self.dropout_rate)

            # ReLU梯度
            Z = self.cache[f'Z{i}']
            dZ = dA * relu_derivative(Z)

            # 计算权重和偏置梯度
            A_prev = self.cache[f'A{i-1}'] if i > 0 else self.cache['input']
            dW = A_prev.T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)

            grads_w.insert(0, dW)
            grads_b.insert(0, db)

            # 继续反向传播
            if i > 0:
                dA = dZ @ self.weights[i].T

        return grads_w, grads_b

    def predict(self, X):
        """预测（推理模式）"""
        self.training = False
        probs = self.forward(X)
        self.training = True
        return np.argmax(probs, axis=1)


# =============================================================================
# 第三部分：Adam优化器
# =============================================================================

class Adam:
    """
    Adam优化器（自适应矩估计）

    结合动量和RMSprop的优点：
    - m: 梯度的一阶矩估计（动量）
    - v: 梯度的二阶矩估计（自适应学习率）
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None  # 权重一阶矩
        self.v_w = None  # 权重二阶矩
        self.m_b = None  # 偏置一阶矩
        self.v_b = None  # 偏置二阶矩
        self.t = 0       # 时间步

    def update(self, weights, biases, grads_w, grads_b):
        """
        执行一步Adam更新（同时更新权重和偏置）

        参数:
            weights: 权重列表
            biases: 偏置列表
            grads_w: 权重梯度列表
            grads_b: 偏置梯度列表

        返回:
            更新后的权重列表, 更新后的偏置列表
        """
        # 首次调用时初始化
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

        self.t += 1
        updated_weights = []
        updated_biases = []

        # 更新权重
        for i, (w, gw) in enumerate(zip(weights, grads_w)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gw
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gw ** 2)

            m_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_w[i] / (1 - self.beta2 ** self.t)

            updated_w = w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_weights.append(updated_w)

        # 更新偏置
        for i, (b, gb) in enumerate(zip(biases, grads_b)):
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gb
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gb ** 2)

            m_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            updated_b = b - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_biases.append(updated_b)

        return updated_weights, updated_biases


# =============================================================================
# 第四部分：训练工具
# =============================================================================

def create_batches(X, y, batch_size, shuffle=True):
    """
    创建小批量数据

    参数:
        X: 特征矩阵
        y: 标签矩阵
        batch_size: 批量大小
        shuffle: 是否打乱数据

    返回:
        批量数据的生成器
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def one_hot_encode(y, num_classes):
    """
    将标签转换为one-hot编码

    参数:
        y: 标签数组, shape (n_samples,)
        num_classes: 类别数

    返回:
        one_hot: shape (n_samples, num_classes)
    """
    n_samples = len(y)
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), y.astype(int)] = 1
    return one_hot


class EarlyStopping:
    """
    早停机制（防止过拟合）

    当验证集损失连续patience个epoch不下降时，停止训练
    """

    def __init__(self, patience=5, min_delta=0.001):
        """
        参数:
            patience: 容忍的epoch数
            min_delta: 最小改善量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_weights = None
        self.best_biases = None

    def check(self, val_loss, weights, biases):
        """
        检查是否应该早停

        参数:
            val_loss: 当前验证损失
            weights: 当前权重
            biases: 当前偏置

        返回:
            是否应该停止训练
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳权重
            self.best_weights = [w.copy() for w in weights]
            self.best_biases = [b.copy() for b in biases]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class StepLR:
    """
    学习率阶梯衰减

    每隔step_size个epoch，学习率乘以gamma
    """

    def __init__(self, optimizer, step_size=10, gamma=0.5):
        """
        参数:
            optimizer: 优化器实例
            step_size: 衰减间隔（epoch数）
            gamma: 衰减因子
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0
        self.base_lr = optimizer.lr

    def step(self):
        """执行一步学习率调度"""
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            print(f"  学习率衰减至: {self.optimizer.lr:.6f}")


# =============================================================================
# 第五部分：评估和可视化
# =============================================================================

def compute_metrics(y_true, y_pred, class_names):
    """
    计算分类指标

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    返回:
        metrics: 指标字典
    """
    metrics = {}

    # 总体准确率
    metrics['accuracy'] = np.mean(y_true == y_pred)

    # 每个类别的指标
    num_classes = len(class_names)
    for i in range(num_classes):
        true_positives = np.sum((y_true == i) & (y_pred == i))
        false_positives = np.sum((y_true != i) & (y_pred == i))
        false_negatives = np.sum((y_true == i) & (y_pred != i))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f'{class_names[i]}_precision'] = precision
        metrics[f'{class_names[i]}_recall'] = recall
        metrics[f'{class_names[i]}_f1'] = f1

    return metrics


def confusion_matrix(y_true, y_pred, num_classes):
    """
    计算混淆矩阵

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数

    返回:
        cm: 混淆矩阵, shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, title_suffix=''):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(train_losses, label='训练损失', color='blue')
    axes[0].plot(val_losses, label='验证损失', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title(f'损失曲线{title_suffix}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(train_accs, label='训练准确率', color='blue')
    axes[1].plot(val_accs, label='验证准确率', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].set_title(f'准确率曲线{title_suffix}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, title='混淆矩阵'):
    """绘制混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, cmap='Blues')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)

    # 设置刻度
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 添加数值标注
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center",
                          color="white" if cm[i, j] > cm.max()/2 else "black")

    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(title)

    plt.tight_layout()
    return fig


# =============================================================================
# 第六部分：数据加载
# =============================================================================

def load_weibo_data(data_path):
    """
    加载微博情感数据集

    参数:
        data_path: 数据文件路径

    返回:
        texts: 文本列表
        labels: 标签数组 (0: 喜悦, 1: 愤怒, 2: 厌恶, 3: 低落)
    """
    texts = []
    labels = []

    # 情绪标签映射
    label_map = {'喜悦': 0, '愤怒': 1, '厌恶': 2, '低落': 3}

    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过表头
        next(f)
        for line in f:
            parts = line.strip().split(',', 1)  # 只分割第一个逗号
            if len(parts) == 2:
                label_str = parts[0]
                text = parts[1]
                if label_str in label_map:
                    labels.append(label_map[label_str])
                    texts.append(text)

    return texts, np.array(labels)


def create_synthetic_data(n_samples=10000):
    """
    创建模拟数据（用于演示）

    当真实数据集不可用时，使用模拟数据演示流程

    参数:
        n_samples: 样本数量

    返回:
        texts: 模拟文本列表
        labels: 标签数组
    """
    print("使用模拟数据进行演示...")

    # 情绪关键词（模拟）
    emotion_keywords = {
        0: ['开心', '快乐', '高兴', '幸福', '美好', '喜欢', '爱', '棒', '赞', '好'],  # 喜悦
        1: ['生气', '愤怒', '讨厌', '烦', '恨', '可恶', '该死', '混蛋', '无语', '坑'],  # 愤怒
        2: ['恶心', '厌恶', '反感', '无聊', '烦躁', '讨厌', '难受', '不喜欢', '差劲', '糟糕'],  # 厌恶
        3: ['难过', '伤心', '失望', '痛苦', '悲伤', '哭', '委屈', '遗憾', '心痛', '孤独']  # 低落
    }

    # 模拟类别不平衡（喜悦最多）
    class_distribution = [0.55, 0.15, 0.15, 0.15]  # 喜悦55%, 其他各15%

    texts = []
    labels = []

    for _ in range(n_samples):
        # 按分布选择类别
        label = np.random.choice(4, p=class_distribution)

        # 生成模拟文本（随机组合关键词）
        keywords = emotion_keywords[label]
        n_words = np.random.randint(3, 10)
        text = ''.join(np.random.choice(keywords, size=n_words))

        # 添加一些噪声词
        noise_words = ['今天', '我', '你', '他', '很', '真的', '啊', '呢', '吧', '了']
        n_noise = np.random.randint(1, 4)
        for noise in np.random.choice(noise_words, size=n_noise):
            insert_pos = np.random.randint(0, len(text) + 1)
            text = text[:insert_pos] + noise + text[insert_pos:]

        texts.append(text)
        labels.append(label)

    return texts, np.array(labels)


# =============================================================================
# 第七部分：主训练流程
# =============================================================================

def train_emotion_model(X_train, y_train, X_val, y_val,
                        loss_type='cross_entropy',
                        epochs=50, batch_size=64, learning_rate=0.001):
    """
    训练情绪识别模型

    参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        loss_type: 'cross_entropy' 或 'focal'
        epochs: 训练轮数
        batch_size: 批量大小
        learning_rate: 学习率

    返回:
        model: 训练好的模型
        history: 训练历史
    """
    input_dim = X_train.shape[1]
    num_classes = 4

    print(f"\n{'='*60}")
    print(f"开始训练 - 损失函数: {loss_type}")
    print(f"{'='*60}")

    # 创建模型
    model = EmotionMLP(input_dim, hidden_dims=[256, 128], num_classes=num_classes, dropout_rate=0.3)

    # 创建优化器
    optimizer = Adam(learning_rate=learning_rate)

    # 学习率调度
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    # 早停
    early_stopping = EarlyStopping(patience=8, min_delta=0.001)

    # one-hot编码
    y_train_onehot = one_hot_encode(y_train, num_classes)
    y_val_onehot = one_hot_encode(y_val, num_classes)

    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(epochs):
        # 训练模式
        model.training = True
        epoch_losses = []

        # 小批量训练
        for X_batch, y_batch in create_batches(X_train, y_train_onehot, batch_size):
            # 前向传播
            y_pred = model.forward(X_batch)

            # 计算损失
            if loss_type == 'cross_entropy':
                loss = cross_entropy_loss(y_pred, y_batch)
            else:
                loss = focal_loss(y_pred, y_batch, gamma=2.0)
            epoch_losses.append(loss)

            # 反向传播
            grads_w, grads_b = model.backward(y_batch, loss_type=loss_type)

            # 更新参数（同时更新权重和偏置）
            model.weights, model.biases = optimizer.update(
                model.weights, model.biases, grads_w, grads_b
            )

        # 计算训练指标
        train_loss = np.mean(epoch_losses)
        train_pred = model.predict(X_train)
        train_acc = np.mean(train_pred == y_train)

        # 验证
        model.training = False
        val_pred_probs = model.forward(X_val)
        if loss_type == 'cross_entropy':
            val_loss = cross_entropy_loss(val_pred_probs, y_val_onehot)
        else:
            val_loss = focal_loss(val_pred_probs, y_val_onehot, gamma=2.0)
        val_pred = np.argmax(val_pred_probs, axis=1)
        val_acc = np.mean(val_pred == y_val)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # 学习率调度
        scheduler.step()

        # 早停检查
        if early_stopping.check(val_loss, model.weights, model.biases):
            print(f"\n早停触发于 Epoch {epoch+1}")
            # 恢复最佳权重
            model.weights = early_stopping.best_weights
            model.biases = early_stopping.best_biases
            break

    return model, history


def main():
    """主函数"""
    print("="*60)
    print("项目1：中文微博情绪识别系统")
    print("="*60)

    # 类别名称
    class_names = ['喜悦', '愤怒', '厌恶', '低落']

    # 尝试加载真实数据，否则使用模拟数据
    data_path = '/Users/lyh/Desktop/ Machine Learning/neural_networks/data/simplifyweibo_4_moods.csv'

    if os.path.exists(data_path):
        print(f"\n加载真实数据: {data_path}")
        texts, labels = load_weibo_data(data_path)
        print(f"加载完成: {len(texts)} 条样本")
    else:
        print(f"\n未找到数据文件: {data_path}")
        print("数据集下载说明:")
        print("1. 访问 https://github.com/SophonPlus/ChineseNlpCorpus")
        print("2. 下载 simplifyweibo_4_moods.csv")
        print("3. 放到 neural_networks/data/ 目录下")
        print()
        texts, labels = create_synthetic_data(n_samples=8000)

    # 显示类别分布
    print("\n类别分布:")
    for i, name in enumerate(class_names):
        count = np.sum(labels == i)
        print(f"  {name}: {count} ({100*count/len(labels):.1f}%)")

    # 划分数据集
    # 使用简单的随机划分
    n_samples = len(texts)
    indices = np.random.permutation(n_samples)

    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    test_texts = [texts[i] for i in test_indices]

    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]

    print(f"\n数据划分: 训练={len(train_texts)}, 验证={len(val_texts)}, 测试={len(test_texts)}")

    # 构建词汇表和TF-IDF特征
    print("\n构建词汇表和TF-IDF特征...")
    word_to_idx = build_vocabulary(train_texts, max_features=3000)

    X_train = text_to_tfidf(train_texts, word_to_idx)
    X_val = text_to_tfidf(val_texts, word_to_idx)
    X_test = text_to_tfidf(test_texts, word_to_idx)

    print(f"特征维度: {X_train.shape[1]}")

    # ==========================================================
    # 对比实验：CrossEntropy vs Focal Loss
    # ==========================================================

    results = {}

    # 1. 使用交叉熵训练
    model_ce, history_ce = train_emotion_model(
        X_train, y_train, X_val, y_val,
        loss_type='cross_entropy',
        epochs=40, batch_size=64, learning_rate=0.001
    )
    results['CrossEntropy'] = {'model': model_ce, 'history': history_ce}

    # 2. 使用Focal Loss训练
    model_fl, history_fl = train_emotion_model(
        X_train, y_train, X_val, y_val,
        loss_type='focal',
        epochs=40, batch_size=64, learning_rate=0.001
    )
    results['Focal Loss'] = {'model': model_fl, 'history': history_fl}

    # ==========================================================
    # 测试集评估
    # ==========================================================

    print("\n" + "="*60)
    print("测试集评估结果")
    print("="*60)

    for name, result in results.items():
        model = result['model']
        y_pred = model.predict(X_test)

        # 计算指标
        metrics = compute_metrics(y_test, y_pred, class_names)

        print(f"\n【{name}】")
        print(f"  总体准确率: {metrics['accuracy']:.4f}")
        print(f"  各类别F1分数:")
        for class_name in class_names:
            f1 = metrics[f'{class_name}_f1']
            recall = metrics[f'{class_name}_recall']
            print(f"    {class_name}: F1={f1:.4f}, Recall={recall:.4f}")

        result['test_metrics'] = metrics
        result['test_pred'] = y_pred

    # ==========================================================
    # 可视化
    # ==========================================================

    print("\n生成可视化结果...")

    # 1. 训练曲线对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # CrossEntropy损失曲线
    axes[0, 0].plot(history_ce['train_loss'], label='训练', color='blue')
    axes[0, 0].plot(history_ce['val_loss'], label='验证', color='red')
    axes[0, 0].set_title('CrossEntropy - 损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Focal Loss损失曲线
    axes[0, 1].plot(history_fl['train_loss'], label='训练', color='blue')
    axes[0, 1].plot(history_fl['val_loss'], label='验证', color='red')
    axes[0, 1].set_title('Focal Loss - 损失曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # CrossEntropy准确率曲线
    axes[1, 0].plot(history_ce['train_acc'], label='训练', color='blue')
    axes[1, 0].plot(history_ce['val_acc'], label='验证', color='red')
    axes[1, 0].set_title('CrossEntropy - 准确率曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Focal Loss准确率曲线
    axes[1, 1].plot(history_fl['train_acc'], label='训练', color='blue')
    axes[1, 1].plot(history_fl['val_acc'], label='验证', color='red')
    axes[1, 1].set_title('Focal Loss - 准确率曲线')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/project_01_training_curves.png', dpi=150)
    plt.close()

    # 2. 混淆矩阵对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['test_pred'], 4)

        im = axes[idx].imshow(cm, cmap='Blues')
        axes[idx].figure.colorbar(im, ax=axes[idx])
        axes[idx].set_xticks(np.arange(4))
        axes[idx].set_yticks(np.arange(4))
        axes[idx].set_xticklabels(class_names)
        axes[idx].set_yticklabels(class_names)

        for i in range(4):
            for j in range(4):
                axes[idx].text(j, i, cm[i, j], ha="center", va="center",
                              color="white" if cm[i, j] > cm.max()/2 else "black")

        axes[idx].set_xlabel('预测标签')
        axes[idx].set_ylabel('真实标签')
        axes[idx].set_title(f'{name} - 混淆矩阵')

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/project_01_confusion_matrix.png', dpi=150)
    plt.close()

    # 3. 各类别F1分数对比条形图
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(class_names))
    width = 0.35

    f1_ce = [results['CrossEntropy']['test_metrics'][f'{c}_f1'] for c in class_names]
    f1_fl = [results['Focal Loss']['test_metrics'][f'{c}_f1'] for c in class_names]

    bars1 = ax.bar(x - width/2, f1_ce, width, label='CrossEntropy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1_fl, width, label='Focal Loss', color='coral')

    ax.set_xlabel('情绪类别')
    ax.set_ylabel('F1分数')
    ax.set_title('各类别F1分数对比（Focal Loss vs CrossEntropy）')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标注
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/project_01_f1_comparison.png', dpi=150)
    plt.close()

    print("\n可视化结果已保存:")
    print("  - project_01_training_curves.png")
    print("  - project_01_confusion_matrix.png")
    print("  - project_01_f1_comparison.png")

    # ==========================================================
    # 总结
    # ==========================================================

    print("\n" + "="*60)
    print("项目总结")
    print("="*60)

    print("\n【知识点应用】")
    print("  - MLP结构: TF-IDF -> 256 -> 128 -> 4")
    print("  - 权重初始化: He初始化（适配ReLU）")
    print("  - 激活函数: ReLU（隐藏层）+ Softmax（输出层）")
    print("  - 损失函数: CrossEntropy vs Focal Loss（对比实验）")
    print("  - 优化器: Adam")
    print("  - 正则化: Dropout(0.3) + Early Stopping")
    print("  - 学习率调度: StepLR（每15个epoch衰减50%）")

    print("\n【Focal Loss效果分析】")
    print("  对于少数类（愤怒、厌恶、低落）：")
    for class_name in class_names[1:]:  # 跳过喜悦
        f1_ce = results['CrossEntropy']['test_metrics'][f'{class_name}_f1']
        f1_fl = results['Focal Loss']['test_metrics'][f'{class_name}_f1']
        improvement = (f1_fl - f1_ce) / f1_ce * 100 if f1_ce > 0 else 0
        print(f"    {class_name}: CE={f1_ce:.4f} -> FL={f1_fl:.4f} ({improvement:+.1f}%)")

    print("\n【业务应用建议】")
    print("  1. 对于类别不平衡的情绪数据，Focal Loss通常表现更好")
    print("  2. 可以根据业务需求调整gamma参数（更大的gamma更关注难样本）")
    print("  3. 实际部署时可以使用jieba进行更精确的分词")
    print("  4. 可以考虑使用预训练词向量替代TF-IDF")


if __name__ == "__main__":
    main()
