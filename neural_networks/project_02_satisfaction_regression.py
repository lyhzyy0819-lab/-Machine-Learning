"""
项目2：航空客户满意度预测系统（回归）

业务场景：预测用户满意度评分，可迁移到对话满意度预测场景
数据集：Airline Passenger Satisfaction（13万条，22个特征）

知识点覆盖：
- MLP多层感知机（回归任务）
- 损失函数（MSE、Huber Loss）
- 优化器（Adam）
- 正则化（Dropout、L2、Early Stopping）
- 权重初始化（He初始化）
- 学习率调度（StepLR）
- 特征工程（标准化、类别编码）

数据集下载：
1. 访问 https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
2. 下载 train.csv
3. 放到 neural_networks/data/ 目录下

如果没有数据集，程序会使用模拟数据进行演示
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# =============================================================================
# 第一部分：数据预处理工具
# =============================================================================

class StandardScaler:
    """
    标准化（Z-score归一化）

    公式: x_scaled = (x - mean) / std

    作用：将特征缩放到均值为0、标准差为1的分布
    对神经网络训练很重要，可以加速收敛
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """
        计算均值和标准差

        参数:
            X: 特征矩阵, shape (n_samples, n_features)
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # 防止除零
        self.std = np.where(self.std == 0, 1, self.std)

    def transform(self, X):
        """
        应用标准化

        参数:
            X: 特征矩阵

        返回:
            X_scaled: 标准化后的特征矩阵
        """
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """拟合并转换"""
        self.fit(X)
        return self.transform(X)


def one_hot_encode_column(values, categories=None):
    """
    对单个类别列进行one-hot编码

    参数:
        values: 类别值数组
        categories: 类别列表（如果为None则自动推断）

    返回:
        encoded: one-hot编码矩阵
        categories: 类别列表
    """
    if categories is None:
        categories = sorted(list(set(values)))

    n_samples = len(values)
    n_categories = len(categories)

    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    encoded = np.zeros((n_samples, n_categories))
    for i, val in enumerate(values):
        if val in cat_to_idx:
            encoded[i, cat_to_idx[val]] = 1

    return encoded, categories


# =============================================================================
# 第二部分：神经网络模型（回归版本）
# =============================================================================

def he_init(fan_in, fan_out):
    """
    He初始化

    公式: W ~ N(0, sqrt(2/fan_in))
    适用于ReLU激活函数
    """
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std


def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU导数"""
    return (x > 0).astype(float)


def mse_loss(y_pred, y_true):
    """
    均方误差损失（MSE）

    公式: MSE = mean((y_pred - y_true)^2)

    最常用的回归损失函数，对大误差敏感
    """
    return np.mean((y_pred - y_true) ** 2)


def huber_loss(y_pred, y_true, delta=1.0):
    """
    Huber损失（平滑L1损失）

    结合了MSE和MAE的优点：
    - 当误差小于delta时，使用MSE（二次惩罚）
    - 当误差大于delta时，使用MAE（线性惩罚）

    优点：对异常值更加鲁棒

    公式:
        L = 0.5 * (y - y_hat)^2           if |y - y_hat| <= delta
        L = delta * (|y - y_hat| - 0.5 * delta)  otherwise
    """
    error = y_pred - y_true
    abs_error = np.abs(error)

    # 分情况计算
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)

    loss = np.where(abs_error <= delta, quadratic, linear)
    return np.mean(loss)


class RegressionMLP:
    """
    回归多层感知机

    网络结构: Input -> Dense(128) -> ReLU -> Dropout -> Dense(64) -> ReLU -> Dropout -> Dense(1) -> Linear

    与分类MLP的区别：
    - 输出层使用线性激活（不是Softmax）
    - 损失函数使用MSE/Huber（不是CrossEntropy）
    - 输出单个连续值（不是概率分布）
    """

    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.2, l2_lambda=0.001):
        """
        初始化网络

        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout概率
            l2_lambda: L2正则化系数
        """
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.training = True

        # 构建层维度: [input_dim, 128, 64, 1]
        layer_dims = [input_dim] + hidden_dims + [1]

        # 初始化权重和偏置
        self.weights = []
        self.biases = []

        print("\n网络结构（回归）:")
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]

            W = he_init(fan_in, fan_out)
            b = np.zeros((1, fan_out))

            self.weights.append(W)
            self.biases.append(b)

            activation = "ReLU" if i < len(layer_dims) - 2 else "Linear"
            print(f"  Layer {i+1}: {fan_in} -> {fan_out} ({activation})")

        self.cache = {}

    def forward(self, X):
        """
        前向传播

        参数:
            X: 输入特征, shape (batch_size, input_dim)

        返回:
            output: 预测值, shape (batch_size, 1)
        """
        self.cache['input'] = X
        A = X

        # 隐藏层（ReLU + Dropout）
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self.cache[f'Z{i}'] = Z

            A = relu(Z)
            self.cache[f'A{i}'] = A

            # Dropout（仅训练时）
            if self.training and self.dropout_rate > 0:
                mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
                A = A * mask / (1 - self.dropout_rate)
                self.cache[f'dropout_mask{i}'] = mask

        # 输出层（线性激活）
        Z_out = A @ self.weights[-1] + self.biases[-1]
        self.cache['Z_out'] = Z_out
        self.cache['output'] = Z_out  # 回归任务不需要激活函数

        return Z_out

    def backward(self, y_true, loss_type='mse', delta=1.0):
        """
        反向传播

        参数:
            y_true: 真实值, shape (batch_size, 1)
            loss_type: 'mse' 或 'huber'
            delta: Huber损失的delta参数

        返回:
            grads_w: 权重梯度列表
            grads_b: 偏置梯度列表
        """
        batch_size = y_true.shape[0]
        y_pred = self.cache['output']

        grads_w = []
        grads_b = []

        # 输出层梯度
        if loss_type == 'mse':
            # MSE梯度: 2 * (y_pred - y_true) / batch_size
            dZ = 2 * (y_pred - y_true) / batch_size
        else:
            # Huber梯度
            error = y_pred - y_true
            abs_error = np.abs(error)
            dZ = np.where(abs_error <= delta,
                         error / batch_size,  # MSE梯度
                         delta * np.sign(error) / batch_size)  # MAE梯度

        # 计算输出层梯度
        A_prev = self.cache[f'A{len(self.weights)-2}'] if len(self.weights) > 1 else self.cache['input']
        dW = A_prev.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)

        # 添加L2正则化梯度
        dW += self.l2_lambda * self.weights[-1]

        grads_w.insert(0, dW)
        grads_b.insert(0, db)

        # 反向传播到输入
        dA = dZ @ self.weights[-1].T

        # 隐藏层反向传播
        for i in range(len(self.weights) - 2, -1, -1):
            # Dropout梯度
            if self.training and self.dropout_rate > 0:
                mask = self.cache[f'dropout_mask{i}']
                dA = dA * mask / (1 - self.dropout_rate)

            # ReLU梯度
            Z = self.cache[f'Z{i}']
            dZ = dA * relu_derivative(Z)

            # 计算梯度
            A_prev = self.cache[f'A{i-1}'] if i > 0 else self.cache['input']
            dW = A_prev.T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)

            # L2正则化
            dW += self.l2_lambda * self.weights[i]

            grads_w.insert(0, dW)
            grads_b.insert(0, db)

            if i > 0:
                dA = dZ @ self.weights[i].T

        return grads_w, grads_b

    def predict(self, X):
        """预测"""
        self.training = False
        output = self.forward(X)
        self.training = True
        return output

    def l2_reg_loss(self):
        """计算L2正则化损失"""
        reg_loss = 0
        for w in self.weights:
            reg_loss += np.sum(w ** 2)
        return 0.5 * self.l2_lambda * reg_loss


# =============================================================================
# 第三部分：Adam优化器
# =============================================================================

class Adam:
    """Adam优化器"""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def update(self, weights, biases, grads_w, grads_b):
        """执行Adam更新"""
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

        self.t += 1
        updated_weights = []
        updated_biases = []

        for i, (w, gw) in enumerate(zip(weights, grads_w)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gw
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gw ** 2)

            m_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_w[i] / (1 - self.beta2 ** self.t)

            updated_w = w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_weights.append(updated_w)

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
    """创建小批量数据"""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_weights = None
        self.best_biases = None

    def check(self, val_loss, weights, biases):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = [w.copy() for w in weights]
            self.best_biases = [b.copy() for b in biases]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class StepLR:
    """学习率阶梯衰减"""

    def __init__(self, optimizer, step_size=15, gamma=0.5):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            print(f"  学习率衰减至: {self.optimizer.lr:.6f}")


# =============================================================================
# 第五部分：回归评估指标
# =============================================================================

def compute_regression_metrics(y_true, y_pred):
    """
    计算回归评估指标

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        metrics: 指标字典
    """
    # 确保是一维数组
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # 均方误差 (MSE)
    mse = np.mean((y_true - y_pred) ** 2)

    # 均方根误差 (RMSE)
    rmse = np.sqrt(mse)

    # 平均绝对误差 (MAE)
    mae = np.mean(np.abs(y_true - y_pred))

    # 决定系数 (R²)
    ss_res = np.sum((y_true - y_pred) ** 2)  # 残差平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 总平方和
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


# =============================================================================
# 第六部分：数据加载
# =============================================================================

def load_airline_data(data_path):
    """
    加载航空满意度数据集

    参数:
        data_path: CSV文件路径

    返回:
        X: 特征矩阵
        y: 目标变量（满意度评分）
        feature_names: 特征名称列表
    """
    # 这里简化处理，实际应该用pandas
    # 由于可能没有安装pandas，我们创建一个简化版本

    print("  注意：完整版本应使用pandas加载数据")
    print("  这里使用模拟数据演示")

    return None, None, None


def create_synthetic_regression_data(n_samples=10000):
    """
    创建模拟的满意度数据（用于演示）

    模拟航空满意度数据的特征：
    - 服务质量评分（1-5）
    - 延误时间
    - 舱位等级
    - 乘客类型等

    参数:
        n_samples: 样本数量

    返回:
        X: 特征矩阵
        y: 满意度评分 (1-5)
        feature_names: 特征名称
    """
    print("使用模拟数据进行演示...")

    # 定义特征
    feature_names = [
        '登机服务', '机上服务', '座位舒适度', '餐饮服务',
        '机上娱乐', 'WiFi服务', '行李处理', '出发延误(分钟)',
        '到达延误(分钟)', '飞行距离(km)', '年龄'
    ]

    n_features = len(feature_names)
    X = np.zeros((n_samples, n_features))

    # 生成各特征
    # 服务评分 (1-5)
    X[:, 0] = np.random.randint(1, 6, n_samples)  # 登机服务
    X[:, 1] = np.random.randint(1, 6, n_samples)  # 机上服务
    X[:, 2] = np.random.randint(1, 6, n_samples)  # 座位舒适度
    X[:, 3] = np.random.randint(1, 6, n_samples)  # 餐饮服务
    X[:, 4] = np.random.randint(1, 6, n_samples)  # 机上娱乐
    X[:, 5] = np.random.randint(1, 6, n_samples)  # WiFi服务
    X[:, 6] = np.random.randint(1, 6, n_samples)  # 行李处理

    # 延误时间 (0-300分钟，右偏分布)
    X[:, 7] = np.abs(np.random.exponential(30, n_samples))  # 出发延误
    X[:, 8] = np.abs(np.random.exponential(30, n_samples))  # 到达延误

    # 飞行距离 (100-5000km)
    X[:, 9] = np.random.uniform(100, 5000, n_samples)

    # 年龄 (18-80)
    X[:, 10] = np.random.randint(18, 81, n_samples)

    # 生成目标变量（满意度评分）
    # 满意度与服务评分正相关，与延误负相关
    y = (
        0.15 * X[:, 0] +  # 登机服务
        0.20 * X[:, 1] +  # 机上服务
        0.15 * X[:, 2] +  # 座位舒适度
        0.10 * X[:, 3] +  # 餐饮服务
        0.10 * X[:, 4] +  # 机上娱乐
        0.05 * X[:, 5] +  # WiFi
        0.10 * X[:, 6] +  # 行李处理
        -0.01 * X[:, 7] +  # 出发延误（负影响）
        -0.01 * X[:, 8] +  # 到达延误（负影响）
        0.0001 * X[:, 9] +  # 飞行距离（微弱正影响）
        np.random.normal(0, 0.3, n_samples)  # 噪声
    )

    # 归一化到1-5范围
    y = np.clip(y, 1, 5)

    return X, y.reshape(-1, 1), feature_names


# =============================================================================
# 第七部分：主训练流程
# =============================================================================

def train_regression_model(X_train, y_train, X_val, y_val,
                           loss_type='mse',
                           epochs=60, batch_size=64, learning_rate=0.001):
    """
    训练回归模型

    参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        loss_type: 'mse' 或 'huber'
        epochs: 训练轮数
        batch_size: 批量大小
        learning_rate: 学习率

    返回:
        model: 训练好的模型
        history: 训练历史
    """
    input_dim = X_train.shape[1]

    print(f"\n{'='*60}")
    print(f"开始训练 - 损失函数: {loss_type.upper()}")
    print(f"{'='*60}")

    # 创建模型
    model = RegressionMLP(input_dim, hidden_dims=[128, 64], dropout_rate=0.2, l2_lambda=0.001)

    # 优化器
    optimizer = Adam(learning_rate=learning_rate)

    # 学习率调度
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # 早停
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_rmse': [], 'val_rmse': []
    }

    for epoch in range(epochs):
        model.training = True
        epoch_losses = []

        # 小批量训练
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            # 前向传播
            y_pred = model.forward(X_batch)

            # 计算损失
            if loss_type == 'mse':
                loss = mse_loss(y_pred, y_batch)
            else:
                loss = huber_loss(y_pred, y_batch, delta=1.0)

            # 加上L2正则化损失
            loss += model.l2_reg_loss()
            epoch_losses.append(loss)

            # 反向传播
            grads_w, grads_b = model.backward(y_batch, loss_type=loss_type)

            # 更新参数
            model.weights, model.biases = optimizer.update(
                model.weights, model.biases, grads_w, grads_b
            )

        # 计算训练指标
        train_loss = np.mean(epoch_losses)
        train_pred = model.predict(X_train)
        train_metrics = compute_regression_metrics(y_train, train_pred)

        # 验证
        model.training = False
        val_pred = model.forward(X_val)
        if loss_type == 'mse':
            val_loss = mse_loss(val_pred, y_val)
        else:
            val_loss = huber_loss(val_pred, y_val)
        val_loss += model.l2_reg_loss()
        val_metrics = compute_regression_metrics(y_val, val_pred)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_rmse'].append(train_metrics['rmse'])
        history['val_rmse'].append(val_metrics['rmse'])

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}, train_rmse={train_metrics['rmse']:.4f}, "
                  f"val_loss={val_loss:.4f}, val_rmse={val_metrics['rmse']:.4f}")

        # 学习率调度
        scheduler.step()

        # 早停检查
        if early_stopping.check(val_loss, model.weights, model.biases):
            print(f"\n早停触发于 Epoch {epoch+1}")
            model.weights = early_stopping.best_weights
            model.biases = early_stopping.best_biases
            break

    return model, history


def analyze_feature_importance(model, feature_names):
    """
    分析特征重要性（基于第一层权重）

    这是一个简单的方法：计算第一层权重的绝对值之和
    更复杂的方法包括：排列重要性、SHAP值等
    """
    # 第一层权重, shape: (input_dim, hidden_dim)
    W1 = model.weights[0]

    # 计算每个特征对应权重的L2范数
    importance = np.linalg.norm(W1, axis=1)

    # 归一化
    importance = importance / np.sum(importance)

    # 排序
    sorted_idx = np.argsort(importance)[::-1]

    return importance, sorted_idx


def main():
    """主函数"""
    print("="*60)
    print("项目2：航空客户满意度预测系统（回归）")
    print("="*60)

    # 尝试加载真实数据
    data_path = '/Users/lyh/Desktop/ Machine Learning/neural_networks/data/train.csv'

    if os.path.exists(data_path):
        print(f"\n加载真实数据: {data_path}")
        X, y, feature_names = load_airline_data(data_path)
    else:
        print(f"\n未找到数据文件: {data_path}")
        print("数据集下载说明:")
        print("1. 访问 https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction")
        print("2. 下载 train.csv")
        print("3. 放到 neural_networks/data/ 目录下")
        print()

    # 使用模拟数据
    X, y, feature_names = create_synthetic_regression_data(n_samples=10000)

    print(f"\n数据集信息:")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  目标变量范围: {y.min():.2f} - {y.max():.2f}")

    # 显示特征
    print(f"\n特征列表:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}: mean={X[:, i].mean():.2f}, std={X[:, i].std():.2f}")

    # 划分数据集
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n数据划分: 训练={len(train_idx)}, 验证={len(val_idx)}, 测试={len(test_idx)}")

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("特征标准化完成")

    # ==========================================================
    # 对比实验：MSE vs Huber Loss
    # ==========================================================

    results = {}

    # 1. 使用MSE训练
    model_mse, history_mse = train_regression_model(
        X_train, y_train, X_val, y_val,
        loss_type='mse',
        epochs=60, batch_size=64, learning_rate=0.001
    )
    results['MSE'] = {'model': model_mse, 'history': history_mse}

    # 2. 使用Huber Loss训练
    model_huber, history_huber = train_regression_model(
        X_train, y_train, X_val, y_val,
        loss_type='huber',
        epochs=60, batch_size=64, learning_rate=0.001
    )
    results['Huber'] = {'model': model_huber, 'history': history_huber}

    # ==========================================================
    # 测试集评估
    # ==========================================================

    print("\n" + "="*60)
    print("测试集评估结果")
    print("="*60)

    for name, result in results.items():
        model = result['model']
        y_pred = model.predict(X_test)

        metrics = compute_regression_metrics(y_test, y_pred)

        print(f"\n【{name} Loss】")
        print(f"  R²分数: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")

        result['test_metrics'] = metrics
        result['test_pred'] = y_pred

    # ==========================================================
    # 特征重要性分析
    # ==========================================================

    print("\n" + "="*60)
    print("特征重要性分析（基于MSE模型）")
    print("="*60)

    importance, sorted_idx = analyze_feature_importance(model_mse, feature_names)

    print("\n特征重要性排名:")
    for rank, idx in enumerate(sorted_idx):
        print(f"  {rank+1}. {feature_names[idx]}: {importance[idx]:.4f}")

    # ==========================================================
    # 可视化
    # ==========================================================

    print("\n生成可视化结果...")

    # 1. 训练曲线对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE损失曲线
    axes[0, 0].plot(history_mse['train_loss'], label='训练', color='blue')
    axes[0, 0].plot(history_mse['val_loss'], label='验证', color='red')
    axes[0, 0].set_title('MSE Loss - 损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Huber损失曲线
    axes[0, 1].plot(history_huber['train_loss'], label='训练', color='blue')
    axes[0, 1].plot(history_huber['val_loss'], label='验证', color='red')
    axes[0, 1].set_title('Huber Loss - 损失曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MSE RMSE曲线
    axes[1, 0].plot(history_mse['train_rmse'], label='训练', color='blue')
    axes[1, 0].plot(history_mse['val_rmse'], label='验证', color='red')
    axes[1, 0].set_title('MSE Loss - RMSE曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Huber RMSE曲线
    axes[1, 1].plot(history_huber['train_rmse'], label='训练', color='blue')
    axes[1, 1].plot(history_huber['val_rmse'], label='验证', color='red')
    axes[1, 1].set_title('Huber Loss - RMSE曲线')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/project_02_training_curves.png', dpi=150)
    plt.close()

    # 2. 预测vs实际散点图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, result) in enumerate(results.items()):
        y_pred = result['test_pred'].flatten()
        y_true = y_test.flatten()

        axes[idx].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[idx].plot([y_true.min(), y_true.max()],
                      [y_true.min(), y_true.max()],
                      'r--', label='理想预测线')

        r2 = result['test_metrics']['r2']
        axes[idx].set_xlabel('实际满意度')
        axes[idx].set_ylabel('预测满意度')
        axes[idx].set_title(f'{name} Loss - 预测vs实际 (R²={r2:.4f})')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/project_02_prediction_scatter.png', dpi=150)
    plt.close()

    # 3. 特征重要性条形图
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(feature_names))
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    bars = ax.barh(y_pos, sorted_importance, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('重要性')
    ax.set_title('特征重要性分析')
    ax.grid(True, alpha=0.3, axis='x')

    # 添加数值标注
    for bar, val in zip(bars, sorted_importance):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/project_02_feature_importance.png', dpi=150)
    plt.close()

    # 4. 残差分析
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, result) in enumerate(results.items()):
        y_pred = result['test_pred'].flatten()
        y_true = y_test.flatten()
        residuals = y_true - y_pred

        # 残差分布直方图
        axes[idx].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].axvline(x=0, color='r', linestyle='--', label='零残差线')
        axes[idx].set_xlabel('残差 (实际 - 预测)')
        axes[idx].set_ylabel('频数')
        axes[idx].set_title(f'{name} Loss - 残差分布')
        axes[idx].legend()

        # 添加统计信息
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        axes[idx].text(0.95, 0.95, f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}',
                      transform=axes[idx].transAxes, ha='right', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/Users/lyh/Desktop/ Machine Learning/neural_networks/project_02_residual_analysis.png', dpi=150)
    plt.close()

    print("\n可视化结果已保存:")
    print("  - project_02_training_curves.png")
    print("  - project_02_prediction_scatter.png")
    print("  - project_02_feature_importance.png")
    print("  - project_02_residual_analysis.png")

    # ==========================================================
    # 总结
    # ==========================================================

    print("\n" + "="*60)
    print("项目总结")
    print("="*60)

    print("\n【知识点应用】")
    print("  - MLP结构: 特征 -> 128 -> 64 -> 1（回归输出）")
    print("  - 权重初始化: He初始化")
    print("  - 激活函数: ReLU（隐藏层）+ Linear（输出层）")
    print("  - 损失函数: MSE vs Huber Loss（对比实验）")
    print("  - 优化器: Adam")
    print("  - 正则化: L2(λ=0.001) + Dropout(0.2) + Early Stopping")
    print("  - 学习率调度: StepLR（每20个epoch衰减50%）")

    print("\n【MSE vs Huber Loss对比】")
    mse_r2 = results['MSE']['test_metrics']['r2']
    huber_r2 = results['Huber']['test_metrics']['r2']
    mse_rmse = results['MSE']['test_metrics']['rmse']
    huber_rmse = results['Huber']['test_metrics']['rmse']
    print(f"  MSE Loss:   R²={mse_r2:.4f}, RMSE={mse_rmse:.4f}")
    print(f"  Huber Loss: R²={huber_r2:.4f}, RMSE={huber_rmse:.4f}")

    print("\n【业务应用建议】")
    print("  1. 当数据存在异常值时，Huber Loss通常更稳健")
    print("  2. 特征重要性分析可帮助识别影响满意度的关键因素")
    print("  3. 可以迁移到对话满意度预测场景（替换特征即可）")
    print("  4. 建议在真实数据上调整网络结构和超参数")


if __name__ == "__main__":
    main()
