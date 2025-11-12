# 数学符号 → Python代码 对照表

> 看到公式立刻知道怎么写代码！

---

## 🎯 快速查找

遇到数学公式？按`Ctrl+F`搜索符号，立即找到对应代码！

---

## 基础运算

| 数学符号 | 含义 | Python代码 | 例子 |
|---------|------|-----------|------|
| $x + y$ | 加法 | `x + y` | `3 + 2 = 5` |
| $x - y$ | 减法 | `x - y` | `5 - 2 = 3` |
| $x \times y$ | 乘法 | `x * y` | `3 * 4 = 12` |
| $x / y$ | 除法 | `x / y` | `10 / 2 = 5.0` |
| $x^2$ | 平方 | `x**2` 或 `np.square(x)` | `3**2 = 9` |
| $x^n$ | n次方 | `x**n` 或 `np.power(x, n)` | `2**3 = 8` |
| $\sqrt{x}$ | 平方根 | `np.sqrt(x)` | `np.sqrt(9) = 3.0` |
| $\sqrt[n]{x}$ | n次根 | `x**(1/n)` | `8**(1/3) ≈ 2.0` |
| $\|x\|$ | 绝对值 | `abs(x)` 或 `np.abs(x)` | `abs(-5) = 5` |
| $e^x$ | 指数 | `np.exp(x)` | `np.exp(1) ≈ 2.718` |
| $\ln(x)$ | 自然对数 | `np.log(x)` | `np.log(e) = 1` |
| $\log_{10}(x)$ | 常用对数 | `np.log10(x)` | `np.log10(100) = 2` |

---

## 向量运算

### 向量基础

| 数学符号 | 含义 | NumPy代码 | 例子 |
|---------|------|----------|------|
| $\vec{v} = [v_1, v_2, v_3]$ | 向量 | `v = np.array([v1, v2, v3])` | `v = np.array([1, 2, 3])` |
| $v_i$ | 第i个元素 | `v[i]` | `v[0] = 1` |
| $\vec{v} + \vec{w}$ | 向量加法 | `v + w` | `[1,2] + [3,4] = [4,6]` |
| $c\vec{v}$ | 标量乘法 | `c * v` | `2 * [1,2] = [2,4]` |

### 向量点积

| 数学符号 | 含义 | NumPy代码 |
|---------|------|----------|
| $\vec{a} \cdot \vec{b}$ | 点积/内积 | `np.dot(a, b)` 或 `a @ b` |
| $\langle a, b \rangle$ | 内积（另一种写法） | `np.dot(a, b)` |
| $\sum_{i=1}^{n} a_i b_i$ | 点积定义 | `np.sum(a * b)` |

**代码示例：**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 三种等价写法
dot1 = np.dot(a, b)      # 32
dot2 = a @ b             # 32
dot3 = np.sum(a * b)     # 32
```

### 向量模（长度）

| 数学符号 | 含义 | NumPy代码 |
|---------|------|----------|
| $\|\vec{v}\|$ | 向量的长度 | `np.linalg.norm(v)` |
| $\|\vec{v}\|_2$ | L2范数（欧氏距离） | `np.linalg.norm(v)` 或 `np.sqrt(np.sum(v**2))` |
| $\|\vec{v}\|_1$ | L1范数 | `np.linalg.norm(v, 1)` 或 `np.sum(np.abs(v))` |
| $\|\vec{v}\|_\infty$ | 无穷范数 | `np.linalg.norm(v, np.inf)` 或 `np.max(np.abs(v))` |

**代码示例：**
```python
v = np.array([3, 4])

# L2范数（欧氏距离）
l2 = np.linalg.norm(v)           # 5.0
l2_manual = np.sqrt(np.sum(v**2)) # 5.0

# L1范数
l1 = np.linalg.norm(v, 1)         # 7.0
l1_manual = np.sum(np.abs(v))     # 7.0
```

---

## 矩阵运算

### 矩阵基础

| 数学符号 | 含义 | NumPy代码 | 例子 |
|---------|------|----------|------|
| $A = \begin{bmatrix}a&b\\c&d\end{bmatrix}$ | 矩阵 | `A = np.array([[a,b],[c,d]])` | `A = np.array([[1,2],[3,4]])` |
| $A_{ij}$ | 第i行j列元素 | `A[i, j]` | `A[0, 1] = 2` |
| $A^T$ | 转置 | `A.T` 或 `np.transpose(A)` | `A.T` |
| $A^{-1}$ | 逆矩阵 | `np.linalg.inv(A)` | `np.linalg.inv(A)` |

**代码示例：**
```python
A = np.array([[1, 2],
              [3, 4]])

# 转置
AT = A.T
# [[1 3]
#  [2 4]]

# 逆矩阵
A_inv = np.linalg.inv(A)

# 验证 A @ A^(-1) = I
I = A @ A_inv  # 单位矩阵
```

### 矩阵乘法

| 数学符号 | 含义 | NumPy代码 |
|---------|------|----------|
| $AB$ | 矩阵乘法 | `A @ B` 或 `np.matmul(A, B)` 或 `np.dot(A, B)` |
| $A \odot B$ | 对应元素相乘 | `A * B` 或 `np.multiply(A, B)` |

**重要区分：**
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
C = A @ B
# [[19 22]
#  [43 50]]

# 对应元素相乘（不是矩阵乘法！）
D = A * B
# [[ 5 12]
#  [21 32]]
```

### 特殊矩阵

| 数学符号 | 含义 | NumPy代码 |
|---------|------|----------|
| $I_n$ | n×n单位矩阵 | `np.eye(n)` 或 `np.identity(n)` |
| $\mathbf{0}_{m \times n}$ | m×n零矩阵 | `np.zeros((m, n))` |
| $\mathbf{1}_{m \times n}$ | m×n全1矩阵 | `np.ones((m, n))` |
| $\text{diag}(v)$ | 对角矩阵 | `np.diag(v)` |

---

## 求和与连乘

### 求和符号 Σ

| 数学符号 | 含义 | NumPy代码 |
|---------|------|----------|
| $\sum_{i=1}^{n} x_i$ | 求和 | `np.sum(x)` |
| $\sum_{i=1}^{n} x_i^2$ | 平方和 | `np.sum(x**2)` |
| $\sum_{i,j} A_{ij}$ | 矩阵所有元素求和 | `np.sum(A)` |
| $\sum_{i} A_{ij}$ | 按列求和 | `np.sum(A, axis=0)` |
| $\sum_{j} A_{ij}$ | 按行求和 | `np.sum(A, axis=1)` |

**代码示例：**
```python
x = np.array([1, 2, 3, 4, 5])

# Σx_i
total = np.sum(x)  # 15

# Σx_i²
sum_squares = np.sum(x**2)  # 55

# 矩阵求和
A = np.array([[1, 2, 3],
              [4, 5, 6]])

total = np.sum(A)        # 21 (所有元素)
col_sum = np.sum(A, axis=0)  # [5, 7, 9] (按列)
row_sum = np.sum(A, axis=1)  # [6, 15] (按行)
```

### 连乘符号 Π

| 数学符号 | 含义 | NumPy代码 |
|---------|------|----------|
| $\prod_{i=1}^{n} x_i$ | 连乘 | `np.prod(x)` |

**代码示例：**
```python
x = np.array([1, 2, 3, 4])

# Πx_i = 1×2×3×4
product = np.prod(x)  # 24
```

---

## 微积分符号

### 导数

| 数学符号 | 含义 | 数值计算代码 |
|---------|------|-------------|
| $\frac{df}{dx}$ | 导数 | `(f(x+h) - f(x)) / h` |
| $f'(x)$ | 一阶导数 | `numerical_derivative(f, x)` |
| $\frac{\partial f}{\partial x}$ | 偏导数 | `(f(x+h, y) - f(x, y)) / h` |

**数值微分代码：**
```python
def numerical_derivative(f, x, h=1e-5):
    """计算导数"""
    return (f(x + h) - f(x)) / h

# 例子
f = lambda x: x**2
derivative_at_3 = numerical_derivative(f, 3)  # ≈ 6

# 偏导数
def partial_derivative(f, x, i, h=1e-5):
    """计算对第i个变量的偏导数"""
    x_plus_h = x.copy()
    x_plus_h[i] += h
    return (f(x_plus_h) - f(x)) / h

f = lambda x: x[0]**2 + x[1]**2
point = np.array([3.0, 4.0])
df_dx = partial_derivative(f, point, 0)  # ≈ 6
df_dy = partial_derivative(f, point, 1)  # ≈ 8
```

### 梯度

| 数学符号 | 含义 | NumPy代码 |
|---------|------|----------|
| $\nabla f$ | 梯度向量 | `numerical_gradient(f, x)` |
| $\nabla f = [\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ...]$ | 梯度定义 | 见下方代码 |

**梯度计算代码：**
```python
def numerical_gradient(f, x, h=1e-5):
    """计算梯度"""
    grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        grad[i] = (f(x_plus_h) - f(x)) / h

    return grad

# 使用
f = lambda x: x[0]**2 + x[1]**2  # f(x,y) = x² + y²
point = np.array([3.0, 4.0])
grad = numerical_gradient(f, point)  # [6.0, 8.0]
```

---

## 概率统计符号

### 概率

| 数学符号 | 含义 | Python代码 |
|---------|------|-----------|
| $P(A)$ | 概率 | `count_A / total` |
| $P(A\|B)$ | 条件概率 | `P_AB / P_B` |
| $P(A, B)$ | 联合概率 | `P_A * P_B_given_A` |

### 期望和方差

| 数学符号 | 含义 | NumPy代码 |
|---------|------|----------|
| $E[X]$ 或 $\mu$ | 期望/均值 | `np.mean(X)` |
| $\text{Var}(X)$ 或 $\sigma^2$ | 方差 | `np.var(X)` |
| $\sigma$ | 标准差 | `np.std(X)` |
| $\text{Cov}(X,Y)$ | 协方差 | `np.cov(X, Y)[0,1]` |

**代码示例：**
```python
data = np.array([1, 2, 3, 4, 5])

# 期望（均值）
mean = np.mean(data)  # 3.0

# 方差
variance = np.var(data)  # 2.0

# 标准差
std = np.std(data)  # 1.414

# 协方差
X = np.array([1, 2, 3, 4])
Y = np.array([2, 4, 6, 8])
cov_matrix = np.cov(X, Y)
covariance = cov_matrix[0, 1]
```

### 分布

| 数学符号 | 含义 | SciPy代码 |
|---------|------|----------|
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | 正态分布 | `stats.norm(mu, sigma)` |
| $X \sim \text{Uniform}(a, b)$ | 均匀分布 | `stats.uniform(a, b-a)` |
| $X \sim \text{Bernoulli}(p)$ | 伯努利分布 | `stats.bernoulli(p)` |

**代码示例：**
```python
from scipy import stats

# 正态分布 N(0, 1)
normal = stats.norm(0, 1)
samples = normal.rvs(size=1000)  # 生成样本
pdf_value = normal.pdf(0)         # 概率密度

# 均匀分布 Uniform(0, 1)
uniform = stats.uniform(0, 1)
samples = uniform.rvs(size=1000)

# 或用NumPy
samples = np.random.normal(0, 1, 1000)  # 正态分布
samples = np.random.uniform(0, 1, 1000) # 均匀分布
```

---

## 机器学习常用公式

### 1. 线性回归

**数学公式：**
$$\hat{y} = w^T x + b = \sum_{i=1}^{n} w_i x_i + b$$

**代码：**
```python
# 向量化版本
y_pred = X @ w + b

# 或循环版本
y_pred = np.sum(w * x) + b

# sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
```

### 2. 均方误差（MSE）

**数学公式：**
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**代码：**
```python
# 手动计算
mse = np.mean((y - y_pred)**2)

# sklearn
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred)
```

### 3. 梯度下降

**数学公式：**
$$\theta \leftarrow \theta - \alpha \nabla L(\theta)$$

**代码：**
```python
# 单次更新
theta = theta - learning_rate * gradient

# 完整训练循环
for epoch in range(n_epochs):
    # 计算梯度
    gradient = compute_gradient(X, y, theta)

    # 更新参数
    theta = theta - learning_rate * gradient

    # 计算损失
    loss = compute_loss(X, y, theta)
```

### 4. Sigmoid函数

**数学公式：**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**代码：**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 向量化版本自动处理数组
z = np.array([-1, 0, 1])
s = sigmoid(z)  # [0.268, 0.5, 0.731]
```

### 5. Softmax函数

**数学公式：**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**代码：**
```python
def softmax(z):
    # 数值稳定版本
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# 使用
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
# [0.659, 0.242, 0.099]
```

### 6. 余弦相似度

**数学公式：**
$$\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}$$

**代码：**
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 或使用sklearn
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([a], [b])[0, 0]
```

---

## 常见模式速查

### 模式1：按行/列操作

```python
# 按行求和: Σ_j A_ij
row_sum = np.sum(A, axis=1)

# 按列求和: Σ_i A_ij
col_sum = np.sum(A, axis=0)

# 记忆方法：axis=0沿着第0维（行）折叠，得到列
#          axis=1沿着第1维（列）折叠，得到行
```

### 模式2：广播

```python
# 数学: 每行减去均值
# X - mean(X, axis=0)

X_centered = X - np.mean(X, axis=0)

# 数学: 每列除以标准差
# X / std(X, axis=1)

X_normalized = X / np.std(X, axis=1, keepdims=True)
```

### 模式3：条件索引

```python
# 数学: {x | x > 0}
positive = X[X > 0]

# 数学: Σ_{x_i > 0} x_i
sum_positive = np.sum(X[X > 0])

# 数学: count({x | x > threshold})
count = np.sum(X > threshold)
```

---

## 🔧 调试技巧

### 检查形状
```python
print(f'X.shape = {X.shape}')
print(f'w.shape = {w.shape}')
print(f'y.shape = {y.shape}')

# 预期: (n_samples, n_features) @ (n_features, 1) = (n_samples, 1)
```

### 检查数值范围
```python
print(f'min={X.min()}, max={X.max()}, mean={X.mean():.2f}')
```

### 验证实现
```python
# 用简单例子手算验证
X_simple = np.array([[1, 2], [3, 4]])
# 手算结果...
# 对比代码结果
```

---

## 📝 快速参考卡

### NumPy核心函数

| 操作 | 函数 |
|------|------|
| 求和 | `np.sum()` |
| 均值 | `np.mean()` |
| 最大 | `np.max()` |
| 最小 | `np.min()` |
| 点积 | `np.dot()` 或 `@` |
| 范数 | `np.linalg.norm()` |
| 转置 | `.T` |
| 逆矩阵 | `np.linalg.inv()` |
| 指数 | `np.exp()` |
| 对数 | `np.log()` |
| 平方根 | `np.sqrt()` |
| 平方 | `np.square()` 或 `**2` |
| 绝对值 | `np.abs()` |

---

**看到公式不再迷茫，直接写代码！** 💻
