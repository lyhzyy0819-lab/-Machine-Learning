# 机器学习核心公式速查表

> 每个公式都配有：通俗解释 + 代码实现 + 实际例子

---

## 📚 目录

- [线性代数公式](#线性代数公式)
- [微积分公式](#微积分公式)
- [概率统计公式](#概率统计公式)
- [机器学习算法公式](#机器学习算法公式)

---

## 线性代数公式

### 1. 向量点积

**公式：**
$$\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + ... + a_nb_n$$

**通俗解释：**
对应位置相乘再求和

**代码：**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 方法1：直接用@
dot_product = a @ b

# 方法2：np.dot
dot_product = np.dot(a, b)

# 方法3：手动计算
dot_product = np.sum(a * b)

print(dot_product)  # 32
```

**ML应用：**
- 计算相似度
- 神经网络中的加权求和

---

### 2. 矩阵乘法

**公式：**
$$C_{ij} = \sum_{k=1}^{m} A_{ik} B_{kj}$$

**通俗解释：**
C的第i行第j列 = A的第i行 点乘 B的第j列

**代码：**
```python
A = np.array([[1, 2],
              [3, 4]])  # 2x2

B = np.array([[5, 6],
              [7, 8]])  # 2x2

# 矩阵乘法
C = A @ B  # 或 np.matmul(A, B) 或 np.dot(A, B)

print(C)
# [[19 22]
#  [43 50]]

# 验证C[0,0] = 1*5 + 2*7 = 19 ✓
```

**ML应用：**
- 神经网络前向传播
- 线性变换

---

### 3. 向量的模（长度）

**公式：**
$$\|\vec{v}\| = \sqrt{v_1^2 + v_2^2 + ... + v_n^2}$$

**通俗解释：**
向量的长度

**代码：**
```python
v = np.array([3, 4])

# 方法1：np.linalg.norm
length = np.linalg.norm(v)

# 方法2：手动计算
length = np.sqrt(np.sum(v**2))

print(length)  # 5.0
```

**ML应用：**
- 归一化
- 距离计算
- 正则化

---

### 4. 余弦相似度

**公式：**
$$\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}$$

**通俗解释：**
两个向量夹角的余弦值，衡量方向的相似度

**代码：**
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

sim = cosine_similarity(a, b)
print(f'相似度: {sim:.4f}')  # 0.9746
```

**ML应用：**
- 推荐系统
- 文本相似度
- 图像检索

---

### 5. 矩阵的迹

**公式：**
$$\text{tr}(A) = \sum_{i=1}^{n} A_{ii}$$

**通俗解释：**
对角线元素之和

**代码：**
```python
A = np.array([[1, 2],
              [3, 4]])

trace = np.trace(A)  # 或 A.diagonal().sum()

print(trace)  # 5 (1 + 4)
```

---

## 微积分公式

### 1. 导数定义

**公式：**
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**通俗解释：**
函数在某点的斜率

**代码：**
```python
def numerical_derivative(f, x, h=1e-5):
    """数值计算导数"""
    return (f(x + h) - f(x)) / h

# 例子：f(x) = x^2，导数是2x
f = lambda x: x**2

x = 3
derivative = numerical_derivative(f, x)
print(f'f(x)=x²在x={x}处的导数: {derivative:.4f}')  # 6.0
print(f'解析解: {2*x}')  # 6
```

---

### 2. 常见函数的导数

| 函数 | 导数 | 代码验证 |
|------|------|----------|
| $f(x) = c$ | $f'(x) = 0$ | `derivative(lambda x: 5, 3) ≈ 0` |
| $f(x) = x$ | $f'(x) = 1$ | `derivative(lambda x: x, 3) ≈ 1` |
| $f(x) = x^n$ | $f'(x) = nx^{n-1}$ | `derivative(lambda x: x**3, 2) ≈ 12` |
| $f(x) = e^x$ | $f'(x) = e^x$ | `derivative(np.exp, 1) ≈ e` |
| $f(x) = \ln(x)$ | $f'(x) = 1/x$ | `derivative(np.log, 2) ≈ 0.5` |
| $f(x) = \sin(x)$ | $f'(x) = \cos(x)$ | `derivative(np.sin, 0) ≈ 1` |

**代码示例：**
```python
# 验证x³的导数是3x²
f = lambda x: x**3
x = 2

numerical = numerical_derivative(f, x)
analytical = 3 * x**2

print(f'数值导数: {numerical:.4f}')  # 12.0
print(f'解析导数: {analytical}')     # 12
```

---

### 3. 梯度

**公式：**
$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**通俗解释：**
多变量函数在各个方向上的导数组成的向量

**代码：**
```python
def numerical_gradient(f, x, h=1e-5):
    """
    计算多变量函数的数值梯度
    """
    grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h

        grad[i] = (f(x_plus_h) - f(x)) / h
    print(x_plus_h)

    return grad

# 例子：f(x,y) = x² + y²
# 梯度 = [2x, 2y]
f = lambda x: x[0]**2 + x[1]**2

point = np.array([3.0, 4.0])
grad = numerical_gradient(f, point)

print(f'数值梯度: {grad}')      # [6.0, 8.0]
print(f'解析梯度: [6.0, 8.0]')  # [2*3, 2*4]
```

**ML应用：**
- 梯度下降优化
- 反向传播

---

### 4. 链式法则

**公式：**
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**通俗解释：**
复合函数的导数 = 外层导数 × 内层导数

**代码：**
```python
# 例子：y = (x² + 1)³
# 令u = x² + 1，则y = u³
# dy/dx = dy/du * du/dx = 3u² * 2x

def f(x):
    """y = (x² + 1)³"""
    return (x**2 + 1)**3

def df(x):
    """导数（链式法则）"""
    u = x**2 + 1
    return 3 * u**2 * 2 * x

x = 2
numerical = numerical_derivative(f, x)
analytical = df(x)

print(f'数值导数: {numerical:.4f}')
print(f'解析导数: {analytical}')
```

**ML应用：**
- 反向传播的核心
- 计算复杂网络的梯度

---

### 5. 梯度下降更新规则

**公式：**
$$\theta_{new} = \theta_{old} - \alpha \nabla L(\theta)$$

**通俗解释：**
新参数 = 旧参数 - 学习率 × 梯度

**代码：**
```python
def gradient_descent(f, grad_f, theta_init, learning_rate=0.1, n_iter=100):
    """
    梯度下降优化
    """
    theta = theta_init.copy()
    history = [theta.copy()]

    for i in range(n_iter):
        grad = grad_f(theta)
        theta = theta - learning_rate * grad  # 更新规则
        history.append(theta.copy())

    return theta, np.array(history)

# 例子：最小化f(x,y) = (x-1)² + (y-2)²
f = lambda x: (x[0]-1)**2 + (x[1]-2)**2
grad_f = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])

theta_init = np.array([0.0, 0.0])
theta_final, history = gradient_descent(f, grad_f, theta_init, 0.1, 50)

print(f'初始点: {theta_init}')
print(f'最终点: {theta_final}')
print(f'真实最小值: [1.0, 2.0]')
```

---

## 概率统计公式

### 1. 期望（均值）

**公式：**

离散：$E[X] = \sum_{i} x_i p(x_i)$

连续：$E[X] = \int x p(x) dx$

**通俗解释：**
加权平均值

**代码：**
```python
# 离散情况
values = np.array([1, 2, 3, 4, 5])
probabilities = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

expectation = np.sum(values * probabilities)
print(f'期望: {expectation}')  # 2.95

# 或者从样本估计
samples = np.random.choice(values, size=10000, p=probabilities)
estimated_mean = np.mean(samples)
print(f'样本均值: {estimated_mean:.2f}')
```

---

### 2. 方差

**公式：**
$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

**通俗解释：**
数据的离散程度

**代码：**
```python
data = np.array([1, 2, 3, 4, 5])

# 方法1：使用公式
mean = np.mean(data)
variance = np.mean((data - mean)**2)

# 方法2：NumPy函数
variance = np.var(data)

# 标准差
std = np.std(data)

print(f'均值: {mean}')
print(f'方差: {variance}')
print(f'标准差: {std}')
```

---

### 3. 协方差

**公式：**
$$\text{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])]$$

**通俗解释：**
两个变量一起变化的程度

**代码：**
```python
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

# 方法1：手动计算
cov = np.mean((X - np.mean(X)) * (Y - np.mean(Y)))

# 方法2：NumPy函数
cov_matrix = np.cov(X, Y)
cov = cov_matrix[0, 1]

print(f'协方差: {cov}')
```

---

### 4. 正态分布

**公式：**
$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**通俗解释：**
钟形曲线分布

**代码：**
```python
from scipy import stats

# 创建正态分布对象
mu = 0
sigma = 1
normal = stats.norm(mu, sigma)

# 概率密度
x = 0
pdf = normal.pdf(x)
print(f'在x={x}的概率密度: {pdf:.4f}')

# 生成随机样本
samples = normal.rvs(size=1000)

# 或用NumPy
samples = np.random.normal(mu, sigma, 1000)

# 验证
print(f'样本均值: {np.mean(samples):.2f}')
print(f'样本标准差: {np.std(samples):.2f}')
```

---

### 5. 贝叶斯定理

**公式：**
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

**通俗解释：**
已知B发生，更新A的概率

**代码示例：**
```python
# 例子：疾病检测
# P(病) = 0.01（患病率）
# P(阳性|病) = 0.95（真阳性率）
# P(阳性|健康) = 0.05（假阳性率）
# 求：检测阳性时，真的患病的概率？

P_disease = 0.01
P_positive_given_disease = 0.95
P_positive_given_healthy = 0.05
P_healthy = 1 - P_disease

# P(阳性)
P_positive = (P_positive_given_disease * P_disease +
              P_positive_given_healthy * P_healthy)

# 贝叶斯定理
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print(f'检测阳性时真患病的概率: {P_disease_given_positive:.2%}')
# 只有16%！说明假阳性很常见
```

---

## 机器学习算法公式

### 1. 线性回归

**公式：**
$$\hat{y} = w^T x + b$$

**损失函数（MSE）：**
$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**正规方程：**
$$w = (X^TX)^{-1}X^Ty$$

**代码：**
```python
# 生成数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 方法1：正规方程
X_b = np.c_[np.ones((100, 1)), X]  # 添加偏置项
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f'参数 (正规方程): {theta.ravel()}')

# 方法2：梯度下降
theta = np.random.randn(2, 1)
learning_rate = 0.1
n_iterations = 1000

for iteration in range(n_iterations):
    gradients = 2/100 * X_b.T @ (X_b @ theta - y)
    theta = theta - learning_rate * gradients

print(f'参数 (梯度下降): {theta.ravel()}')

# 方法3：sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
print(f'参数 (sklearn): [{model.intercept_[0]:.4f}, {model.coef_[0][0]:.4f}]')
```

---

### 2. 逻辑回归

**Sigmoid函数：**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**预测概率：**
$$P(y=1|x) = \sigma(w^T x + b)$$

**交叉熵损失：**
$$L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**代码：**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_predict(X, theta):
    return sigmoid(X @ theta)

# Sigmoid示例
z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.title('Sigmoid函数')
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.grid(True)
```

---

### 3. Softmax（多分类）

**公式：**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**通俗解释：**
将logits转换为概率分布

**代码：**
```python
def softmax(z):
    # 数值稳定版本
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# 例子
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print(f'Logits: {logits}')
print(f'概率: {probs}')
print(f'概率和: {np.sum(probs):.4f}')  # 应该为1
```

---

### 4. 正则化

**L1正则化（Lasso）：**
$$L = \text{MSE} + \lambda \sum_{j=1}^{n}|w_j|$$

**L2正则化（Ridge）：**
$$L = \text{MSE} + \lambda \sum_{j=1}^{n}w_j^2$$

**代码：**
```python
from sklearn.linear_model import Ridge, Lasso

# Ridge回归
ridge = Ridge(alpha=1.0)  # alpha就是λ
ridge.fit(X_train, y_train)

# Lasso回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

print(f'Ridge权重: {ridge.coef_}')
print(f'Lasso权重: {lasso.coef_}')
print(f'Lasso有{np.sum(lasso.coef_ == 0)}个权重被置为0（特征选择）')
```

---

### 5. K-近邻距离

**欧氏距离：**
$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**曼哈顿距离：**
$$d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$$

**代码：**
```python
from scipy.spatial.distance import euclidean, cityblock

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# 欧氏距离
dist_euclidean = euclidean(x, y)
# 或
dist_euclidean = np.linalg.norm(x - y)

# 曼哈顿距离
dist_manhattan = cityblock(x, y)
# 或
dist_manhattan = np.sum(np.abs(x - y))

print(f'欧氏距离: {dist_euclidean:.4f}')
print(f'曼哈顿距离: {dist_manhattan}')
```

---

## 💡 使用技巧

### 1. 公式转代码的步骤
```
1. 识别符号 → 查math_symbols_guide.md
2. 理解公式 → 看本文档的"通俗解释"
3. 写伪代码 → 用自然语言描述步骤
4. 实现代码 → 用NumPy实现
5. 验证结果 → 用简单例子测试
```

### 2. 调试公式实现
```python
# 技巧1：打印中间结果
def my_function(x):
    step1 = x**2
    print(f'Step 1: {step1}')
    step2 = step1 + 1
    print(f'Step 2: {step2}')
    return step2

# 技巧2：用简单例子验证
# 用手算能验证的简单数字
x = 2  # 而不是复杂的浮点数

# 技巧3：对比库函数
# 你的实现 vs NumPy/SciPy/sklearn
```

### 3. 常见错误
```python
# ❌ 错误：忘记轴
np.sum(X**2)  # 所有元素求和

# ✅ 正确：指定轴
np.sum(X**2, axis=1)  # 按行求和

# ❌ 错误：维度不匹配
a = np.array([1,2,3])      # (3,)
b = np.array([[1],[2],[3]]) # (3,1)
a @ b  # 错误！

# ✅ 正确：调整形状
a.reshape(-1, 1) @ b.T  # 或 a[:, None] @ b.T
```

---

## 🔗 扩展阅读

- **Matrix Cookbook**：矩阵运算公式大全
- **Andrew Ng的ML课程**：公式讲解清晰
- **Deep Learning Book**：深度学习数学基础

---

**掌握这些公式，机器学习算法不再神秘！** 🎯
