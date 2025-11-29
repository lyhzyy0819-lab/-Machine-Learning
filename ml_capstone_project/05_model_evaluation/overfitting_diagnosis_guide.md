# 🩺 过拟合诊断与改进指南

> **核心目标**：识别过拟合/欠拟合问题，给出具体改进策略
>
> ⏱️ **预计用时**：1-1.5小时掌握诊断和改进方法

---

## 🎯 使用场景

**本文档适用于：**
- ✅ 模型训练完成，需要诊断性能问题
- ✅ 训练集和测试集性能差距大
- ✅ 模型性能不理想，需要改进方向

**常见症状：**
- ❌ 训练集AUC=0.95，测试集AUC=0.70（过拟合）
- ❌ 训练集和测试集都很差（欠拟合）
- ❌ 不知道如何改进模型

---

## 📋 目录

1. [过拟合/欠拟合识别](#过拟合欠拟合识别)
2. [诊断工具](#诊断工具)
3. [过拟合解决方案](#过拟合解决方案)
4. [欠拟合解决方案](#欠拟合解决方案)
5. [实战案例](#实战案例)
6. [代码实现](#代码实现)

---

## 过拟合/欠拟合识别

### 快速诊断表

| 症状 | 训练集表现 | 验证集表现 | Gap | 诊断 |
|------|------------|------------|-----|------|
| **欠拟合** | 差 | 差 | 小 | 模型太简单，拟合能力不足 |
| **✅ 理想** | 好 | 好 | 小 | 模型刚好，泛化能力强 |
| **过拟合** | 很好 | 差 | 大 | 模型太复杂，记住了训练数据 |

**Gap计算**：
```
Gap = 训练集性能 - 验证集性能

分类：Gap = Train AUC - Val AUC
回归：Gap = Val RMSE - Train RMSE（越大越过拟合）
```

**判断标准**：
- Gap < 0.05：✅ 合适
- Gap 0.05-0.10：⚠️ 轻度过拟合
- Gap > 0.10：❌ 严重过拟合

---

### 具体诊断示例

#### 示例1：严重过拟合（分类）
```
训练集 AUC: 0.95
验证集 AUC: 0.70
Gap: 0.25 ❌ 严重过拟合

原因：模型记住了训练数据的噪声
解决：正则化、简化模型、增加数据
```

#### 示例2：欠拟合（回归）
```
训练集 RMSE: 50
验证集 RMSE: 52
Gap: 2（小，但都很高）❌ 欠拟合

原因：模型太简单，无法捕获数据规律
解决：增加模型复杂度、添加特征
```

#### 示例3：理想状态
```
训练集 AUC: 0.82
验证集 AUC: 0.80
Gap: 0.02 ✅ 理想

泛化能力良好，无需调整
```

---

## 诊断工具

### 工具1：学习曲线（Learning Curve）

**含义**：观察训练集和验证集误差随**样本量**的变化

**原理**：
```
X轴：训练样本数（从少到多）
Y轴：模型误差（RMSE或1-AUC）

两条曲线：
- 训练集误差（Train Error）
- 验证集误差（Validation Error）
```

**诊断模式**：

```
1️⃣ 欠拟合模式
    误差 ↑
    │   [Train]────────────────
    │   [Val  ]────────────────
    │    两条曲线都很高且接近
    └────────────────────→ 样本量
    解决：增加模型复杂度

2️⃣ 理想模式
    误差 ↑
    │   [Train]────────────
    │   [Val  ]──────────── 两条曲线都低且接近
    │
    └────────────────────→ 样本量
    继续使用！

3️⃣ 过拟合模式
    误差 ↑
    │         [Val  ]────────────── 验证误差高
    │                    ↑ 大Gap
    │   [Train]────────────────── 训练误差低
    └────────────────────→ 样本量
    解决：正则化、增加数据

4️⃣ 高方差（需要更多数据）
    误差 ↑
    │         [Val  ]──────╲
    │                    ↓ Gap缩小中
    │   [Train]──────────╲
    └────────────────────→ 样本量
    解决：增加训练数据（Gap还在缩小）
```

**代码实现**：
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    """
    绘制学习曲线

    参数：
        model: 模型对象
        X: 特征矩阵
        y: 目标变量
        cv: 交叉验证折数
        scoring: 评估指标（需要是负数形式，如neg_mean_squared_error）
    """
    # 计算学习曲线
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),  # 10个样本量点
        n_jobs=-1
    )

    # 计算均值和标准差
    train_mean = -train_scores.mean(axis=1)  # 转为正数
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Error', marker='o')
    plt.plot(train_sizes, val_mean, label='Validation Error', marker='s')

    # 添加标准差阴影
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)

    plt.xlabel('Number of Training Samples')
    plt.ylabel('Error (RMSE)')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # 诊断
    final_gap = val_mean[-1] - train_mean[-1]
    if final_gap > 0.1 * val_mean[-1]:  # Gap > 10%
        print("⚠️ 诊断：过拟合（训练和验证误差差距大）")
        print(f"   Train Error: {train_mean[-1]:.3f}")
        print(f"   Val Error:   {val_mean[-1]:.3f}")
        print(f"   Gap:         {final_gap:.3f}")
    elif train_mean[-1] > 0.5 * val_mean[-1]:  # 训练误差也很高
        print("⚠️ 诊断：欠拟合（训练和验证误差都很高）")
    else:
        print("✅ 诊断：模型状态良好")

# 使用示例
plot_learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error')
```

---

### 工具2：验证曲线（Validation Curve）

**含义**：观察模型复杂度对性能的影响

**原理**：
```
X轴：模型复杂度参数（如树深度、正则化强度）
Y轴：模型性能（AUC或1-RMSE）

两条曲线：
- 训练集性能
- 验证集性能
```

**诊断模式**：
```
    性能 ↑
    │        [Train]
    │            ╱────────────
    │          ╱
    │        ╱     [Val] ╱╲
    │      ╱           ╱    ╲  过拟合区
    │    ╱           ╱        ╲
    │  ╱───────────╱            ╲
    │  欠拟合区    ↑ 最优点
    └────────────────────────→ 模型复杂度
                           （如：树深度）

找到验证集性能最高点 = 最优复杂度
```

**代码实现**：
```python
from sklearn.model_selection import validation_curve

def plot_validation_curve(model, X, y, param_name, param_range, cv=5, scoring='roc_auc'):
    """
    绘制验证曲线

    参数：
        model: 模型对象
        X: 特征矩阵
        y: 目标变量
        param_name: 参数名（如'max_depth'）
        param_range: 参数范围（如[1, 2, 3, 5, 10, 20]）
        cv: 交叉验证折数
        scoring: 评估指标
    """
    # 计算验证曲线
    train_scores, val_scores = validation_curve(
        model, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    # 计算均值和标准差
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label='Training Score', marker='o')
    plt.plot(param_range, val_mean, label='Validation Score', marker='s')

    # 标准差阴影
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)

    plt.xlabel(param_name)
    plt.ylabel(f'Score ({scoring})')
    plt.title(f'Validation Curve ({param_name})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # 找到最优参数
    best_idx = val_mean.argmax()
    best_param = param_range[best_idx]
    print(f"✅ 最优 {param_name}: {best_param}")
    print(f"   Validation Score: {val_mean[best_idx]:.3f}")

# 使用示例（树深度）
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
param_range = [1, 2, 3, 5, 10, 15, 20, 30]

plot_validation_curve(
    model, X, y,
    param_name='max_depth',
    param_range=param_range,
    cv=5,
    scoring='roc_auc'
)
```

---

### 工具3：偏差-方差分解

**理论**：
```
总误差 = 偏差² + 方差 + 噪声

偏差（Bias）：欠拟合
- 模型太简单，无法捕获真实关系

方差（Variance）：过拟合
- 模型太复杂，对训练数据过敏感

噪声（Irreducible Error）：不可避免
- 数据本身的随机性
```

**权衡关系**：
```
    误差 ↑
    │
    │  [总误差]    ╲ ╱
    │              ╳
    │  [方差]    ╱   ╲
    │  [偏差]  ╲       ╱
    │
    │         欠拟合 ↑ 过拟合
    └────────────────────→ 模型复杂度
                   最优点
```

---

## 过拟合解决方案

### 策略1：正则化（Regularization）

**原理**：在损失函数中添加惩罚项，限制模型复杂度

**方法**：

| 正则化类型 | 惩罚项 | 效果 | 适用模型 |
|------------|--------|------|----------|
| **L1（Lasso）** | λ·\|w\| | 权重稀疏化，特征选择 | 线性模型 |
| **L2（Ridge）** | λ·w² | 权重平滑化 | 线性模型、NN |
| **Elastic Net** | λ₁·\|w\| + λ₂·w² | L1+L2组合 | 线性模型 |
| **Dropout** | 随机失活神经元 | 防止神经元共适应 | 神经网络 |
| **树剪枝** | 限制树深度/叶子节点 | 简化决策树 | 树模型 |

**代码示例**：

**线性模型正则化**：
```python
from sklearn.linear_model import Ridge, Lasso

# L2正则化（Ridge）
ridge = Ridge(alpha=1.0)  # alpha越大，正则化越强
ridge.fit(X_train, y_train)

# L1正则化（Lasso）
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 对比不同正则化强度
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"Alpha={alpha:5.3f}: Train R²={train_score:.3f}, Val R²={val_score:.3f}")

# 输出示例：
# Alpha=0.001: Train R²=0.850, Val R²=0.720  过拟合
# Alpha=0.010: Train R²=0.830, Val R²=0.780
# Alpha=0.100: Train R²=0.810, Val R²=0.805  ← 最优
# Alpha=1.000: Train R²=0.750, Val R²=0.760
# Alpha=10.00: Train R²=0.650, Val R²=0.670  欠拟合
```

**树模型正则化**：
```python
from sklearn.ensemble import RandomForestClassifier

# 限制树深度
rf = RandomForestClassifier(
    max_depth=10,          # 限制树深度（防止过拟合）
    min_samples_split=20,  # 节点分裂最小样本数
    min_samples_leaf=10,   # 叶子节点最小样本数
    max_features='sqrt',   # 特征子集大小
    n_estimators=100
)
rf.fit(X_train, y_train)
```

---

### 策略2：增加训练数据

**原理**：更多数据帮助模型学习真实规律而非噪声

**方法**：

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| **收集更多数据** | 最直接有效 | 数据可获取 |
| **数据增强** | 图像旋转、文本同义替换 | 图像、文本、音频 |
| **合成数据** | SMOTE（过采样） | 不平衡数据 |
| **迁移学习** | 使用预训练模型 | 小数据集 + 大模型 |

**数据增强示例（图像）**：
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 图像数据增强
datagen = ImageDataGenerator(
    rotation_range=20,      # 随机旋转±20度
    width_shift_range=0.2,  # 水平平移
    height_shift_range=0.2, # 垂直平移
    horizontal_flip=True,   # 水平翻转
    zoom_range=0.2          # 随机缩放
)

# 生成增强数据
datagen.fit(X_train)
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50)
```

---

### 策略3：简化模型

**原理**：降低模型复杂度，减少拟合噪声的能力

**方法**：

| 简化方式 | 示例 | 效果 |
|----------|------|------|
| **减少特征** | 100个特征 → 50个 | 降低维度 |
| **降低模型复杂度** | 决策树深度30 → 10 | 简化结构 |
| **使用更简单模型** | XGBoost → 逻辑回归 | 降低容量 |
| **Early Stopping** | 训练时监控验证误差 | 防止过度训练 |

**代码示例**：

**特征选择（减少特征）**：
```python
from sklearn.feature_selection import SelectKBest, f_classif

# 方法1：基于统计检验选择Top K特征
selector = SelectKBest(f_classif, k=50)  # 选择50个最佳特征
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

# 方法2：基于模型的特征选择
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 选择重要性高于阈值的特征
selector = SelectFromModel(rf, threshold='median')
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)

print(f"原始特征数: {X_train.shape[1]}")
print(f"选择后特征数: {X_train_selected.shape[1]}")
```

**Early Stopping（提前停止）**：
```python
from sklearn.ensemble import GradientBoostingClassifier

# 使用Early Stopping
model = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    validation_fraction=0.2,  # 20%数据用于验证
    n_iter_no_change=10,       # 10轮不改进则停止
    tol=0.001
)
model.fit(X_train, y_train)

print(f"训练轮数: {model.n_estimators_}")  # 实际训练的轮数
```

---

### 策略4：集成方法

**原理**：组合多个模型的预测，减少单个模型的方差

**方法**：

| 集成方法 | 说明 | 适用场景 |
|----------|------|----------|
| **Bagging** | 多个模型独立训练，取平均 | 高方差模型（决策树） |
| **Boosting** | 顺序训练，纠正前一个模型错误 | 高偏差模型 |
| **Stacking** | 用元模型组合多个基模型 | 多种算法组合 |

**代码示例**：
```python
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging（减少方差）
base_model = DecisionTreeClassifier(max_depth=None)  # 深度不限
bagging = BaggingClassifier(
    base_estimator=base_model,
    n_estimators=10,
    max_samples=0.8,
    random_state=42
)
bagging.fit(X_train, y_train)

# Random Forest就是Bagging的特例
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
```

---

### 策略5：交叉验证

**原理**：使用多次数据划分评估模型，更可靠地检测过拟合

**代码示例**：
```python
from sklearn.model_selection import cross_val_score

# K-Fold交叉验证
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 如果CV标准差很大（>0.05），说明模型不稳定
if cv_scores.std() > 0.05:
    print("⚠️ 模型不稳定，可能过拟合")
```

---

## 欠拟合解决方案

### 策略1：增加模型复杂度

**原理**：使用更强大的模型，增加拟合能力

**方法**：

| 增加复杂度方式 | 示例 | 效果 |
|----------------|------|------|
| **增加树深度** | max_depth=5 → 15 | 捕获更复杂模式 |
| **增加神经网络层数** | 2层 → 5层 | 增加表达能力 |
| **使用更复杂模型** | 线性回归 → XGBoost | 提升拟合能力 |
| **增加多项式特征** | x → [x, x², x³] | 捕获非线性关系 |

**代码示例**：

**增加树深度**：
```python
from sklearn.tree import DecisionTreeClassifier

# 欠拟合：树太浅
model_simple = DecisionTreeClassifier(max_depth=2)
model_simple.fit(X_train, y_train)
print(f"Train AUC: {model_simple.score(X_train, y_train):.3f}")
print(f"Val AUC: {model_simple.score(X_val, y_val):.3f}")
# 输出：Train AUC: 0.72, Val AUC: 0.70（都很低，欠拟合）

# 增加复杂度
model_complex = DecisionTreeClassifier(max_depth=10)
model_complex.fit(X_train, y_train)
print(f"Train AUC: {model_complex.score(X_train, y_train):.3f}")
print(f"Val AUC: {model_complex.score(X_val, y_val):.3f}")
# 输出：Train AUC: 0.85, Val AUC: 0.82（改善）
```

**多项式特征**：
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 原始线性模型（欠拟合）
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear - Train R²: {lr.score(X_train, y_train):.3f}")
print(f"Linear - Val R²: {lr.score(X_val, y_val):.3f}")

# 添加多项式特征
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
print(f"Poly - Train R²: {lr_poly.score(X_train_poly, y_train):.3f}")
print(f"Poly - Val R²: {lr_poly.score(X_val_poly, y_val):.3f}")
```

---

### 策略2：添加特征

**原理**：增加特征数量，帮助模型捕获更多信息

**方法**：

| 特征添加方式 | 说明 | 示例 |
|--------------|------|------|
| **特征工程** | 手动构造有意义特征 | 年龄 → [年龄段, 是否成年] |
| **交叉特征** | 特征之间的组合 | [身高, 体重] → BMI |
| **聚合特征** | 分组统计特征 | 用户历史平均消费 |
| **嵌入特征** | 类别特征的稠密表示 | Word2Vec、实体嵌入 |

**代码示例**：
```python
import pandas as pd

# 假设数据
df = pd.DataFrame({
    'age': [25, 35, 45, 55],
    'income': [30000, 50000, 70000, 90000],
    'education': ['高中', '本科', '硕士', '博士']
})

# 特征工程
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['青年', '中年', '老年'])
df['high_income'] = (df['income'] > 60000).astype(int)

# 交叉特征
df['income_per_age'] = df['income'] / df['age']

# 类别编码
df['education_encoded'] = df['education'].map({
    '高中': 1, '本科': 2, '硕士': 3, '博士': 4
})

print(df)
```

---

### 策略3：减少正则化强度

**原理**：如果模型被过度正则化，可能导致欠拟合

**代码示例**：
```python
from sklearn.linear_model import Ridge

# 过度正则化（欠拟合）
ridge_strong = Ridge(alpha=100.0)
ridge_strong.fit(X_train, y_train)
print(f"Strong Reg - Train R²: {ridge_strong.score(X_train, y_train):.3f}")
print(f"Strong Reg - Val R²: {ridge_strong.score(X_val, y_val):.3f}")
# 输出：都很低

# 减少正则化
ridge_weak = Ridge(alpha=0.1)
ridge_weak.fit(X_train, y_train)
print(f"Weak Reg - Train R²: {ridge_weak.score(X_train, y_train):.3f}")
print(f"Weak Reg - Val R²: {ridge_weak.score(X_val, y_val):.3f}")
# 输出：改善
```

---

### 策略4：训练更长时间

**原理**：增加训练轮数，让模型充分学习

**代码示例**：
```python
from sklearn.neural_network import MLPClassifier

# 训练轮数不足（欠拟合）
mlp_short = MLPClassifier(max_iter=10)
mlp_short.fit(X_train, y_train)

# 增加训练轮数
mlp_long = MLPClassifier(max_iter=200)
mlp_long.fit(X_train, y_train)
```

---

## 实战案例

### 案例：客户流失预测的诊断和改进

#### 背景
- 数据：5000个客户，15个特征
- 问题：二分类（流失/不流失）
- 初始模型：逻辑回归

#### Step 1：初始诊断

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 训练初始模型
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)

# 评估
train_auc = roc_auc_score(y_train, lr.predict_proba(X_train)[:, 1])
val_auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])

print(f"Train AUC: {train_auc:.3f}")
print(f"Val AUC: {val_auc:.3f}")
print(f"Gap: {train_auc - val_auc:.3f}")

# 输出：
# Train AUC: 0.720
# Val AUC: 0.715
# Gap: 0.005
```

**诊断**：Gap很小，但性能都不高 → **欠拟合**

#### Step 2：绘制学习曲线确认

```python
from src.model_evaluation import plot_learning_curve

plot_learning_curve(lr, X, y, cv=5, scoring='roc_auc')
```

**观察**：训练和验证曲线都很高且接近 → 确认欠拟合

#### Step 3：改进方案

**方案1：添加多项式特征**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('lr', LogisticRegression())
])
pipeline.fit(X_train, y_train)

train_auc = roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1])
val_auc = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])

print(f"Train AUC: {train_auc:.3f}")  # 0.780
print(f"Val AUC: {val_auc:.3f}")      # 0.765
print(f"Gap: {train_auc - val_auc:.3f}")  # 0.015
```

**结果**：性能提升，但可能可以做得更好

**方案2：使用更复杂模型（随机森林）**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

train_auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
val_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])

print(f"Train AUC: {train_auc:.3f}")  # 0.920
print(f"Val AUC: {val_auc:.3f}")      # 0.785
print(f"Gap: {train_auc - val_auc:.3f}")  # 0.135
```

**新问题**：Gap=0.135 → **过拟合！**

#### Step 4：解决过拟合

```python
# 增加正则化（限制树深度）
rf_tuned = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,           # 限制深度
    min_samples_split=20,  # 增加分裂限制
    min_samples_leaf=10    # 增加叶子限制
)
rf_tuned.fit(X_train, y_train)

train_auc = roc_auc_score(y_train, rf_tuned.predict_proba(X_train)[:, 1])
val_auc = roc_auc_score(y_val, rf_tuned.predict_proba(X_val)[:, 1])

print(f"Train AUC: {train_auc:.3f}")  # 0.850
print(f"Val AUC: {val_auc:.3f}")      # 0.820
print(f"Gap: {train_auc - val_auc:.3f}")  # 0.030
```

**最终结果**：✅ Gap<0.05，性能良好

---

## 代码实现

### 完整的诊断和改进流程

```python
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import numpy as np

class ModelDiagnostics:
    """模型诊断工具类"""

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def diagnose(self):
        """
        完整诊断流程

        返回诊断报告
        """
        print("=" * 60)
        print("模型诊断报告")
        print("=" * 60)

        # 1. 基础性能
        self._check_basic_performance()

        # 2. 学习曲线
        self._plot_learning_curve()

        # 3. 诊断结论
        self._diagnose_conclusion()

        # 4. 改进建议
        self._suggest_improvements()

    def _check_basic_performance(self):
        """检查基础性能"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        # 计算AUC
        train_auc = roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1])
        val_auc = roc_auc_score(y_val, self.model.predict_proba(X_val)[:, 1])
        gap = train_auc - val_auc

        print(f"\n1. 基础性能")
        print(f"   Train AUC: {train_auc:.3f}")
        print(f"   Val AUC:   {val_auc:.3f}")
        print(f"   Gap:       {gap:.3f}")

        self.train_auc = train_auc
        self.val_auc = val_auc
        self.gap = gap

    def _plot_learning_curve(self):
        """绘制学习曲线"""
        print(f"\n2. 学习曲线分析")

        train_sizes, train_scores, val_scores = learning_curve(
            self.model, self.X, self.y,
            cv=5,
            scoring='roc_auc',
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
        plt.plot(train_sizes, val_mean, label='Validation Score', marker='s')
        plt.xlabel('Training Set Size')
        plt.ylabel('AUC Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        print("   ✅ 学习曲线已生成")

    def _diagnose_conclusion(self):
        """诊断结论"""
        print(f"\n3. 诊断结论")

        if self.gap > 0.10:
            print("   ❌ 严重过拟合")
        elif self.gap > 0.05:
            print("   ⚠️  轻度过拟合")
        elif self.train_auc < 0.70 and self.val_auc < 0.70:
            print("   ❌ 欠拟合")
        else:
            print("   ✅ 模型状态良好")

    def _suggest_improvements(self):
        """改进建议"""
        print(f"\n4. 改进建议")

        if self.gap > 0.05:
            print("   过拟合改进建议：")
            print("   - 增加正则化强度")
            print("   - 减少模型复杂度（降低树深度）")
            print("   - 增加训练数据")
            print("   - 使用Dropout（神经网络）")
        elif self.train_auc < 0.70:
            print("   欠拟合改进建议：")
            print("   - 增加模型复杂度")
            print("   - 添加特征（特征工程）")
            print("   - 减少正则化")
            print("   - 使用更复杂的模型")
        else:
            print("   ✅ 模型已优化，可以继续")

# 使用示例
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
diagnostics = ModelDiagnostics(model, X, y)
diagnostics.diagnose()
```

---

## ✅ 快速参考卡片

### 过拟合/欠拟合快速判断

| 症状 | 训练集 | 验证集 | Gap | 诊断 | 解决方案 |
|------|--------|--------|-----|------|----------|
| 两者都差 | 低 | 低 | 小 | 欠拟合 | 增加复杂度、添加特征 |
| 训练好验证差 | 高 | 低 | 大 | 过拟合 | 正则化、简化模型 |
| 两者都好 | 高 | 高 | 小 | ✅ 理想 | 继续使用 |

### 改进策略速查

**过拟合 → 减少方差**：
1. 正则化（L1/L2/Dropout）
2. 简化模型（降低深度、减少特征）
3. 增加数据
4. Early Stopping
5. 集成方法（Bagging）

**欠拟合 → 减少偏差**：
1. 增加模型复杂度
2. 添加特征
3. 减少正则化
4. 训练更长时间
5. 使用更强大模型

---

## 📚 延伸阅读

**相关文档**：
- **模型比较**：[model_comparison_and_selection.md](model_comparison_and_selection.md)
- **指标计算**：[metrics_calculation_guide.md](metrics_calculation_guide.md)
- **业务转化**：[business_value_translation.md](business_value_translation.md)

**推荐资源**：
- 《The Elements of Statistical Learning》 - Bias-Variance Trade-off
- sklearn文档：[Learning Curves](https://scikit-learn.org/stable/modules/learning_curve.html)

---

**最后更新**：2024年11月
**代码模块**：src/model_evaluation.py - `plot_learning_curve()`, `plot_validation_curve()`