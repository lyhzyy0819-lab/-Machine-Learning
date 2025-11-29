# 数据预处理快速参考⭐

> **核心理念**："15分钟完成基础预处理决策，快速进入建模阶段"
>
> **适用场景**：拿到新数据后的快速预处理、项目初期的Baseline建立、时间紧急的快速实验

---

## 📋 目录

1. [快速使用指南（3分钟必读）](#1-快速使用指南3分钟必读)
2. [5步快速预处理（15分钟）](#2-5步快速预处理15分钟)
3. [常用决策矩阵速查](#3-常用决策矩阵速查)
4. [实战示例](#4-实战示例)
5. [快速代码模板](#5-快速代码模板)

---

## 1. 快速使用指南（3分钟必读）

### 1.1 本文档定位

**不是什么**：
- ❌ 不是完整的预处理教程（请看其他文档）
- ❌ 不是最优方案（只是快速可用方案）
- ❌ 不替代深入诊断（重要项目请用完整流程）

**是什么**：
- ✅ 15分钟快速决策工具
- ✅ 基于经验的常用方案集合
- ✅ 快速建立Baseline的起点

### 1.2 何时使用本文档

| 场景 | 推荐使用 | 预计时间 |
|------|---------|---------|
| 新数据快速预处理 | ✅ 本文档 | 15分钟 |
| 快速Baseline建立 | ✅ 本文档 | 15-20分钟 |
| Kaggle快速试验 | ✅ 本文档 | 10-15分钟 |
| 重要项目 | ⚠️ 用完整流程 | 1-2小时 |
| 生产环境 | ⚠️ 用完整流程 | 2-3小时 |

### 1.3 使用流程

```
Step 1: 扫描数据概况（1分钟）
  └─ 运行快速诊断脚本

Step 2: 按5步流程决策（12分钟）
  └─ 逐步查询决策表

Step 3: 执行预处理代码（2分钟）
  └─ 使用快速代码模板

总计：15分钟
```

---

## 2. 5步快速预处理（15分钟）

### Step 1: 缺失值快速处理（3分钟）⭐

#### 决策表：缺失值处理方案

| 缺失率 | 缺失机制 | 推荐方案 | 代码位置 | 优先级 |
|--------|---------|---------|----------|--------|
| **<5%** | MCAR | 删除行 | `data_preprocessing.py:519` | P0 |
| **5-20%** | MCAR/MAR | 中位数填充（数值）<br>众数填充（分类） | `data_preprocessing.py:130`<br>`data_preprocessing.py:176` | P0 |
| **20-50%** | MAR | KNN填充 | SimpleImputer/KNNImputer | P1 |
| **>50%** | 任意 | **删除列** | `data_preprocessing.py:157` | P0 |
| 特殊：MNAR | - | 建模为特殊类别 | 手动处理 | P2 |

#### 快速判断缺失机制

**MCAR（完全随机）**：
- 缺失与任何变量无关
- 例如：传感器随机故障

**MAR（随机缺失）**：
- 缺失依赖于其他变量
- 例如：年轻人不愿填收入

**MNAR（非随机）**：
- 缺失本身有意义
- 例如：高收入者故意不填

#### 代码模板

```python
from src.data_preprocessing import handle_missing_values

# 快速处理：中位数（数值）+ 众数（分类）
df_filled = handle_missing_values(
    df,
    numeric_strategy='median',
    categorical_strategy='mode',
    drop_threshold=0.5  # 缺失>50%的列直接删除
)

print(f"✓ 缺失值处理完成")
```

---

### Step 2: 异常值快速处理（3分钟）⭐

#### 决策表：异常值处理方案

| 异常值性质 | 推荐方法 | 代码位置 | 适用场景 |
|-----------|---------|----------|---------|
| **数据错误** | 删除 | `data_preprocessing.py:234` | 年龄200岁、负数收入 |
| **真实极值+线性模型** | 鲁棒标准化 | `data_preprocessing.py:413` | 富豪收入、豪宅价格 |
| **真实极值+树模型** | 保留 | - | 树模型对异常值鲁棒 |
| **潜在欺诈** | 单独建模 | 异常检测模块 | 欺诈交易、异常行为 |

#### 快速检测方法

| 方法 | 何时使用 | 代码位置 | 计算速度 |
|------|---------|----------|---------|
| **IQR方法** | 偏态分布、快速筛查 | `data_preprocessing.py:193` | ⚡ 快 |
| **3σ法** | 正态分布 | 手动计算 | ⚡ 快 |
| **Isolation Forest** | 高维数据、多变量检测 | sklearn.ensemble | ⚠️ 中等 |

#### 代码模板

```python
from src.data_preprocessing import handle_outliers_iqr

# 快速处理：IQR方法 + 截断策略
df_clean = handle_outliers_iqr(
    df,
    columns=['income', 'age', 'transaction_amount'],  # 数值列
    method='clip',  # 截断到上下界（保留样本）
    k=1.5  # IQR倍数（1.5为标准值）
)

print(f"✓ 异常值处理完成")
```

---

### Step 3: 特征编码（3分钟）⭐

#### 决策表：分类特征编码方案

| 特征类型 | 类别数（基数） | 推荐编码 | 代码位置 | 适用算法 |
|---------|--------------|---------|----------|---------|
| **无序分类** | <10 | **One-Hot** | `data_preprocessing.py:374` | 线性模型、NN |
| **无序分类** | 10-50 | Target编码 | `data_preprocessing.py:299` | 任意 |
| **无序分类** | >50 | Target编码 + Frequency | 组合使用 | 树模型 |
| **有序分类** | 任意 | Label编码 | `data_preprocessing.py:383` | 树模型 |

#### 快速判断编码方式

**基数判断**：
```python
# 查看类别数
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    n_unique = df[col].nunique()
    print(f"{col}: {n_unique}个类别")
```

**建议**：
- 基数<10：One-Hot（安全选择）
- 基数10-50：Target编码（适合树模型）
- 基数>50：考虑特征哈希或删除

#### 代码模板

```python
from src.data_preprocessing import encode_categorical_features

# 方式1：快速One-Hot编码（适合低基数）
df_encoded = encode_categorical_features(
    df,
    columns=['gender', 'city', 'product_type'],
    method='onehot',
    drop_first=True  # 避免多重共线性
)

# 方式2：Label编码（适合树模型）
df_encoded = encode_categorical_features(
    df,
    columns=['education_level', 'rank'],  # 有序类别
    method='label'
)

print(f"✓ 特征编码完成")
```

---

### Step 4: 特征缩放（3分钟）⭐

#### 决策表：根据算法选择缩放方法

| 算法类型 | 是否需要缩放 | 推荐方法 | 代码位置 | 原因 |
|---------|------------|---------|----------|------|
| **线性回归/逻辑回归** | ✅ 必须 | StandardScaler | `data_preprocessing.py:409` | 基于梯度下降 |
| **SVM** | ✅ 必须 | StandardScaler | 同上 | 对特征尺度敏感 |
| **神经网络** | ✅ 必须 | MinMaxScaler [0,1] | `data_preprocessing.py:411` | 激活函数需要 |
| **KNN** | ✅ 必须 | StandardScaler | `data_preprocessing.py:409` | 基于距离 |
| **PCA** | ✅ 必须 | StandardScaler | 同上 | 方差敏感 |
| **树模型**（RF/XGB） | ❌ 不需要 | - | - | 对尺度不敏感 |
| **朴素贝叶斯** | ❌ 不需要 | - | - | 基于概率 |

#### 快速缩放方法对比

| 方法 | 公式 | 结果范围 | 何时使用 |
|------|------|---------|---------|
| **StandardScaler** | (X-μ)/σ | 均值0，方差1 | 正态分布、线性模型 |
| **MinMaxScaler** | (X-min)/(max-min) | [0,1] | 有界数据、神经网络 |
| **RobustScaler** | (X-median)/IQR | - | 有异常值 |

#### 代码模板

```python
from src.data_preprocessing import FeatureScaler

# 方式1：标准化（最常用）
scaler = FeatureScaler(method='standard')
df_scaled = scaler.fit_transform(df)

# 方式2：归一化（神经网络）
scaler = FeatureScaler(method='minmax')
df_scaled = scaler.fit_transform(df)

# 方式3：指定列缩放
scaler = FeatureScaler(method='standard')
df_scaled = scaler.fit_transform(df, columns=['age', 'income', 'score'])

print(f"✓ 特征缩放完成")
```

---

### Step 5: 快速特征工程（3分钟）⭐

#### P0优先级特征（必做）

| 特征类型 | 操作 | 代码位置 | 价值 |
|---------|------|----------|------|
| **时间特征** | 提取年/月/日/星期 | `feature_engineering.py:525` | ⭐⭐⭐ |
| **缺失值标记** | is_missing列 | 手动：`df['col_missing'] = df['col'].isnull()` | ⭐⭐⭐ |
| **交互特征（乘法）** | 重要特征相乘 | `feature_engineering.py:363` | ⭐⭐⭐ |

#### P1优先级特征（建议）

| 特征类型 | 操作 | 代码位置 | 价值 |
|---------|------|----------|------|
| **比例特征** | 部分/总计 | 手动计算 | ⭐⭐ |
| **聚合特征** | 分组统计 | `feature_engineering.py:411` | ⭐⭐ |
| **多项式特征** | X² | `feature_engineering.py:211` | ⭐ |

#### 代码模板

```python
from src.feature_engineering import extract_datetime_features, create_interaction_features

# 1. 时间特征提取（如有datetime列）
if 'signup_date' in df.columns:
    df = extract_datetime_features(df, datetime_column='signup_date')
    print(f"✓ 时间特征提取完成")

# 2. 缺失值标记（重要！）
for col in ['income', 'age']:  # 缺失较多的列
    if df[col].isnull().sum() > 0:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)

# 3. 交互特征（核心特征的乘法）
df = create_interaction_features(
    df,
    columns=['age', 'income', 'tenure'],  # 选择2-3个核心特征
    operations=['*']  # 只做乘法（最有用）
)

print(f"✓ 快速特征工程完成")
```

---

## 3. 常用决策矩阵速查

### 3.1 数据量与预处理策略

| 样本数 | 特征数 | 推荐策略 | 注意事项 |
|--------|--------|---------|---------|
| <1K | <20 | 简单填充 + 标准化 | 避免复杂方法（过拟合） |
| 1K-10K | 20-100 | 标准流程 | 可以尝试特征工程 |
| 10K-100K | 100-1000 | 完整流程 + 特征选择 | 注意计算效率 |
| >100K | >1000 | 自动化Pipeline | 使用并行处理 |

### 3.2 问题类型与预处理重点

| 问题类型 | 预处理重点 | 关键步骤 |
|---------|----------|---------|
| **分类（平衡）** | 标准化 + 编码 | 缺失值、编码、缩放 |
| **分类（不平衡）** | 数据平衡⭐ | SMOTE/权重调整 |
| **回归** | 异常值处理⭐ | 鲁棒标准化、log变换 |
| **时间序列** | 时间特征⭐ | 滑动窗口、滞后特征 |
| **文本分类** | 编码+降维 | TF-IDF、词嵌入 |

### 3.3 算法与预处理要求

| 算法 | 缺失值 | 异常值 | 编码 | 缩放 |
|------|--------|--------|------|------|
| **线性回归** | ❌ 不允许 | ⚠️ 敏感 | One-Hot | ✅ 必须 |
| **逻辑回归** | ❌ 不允许 | ⚠️ 敏感 | One-Hot | ✅ 必须 |
| **决策树** | ✅ 可选 | ✅ 鲁棒 | Label | ❌ 不需要 |
| **随机森林** | ✅ 可选 | ✅ 鲁棒 | Label | ❌ 不需要 |
| **XGBoost** | ✅ 原生支持 | ✅ 鲁棒 | Label | ❌ 不需要 |
| **SVM** | ❌ 不允许 | ❌ 敏感 | One-Hot | ✅ 必须 |
| **神经网络** | ❌ 不允许 | ⚠️ 敏感 | One-Hot | ✅ 必须 |
| **KNN** | ❌ 不允许 | ❌ 敏感 | One-Hot | ✅ 必须 |

---

## 4. 实战示例

### 示例1：客户流失预测（15分钟完整流程）

**数据概况**：
- 7043行 × 21列
- 目标：预测客户是否流失（二分类）
- 问题：11个缺失值，部分高基数分类特征

**快速预处理流程**：

```python
import pandas as pd
from src.data_preprocessing import *
from src.feature_engineering import *

# Step 1: 加载数据
df = pd.read_csv('telco_customer_churn.csv')
print(f"数据形状: {df.shape}")

# Step 2: 缺失值处理（3分钟）
# - TotalCharges列有11个缺失（0.16%）→ 删除行
df = df.dropna()
print(f"✓ Step 1: 删除11行缺失数据")

# Step 3: 异常值检查（快速跳过）
# - 业务字段无明显异常值
print(f"✓ Step 2: 无异常值需处理")

# Step 4: 特征编码（3分钟）
# - 低基数分类（gender, Partner等）→ One-Hot
# - 高基数分类（无）
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                    'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']

df_encoded = encode_categorical_features(
    df,
    columns=categorical_cols,
    method='onehot',
    drop_first=True
)
print(f"✓ Step 3: One-Hot编码完成，特征数：{df_encoded.shape[1]}")

# Step 5: 特征缩放（3分钟）
# - 使用逻辑回归 → 标准化
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = FeatureScaler(method='standard')
df_scaled = scaler.fit_transform(df_encoded, columns=numeric_cols)
print(f"✓ Step 4: 标准化完成")

# Step 6: 快速特征工程（3分钟）
# - 交互特征：tenure × MonthlyCharges = TotalValue
df_scaled['TotalValue'] = df_scaled['tenure'] * df_scaled['MonthlyCharges']
# - 缺失值标记（虽然已删除，但展示方法）
print(f"✓ Step 5: 特征工程完成")

# 最终结果
print(f"\n✅ 预处理完成！")
print(f"   最终形状: {df_scaled.shape}")
print(f"   总耗时: 约15分钟")
```

**结果**：
- 输入：7043行 × 21列
- 输出：7032行 × 47列（One-Hot后）
- 耗时：15分钟
- 可以直接用于建模

---

### 示例2：房价预测（10分钟快速处理）

**数据概况**：
- 1460行 × 80列
- 目标：预测房价（回归）
- 问题：多列缺失、偏态分布

**快速预处理流程**：

```python
# Step 1: 缺失值处理（5分钟）
df_filled = handle_missing_values(
    df,
    numeric_strategy='median',
    categorical_strategy='mode',
    drop_threshold=0.5  # 删除缺失>50%的列
)

# Step 2: 异常值处理（2分钟）
# - SalePrice右偏 → log变换
df_filled['SalePrice_log'] = np.log1p(df_filled['SalePrice'])

# - 其他数值列 → IQR截断
numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
df_clean = handle_outliers_iqr(
    df_filled,
    columns=numeric_cols,
    method='clip'
)

# Step 3: 编码+缩放（3分钟）
# - 分类特征 → Label编码（准备用树模型）
categorical_cols = df_clean.select_dtypes(include=['object']).columns
df_encoded = encode_categorical_features(
    df_clean,
    columns=categorical_cols,
    method='label'
)

# - 数值特征 → 标准化（备用）
scaler = FeatureScaler(method='standard')
df_final = scaler.fit_transform(df_encoded)

print(f"✅ 快速预处理完成！可用于树模型")
```

---

## 5. 快速代码模板

### 5.1 一键预处理Pipeline（最快）

```python
from src.data_preprocessing import build_preprocessing_pipeline
from sklearn.model_selection import train_test_split

# 分离特征和目标
X = df.drop('target', axis=1)
y = df['target']

# 识别特征类型
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 构建Pipeline
preprocessor = build_preprocessing_pipeline(
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    numeric_strategy='median',
    categorical_strategy='most_frequent',
    scaling_method='standard'
)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 拟合并转换
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"✅ 一键预处理完成！")
```

### 5.2 手动Pipeline（灵活）

```python
# 1. 缺失值
df = handle_missing_values(df, numeric_strategy='median', categorical_strategy='mode')

# 2. 异常值
df = handle_outliers_iqr(df, columns=numeric_cols, method='clip')

# 3. 编码
df = encode_categorical_features(df, columns=categorical_cols, method='onehot')

# 4. 缩放
scaler = FeatureScaler(method='standard')
df = scaler.fit_transform(df)

# 5. 特征工程（可选）
df = create_interaction_features(df, columns=['col1', 'col2'], operations=['*'])

print(f"✅ 手动预处理完成！")
```

---

## 📚 延伸阅读

### 需要更深入？

- **理论学习**：
  - [missing_values_strategies.md](missing_values_strategies.md) - 缺失值完整指南
  - [outlier_detection_methods.md](outlier_detection_methods.md) - 异常值检测方法
  - [feature_engineering_cookbook.md](feature_engineering_cookbook.md) - 特征工程详解

- **系统化流程**：
  - [preprocessing_decision_tree.md](preprocessing_decision_tree.md) - 完整决策树
  - [preprocessing_checklist.md](preprocessing_checklist.md) - 检查清单

### 常见问题

**Q1: 15分钟够吗？**
A: 对于快速Baseline够了，重要项目请用完整流程（1-2小时）

**Q2: 这是最优方案吗？**
A: 不是，但是快速可用的方案，足以建立Baseline

**Q3: 可以跳过某些步骤吗？**
A: 可以，但P0步骤（缺失值、编码、缩放）建议都做

**Q4: 树模型是否需要预处理？**
A: 缺失值和编码必须处理，缩放可以跳过

---

**最后更新**：2024年11月
**预计使用时间**：15分钟
**后续文档**：[preprocessing_decision_tree.md](preprocessing_decision_tree.md)
