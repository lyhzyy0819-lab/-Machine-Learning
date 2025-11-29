# 🔧 常见数据问题及解决方案

> 实战中最常遇到的数据问题及处理方法

## 🚀 快速导航

**来自速查表** → 如果从 [data_diagnosis_quick_reference.md](data_diagnosis_quick_reference.md) 跳转过来，本文档提供每个问题的**深入理解和详细方案**

**查找方案** → 如果需要直接可用的代码，查看 [data_problem_to_solution_mapping.md](data_problem_to_solution_mapping.md)

**系统化诊断** → 如果需要完整的诊断流程，查看 [data_diagnosis_decision_tree.md](data_diagnosis_decision_tree.md)

---

## 📚 目录

1. [缺失值问题](#1-缺失值问题) - 最常见，影响大
2. [异常值问题](#2-异常值问题) - 需判断性质
3. [数据不平衡](#3-数据不平衡) - 分类问题必查
4. [特征相关性问题](#4-特征相关性问题) - 线性模型敏感
5. [数据泄漏](#5-数据泄漏) - 最危险
6. [数据类型问题](#6-数据类型问题) - 基础问题
7. [特殊值问题](#7-特殊值问题) - 易被忽略
8. [时间相关问题](#8-时间相关问题) - 时序数据专用

---

## 1. 缺失值问题

### ❓ 问题描述

数据集中某些值为空（NaN、NULL、None等），导致无法直接建模。

### 📊 常见场景

```python
# 示例数据
age     income    city
25      50000     北京
NaN     60000     上海
30      NaN       广州
28      55000     NaN
```

### 🔍 诊断方法

```python
# 检查缺失值
print(df.isnull().sum())
print(df.isnull().sum() / len(df) * 100)  # 缺失率

# 可视化缺失模式
import missingno as msno
msno.matrix(df)
msno.heatmap(df)  # 缺失值相关性
```

### ✅ 解决方案

#### 方案1：删除法

**适用场景：** 缺失率<5% 且数据量充足

```python
# 删除含缺失值的行
df_cleaned = df.dropna()

# 删除缺失率>50%的列
threshold = 0.5
df_cleaned = df.loc[:, df.isnull().mean() < threshold]
```

**优点：** 简单直接
**缺点：** 丢失信息

#### 方案2：简单填充

**适用场景：** 轻度缺失，数据分布简单

```python
# 数值型：均值/中位数
df['age'].fillna(df['age'].median(), inplace=True)

# 类别型：众数
df['city'].fillna(df['city'].mode()[0], inplace=True)

# 常数填充
df['income'].fillna(0, inplace=True)  # 用0填充
df['city'].fillna('Unknown', inplace=True)  # 用特殊标记填充
```

**优点：** 快速，保留数据量
**缺点：** 可能引入偏差

#### 方案3：KNN填充

**适用场景：** 中度缺失，特征间有相关性

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_filled = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
```

**优点：** 考虑特征关系
**缺点：** 计算开销大

#### 方案4：建模填充

**适用场景：** 重度缺失，特征关系复杂

```python
# 使用其他特征预测缺失特征
from sklearn.ensemble import RandomForestRegressor

# 分离有/无缺失的数据
df_with_age = df[df['age'].notna()]
df_without_age = df[df['age'].isna()]

# 训练模型
X_train = df_with_age.drop('age', axis=1)
y_train = df_with_age['age']

model = RandomForestRegressor()
model.fit(X_train, y_train)

# 预测缺失值
X_pred = df_without_age.drop('age', axis=1)
df.loc[df['age'].isna(), 'age'] = model.predict(X_pred)
```

**优点：** 最准确
**缺点：** 复杂，可能过拟合

#### 方案5：保留缺失信息

**适用场景：** 缺失本身有意义

```python
# 创建缺失指示列
df['age_missing'] = df['age'].isna().astype(int)

# 然后填充缺失值
df['age'].fillna(df['age'].median(), inplace=True)
```

**适用案例：**
- 医疗数据：某些检查未做 → 可能表示健康
- 用户数据：某些字段未填 → 可能表示隐私意识

### 📈 效果对比

| 方法 | 信息保留 | 计算成本 | 引入偏差风险 |
|------|----------|----------|--------------|
| 删除 | ★☆☆☆☆ | ★★★★★ | ★☆☆☆☆ |
| 简单填充 | ★★☆☆☆ | ★★★★★ | ★★★☆☆ |
| KNN填充 | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ |
| 建模填充 | ★★★★☆ | ★☆☆☆☆ | ★★★★☆ |
| 保留标记 | ★★★★★ | ★★★★☆ | ★☆☆☆☆ |

---

## 2. 异常值问题

### ❓ 问题描述

数据中存在明显偏离正常范围的值，可能是错误或真实极值。

### 📊 常见场景

```python
# 年龄数据中出现负数或超大值
age: [25, 28, 30, -5, 150, 27, 26]

# 价格数据中的极端值
price: [100, 150, 200, 9999999, 180, 160]
```

### 🔍 诊断方法

```python
# 统计方法
print(df.describe())

# IQR方法
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['price'] < Q1 - 1.5*IQR) | (df['price'] > Q3 + 1.5*IQR)]

# Z-Score方法
from scipy import stats
z_scores = np.abs(stats.zscore(df['price']))
outliers = df[z_scores > 3]

# 可视化
df.boxplot(column='price')
```

### ✅ 解决方案

#### 方案1：删除异常值

**适用场景：** 明确的数据错误

```python
# 删除不合理的值
df = df[df['age'] >= 0]  # 年龄不能为负
df = df[df['age'] <= 120]  # 年龄不超过120

# 使用IQR删除
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]
```

**决策标准：**
- 业务上不可能（如负数年龄）→ 删除
- 明显的录入错误（如价格多打一个0）→ 删除或修正

#### 方案2：截断（Winsorization）

**适用场景：** 保留数据量，但限制极值影响

```python
# 设置上下限
lower_bound = df['price'].quantile(0.01)
upper_bound = df['price'].quantile(0.99)

df['price'] = df['price'].clip(lower=lower_bound, upper=upper_bound)
```

**优点：** 保留所有样本
**缺点：** 改变了数据分布

#### 方案3：转换

**适用场景：** 真实极值，需要保留但降低影响

```python
# 对数转换（适合右偏数据）
df['price_log'] = np.log1p(df['price'])

# 平方根转换
df['price_sqrt'] = np.sqrt(df['price'])

# Box-Cox转换（自动找最佳λ）
from scipy.stats import boxcox
df['price_boxcox'], lambda_param = boxcox(df['price'] + 1)
```

#### 方案4：使用鲁棒模型

**适用场景：** 异常值是真实数据的一部分

```python
# 使用对异常值不敏感的算法
from sklearn.ensemble import RandomForestRegressor  # 树模型对异常值不敏感
from sklearn.linear_model import HuberRegressor  # 鲁棒回归

# 或使用鲁棒缩放
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # 使用中位数和IQR，不受异常值影响
df_scaled = scaler.fit_transform(df)
```

### 🎯 决策流程图

```
发现异常值
    │
    ├─→ 是数据错误？ ─→ 是 ─→ 删除或修正
    │
    └─→ 否（真实极值）
            │
            ├─→ 样本量充足？ ─→ 是 ─→ 可以删除部分
            │
            └─→ 否 ─→ 保留 ─→ 选择：
                              1. 转换
                              2. 鲁棒模型
                              3. 截断
```

---

## 3. 数据不平衡

### ❓ 问题描述

分类问题中，某些类别样本数远少于其他类别。

### 📊 常见场景

```python
# 信用卡欺诈检测
正常交易: 99,500 (99.5%)
欺诈交易: 500 (0.5%)

# 疾病诊断
健康: 9,000 (90%)
患病: 1,000 (10%)
```

### 🔍 诊断方法

```python
# 统计类别分布
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True))

# 可视化
df['label'].value_counts().plot(kind='bar')
```

### ✅ 解决方案

#### 方案1：重采样

##### 过采样（增加少数类）

```python
from imblearn.over_sampling import SMOTE, RandomOverSampler

# 随机过采样
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# SMOTE（合成少数类样本）
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**优点：** 增加少数类样本
**缺点：** 可能过拟合

##### 欠采样（减少多数类）

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

**优点：** 快速平衡
**缺点：** 丢失信息

##### 组合采样

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)
```

#### 方案2：调整类权重

```python
from sklearn.ensemble import RandomForestClassifier

# 自动计算类权重
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# 手动设置类权重
model = RandomForestClassifier(class_weight={0: 1, 1: 10})
```

#### 方案3：改变评估指标

```python
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

# 不要只看准确率！
# 使用：F1-Score, AUC, Precision, Recall

# F1-Score
f1 = f1_score(y_true, y_pred)

# AUC
auc = roc_auc_score(y_true, y_pred_proba)

# PR曲线（不平衡数据更适合）
precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
```

#### 方案4：使用特殊算法

```python
# XGBoost/LightGBM有内置的不平衡处理
import xgboost as xgb

# 计算scale_pos_weight
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)
```

### 📈 方法对比

| 方法 | 数据量 | 过拟合风险 | 效果 | 难度 |
|------|--------|------------|------|------|
| 随机过采样 | 增加 | 高 | ★★☆ | ★☆☆ |
| SMOTE | 增加 | 中 | ★★★ | ★★☆ |
| 随机欠采样 | 减少 | 低 | ★★☆ | ★☆☆ |
| 类权重 | 不变 | 中 | ★★★ | ★☆☆ |
| 特殊算法 | 不变 | 低 | ★★★★ | ★★☆ |

---

## 4. 特征相关性问题

### ❓ 问题描述

多个特征高度相关（多重共线性），导致模型不稳定。

### 📊 常见场景

```python
# 相关特征
总面积 = 卧室面积 + 客厅面积 + 厨房面积  # 完全线性相关
BMI = 体重 / 身高²  # 数学关系
```

### 🔍 诊断方法

```python
# 相关系数矩阵
corr_matrix = df.corr()
print(corr_matrix)

# 可视化
import seaborn as sns
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# VIF（方差膨胀因子）
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)
```

### ✅ 解决方案

#### 方案1：删除高相关特征

```python
# 找出高相关特征对
threshold = 0.9
to_drop = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            to_drop.add(colname)

df_cleaned = df.drop(columns=to_drop)
```

**决策标准：**
1. 保留与目标相关性更高的
2. 保留业务意义更重要的
3. 保留更容易获取的

#### 方案2：PCA降维

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # 保留95%方差
X_pca = pca.fit_transform(X)
```

**优点：** 自动消除相关性
**缺点：** 特征不可解释

#### 方案3：正则化

```python
from sklearn.linear_model import Ridge, Lasso

# L2正则化（Ridge）
model = Ridge(alpha=1.0)

# L1正则化（Lasso）- 自动特征选择
model = Lasso(alpha=0.1)
```

#### 方案4：使用树模型

```python
# 树模型对多重共线性不敏感
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
# 无需处理多重共线性
```

---

## 5. 数据泄漏

### ❓ 问题描述

测试集信息"泄漏"到训练过程，导致评估结果过于乐观。

### 📊 常见场景

#### 场景1：ID列泄漏

```python
# ❌ 错误：用户ID被用作特征
user_id  feature1  target
1001     100       1
1002     200       0

# 如果测试集ID>2000，模型会过拟合ID范围
```

#### 场景2：未来信息泄漏

```python
# ❌ 错误：用未来数据预测过去
# 预测t时刻的销量，却使用了t+1时刻的库存数据
```

#### 场景3：目标变量泄漏

```python
# ❌ 错误：特征是目标的变种
target: 是否购买
feature: 购买金额  # 只有购买了才有金额！
```

#### 场景4：数据预处理泄漏

```python
# ❌ 错误：在划分前标准化
scaler.fit(X)  # 包含了测试集信息！
X_train, X_test = train_test_split(X)

# ✅ 正确：先划分，再标准化
X_train, X_test = train_test_split(X)
scaler.fit(X_train)  # 只用训练集
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 🔍 诊断方法

```python
# 1. 检查特征重要性
# 如果ID列重要性很高 → 可能泄漏

# 2. 检查训练/测试集性能差异
# 训练集AUC=0.99, 测试集AUC=0.6 → 可能过拟合/泄漏

# 3. 时间验证
# 对时间序列数据，用过去预测未来
```

### ✅ 解决方案

#### 通用原则

1. **删除ID列**
```python
df = df.drop(['user_id', 'order_id'], axis=1)
```

2. **时间顺序验证**
```python
# 不要随机划分时间序列数据！
# 用2019-2020训练，2021测试
train = df[df['date'] < '2021-01-01']
test = df[df['date'] >= '2021-01-01']
```

3. **严格的train-test分离**
```python
# 所有数据处理都要分开做
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 填充
imputer.fit(X_train)
X_train_filled = imputer.transform(X_train)
X_test_filled = imputer.transform(X_test)

# 编码
encoder.fit(X_train)
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)

# 缩放
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

4. **特征工程pipeline化**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

# Pipeline自动处理train-test分离
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

---

## 6. 数据类型问题

### ❓ 问题描述

数据类型识别错误，影响后续处理。

### 📊 常见场景

```python
# 数值型被识别为字符串
'12.5' → 字符串，无法计算均值

# 类别型被识别为数值
邮编: 100000 → 被当作数值，实际是类别

# 日期被识别为字符串
'2024-01-01' → 字符串，无法提取年月日
```

### ✅ 解决方案

```python
# 1. 检查数据类型
print(df.dtypes)

# 2. 转换数值类型
df['price'] = pd.to_numeric(df['price'], errors='coerce')  # 无法转换的变成NaN

# 3. 转换类别类型
df['zipcode'] = df['zipcode'].astype('category')

# 4. 转换日期类型
df['date'] = pd.to_datetime(df['date'])

# 5. 批量转换
df = df.convert_dtypes()  # pandas自动推断
```

---

## 7. 特殊值问题

### 📊 常见特殊值

```python
# -999, -99, 0, 9999 → 常用作缺失值标记
# inf, -inf → 除以0或计算溢出
# 空字符串 '' → 文本缺失
# 'Unknown', 'N/A', 'NULL' → 显式缺失标记
```

### ✅ 解决方案

```python
# 1. 替换特殊值为NaN
df.replace([-999, -99, 9999], np.nan, inplace=True)

# 2. 处理无穷值
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 3. 统一缺失标记
df.replace(['Unknown', 'N/A', 'NULL', ''], np.nan, inplace=True)

# 4. 检查所有数值是否有限
df = df[np.isfinite(df).all(axis=1)]
```

---

## 8. 时间相关问题

### ❓ 问题描述

时间数据处理不当，导致信息丢失或泄漏。

### ✅ 解决方案

```python
# 1. 提取时间特征
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

# 2. 时间差特征
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days

# 3. 周期性特征（正弦/余弦编码）
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# 4. 时间窗口特征
df['sales_7d_avg'] = df.groupby('user_id')['sales'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
```

---

## 🎯 快速诊断流程

```
1. 加载数据 → 检查dtypes
2. 基础统计 → df.describe(), df.info()
3. 缺失值 → df.isnull().sum()
4. 重复值 → df.duplicated().sum()
5. 异常值 → df.boxplot()
6. 分布 → df.hist()
7. 相关性 → df.corr()
8. 目标变量 → df[target].value_counts()
```

---

## 📚 推荐工具

```python
# 自动化数据分析报告
import pandas_profiling
profile = df.profile_report(title='Data Report')
profile.to_file("report.html")

# 数据清洗库
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
```

---

**下一步：** 查看 [02_问题定义指南](../02_problem_definition_guide/)，学习如何正确定义机器学习问题！
