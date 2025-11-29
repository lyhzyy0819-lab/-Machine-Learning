# 🔧 数据问题→解决方案映射表

> **用途**：已诊断出问题，快速查找处理方案
> **使用**：根据问题类型，查表获得代码和方案

---

## 📋 使用说明

本文档按**问题类型**组织，每个问题包含：
1. ⚡ 快速识别方法（检测代码）
2. 📊 严重程度判断（查表）
3. 💡 解决方案矩阵（多种方案对比）
4. 💻 代码示例（直接可用）
5. ⚠️ 注意事项（避坑指南）

**快速导航**：
- [1. 缺失值问题](#1-缺失值问题) - 最常见
- [2. 异常值问题](#2-异常值问题) - 影响大
- [3. 类别不平衡](#3-类别不平衡问题) - 分类必查
- [4. 重复值问题](#4-重复值问题) - 简单
- [5. 数据泄漏](#5-数据泄漏风险) - 严重
- [6. 数据类型](#6-数据类型问题) - 基础

---

## 1. 缺失值问题

### ⚡ 快速识别

```python
# 检测缺失值
missing_summary = df.isnull().sum()
missing_ratio = (df.isnull().sum() / len(df) * 100).round(2)

# 只显示有缺失的列
print("缺失值统计:")
print(missing_summary[missing_summary > 0])
print("\n缺失率(%):")
print(missing_ratio[missing_ratio > 0])

# 可视化（推荐）
import matplotlib.pyplot as plt
missing_cols = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
missing_cols.plot(kind='barh', figsize=(10, 6))
plt.xlabel('缺失率 (%)')
plt.title('各特征缺失值比例')
plt.show()
```

### 📊 严重程度判断

| 缺失率 | 严重程度 | 推荐方案 | 说明 |
|-------|---------|---------|------|
| <5% | ✅ 轻度 | 删除行或简单填充 | 信息损失小，快速处理 |
| 5-20% | ⚠️ 中度 | 智能填充（KNN/中位数） | 需要保留信息 |
| 20-50% | ❌ 严重 | 建模填充或删除列 | 权衡信息vs准确性 |
| >50% | ❌❌ 极严重 | 删除列 | 信息太少，无意义 |

### 💡 解决方案矩阵

#### 方案1：删除法（缺失<5%，数据量充足）

**代码**：
```python
# 删除含缺失值的行
df_clean = df.dropna()

print(f"删除前: {len(df)} 行")
print(f"删除后: {len(df_clean)} 行")
print(f"损失: {len(df) - len(df_clean)} 行 ({(len(df) - len(df_clean))/len(df)*100:.1f}%)")

# 或删除缺失率>50%的列
threshold = 0.5
cols_to_drop = df.columns[df.isnull().mean() > threshold]
df_clean = df.drop(columns=cols_to_drop)
print(f"删除列: {list(cols_to_drop)}")
```

✅ **适用**：MCAR（完全随机缺失），数据量充足（>10K）
⚠️ **注意**：删除会损失信息，确保损失<10%

#### 方案2：简单填充（缺失5-20%）

**代码**：
```python
import pandas as pd
import numpy as np

# 数值型：中位数填充（比均值更鲁棒，不受异常值影响）
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"{col}: 用中位数 {median_value:.2f} 填充")

# 类别型：众数填充
for col in df.select_dtypes(include=['object', 'category']).columns:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"{col}: 用众数 '{mode_value}' 填充")

# 或常数填充（如果缺失本身有意义）
df['income'].fillna(0, inplace=True)  # 收入缺失可能表示无收入
df['city'].fillna('Unknown', inplace=True)  # 城市缺失标记为未知
```

✅ **适用**：MAR（随机缺失），分布简单，快速处理
⚠️ **注意**：会低估方差，不适合缺失率>20%

#### 方案3：KNN填充（缺失>20%，特征相关）

**代码**：
```python
from sklearn.impute import KNNImputer
import pandas as pd

# 只对数值列使用KNN填充
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols]

# KNN填充（使用最近的5个邻居）
imputer = KNNImputer(n_neighbors=5)
df_filled_numeric = pd.DataFrame(
    imputer.fit_transform(df_numeric),
    columns=numeric_cols,
    index=df.index
)

# 合并回原DataFrame
df[numeric_cols] = df_filled_numeric

print("KNN填充完成")
```

✅ **适用**：MAR，特征间有相关性（如身高体重相关）
⚠️ **注意**：计算成本高，数据量>50K可能很慢

#### 方案4：建模填充（缺失>30%，最准确）

**代码**：
```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np

def model_imputation(df, col_to_fill):
    """
    使用机器学习模型填充缺失值

    原理：用其他特征预测缺失特征
    """
    # 分离有/无缺失的数据
    df_with_value = df[df[col_to_fill].notna()].copy()
    df_without_value = df[df[col_to_fill].isna()].copy()

    # 选择预测特征（除了待填充列，其他数值列）
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != col_to_fill]

    X_train = df_with_value[feature_cols].fillna(0)  # 简单处理其他缺失
    y_train = df_with_value[col_to_fill]
    X_pred = df_without_value[feature_cols].fillna(0)

    # 选择模型（数值用回归，类别用分类）
    if df[col_to_fill].dtype in [np.float64, np.int64]:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练并预测
    model.fit(X_train, y_train)
    predictions = model.predict(X_pred)

    # 填充
    df.loc[df[col_to_fill].isna(), col_to_fill] = predictions
    print(f"{col_to_fill}: 模型填充 {len(predictions)} 个缺失值")

    return df

# 使用示例
df = model_imputation(df, 'age')
```

✅ **适用**：缺失率高但特征重要，需要最准确的填充
⚠️ **注意**：可能过拟合，计算成本最高

#### 方案5：保留缺失信息（MNAR）

**代码**：
```python
# 创建缺失指示列（缺失本身可能有意义）
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[f'{col}_missing'] = df[col].isna().astype(int)
        print(f"创建指示列: {col}_missing")

# 然后用简单方法填充原列
df[col].fillna(df[col].median(), inplace=True)
```

✅ **适用**：MNAR（缺失本身有信息，如富人不愿填收入）
⚠️ **注意**：增加特征维度，可能引入噪声

### ⚠️ 注意事项

1. **先划分train/test，再填充**（避免数据泄漏）
```python
# ❌ 错误：在划分前填充
df_filled = df.fillna(df.median())
X_train, X_test = train_test_split(df_filled)

# ✅ 正确：先划分，再填充
X_train, X_test = train_test_split(df)
# 只用训练集的中位数
fill_values = X_train.median()
X_train_filled = X_train.fillna(fill_values)
X_test_filled = X_test.fillna(fill_values)  # 用训练集的统计量
```

2. **某些算法原生支持缺失值**
- XGBoost、LightGBM：可直接处理NaN
- 如果用这些算法，可以跳过填充

**延伸阅读** → [common_data_issues.md](common_data_issues.md) 第1节

---

## 2. 异常值问题

### ⚡ 快速识别

```python
import numpy as np
from scipy import stats

# 方法1：箱线图可视化（推荐）
import matplotlib.pyplot as plt
df.boxplot(figsize=(15, 10), rot=45)
plt.title('异常值检测（箱线图）')
plt.show()

# 方法2：IQR方法检测
def detect_outliers_iqr(df):
    outlier_summary = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_ratio = outliers / len(df) * 100

        if outliers > 0:
            outlier_summary[col] = {
                'count': outliers,
                'ratio': outlier_ratio,
                'bounds': (lower_bound, upper_bound)
            }

    return outlier_summary

outliers = detect_outliers_iqr(df)
for col, info in outliers.items():
    print(f"{col}: {info['count']} 个异常值 ({info['ratio']:.1f}%)")
    print(f"  正常范围: [{info['bounds'][0]:.2f}, {info['bounds'][1]:.2f}]")
```

### 📊 严重程度判断 & 性质判断

先判断**性质**，再决定方案：

| 异常值性质 | 判断依据 | 示例 | 推荐方案 |
|-----------|---------|------|---------|
| **数据错误** | 不符合业务逻辑 | 年龄200岁、收入为负 | 删除或修正 |
| **真实极值** | 符合业务逻辑但极端 | 富豪收入是普通人100倍 | 鲁棒标准化/log变换 |
| **潜在异常** | 可能是欺诈/异常事件 | 信用卡异常交易 | 单独建模（异常检测） |

### 💡 解决方案矩阵

#### 方案A：删除或修正（数据错误）

**代码**：
```python
# 示例：处理年龄异常
print(f"修正前: min={df['age'].min()}, max={df['age'].max()}")

# 删除不合理的值
df = df[(df['age'] >= 0) & (df['age'] <= 120)]

# 或修正明显的错误（如输入错误：200 → 20）
df.loc[df['age'] > 120, 'age'] = df[df['age'] > 120]['age'] / 10

print(f"修正后: min={df['age'].min()}, max={df['age'].max()}")
```

✅ **适用**：明确的数据错误
⚠️ **注意**：需要业务知识判断

#### 方案B：鲁棒标准化（真实极值）

**代码**：
```python
from sklearn.preprocessing import RobustScaler

# RobustScaler：使用中位数和IQR，不受极值影响
scaler = RobustScaler()

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("使用RobustScaler标准化完成")
```

✅ **适用**：真实极值，需要保留但降低影响
⚠️ **注意**：比StandardScaler更鲁棒

#### 方案C：log变换（右偏数据）

**代码**：
```python
# log变换（处理右偏数据，如收入、房价）
df['income_log'] = np.log1p(df['income'])  # log1p = log(1+x)，处理0值

# 对比原始和变换后的分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
df['income'].hist(bins=50, ax=ax1)
ax1.set_title('原始分布（右偏）')
df['income_log'].hist(bins=50, ax=ax2)
ax2.set_title('log变换后（正态）')
plt.show()
```

✅ **适用**：右偏分布（收入、房价、点击量等）
⚠️ **注意**：变换后要记得在预测时逆变换

#### 方案D：截断法（Winsorization）

**代码**：
```python
# 设置上下限（1%和99%分位数）
lower = df['income'].quantile(0.01)
upper = df['income'].quantile(0.99)

print(f"截断前: min={df['income'].min():.2f}, max={df['income'].max():.2f}")

# 截断（将超出范围的值设为边界值）
df['income'] = df['income'].clip(lower=lower, upper=upper)

print(f"截断后: min={df['income'].min():.2f}, max={df['income'].max():.2f}")
```

✅ **适用**：保留所有样本，但限制极值影响
⚠️ **注意**：改变了数据分布

#### 方案E：使用鲁棒算法（无需处理异常值）

**推荐算法**：
- **树模型**：RandomForest、XGBoost、LightGBM（对异常值不敏感）
- **鲁棒回归**：HuberRegressor、RANSACRegressor

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor

# 方案1：使用树模型（推荐）
model = RandomForestRegressor()  # 无需处理异常值

# 方案2：鲁棒回归
model = HuberRegressor()  # 对异常值鲁棒的线性回归
```

✅ **适用**：异常值是数据的一部分，需要保留
⚠️ **注意**：树模型对异常值最鲁棒

### ⚠️ 注意事项

**异常值 ≠ 一定要删除**

判断流程：
```
发现异常值
   ↓
业务逻辑判断
   ↓
┌─ 不合理（年龄负数、收入负数） → 数据错误 → 删除
│
├─ 合理但极端（富豪、豪宅） → 真实极值 → 鲁棒处理
│
└─ 可能是目标（欺诈检测） → 潜在异常 → 单独建模
```

**延伸阅读** → [common_data_issues.md](common_data_issues.md) 第2节

---

## 3. 类别不平衡问题

### ⚡ 快速识别

```python
# 检测目标变量分布
print("类别分布（数量）:")
print(df['target'].value_counts())

print("\n类别分布（比例）:")
print(df['target'].value_counts(normalize=True))

# 可视化
import matplotlib.pyplot as plt
df['target'].value_counts().plot(kind='bar')
plt.title('目标变量分布')
plt.xlabel('类别')
plt.ylabel('数量')
plt.show()

# 计算不平衡比例
value_counts = df['target'].value_counts()
max_ratio = value_counts.max() / value_counts.sum()
min_ratio = value_counts.min() / value_counts.sum()
imbalance_ratio = max_ratio / min_ratio
print(f"\n不平衡比例: {imbalance_ratio:.1f}:1")
```

### 📊 严重程度判断

| 比例 | 不平衡程度 | 推荐方案 | 评估指标 |
|------|----------|---------|---------|
| 4:6 ~ 5:5 | ✅ 平衡 | 无需特殊处理 | Accuracy |
| 3:7 | ⚠️ 轻度 | 调整评估指标 | F1-Score |
| 2:8 ~ 1:9 | ❌ 中度 | SMOTE或类权重 | F1/Precision/Recall |
| <1:10 | ❌❌ 严重 | 特殊算法+采样+集成 | AUC-ROC、PR-AUC |

### 💡 解决方案矩阵

#### 方案1：SMOTE过采样（中度不平衡）

**代码**：
```python
from imblearn.over_sampling import SMOTE
from collections import Counter

print("重采样前:", Counter(y))

# SMOTE：合成少数类样本（推荐）
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("重采样后:", Counter(y_resampled))
```

✅ **适用**：中度不平衡（1:5到1:10）
⚠️ **注意**：可能过拟合，仅用于训练集

#### 方案2：类权重调整（推荐，最简单）

**代码**：
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 方法A：自动计算类权重（推荐）
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# 方法B：手动设置类权重
# 如果0类:1类 = 9:1，给1类10倍权重
model = LogisticRegression(class_weight={0: 1, 1: 10})
model.fit(X_train, y_train)

# 方法C：计算类权重
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
print("类权重:", dict(enumerate(class_weights)))
```

✅ **适用**：所有不平衡场景，无需改变数据
⚠️ **注意**：不是所有算法都支持class_weight参数

#### 方案3：改变评估指标（必须）

**代码**：
```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)

# 不要只看Accuracy！
print("Accuracy:", accuracy_score(y_true, y_pred))  # ❌ 不平衡数据无意义

# ✅ 使用以下指标
print("F1-Score:", f1_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("AUC-ROC:", roc_auc_score(y_true, y_pred_proba))
print("PR-AUC:", average_precision_score(y_true, y_pred_proba))

# 混淆矩阵（详细分析）
print("\n混淆矩阵:")
print(confusion_matrix(y_true, y_pred))

# 分类报告（综合）
print("\n分类报告:")
print(classification_report(y_true, y_pred))
```

✅ **适用**：所有不平衡场景（必须做）
⚠️ **注意**：根据业务目标选择指标（Precision vs Recall）

#### 方案4：使用专门算法（严重不平衡）

**代码**：
```python
import xgboost as xgb
from sklearn.ensemble import BalancedRandomForestClassifier

# 方法A：XGBoost with scale_pos_weight
scale_pos_weight = (y == 0).sum() / (y == 1).sum()  # 负类/正类
model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)

# 方法B：BalancedRandomForest
model = BalancedRandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)
```

✅ **适用**：严重不平衡（>1:10）
⚠️ **注意**：配合方案2和方案3一起使用

#### 方案5：集成多种方法（最佳实践）

**代码**：
```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# 组合：SMOTE + 类权重 + 正确的评估指标
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(class_weight='balanced', n_estimators=100)
model.fit(X_train_res, y_train_res)

# 使用F1-Score评估
y_pred = model.predict(X_test)
print("F1-Score:", f1_score(y_test, y_pred))
```

✅ **适用**：严重不平衡，追求最佳性能
⚠️ **注意**：计算成本高，但效果最好

### ⚠️ 注意事项

1. **仅对训练集采样**（测试集保持原始分布）
```python
# ❌ 错误
X_res, y_res = SMOTE().fit_resample(X, y)
X_train, X_test = train_test_split(X_res, y_res)

# ✅ 正确
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)
# X_test保持不变
```

2. **根据业务目标选择评估指标**
- 欺诈检测：Recall优先（不能漏掉欺诈）
- 垃圾邮件：Precision优先（不能误判正常邮件）
- 平衡：F1-Score

**延伸阅读** → [common_data_issues.md](common_data_issues.md) 第3节

---

## 4. 重复值问题

### ⚡ 快速识别

```python
# 检测完全重复的行
n_duplicates = df.duplicated().sum()
print(f"重复行数: {n_duplicates} ({n_duplicates/len(df)*100:.1f}%)")

# 查看重复样本
duplicates = df[df.duplicated(keep=False)]  # keep=False显示所有重复
print(f"\n重复样本示例:")
print(duplicates.head(10))

# 检测特定列重复（如ID列）
if 'user_id' in df.columns:
    id_duplicates = df['user_id'].duplicated().sum()
    print(f"\nID重复数: {id_duplicates}")
```

### 💡 解决方案

**直接删除**（大多数情况）

```python
# 删除重复行（保留第一次出现）
print(f"删除前: {len(df)} 行")
df_clean = df.drop_duplicates()
print(f"删除后: {len(df_clean)} 行")
print(f"删除了 {len(df) - len(df_clean)} 行")

# 或指定列判断重复（如只看ID列）
df_clean = df.drop_duplicates(subset=['user_id'])

# 或保留最后一次出现
df_clean = df.drop_duplicates(keep='last')
```

### ⚠️ 注意事项

1. 确认是真重复还是数据错误
2. 某些业务场景重复是正常的（如用户多次购买）

---

## 5. 数据泄漏风险

### ⚡ 快速识别

```python
# 检查1：唯一值比例>95%的列（可能是ID）
unique_ratios = df.nunique() / len(df)
potential_ids = unique_ratios[unique_ratios > 0.95].index.tolist()
print(f"疑似ID列: {potential_ids}")

# 检查2：与目标完全相关（r≈1.0）
if 'target' in df.columns:
    corr_with_target = df.corr()['target'].abs().sort_values(ascending=False)
    potential_leakage = corr_with_target[corr_with_target > 0.99].index.tolist()
    print(f"\n疑似泄漏特征（与目标相关r>0.99）:")
    print(potential_leakage)

# 检查3：常量或准常量（方差≈0）
low_variance = df.var()
potential_constants = low_variance[low_variance < 0.01].index.tolist()
print(f"\n准常量特征（方差<0.01）: {potential_constants}")
```

### 💡 解决方案

**删除风险特征**

```python
# 删除ID列
id_cols = ['user_id', 'order_id', 'transaction_id']
df = df.drop(columns=[c for c in id_cols if c in df.columns])

# 删除泄漏特征
# 例如：预测是否购买，但有"购买金额"列（只有购买了才有金额）
df = df.drop(columns=['purchase_amount'])

# 删除常量列
constant_cols = df.columns[df.nunique() == 1]
df = df.drop(columns=constant_cols)

print(f"删除后剩余特征: {df.shape[1]}")
```

### ⚠️ 注意事项

**常见数据泄漏场景**：
1. ID列被用作特征
2. 未来信息（预测t时刻，使用了t+1时刻的数据）
3. 目标变量的变种（预测购买，包含购买金额）
4. 测试集统计量泄漏到训练集（错误的标准化）

**延伸阅读** → [common_data_issues.md](common_data_issues.md) 第5节

---

## 6. 数据类型问题

### ⚡ 快速识别

```python
# 检查数据类型
print(df.dtypes)

# 找出数值型被识别为字符串的列
for col in df.select_dtypes(include=['object']).columns:
    try:
        pd.to_numeric(df[col])
        print(f"{col}: 数值型被识别为字符串")
    except:
        pass
```

### 💡 解决方案矩阵

| 问题 | 检测 | 解决方案 |
|------|------|---------|
| 数值被识别为字符串 | `df.dtypes` 显示object | `pd.to_numeric(df['col'], errors='coerce')` |
| 类别被识别为数值 | 业务判断 | `df['zipcode'].astype('category')` |
| 日期被识别为字符串 | `df.dtypes` 显示object | `pd.to_datetime(df['date'])` |

**代码**：
```python
# 转换数值类型
df['price'] = pd.to_numeric(df['price'], errors='coerce')  # 无法转换的变NaN

# 转换类别类型
df['zipcode'] = df['zipcode'].astype('category')

# 转换日期类型
df['date'] = pd.to_datetime(df['date'])

# 批量自动推断
df = df.convert_dtypes()  # pandas自动推断类型
```

---

## 🎯 快速决策流程

```
遇到数据问题
   ↓
查看本文档目录 → 找到对应问题类型
   ↓
查看"快速识别"代码 → 确认问题
   ↓
查看"严重程度判断"表格 → 评估优先级
   ↓
查看"解决方案矩阵" → 选择合适方案
   ↓
复制"代码示例" → 直接使用
   ↓
完成处理 → 进入下一个问题
```

---

## 📖 相关文档

- **深入了解问题** → [common_data_issues.md](common_data_issues.md)
- **系统化诊断** → [data_diagnosis_decision_tree.md](data_diagnosis_decision_tree.md)
- **完整预处理** → [../04_preprocessing_and_features/](../04_preprocessing_and_features/)

---

**最后更新**：2024年11月
**核心价值**：问题→方案一对一映射，代码直接可用
**使用频率**：每次诊断发现问题后查阅

**下一步** → 查看 [04_preprocessing_and_features](../04_preprocessing_and_features/) 进行系统化预处理！
