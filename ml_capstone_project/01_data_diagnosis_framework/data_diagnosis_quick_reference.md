# 📋 数据诊断速查表

> **用途**：15分钟快速诊断数据问题
> **使用**：根据你的数据特征，查表定位问题

---

## 🚀 快速诊断流程（15分钟）

### Step 1: 基础信息扫描（3-5分钟）

运行以下诊断代码：

```python
import pandas as pd
import numpy as np

# 快速诊断脚本
print("="*60)
print("📊 基础信息")
print("="*60)
print(f"样本数: {len(df):,}")
print(f"特征数: {df.shape[1]}")
print(f"缺失值总数: {df.isnull().sum().sum():,}")
print(f"重复行数: {df.duplicated().sum():,}")
print(f"\n数据类型分布:")
print(df.dtypes.value_counts())
print("\n"+"="*60)
```

**记录关键数字**：
- 样本数: \_\_\_\_\_\_
- 特征数: \_\_\_\_\_\_
- 缺失值: \_\_\_\_\_\_
- 重复行: \_\_\_\_\_\_

### Step 2: 数据量级诊断（查表，2分钟）

根据你的数据量，查找对应行：

| 你的情况 | 诊断结果 | 关注点 | 下一步行动 |
|---------|---------|--------|-----------|
| 样本<1K，特征<20 | ⚠️ **小数据集** | 过拟合风险极高 | • 03章：选择简单模型（线性回归/逻辑回归）<br>• 05章：必须使用交叉验证<br>• 考虑数据增强或迁移学习 |
| 样本1K-10K，特征20-50 | ✅ **小中规模** | 适合大多数经典算法 | • 03章：标准算法流程<br>• 避免深度学习（数据不足） |
| 样本10K-100K，特征50-200 | ✅ **中等数据集** | 适合所有传统ML | • 03章：可尝试集成模型<br>• 标准ML工作流 |
| 样本>100K，特征>200 | ⚠️ **大数据集** | 计算资源/速度 | • 03章：选择高效算法（LightGBM）<br>• 04章：考虑采样加速<br>• 可能需要分布式计算 |
| **特征数>样本数** | ❌ **高维稀疏** | 维度灾难 | • 04章：**必须**降维或特征选择<br>• 03章：使用L1正则化（Lasso） |

### Step 3: 数据质量诊断（查表，5分钟）

检查4个关键质量指标：

| 问题类型 | 快速检测代码 | 严重程度判断 | 查看详细方案 |
|---------|------------|------------|-------------|
| **缺失值** | `df.isnull().sum()`<br>`(df.isnull().sum() / len(df) * 100).round(2)` | • >50% = 严重（删除列）<br>• 5-50% = 中度（智能填充）<br>• <5% = 轻度（简单填充） | [data_problem_to_solution_mapping.md](data_problem_to_solution_mapping.md) → 第1节 |
| **异常值** | `df.describe()` 看 min/max<br>或 `df.boxplot()` | • 异常率>5% = 需处理<br>• min/max不合理 = 数据错误 | [common_data_issues.md](common_data_issues.md) → 第2节 |
| **重复值** | `df.duplicated().sum()` | • >10% = 问题<br>• <1% = 可接受 | [data_problem_to_solution_mapping.md](data_problem_to_solution_mapping.md) → 第4节 |
| **类别不平衡** | `df[target].value_counts(normalize=True)` | • 最大类>80% = 严重<br>• 70-80% = 中度<br>• <70% = 轻度 | [common_data_issues.md](common_data_issues.md) → 第3节 |

#### 快速检测脚本

```python
# 缺失值快速检查
missing_summary = df.isnull().sum()
severe_missing = missing_summary[missing_summary / len(df) > 0.5]
print(f"⚠️  严重缺失列（>50%）: {len(severe_missing)}")
if len(severe_missing) > 0:
    print(severe_missing)

# 异常值快速检查（IQR方法）
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:5]:  # 检查前5个数值列
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    if outliers > 0:
        print(f"{col}: {outliers} 个异常值 ({outliers/len(df)*100:.1f}%)")

# 重复值检查
dup_count = df.duplicated().sum()
print(f"重复行: {dup_count} ({dup_count/len(df)*100:.1f}%)")

# 类别不平衡检查（如果有目标变量）
if 'target' in df.columns:
    target_dist = df['target'].value_counts(normalize=True)
    max_ratio = target_dist.max()
    print(f"目标变量最大类占比: {max_ratio*100:.1f}%")
    if max_ratio > 0.8:
        print("⚠️  严重不平衡！")
```

### Step 4: 数据分布诊断（查表，3分钟）

| 分布特征 | 检测方法 | 影响 | 快速处理方案 |
|---------|---------|------|------------|
| **右偏分布** | `df.skew() > 0.5` | 线性模型性能差 | `np.log1p(df[col])` |
| **左偏分布** | `df.skew() < -0.5` | 线性模型性能差 | `df[col] ** 2` |
| **多峰分布** | 绘制 `df.hist()` | 可能存在子群体 | 04章：聚类分析 |
| **长尾分布** | 极端值多 | 易过拟合 | `RobustScaler()` |

```python
# 快速偏度检查
numeric_df = df.select_dtypes(include=[np.number])
skewness = numeric_df.skew()
high_skew = skewness[abs(skewness) > 0.5]
print(f"高偏度特征（|skew|>0.5）: {len(high_skew)}")
print(high_skew)
```

### Step 5: 特征相关性诊断（查表，2分钟）

| 相关性情况 | 检测代码 | 问题 | 处理方案 |
|-----------|---------|------|---------|
| **特征间高度相关** | `df.corr().abs() > 0.9` | 多重共线性 | 删除冗余特征（保留一个） |
| **特征与目标弱相关** | `df.corr()['target'].abs() < 0.1` | 特征无效 | 04章：特征工程或删除 |
| **特征完全相关** | `df.corr() == 1.0` | 重复特征 | 删除其中一个 |

```python
# 快速相关性检查
corr_matrix = df.corr().abs()

# 找出高相关特征对
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.9:
            high_corr_pairs.append(
                (corr_matrix.columns[i], corr_matrix.columns[j],
                 corr_matrix.iloc[i, j])
            )

print(f"高相关特征对（r>0.9）: {len(high_corr_pairs)}")
for feat1, feat2, corr in high_corr_pairs[:5]:  # 显示前5对
    print(f"  {feat1} <-> {feat2}: {corr:.3f}")
```

---

## 📊 核心决策矩阵

### 矩阵1：缺失值处理决策

| 缺失率 | 缺失机制 | 推荐方案 | 代码示例 | 避免 |
|-------|---------|---------|---------|------|
| <5% | MCAR（完全随机） | 删除行 | `df.dropna()` | - |
| 5-20% | MAR（随机） | 均值/中位数填充 | `df.fillna(df.median())` | 删除列 |
| 20-50% | MAR | KNN/模型填充 | `KNNImputer(n_neighbors=5)` | 简单填充 |
| >50% | MNAR（非随机） | 删除列或特殊建模 | `df.drop(columns=[col])` | 保留使用 |

**缺失机制快速判断**：
- **MCAR**：缺失与任何变量无关（如随机抽样丢失）
- **MAR**：缺失与其他变量相关（如高收入者不愿填收入）
- **MNAR**：缺失与缺失值本身相关（如身体差才不填体检）

### 矩阵2：异常值处理决策

| 异常值性质 | 判断依据 | 推荐方案 | 代码示例 | 避免 |
|-----------|---------|---------|---------|------|
| **数据错误** | 不符合业务逻辑<br>（年龄200岁） | 删除或修正 | `df = df[df['age'] < 120]` | 保留 |
| **真实极值** | 符合业务逻辑<br>（富豪收入高） | 鲁棒标准化 | `RobustScaler().fit_transform(df)` | 简单删除 |
| **潜在异常** | 异常检测模型标记 | 单独建模 | `IsolationForest()` | 混在训练集 |

**快速判断流程**：
```
发现异常值
   ↓
业务逻辑判断
   ↓
┌─ 不合理（年龄负数） → 数据错误 → 删除
│
├─ 合理但极端（百万富翁） → 真实极值 → 鲁棒处理
│
└─ 不确定 → 可视化检查 → 咨询领域专家
```

### 矩阵3：类别不平衡处理决策

| 不平衡程度 | 比例 | 推荐方案 | 评估指标 | 避免 |
|-----------|------|---------|---------|------|
| 平衡 | 4:6 ~ 5:5 | 无需特殊处理 | Accuracy 可用 | 过度采样 |
| 轻度不平衡 | 3:7 | 调整评估指标 | F1-Score | 仍用Accuracy |
| 中度不平衡 | 2:8 ~ 1:9 | SMOTE 或类权重 | F1/Precision/Recall | 简单过采样 |
| 严重不平衡 | <1:10 | 特殊算法+采样+集成 | AUC-ROC | 只用一种方法 |

**快速方案代码**：
```python
# 方案A：SMOTE过采样
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(X, y)

# 方案B：类权重（推荐）
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')

# 方案C：改变评估指标
from sklearn.metrics import f1_score, roc_auc_score
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_proba)
```

### 矩阵4：数据量与算法选择

| 数据量级 | 推荐算法 | 避免算法 | 原因 |
|---------|---------|---------|------|
| 小<1K | 线性回归、逻辑回归、KNN、朴素贝叶斯 | 深度学习、复杂集成模型 | 过拟合风险 |
| 中1K-50K | 决策树、随机森林、XGBoost、SVM | 深度学习 | 数据量不足 |
| 大50K-500K | XGBoost、LightGBM、Random Forest | KNN（太慢）、层次聚类 | 计算效率 |
| 超大>500K | LightGBM、SGD、在线学习 | 全批量训练 | 内存限制 |

### 矩阵5：算法对数据要求的敏感度

| 算法类型 | 缺失值 | 异常值 | 数据标准化 | 类别编码 | 不平衡 |
|---------|-------|-------|-----------|---------|--------|
| **线性模型** | ❌ 不支持 | ⚠️ 敏感 | ✅ 必须 | ✅ 必须 | ⚠️ 敏感 |
| **树模型** | ✅ 支持 | ✅ 鲁棒 | ❌ 不需要 | ✅ 必须 | ⚠️ 中等 |
| **KNN** | ❌ 不支持 | ⚠️ 敏感 | ✅ 必须 | ✅ 必须 | ⚠️ 敏感 |
| **朴素贝叶斯** | ⚠️ 部分支持 | ✅ 鲁棒 | ❌ 不需要 | ✅ 必须 | ✅ 鲁棒 |
| **SVM** | ❌ 不支持 | ⚠️ 敏感 | ✅ 必须 | ✅ 必须 | ⚠️ 敏感 |
| **神经网络** | ❌ 不支持 | ⚠️ 敏感 | ✅ 必须（归一化） | ✅ 必须 | ⚠️ 敏感 |

**说明**：
- ✅ 支持/鲁棒/不需要：无需特殊处理
- ⚠️ 敏感/中等/部分支持：需要处理但不严格
- ❌ 不支持/必须：必须预处理

---

## 🎯 快速诊断检查清单（5分钟版）

快速勾选，确保不遗漏关键项：

### 基础检查
- [ ] 数据加载成功？（无编码错误）
- [ ] 样本数和特征数合理？（记录数字）
- [ ] 内存占用可接受？（<2GB 正常）

### 质量检查
- [ ] 缺失值检查完成？
  - [ ] 缺失率计算完成
  - [ ] 严重缺失列（>50%）已识别：\_\_\_个
- [ ] 异常值检查完成？
  - [ ] 使用IQR或3σ方法
  - [ ] 数值范围合理性检查
- [ ] 重复值检查完成？
  - [ ] 重复率：\_\_\_\_\_%
- [ ] 数据类型正确？
  - [ ] 数值列无字符串混入
  - [ ] 类别列已识别
  - [ ] 日期列已转换

### 目标变量检查（如适用）
- [ ] 目标变量分布检查？
  - [ ] 分类：类别平衡性：\_\_\_\_
  - [ ] 回归：值域范围：\_\_\_\_
- [ ] 评估指标选择？
  - [ ] 平衡数据：Accuracy
  - [ ] 不平衡数据：F1/AUC

### 特征检查
- [ ] 特征相关性检查？
  - [ ] 高相关特征对（r>0.9）：\_\_\_个
  - [ ] 与目标弱相关特征：\_\_\_个
- [ ] 数据泄漏风险评估？
  - [ ] ID列已删除？
  - [ ] 无未来信息？
  - [ ] 无目标泄漏？

### 后续规划
- [ ] 已确定需要处理的问题清单
- [ ] 已查阅对应的解决方案文档
- [ ] 准备好进入 02_problem_definition 或 04_preprocessing

**完成后下一步** → [data_problem_to_solution_mapping.md](data_problem_to_solution_mapping.md) 查找处理方案

---

## 💡 实战使用示例

### 示例1：客户流失预测项目

```
场景：拿到客户流失数据，需要快速诊断

Step 1: 基础扫描（3分钟）
运行快速诊断脚本
→ 结果：
   样本数: 50,000
   特征数: 25
   缺失值: 3,500
   重复行: 0
   数据类型: 20个数值型，5个类别型

Step 2: 数据量级诊断（1分钟）
→ 查表：50K样本×25特征 = 中等数据集
→ 结论：✅ 适合所有传统ML算法，标准流程

Step 3: 数据质量诊断（5分钟）
→ 缺失值：3500/(50000×25) = 0.28%
→ 查表：<5% = 轻度缺失
→ 处理方案：简单填充（中位数/众数）

→ 运行类别不平衡检查：
   流失: 5,000 (10%)
   未流失: 45,000 (90%)
→ 查表：1:9 = 中度不平衡
→ 处理方案：SMOTE 或类权重调整

Step 4: 分布诊断（2分钟）
→ 检查偏度：收入列 skew=2.1（右偏）
→ 处理方案：log变换

Step 5: 相关性诊断（2分钟）
→ 发现：浏览次数 与 点击次数 r=0.95
→ 处理方案：删除点击次数（保留浏览次数）

总结（2分钟）：
问题清单：
1. P1：中度不平衡（需SMOTE或类权重）
2. P2：轻度缺失（简单填充）
3. P2：1列右偏（log变换）
4. P2：1对高相关（删除一个）

下一步：
→ 查看 data_problem_to_solution_mapping.md
→ 复制代码，开始处理
→ 进入 03_algorithm_selection 选择算法

总耗时：15分钟完成诊断！
```

### 示例2：房价预测项目

```
场景：房价预测，已知有异常值问题

Step 1: 问题驱动（直接跳到异常值）
→ 打开本文档，搜索"异常值"
→ 找到"矩阵2：异常值处理决策"

Step 2: 判断异常值性质
→ 房价中有几个豪宅价格是普通房的100倍
→ 业务判断：真实极值（非数据错误）
→ 查表：真实极值 → 鲁棒标准化

Step 3: 复制代码
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])

Step 4: 进入 04章 详细处理

总耗时：5分钟解决问题！
```

---

## 📌 重要提醒

### 诊断 ≠ 处理

- **本文档作用**：**发现**问题，不解决问题
- **实际处理代码**：在 [04_preprocessing_and_features](../04_preprocessing_and_features/) 章节

### 优先级原则

不是所有问题都要处理：

**P0（必须处理）**：
- 数据泄漏
- 严重缺失（>50%）
- 明显数据错误

**P1（建议处理）**：
- 中度缺失（5-50%）
- 异常值（>5%）
- 严重类别不平衡（>80%）

**P2（可选处理）**：
- 轻度缺失（<5%）
- 弱相关特征
- 轻度偏态

### 算法选择关联

诊断结果直接影响算法选择：

| 诊断发现 | 算法选择建议（03章） |
|---------|-------------------|
| 数据量小（<1K） | 避免复杂模型，选线性回归/逻辑回归 |
| 异常值多 | 选择鲁棒算法（树模型） |
| 类别不平衡 | XGBoost（支持类权重）或专门算法 |
| 特征高度相关 | L1正则化（Lasso）自动特征选择 |
| 非线性关系 | 避免线性模型，选SVM/树模型 |

---

## 🔍 常见陷阱与解决

### 陷阱1：只看缺失率，不看缺失模式

**错误做法**：看到30%缺失就删除列

**正确做法**：
1. 判断缺失机制（MCAR/MAR/MNAR）
2. 如果是MAR，用KNN填充可能比删除更好

### 陷阱2：盲目删除异常值

**错误做法**：IQR检测到异常值就直接删除

**正确做法**：
1. 业务判断：是数据错误还是真实极值？
2. 真实极值：使用鲁棒方法而非删除

### 陷阱3：忽略类别不平衡

**错误做法**：不平衡数据仍用Accuracy评估

**正确做法**：
1. 使用F1-Score、AUC-ROC
2. 调整算法（SMOTE或类权重）

### 陷阱4：数据泄漏未发现

**错误做法**：直接用所有特征建模

**正确做法**：
1. 检查唯一值比例>95%的列（可能是ID）
2. 检查与目标完全相关的列（r≈1.0）
3. 检查是否有未来信息

---

## 📖 相关文档快速跳转

| 需求 | 文档 | 章节 |
|------|------|------|
| **查找处理方案** | [data_problem_to_solution_mapping.md](data_problem_to_solution_mapping.md) | 全部 |
| **深入了解某问题** | [common_data_issues.md](common_data_issues.md) | 按问题类型 |
| **系统化完整诊断** | [data_diagnosis_decision_tree.md](data_diagnosis_decision_tree.md) | 14步流程 |
| **检查不遗漏** | [diagnosis_checklist.md](diagnosis_checklist.md) | 完整清单 |
| **实际处理数据** | [04_preprocessing_and_features/](../04_preprocessing_and_features/) | 全章 |
| **选择算法** | [03_algorithm_selection_matrix/](../03_algorithm_selection_matrix/) | 全章 |

---

**最后更新**：2024年11月
**建议使用频率**：每次拿到新数据都用一次（15分钟）
**核心价值**：快速定位问题，避免盲目处理

**下一步** → 根据诊断结果，查阅对应文档获取解决方案！
