# 📊 数据诊断决策模板

> **用途**：快速诊断数据质量，识别关键问题
> **使用场景**：拿到新数据集时的第一步
> **使用时间**：⚡ 快速模式5分钟 | 📊 完整模式15分钟

---

## ⚡ 快速诊断卡片（5分钟）

### 使用说明

1. **运行诊断代码**获取基础信息（2分钟）
2. **填写以下卡片**（2分钟）
3. **查看自动推荐**的处理方案（1分钟）

---

### 诊断卡片（可复制填写）

```
===========================================
数据诊断卡片 - [项目名称]
诊断日期：____年____月____日
===========================================

【基础信息】
□ 数据规模：____行 × ____列
□ 目标变量：____（类型：[ ] 分类 [ ] 回归）

【数据规模诊断】
□ [ ] 小样本（<1,000行）        → ⚠️ 避免复杂模型
□ [ ] 中等样本（1K-100K行）     → ✅ 常规方法适用
□ [ ] 大样本（100K-1M行）       → ✅ 可用复杂模型
□ [ ] 超大样本（>1M行）         → 💡 考虑采样/分布式

【数据质量诊断】
□ 缺失率：____%
  [ ] 低（<5%） [ ] 中（5-20%） [ ] 高（>20%）

□ 重复样本：____%
  [ ] 无 [ ] 少量（<1%） [ ] 较多（1-5%） [ ] 大量（>5%）

□ 异常值情况：
  [ ] 无明显异常
  [ ] 少量异常（<5%特征受影响）
  [ ] 较多异常（5-20%特征受影响）
  [ ] 大量异常（>20%特征受影响）

【类别平衡诊断】（仅分类任务）
□ 最小类别占比：____%
  [ ] 平衡（>20%）
  [ ] 轻度不平衡（10-20%）
  [ ] 中度不平衡（5-10%）
  [ ] 严重不平衡（<5%）

【特征质量诊断】
□ 常数特征：____个（方差=0）
□ 重复特征：____组
□ 高相关特征对（|r|>0.9）：____对
□ 数据类型错误：[ ] 有 [ ] 无

【P0级问题（必须处理）】
1. _________________________________
2. _________________________________
3. _________________________________

【P1级问题（建议处理）】
1. _________________________________
2. _________________________________

===========================================
→ 自动推荐的后续行动（根据上述信息）
===========================================
```

---

### 诊断代码（2分钟运行）

```python
import pandas as pd
import numpy as np

def quick_diagnosis(df, target_col=None):
    """
    快速数据诊断函数

    Parameters:
    -----------
    df : DataFrame
        待诊断的数据集
    target_col : str, optional
        目标变量列名（分类/回归任务）

    Returns:
    --------
    dict : 诊断结果字典
    """

    diagnosis = {}

    # 1. 基础信息
    diagnosis['shape'] = df.shape
    diagnosis['memory_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024

    # 2. 数据规模分类
    n_rows = df.shape[0]
    if n_rows < 1000:
        diagnosis['size_category'] = '小样本(<1K)'
        diagnosis['size_warning'] = '⚠️ 避免复杂模型，考虑数据增强'
    elif n_rows < 100000:
        diagnosis['size_category'] = '中等样本(1K-100K)'
        diagnosis['size_warning'] = '✅ 常规方法适用'
    elif n_rows < 1000000:
        diagnosis['size_category'] = '大样本(100K-1M)'
        diagnosis['size_warning'] = '✅ 可使用复杂模型'
    else:
        diagnosis['size_category'] = '超大样本(>1M)'
        diagnosis['size_warning'] = '💡 考虑采样或分布式计算'

    # 3. 缺失值诊断
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    diagnosis['missing_pct'] = missing_pct[missing_pct > 0]
    diagnosis['overall_missing'] = df.isnull().sum().sum() / df.size * 100

    if diagnosis['overall_missing'] < 5:
        diagnosis['missing_severity'] = '低（<5%）'
    elif diagnosis['overall_missing'] < 20:
        diagnosis['missing_severity'] = '中（5-20%）'
    else:
        diagnosis['missing_severity'] = '高（>20%）'

    # 4. 重复样本
    duplicates = df.duplicated().sum()
    diagnosis['duplicate_count'] = duplicates
    diagnosis['duplicate_pct'] = duplicates / len(df) * 100

    # 5. 常数特征（方差为0）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_features = []
    for col in numeric_cols:
        if df[col].nunique() == 1:
            constant_features.append(col)
    diagnosis['constant_features'] = constant_features

    # 6. 数据类型
    diagnosis['dtypes'] = df.dtypes.value_counts().to_dict()

    # 7. 目标变量分析（如果提供）
    if target_col and target_col in df.columns:
        diagnosis['target_type'] = str(df[target_col].dtype)
        diagnosis['target_nunique'] = df[target_col].nunique()

        # 分类任务的类别平衡分析
        if df[target_col].nunique() < 20:  # 假设<20类为分类任务
            value_counts = df[target_col].value_counts()
            diagnosis['target_distribution'] = value_counts.to_dict()
            diagnosis['min_class_pct'] = (value_counts.min() / len(df) * 100)

            if diagnosis['min_class_pct'] > 20:
                diagnosis['balance'] = '平衡（>20%）'
            elif diagnosis['min_class_pct'] > 10:
                diagnosis['balance'] = '轻度不平衡（10-20%）'
            elif diagnosis['min_class_pct'] > 5:
                diagnosis['balance'] = '中度不平衡（5-10%）'
            else:
                diagnosis['balance'] = '严重不平衡（<5%）'

    # 8. 高相关特征对
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_pairs = [
            (col, row)
            for col in upper_triangle.columns
            for row in upper_triangle.index
            if upper_triangle.loc[row, col] > 0.9
        ]
        diagnosis['high_corr_pairs'] = high_corr_pairs

    return diagnosis


# 使用示例
# df = pd.read_csv('your_data.csv')
# result = quick_diagnosis(df, target_col='target')
# print(result)
```

---

### 自动推荐方案（根据诊断结果）

| 诊断项 | 情况 | 推荐方案 | 优先级 |
|--------|------|---------|--------|
| **数据规模** | 小样本(<1K) | 避免深度学习，优先简单模型（逻辑回归、决策树）<br>考虑数据增强或迁移学习 | P0 |
| | 中等样本(1K-100K) | 常规方法适用（随机森林、XGBoost） | - |
| | 大样本(>100K) | 可使用集成学习、深度学习 | - |
| | 超大样本(>1M) | 考虑采样策略或分布式框架 | P1 |
| **缺失率** | 低(<5%) | 简单删除或填充 | P1 |
| | 中(5-20%) | 根据缺失机制选择：均值/中位数/模型填充 | P0 |
| | 高(>20%) | ⚠️ 评估数据可用性，考虑特征删除或MICE填充 | P0 |
| **重复样本** | <1% | 删除重复 | P1 |
| | >5% | ⚠️ 检查是否为数据错误，删除重复 | P0 |
| **类别不平衡** | 平衡 | 无需特殊处理 | - |
| | 轻度不平衡 | 使用分层抽样，class_weight参数 | P1 |
| | 中度不平衡 | SMOTE过采样或欠采样 | P0 |
| | 严重不平衡 | 考虑异常检测方法（Isolation Forest） | P0 |
| **常数特征** | 存在 | 删除（无信息量） | P0 |
| **高相关特征** | >5对 | 特征选择，移除冗余特征 | P1 |

---

## 📊 完整诊断决策树（15分钟）

### 5步系统化诊断流程

```
新数据集
    ↓
┌─────────────────────────────────────┐
│ Step 1: 基础信息扫描（3分钟）        │
│ ✓ 数据规模、内存占用                │
│ ✓ 特征数量和类型                    │
│ ✓ 目标变量分布                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 2: 数据质量评估（5分钟）        │
│ ✓ 缺失值模式和比例                  │
│ ✓ 重复样本检测                      │
│ ✓ 异常值初步判断                    │
│ ✓ 数据类型正确性                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 3: 分布特征分析（3分钟）        │
│ ✓ 目标变量分布                      │
│ ✓ 数值特征分布（偏态、多峰）        │
│ ✓ 分类特征基数                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 4: 特征关系分析（2分钟）        │
│ ✓ 高相关特征对                      │
│ ✓ 常数和低方差特征                  │
│ ✓ 潜在数据泄漏                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 5: 问题优先级排序（2分钟）      │
│ ✓ P0必须处理的问题                  │
│ ✓ P1建议处理的问题                  │
│ ✓ P2可选优化的问题                  │
└─────────────────────────────────────┘
    ↓
数据诊断报告 + 处理方案
```

---

### Step 1: 基础信息扫描（3分钟）

#### 检查清单

```
- [ ] 数据已成功加载，无报错
- [ ] 记录数据形状：____行 × ____列
- [ ] 记录内存占用：____MB
- [ ] 识别目标变量类型
      [ ] 分类（<20个唯一值）
      [ ] 回归（连续数值）
      [ ] 无监督（无目标变量）
- [ ] 统计特征类型分布
      数值特征：____个
      分类特征：____个
      日期特征：____个
      文本特征：____个
```

#### 诊断代码

```python
# 基础信息扫描
print("="*50)
print("Step 1: 基础信息扫描")
print("="*50)

print(f"\n数据形状: {df.shape[0]:,} 行 × {df.shape[1]} 列")
print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print("\n特征类型分布:")
print(df.dtypes.value_counts())

print("\n前5行数据预览:")
print(df.head())

print("\n基本统计信息:")
print(df.describe())
```

---

### Step 2: 数据质量评估（5分钟）

#### 检查清单

```
【缺失值检查】
- [ ] 统计缺失值比例
- [ ] 识别缺失严重的特征（>50%）
- [ ] 判断缺失模式
      [ ] MCAR（完全随机缺失）
      [ ] MAR（随机缺失）
      [ ] MNAR（非随机缺失）

【重复样本检查】
- [ ] 完全重复：____行
- [ ] 处理建议：[ ] 删除 [ ] 保留（合理重复）

【异常值检查】
- [ ] 使用3σ或IQR法初步检测
- [ ] 记录异常特征：____________
- [ ] 可视化异常分布（箱线图）

【数据类型检查】
- [ ] 数值列误标为object：____个
- [ ] 分类列误标为数值：____个
- [ ] 需要修正：[ ] 是 [ ] 否
```

#### 诊断代码

```python
print("\n" + "="*50)
print("Step 2: 数据质量评估")
print("="*50)

# 2.1 缺失值分析
print("\n【缺失值分析】")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
missing_df = pd.DataFrame({
    '缺失数量': missing,
    '缺失比例(%)': missing_pct
})
print(missing_df[missing_df['缺失数量'] > 0])

# 2.2 重复样本
print(f"\n【重复样本】")
duplicates = df.duplicated().sum()
print(f"重复样本数量: {duplicates} ({duplicates/len(df)*100:.2f}%)")

# 2.3 异常值检测（数值特征）
print(f"\n【异常值检测（3σ法）】")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} 个异常值 ({len(outliers)/len(df)*100:.2f}%)")

# 2.4 数据类型检查
print(f"\n【数据类型】")
print(df.dtypes)
```

---

### Step 3: 分布特征分析（3分钟）

#### 检查清单

```
【目标变量分布】
- [ ] 可视化目标变量分布
- [ ] 分类任务：检查类别平衡
      最小类占比：____%
      [ ] 平衡 [ ] 不平衡
- [ ] 回归任务：检查分布形态
      [ ] 正态分布
      [ ] 偏态分布（偏度：____）
      [ ] 长尾分布

【数值特征分布】
- [ ] 识别偏态分布特征（|skew|>1）：____个
- [ ] 识别长尾分布特征：____个
- [ ] 是否需要变换：[ ] 是 [ ] 否

【分类特征基数】
- [ ] 低基数（<10）：____个
- [ ] 中基数（10-50）：____个
- [ ] 高基数（>50）：____个
```

#### 诊断代码

```python
print("\n" + "="*50)
print("Step 3: 分布特征分析")
print("="*50)

# 3.1 目标变量分布（假设目标列为'target'）
if 'target' in df.columns:
    print("\n【目标变量分布】")
    print(df['target'].value_counts())
    print(f"\n类别占比:")
    print(df['target'].value_counts(normalize=True) * 100)

# 3.2 数值特征偏度
print("\n【数值特征偏度】")
numeric_cols = df.select_dtypes(include=[np.number]).columns
skewness = df[numeric_cols].skew().sort_values(ascending=False)
print("偏度绝对值>1的特征（建议变换）:")
print(skewness[abs(skewness) > 1])

# 3.3 分类特征基数
print("\n【分类特征基数】")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    nunique = df[col].nunique()
    print(f"{col}: {nunique} 个唯一值", end="")
    if nunique < 10:
        print(" [低基数]")
    elif nunique < 50:
        print(" [中基数]")
    else:
        print(" [高基数]")
```

---

### Step 4: 特征关系分析（2分钟）

#### 检查清单

```
【相关性分析】
- [ ] 计算特征相关矩阵
- [ ] 识别高相关特征对（|r|>0.9）：____对
- [ ] 是否需要特征选择：[ ] 是 [ ] 否

【特征质量】
- [ ] 常数特征（方差=0）：____个
- [ ] 低方差特征（<0.01）：____个
- [ ] 建议删除：[ ] 是 [ ] 否

【数据泄漏检查】
- [ ] 检查特征与目标相关性（|r|>0.95可疑）
- [ ] 检查未来信息（如订单日期晚于预测日期）
- [ ] 可疑特征：____________
```

#### 诊断代码

```python
print("\n" + "="*50)
print("Step 4: 特征关系分析")
print("="*50)

# 4.1 高相关特征对
print("\n【高相关特征对（|r|>0.9）】")
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr = [(col, row, upper.loc[row, col])
                 for col in upper.columns
                 for row in upper.index
                 if upper.loc[row, col] > 0.9]

    if high_corr:
        for col1, col2, corr in high_corr:
            print(f"{col1} <-> {col2}: {corr:.3f}")
    else:
        print("无高相关特征对")

# 4.2 常数和低方差特征
print("\n【常数/低方差特征】")
for col in numeric_cols:
    var = df[col].var()
    if var == 0:
        print(f"{col}: 常数特征（方差=0）")
    elif var < 0.01:
        print(f"{col}: 低方差特征（方差={var:.4f}）")
```

---

### Step 5: 问题优先级排序（2分钟）

#### 决策矩阵

| 问题类型 | 严重程度判断 | 优先级 | 推荐行动 |
|---------|------------|--------|---------|
| **缺失率>50%** | 严重 | P0 | 评估特征价值，考虑删除或MICE填充 |
| **严重类别不平衡(<5%)** | 严重 | P0 | SMOTE/异常检测方法 |
| **大量重复样本(>5%)** | 严重 | P0 | 检查数据来源，删除重复 |
| **常数特征** | 严重 | P0 | 立即删除（无信息） |
| **数据泄漏** | 严重 | P0 | 移除泄漏特征 |
| **缺失率10-50%** | 中等 | P1 | 根据缺失机制填充 |
| **轻度不平衡(10-20%)** | 中等 | P1 | 分层抽样+class_weight |
| **高相关特征对** | 中等 | P1 | 特征选择 |
| **偏态分布** | 轻微 | P2 | 对数变换/Box-Cox |
| **高基数分类特征** | 轻微 | P2 | Target编码 |

#### 输出模板

```
===========================================
数据诊断报告 - 问题优先级
===========================================

【P0级问题（必须处理，影响模型可用性）】
1. ___________________________________________
   严重程度：⚠️⚠️⚠️
   推荐方案：_________________________________

2. ___________________________________________
   严重程度：⚠️⚠️⚠️
   推荐方案：_________________________________

【P1级问题（建议处理，影响模型性能）】
1. ___________________________________________
   严重程度：⚠️⚠️
   推荐方案：_________________________________

2. ___________________________________________
   严重程度：⚠️⚠️
   推荐方案：_________________________________

【P2级问题（可选优化，锦上添花）】
1. ___________________________________________
   严重程度：⚠️
   推荐方案：_________________________________

===========================================
→ 下一步行动（按优先级执行）
===========================================
1. 处理P0级问题（必须）
2. 处理P1级问题（建议）
3. 评估是否处理P2级问题
4. 进入算法选择阶段 → algorithm_selection_template.md
```

---

## 📚 实战案例

### 案例1：客户流失预测项目

**背景**：电信公司客户流失预测，数据集7043行×21列

**诊断过程**（使用快速诊断卡）：

```
===========================================
数据诊断卡片 - 电信客户流失预测
诊断日期：2024年11月
===========================================

【基础信息】
□ 数据规模：7,043 行 × 21 列
□ 目标变量：Churn（类型：[✓] 分类）

【数据规模诊断】
□ [✓] 中等样本（1K-100K行） → ✅ 常规方法适用

【数据质量诊断】
□ 缺失率：0.16%
  [✓] 低（<5%）

□ 重复样本：0%
  [✓] 无

□ 异常值情况：
  [✓] 少量异常（TotalCharges列有11个异常值）

【类别平衡诊断】
□ 最小类别占比：26.5%
  [✓] 平衡（>20%）

【特征质量诊断】
□ 常数特征：0个
□ 重复特征：0组
□ 高相关特征对（|r|>0.9）：2对
  - tenure <-> TotalCharges (0.95)
  - MonthlyCharges <-> TotalCharges (0.92)
□ 数据类型错误：[✓] 有（TotalCharges应为数值型）

【P0级问题】
1. TotalCharges列数据类型错误（object应为float）

【P1级问题】
1. 高相关特征对：考虑删除TotalCharges
2. TotalCharges列11个异常值需处理

===========================================
→ 自动推荐的后续行动
===========================================
1. 修正TotalCharges数据类型
2. 删除高相关的TotalCharges特征（保留tenure和MonthlyCharges）
3. 由于类别平衡，可直接使用标准分类算法
4. 数据质量良好，可以进入算法选择阶段
```

**实施结果**：
- 5分钟完成快速诊断
- 识别出1个P0问题和2个P1问题
- 处理后数据质量显著提升，模型AUC从0.82提升到0.86

---

### 案例2：房价预测项目

**背景**：波士顿房价预测，数据集506行×14列

**诊断过程**（使用完整决策树）：

**Step 1: 基础信息**
- 数据形状：506行 × 14列
- 内存占用：0.05 MB
- 目标变量：MEDV（中位房价，回归任务）

**Step 2: 数据质量**
- 缺失率：0%（无缺失值）
- 重复样本：0
- 异常值：CRIM（犯罪率）、RM（房间数）、LSTAT（低收入人口比例）有明显异常

**Step 3: 分布特征**
- 目标变量MEDV：右偏分布（偏度=1.11），建议对数变换
- CRIM、DIS、LSTAT：严重右偏（偏度>2），建议Box-Cox变换

**Step 4: 特征关系**
- 高相关：RAD <-> TAX (0.91)
- 与目标强相关：LSTAT (-0.74)、RM (0.70)

**Step 5: 问题优先级**
```
【P0级问题】
（无P0级问题）

【P1级问题】
1. 目标变量右偏分布
   → 建议：对数变换

2. 多个特征严重右偏
   → 建议：Box-Cox变换

3. RAD与TAX高相关
   → 建议：删除其中一个（保留TAX）

【P2级问题】
1. 样本量较小（506行）
   → 建议：使用简单模型（线性回归、Ridge）
```

**实施结果**：
- 15分钟完成系统化诊断
- 应用对数变换后，模型R²从0.67提升到0.75
- 删除高相关特征后，避免了多重共线性问题

---

### 案例3：信用卡欺诈检测（严重不平衡）

**背景**：信用卡交易欺诈检测，数据集284,807行×31列

**诊断亮点**：

```
【类别平衡诊断】
□ 最小类别占比：0.17%（492个欺诈 / 284,315个正常）
  [✓] 严重不平衡（<5%）

【P0级问题】
1. 严重类别不平衡（欺诈仅占0.17%）
   → 推荐方案：
      选项1：使用异常检测方法（Isolation Forest）
      选项2：SMOTE过采样 + 欠采样组合
      选项3：调整class_weight + 阈值优化
   → 最终选择：选项1（异常检测）

   理由：
   - 欺诈占比<1%，更适合异常检测框架
   - SMOTE可能产生不真实的合成样本
   - 评估指标应使用Precision-Recall而非Accuracy
```

**关键决策**：
- 将分类问题重新定义为异常检测问题
- 使用Isolation Forest而非传统分类器
- F1-Score从0.65（逻辑回归）提升到0.82（Isolation Forest）

---

## ⚠️ 常见陷阱与解决

### 陷阱1：跳过数据诊断直接建模

**错误做法**：
```python
# ❌ 直接开始建模
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

**正确做法**：
```python
# ✅ 先诊断，再建模
diagnosis = quick_diagnosis(df, target_col='target')

# 根据诊断结果处理数据
if diagnosis['overall_missing'] > 5:
    # 处理缺失值
    ...

if diagnosis.get('balance') == '严重不平衡':
    # 使用异常检测或SMOTE
    ...

# 然后再建模
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

**后果**：
- 缺失值导致模型训练失败
- 类别不平衡导致模型无效（Accuracy虚高）
- 数据泄漏导致过拟合

---

### 陷阱2：只看缺失比例，不看缺失模式

**错误做法**：
```python
# ❌ 只看比例，一律删除
df_clean = df.dropna()
```

**正确做法**：
```python
# ✅ 分析缺失模式
# 1. MCAR（完全随机）→ 可以删除或简单填充
# 2. MAR（随机）→ 需要模型预测填充
# 3. MNAR（非随机）→ 建模为特殊类别

# 检查缺失模式：查看缺失与其他变量的关系
import missingno as msno
msno.matrix(df)  # 可视化缺失模式
msno.heatmap(df)  # 缺失相关性
```

---

### 陷阱3：忽略数据泄漏

**错误示例**：
```python
# ❌ 使用了未来信息
# 特征：last_transaction_date（最后交易日期）
# 目标：predict_churn（预测是否流失）
# 问题：流失后才会有"最后交易日期"，这是未来信息！
```

**检测方法**：
```python
# ✅ 检查特征与目标的异常高相关
corr_with_target = df.corr()['target'].abs().sort_values(ascending=False)
print("与目标相关性>0.95的特征（可疑）:")
print(corr_with_target[corr_with_target > 0.95])

# 人工检查：每个特征在预测时是否可获得？
```

---

### 陷阱4：对异常值一刀切

**错误做法**：
```python
# ❌ 删除所有3σ外的点
df_clean = df[(df['age'] > mean - 3*std) & (df['age'] < mean + 3*std)]
```

**正确做法**：
```python
# ✅ 区分数据错误 vs 真实极端值

# 1. 明显数据错误 → 删除
df = df[df['age'] >= 0]  # 年龄不可能为负

# 2. 真实极端值 → 保留或鲁棒处理
# 使用鲁棒的方法（如IQR）而非直接删除
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
# 截断而非删除
df['income_robust'] = df['income'].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
```

---

### 陷阱5：在训练集和测试集上分别诊断

**错误做法**：
```python
# ❌ 分别处理
X_train_clean = handle_missing(X_train)
X_test_clean = handle_missing(X_test)  # 使用测试集的均值！
```

**正确做法**：
```python
# ✅ 基于训练集的统计量处理测试集
from sklearn.impute import SimpleImputer

# 在训练集上fit
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)

# 在训练集和测试集上transform
X_train_clean = imputer.transform(X_train)
X_test_clean = imputer.transform(X_test)  # 使用训练集的均值
```

---

## 🔗 相关资源

### 前置学习

💡 **第一次遇到数据诊断**，建议先学习：
- 📖 01_data_diagnosis_framework/（完整的数据诊断方法论）
- 📖 01_data_diagnosis_framework/data_diagnosis_quick_reference.md（15分钟速查）
- 📖 01_data_diagnosis_framework/data_problem_to_solution_mapping.md（问题→方案Cookbook）

### 后续步骤

📖 **完成数据诊断后**，查看：
- 📋 algorithm_selection_template.md（下一步：选择算法）
- 💻 08_code_templates/preprocessing_templates.py（获取数据处理代码）
- 📊 06_comprehensive_project/phase1_data_diagnosis.ipynb（完整实战案例）

### 深入学习

📚 **想深入理解数据诊断**，参考：
- 📖 ML_WORKFLOW_GUIDE.md - 第1部分：数据诊断阶段
- 📖 01_data_diagnosis_framework/common_data_issues.md（常见问题详细方案）

---

**最后更新**：2024年11月
**适用场景**：所有ML项目的第一步
**建议使用频率**：每次拿到新数据时必用
