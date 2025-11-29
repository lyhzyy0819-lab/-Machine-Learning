# 📈 模型评估决策模板

> **用途**：快速选择评估指标和验证策略
> **使用场景**：开始建模前，确定如何衡量模型好坏
> **使用时间**：⚡ 快速模式3分钟 | 📊 完整模式10分钟

---

## ⚡ 评估指标快速选择卡（3-8分钟）

### 使用说明

1. **确定问题类型**（1分钟）
2. **填写选择卡片**（2分钟）
3. **获得推荐指标组合**（根据卡片自动生成）

---

### 评估指标选择卡片（可复制填写）

```
===========================================
评估指标选择卡片 - [项目名称]
决策日期：____年____月____日
===========================================

【问题类型】（二选一）
□ [ ] 分类任务（预测类别）
□ [ ] 回归任务（预测数值）
□ [ ] 聚类任务（无监督分群）
□ [ ] 排序任务（推荐系统）

===========================================
【分类任务】（如适用）
===========================================

□ 数据平衡性：
  [ ] 平衡（各类占比相近）
  [ ] 轻度不平衡（最小类10-20%）
  [ ] 中度不平衡（最小类5-10%）
  [ ] 严重不平衡（最小类<5%）

□ 业务关注点：
  [ ] 整体准确率（所有预测正确）
  [ ] 召回率（不能漏掉正例，如疾病诊断）
  [ ] 精确率（不能误报，如垃圾邮件）
  [ ] F1平衡（Recall和Precision都重要）
  [ ] AUC（排序能力，如推荐系统）

□ 错误成本：
  假阳性成本（误判为正）：[ ] 高 [ ] 中 [ ] 低
  假阴性成本（漏判正例）：[ ] 高 [ ] 中 [ ] 低
  成本比例：FP成本 / FN成本 ≈ ____

→ 推荐主指标：________________
→ 推荐辅助指标：________________、________________
→ 推荐阈值：____（默认0.5）

===========================================
【回归任务】（如适用）
===========================================

□ 异常值情况：
  [ ] 少（<5%数据点是异常）
  [ ] 多（>5%数据点是异常）

□ 业务关注点：
  [ ] 平均误差（整体偏差，用MAE）
  [ ] 大误差惩罚（防止离谱预测，用RMSE）
  [ ] 解释方差比例（用R²）
  [ ] 相对误差（MAPE，预测销售额等比例问题）

→ 推荐主指标：________________
→ 推荐辅助指标：________________

===========================================
【聚类任务】（如适用）
===========================================

□ 有无真实标签：
  [ ] 有标签（可验证聚类质量）
  [ ] 无标签（仅内部指标）

→ 推荐指标组合：
  内部指标：________________
  外部指标（如有标签）：________________

===========================================
【验证策略】
===========================================

□ 数据量：____行
  [ ] 小样本(<1K) → LOO或k折(k≥5)
  [ ] 中样本(1K-100K) → 5-10折交叉验证
  [ ] 大样本(>100K) → 3-5折或Hold-out

□ 数据特征：
  [ ] 类别平衡 → 标准k折
  [ ] 类别不平衡 → 分层k折
  [ ] 时间序列 → 时序交叉验证（禁止shuffle）

→ 推荐验证策略：________________

===========================================
```

---

### 快速决策矩阵

#### 矩阵1：分类任务指标选择

| 数据平衡性 | 业务关注点 | 主指标 | 辅助指标 | 阈值调整 |
|-----------|-----------|--------|---------|---------|
| **平衡** | 整体准确 | Accuracy | Precision, Recall | 默认0.5 |
| | 排序能力 | AUC-ROC | Precision-Recall曲线 | 默认0.5 |
| **轻度不平衡** | 综合性能 | F1-Score | Precision, Recall, AUC | 调整至最优F1 |
| | 不能漏正例 | Recall | Precision, F1 | 降低阈值 |
| **中度不平衡** | 不能漏正例 | Recall | Precision@K, AUC | 降低阈值 |
| | 不能误报 | Precision | Recall@K, F1 | 提高阈值 |
| **严重不平衡** | 异常检测 | Precision-Recall AUC | F1, MCC | 基于成本优化 |
| | 排序TopK | Precision@K | Recall@K | TopK阈值 |

#### 矩阵2：回归任务指标选择

| 异常值情况 | 业务关注 | 主指标 | 辅助指标 | 说明 |
|-----------|---------|--------|---------|------|
| **少量异常** | 平均误差 | MAE | RMSE, R² | 对异常鲁棒 |
| | 大误差惩罚 | RMSE | MAE, R² | 平方惩罚大误差 |
| | 解释方差 | R² | Adjusted R² | 越接近1越好 |
| **大量异常** | 鲁棒预测 | MAE | Median AE, Huber Loss | 避免被异常影响 |
| | 识别异常 | RMSE | MAE（对比差异） | RMSE远大于MAE说明有异常 |
| **比例问题** | 相对误差 | MAPE | RMSE, R² | 如预测销售额、股价 |

#### 矩阵3：交叉验证策略选择

| 样本量 | 数据特征 | 推荐策略 | 折数k | 注意事项 |
|-------|---------|---------|------|---------|
| **<1K** | 任何 | k折或LOO | k=5-10或LOO | 小样本需要充分利用数据 |
| **1K-10K** | 平衡 | k折 | k=5-10 | 标准方法 |
| | 不平衡 | 分层k折 | k=5-10 | 保持每折类别比例 |
| | 时间序列 | 时序k折 | k=5 | 训练集时间<测试集时间 |
| **10K-100K** | 任何 | k折 | k=3-5 | 减少计算时间 |
| **>100K** | 任何 | Hold-out | 80/20或70/30 | 样本足够大，无需交叉验证 |

---

## 📊 完整评估策略决策树（10分钟）

### 决策流程总览

```
确定评估目标
    ↓
┌──────────────────────────────────────┐
│ Step 1: 识别问题类型（2分钟）         │
│ - 分类/回归/聚类？                    │
│ - 二分类/多分类？                     │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Step 2: 选择主指标（3分钟）           │
│ - 基于业务目标                        │
│ - 考虑数据平衡性                      │
│ - 考虑错误成本                        │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Step 3: 选择辅助指标（2分钟）         │
│ - 多维度评估模型                      │
│ - 避免单一指标的片面性                │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Step 4: 确定验证策略（2分钟）         │
│ - 交叉验证类型                        │
│ - 折数选择                            │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Step 5: 设定性能目标（1分钟）         │
│ - Baseline性能                        │
│ - 目标性能                            │
│ - 可接受范围                          │
└──────────────────────────────────────┘
    ↓
评估方案 + 性能目标
```

---

### Step 1: 识别问题类型（2分钟）

#### 分类任务细分

```
分类任务
├─ 二分类
│   ├─ 平衡二分类
│   │   └─ 主指标：Accuracy, AUC
│   │
│   └─ 不平衡二分类
│       ├─ 轻度不平衡（10-20%）
│       │   └─ 主指标：F1-Score
│       │
│       ├─ 中度不平衡（5-10%）
│       │   └─ 主指标：Precision-Recall AUC
│       │
│       └─ 严重不平衡（<5%）
│           └─ 主指标：Precision@K, Recall@K
│           └─ 考虑：改为异常检测框架
│
└─ 多分类
    ├─ 平衡多分类
    │   └─ 主指标：Accuracy, Macro F1
    │
    └─ 不平衡多分类
        └─ 主指标：Weighted F1, Macro F1
```

#### 回归任务细分

```
回归任务
├─ 数值预测（绝对值问题）
│   ├─ 少量异常值
│   │   └─ 主指标：RMSE（平方惩罚）, R²
│   │
│   └─ 大量异常值
│       └─ 主指标：MAE（鲁棒）, Median AE
│
└─ 比例预测（相对值问题）
    ├─ 销售额、股价等
    │   └─ 主指标：MAPE（相对误差%）
    │
    └─ 注意：MAPE对接近0的值敏感
        └─ 替代：SMAPE（对称MAPE）
```

---

### Step 2: 选择主指标（3分钟）

#### 分类指标速查表

| 指标 | 公式 | 适用场景 | 优点 | 缺点 |
|------|------|---------|------|------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 平衡数据 | 直观易懂 | 不平衡数据失效 |
| **Precision** | TP/(TP+FP) | 不能误报（垃圾邮件） | 关注误报 | 忽略漏报 |
| **Recall** | TP/(TP+FN) | 不能漏报（疾病诊断） | 关注漏报 | 忽略误报 |
| **F1-Score** | 2×P×R/(P+R) | Precision和Recall都重要 | 平衡两者 | 不区分FP和FN成本 |
| **AUC-ROC** | ROC曲线下面积 | 排序问题（推荐） | 阈值无关 | 不平衡数据不准确 |
| **AUC-PR** | PR曲线下面积 | 不平衡数据 | 适合不平衡 | 解释性差 |
| **MCC** | 马修斯相关系数 | 严重不平衡 | 最全面 | 不直观 |

#### 回归指标速查表

| 指标 | 公式 | 适用场景 | 优点 | 缺点 |
|------|------|---------|------|------|
| **MAE** | mean(|y_true - y_pred|) | 平均误差、鲁棒 | 对异常值不敏感 | 不惩罚大误差 |
| **RMSE** | sqrt(mean((y_true - y_pred)²)) | 惩罚大误差 | 平方惩罚 | 对异常值敏感 |
| **R²** | 1 - SS_res/SS_tot | 解释方差比例 | 直观（0-1） | 可能为负 |
| **MAPE** | mean(|y_true - y_pred|/y_true) × 100% | 相对误差% | 相对误差 | 接近0时不稳定 |
| **Median AE** | median(|y_true - y_pred|) | 极端鲁棒 | 最鲁棒 | 损失信息 |

---

### Step 3: 选择辅助指标（2分钟）

#### 推荐指标组合

**分类任务**：
```
【平衡二分类】
主指标：Accuracy
辅助指标：
  - Precision（查看误报率）
  - Recall（查看漏报率）
  - AUC（查看排序能力）
  - 混淆矩阵（详细错误分析）

【不平衡二分类】
主指标：F1-Score 或 AUC-PR
辅助指标：
  - Precision（正例精度）
  - Recall（正例召回）
  - Precision-Recall曲线
  - 不同阈值下的F1变化
  - MCC（综合指标）

【多分类】
主指标：Macro F1（平等对待每类）
辅助指标：
  - Weighted F1（按样本量加权）
  - 每类的Precision和Recall
  - 混淆矩阵热力图
  - Top-K Accuracy（如适用）
```

**回归任务**：
```
【常规回归】
主指标：RMSE
辅助指标：
  - MAE（对比RMSE，判断异常值影响）
  - R²（解释方差比例）
  - 残差分布图（检查偏差）
  - 预测值vs真实值散点图

【有异常值】
主指标：MAE
辅助指标：
  - Median AE（更鲁棒）
  - RMSE（看与MAE的差异）
  - Huber Loss（兼顾）

【比例问题】
主指标：MAPE
辅助指标：
  - RMSE（绝对误差）
  - R²（解释方差）
  - SMAPE（对称版本）
```

---

### Step 4: 确定验证策略（2分钟）

#### 交叉验证代码模板

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def cross_validate_model(model, X, y, cv_strategy='stratified', k=5):
    """
    通用交叉验证函数

    Parameters:
    -----------
    model : 模型对象
    X : 特征矩阵
    y : 目标变量
    cv_strategy : str
        'standard' - 标准k折
        'stratified' - 分层k折（分类推荐）
        'timeseries' - 时序k折
        'loo' - 留一法（小样本）
    k : int, 折数（LOO时忽略）

    Returns:
    --------
    scores : 每折得分
    mean_score : 平均得分
    std_score : 标准差（稳定性）
    """

    # 选择验证策略
    if cv_strategy == 'standard':
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
    elif cv_strategy == 'stratified':
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    elif cv_strategy == 'timeseries':
        cv = TimeSeriesSplit(n_splits=k)
    elif cv_strategy == 'loo':
        cv = LeaveOneOut()
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

    scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)

        # 评估（这里用准确率，可替换为其他指标）
        score = accuracy_score(y_val, y_pred)
        scores.append(score)

        print(f"Fold {fold+1}: {score:.4f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"\n平均得分: {mean_score:.4f} ± {std_score:.4f}")

    return scores, mean_score, std_score

# 使用示例
# scores, mean, std = cross_validate_model(
#     model=RandomForestClassifier(),
#     X=X, y=y,
#     cv_strategy='stratified',
#     k=5
# )
```

#### 时序交叉验证特殊处理

```python
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

def time_series_cv_split(X, y, n_splits=5):
    """
    时序交叉验证的正确分割方式

    ⚠️ 注意：
    - 训练集时间必须早于测试集
    - 不能shuffle（会破坏时序）
    - 每次测试集都在训练集之后
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold+1}:")
        print(f"  训练集: 样本 {train_idx[0]} 到 {train_idx[-1]} ({len(train_idx)}个)")
        print(f"  测试集: 样本 {test_idx[0]} 到 {test_idx[-1]} ({len(test_idx)}个)")

        # 可视化分割
        plt.figure(figsize=(10, 2))
        plt.scatter(train_idx, [fold]*len(train_idx), c='blue', label='训练集')
        plt.scatter(test_idx, [fold]*len(test_idx), c='red', label='测试集')
        plt.ylabel(f'Fold {fold+1}')
        plt.xlabel('样本索引')
        plt.legend()

    plt.tight_layout()
    plt.show()

# 使用示例
# time_series_cv_split(X, y, n_splits=5)
```

---

### Step 5: 设定性能目标（1分钟）

#### 性能目标模板

```
===========================================
模型性能目标设定
===========================================

【Baseline性能】（必须先建立）
算法：________________（如Logistic Regression）
主指标：________ = ____（如Accuracy = 0.75）
辅助指标：________ = ____

【目标性能】
主指标目标：________ ≥ ____
理由：________________（如业务要求、竞品水平）

【可接受性能范围】
最低可接受：________ ≥ ____（低于此值不部署）
理想目标：________ ≥ ____（达到此值优秀）

【性能提升期望】
相比Baseline提升：+____% 或 +____（绝对值）

【性能vs成本权衡】
- 性能提升1%的代价：
  训练时间：+____小时
  计算资源：+____
  模型复杂度：+____
- 是否值得：[ ] 是 [ ] 否

===========================================
```

---

## ✅ 模型对比检查清单

### 标准对比流程

```
===========================================
模型对比检查清单
===========================================

【数据一致性】（确保公平对比）
- [ ] 所有模型使用相同的train/test划分
- [ ] 所有模型使用相同的预处理流程
- [ ] 所有模型使用相同的特征集
- [ ] 所有模型使用相同的随机种子

【性能评估】
- [ ] 计算所有候选模型的主指标
- [ ] 计算所有候选模型的辅助指标
- [ ] 使用交叉验证评估稳定性（均值±标准差）
- [ ] 记录每个模型的训练时间
- [ ] 记录每个模型的预测时间

【统计显著性检验】（如需要）
- [ ] 使用配对t检验比较模型
- [ ] 计算p值，判断差异是否显著
- [ ] 注意：样本量<30时用非参数检验

【可视化对比】
- [ ] 绘制模型性能对比柱状图
- [ ] 绘制学习曲线（性能vs样本量）
- [ ] 绘制ROC曲线对比（分类）
- [ ] 绘制残差图（回归）

【综合决策】
- [ ] 不仅看性能，还看训练/预测时间
- [ ] 考虑模型可解释性
- [ ] 考虑部署难度
- [ ] 考虑维护成本

【最终选择】
选定模型：________________
选择理由：
  性能：________________
  效率：________________
  可解释性：________________
  其他：________________

===========================================
```

---

## 💰 业务价值转化模板

### 从技术指标到业务语言

#### 分类任务转化示例

```
===========================================
业务价值转化报告 - 客户流失预测
===========================================

【技术指标】
Precision = 0.78
Recall = 0.85
F1-Score = 0.81

【业务语言转化】
→ Recall = 0.85 意味着：
  "能成功挽留85%的即将流失客户"

→ Precision = 0.78 意味着：
  "预测为流失的客户中，78%确实会流失"
  "22%是误报（额外挽留成本）"

【ROI计算】
假设：
  - 每月预测100个流失客户
  - 挽留成功率：50%
  - 每个客户年价值：5,000元
  - 挽留成本：500元/人

收益计算：
  成功挽留数 = 100 × 0.85 × 0.50 = 42.5人
  收益 = 42.5 × 5,000 = 212,500元/月

成本计算：
  挽留总人数 = 100人
  成本 = 100 × 500 = 50,000元/月

净收益：
  212,500 - 50,000 = 162,500元/月
  年净收益 ≈ 195万元

投资回报率(ROI)：
  (收益 - 成本) / 成本 = 325%

===========================================
```

#### 回归任务转化示例

```
===========================================
业务价值转化报告 - 销售额预测
===========================================

【技术指标】
RMSE = 5,000元
MAE = 3,500元
R² = 0.88

【业务语言转化】
→ RMSE = 5,000元 意味着：
  "预测销售额的误差约为±5,000元"
  "大约68%的预测在±5,000元范围内"（正态假设）

→ MAE = 3,500元 意味着：
  "平均每次预测偏差3,500元"

→ R² = 0.88 意味着：
  "模型能解释88%的销售额变化"
  "还有12%的变化无法预测（随机因素）"

【业务影响】
场景：月度库存规划

无模型情况：
  - 按历史平均备货
  - 缺货或积压：月均损失50万元

有模型情况：
  - 根据预测精准备货
  - RMSE=5000意味着库存偏差±5000元
  - 减少损失：35万元/月（减少70%）

年度价值：
  35万 × 12 = 420万元/年

模型开发成本：
  开发：20万元（一次性）
  维护：5万元/年

ROI：
  第一年：(420 - 20 - 5) / (20 + 5) = 1580%
  后续年：(420 - 5) / 5 = 8300%

===========================================
```

---

## 📚 实战评估案例

### 案例1：信用卡欺诈检测（不平衡分类）

**背景**：
- 数据集：284,807笔交易
- 欺诈交易：492笔（0.17%）
- 极度不平衡

**评估指标选择**：

**❌ 错误选择：Accuracy**
```python
from sklearn.dummy import DummyClassifier

# 全部预测为"正常"
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
accuracy = dummy.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")  # 0.9983

# 问题：Accuracy高达99.83%，但完全没用！
# 因为全部预测为正常，一个欺诈都没抓到！
```

**✅ 正确选择：Precision-Recall + F1**
```python
from sklearn.metrics import classification_report, precision_recall_curve

# 使用Isolation Forest
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 详细评估
print(classification_report(y_test, y_pred))

# 输出示例:
#               precision    recall  f1-score
# 正常(0)          1.00      0.98      0.99
# 欺诈(1)          0.78      0.85      0.82  ← 关注这个！

# Precision-Recall曲线
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# 选择最优阈值（最大化F1）
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"最优阈值: {best_threshold:.4f}")
```

**业务价值转化**：
```
Recall = 0.85 → 能检测出85%的欺诈交易
假设每笔欺诈平均损失1000元
每月欺诈交易100笔

无模型损失：100 × 1000 = 10万元/月
有模型损失：15 × 1000 = 1.5万元/月（漏掉15%）
月节省：8.5万元
年节省：102万元
```

---

### 案例2：房价预测（回归with异常值）

**背景**：
- 数据集：1,460套房屋
- 目标：预测房价
- 存在高价豪宅（异常值）

**评估指标选择**：

**实验对比**：
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 训练模型
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 计算多个指标
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: ${mae:,.0f}")    # 平均误差
print(f"RMSE: ${rmse:,.0f}")  # 均方根误差
print(f"R²: {r2:.3f}")        # 解释方差

# 对比RMSE和MAE
ratio = rmse / mae
print(f"RMSE/MAE比值: {ratio:.2f}")

if ratio > 1.5:
    print("⚠️ RMSE远大于MAE，存在较多大误差（异常值影响）")
elif ratio > 1.2:
    print("✓ RMSE略大于MAE，少量异常值")
else:
    print("✓ RMSE接近MAE，误差分布均匀")
```

**实际结果**：
```
MAE: $18,500
RMSE: $32,000
R²: 0.88
RMSE/MAE比值: 1.73

→ 解读：
  - 平均误差约1.85万元
  - 但存在较多大误差（RMSE是MAE的1.73倍）
  - 说明对高价房预测不准

→ 改进方向：
  1. 对目标变量做对数变换（缩小数值范围）
  2. 删除极端异常值（>3σ）
  3. 使用鲁棒回归（Huber Loss）
```

**改进后**：
```python
# 对数变换
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

model.fit(X_train, y_train_log)
y_pred_log = model.predict(X_test)

# 转换回原尺度
y_pred = np.expm1(y_pred_log)

# 重新评估
mae_improved = mean_absolute_error(y_test, y_pred)
rmse_improved = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"改进后 MAE: ${mae_improved:,.0f}")  # $15,200（降低18%）
print(f"改进后 RMSE: ${rmse_improved:,.0f}")  # $24,500（降低23%）
print(f"改进后 RMSE/MAE: {rmse_improved/mae_improved:.2f}")  # 1.61（更接近）
```

---

## ⚠️ 常见陷阱与解决

### 陷阱1：不平衡数据用Accuracy

**问题**：
```python
# 欺诈检测：欺诈仅占0.5%
# ❌ 使用Accuracy
accuracy = 0.995  # 看起来很好！

# 但实际：模型全部预测为"正常"
# Recall = 0%（一个欺诈都没抓到）
```

**解决**：
```python
# ✅ 使用Precision-Recall
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
```

---

### 陷阱2：数据泄漏到测试集

**问题**：
```python
# ❌ 在全量数据上计算统计量
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 使用了测试集的均值！

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
# 测试集性能虚高！
```

**解决**：
```python
# ✅ 先划分，再处理
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 只用训练集
X_test_scaled = scaler.transform(X_test)  # 用训练集的统计量

# 或使用Pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)  # 自动处理
```

---

### 陷阱3：时间序列shuffle

**问题**：
```python
# ❌ 时序数据使用标准交叉验证
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True)  # shuffle=True破坏时序！
# 会用未来数据预测过去，性能虚高
```

**解决**：
```python
# ✅ 使用时序交叉验证
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
# 保证训练集时间 < 测试集时间
```

---

## 🔗 相关资源

### 前置学习

💡 **第一次选择评估指标**，建议先学习：
- 📖 05_model_evaluation/metrics_calculation_guide.md（指标计算方法）
- 📖 05_model_evaluation/model_comparison_and_selection.md（模型对比方法）
- 📖 ML_WORKFLOW_GUIDE.md - 第6部分：模型评估阶段

### 后续步骤

📖 **完成评估指标选择后**，查看：
- 📋 hyperparameter_tuning_template.md（调优模型）
- 💻 08_code_templates/evaluation_templates.py（评估代码模板）

### 深入学习

📚 **想深入理解评估方法**，参考：
- 📖 05_model_evaluation/business_value_translation.md（业务价值转化）
- 📖 06_comprehensive_project/phase3_supervised_solution.ipynb（完整评估流程）

---

**最后更新**：2024年11月
**适用场景**：开始建模前的评估策略制定
**建议使用频率**：每个新项目使用一次
