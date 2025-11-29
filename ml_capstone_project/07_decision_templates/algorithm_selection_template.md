# 🎯 算法选择决策模板

> **用途**：快速选择候选算法，避免盲目试错
> **使用场景**：完成数据诊断后，确定问题类型时
> **使用时间**：⚡ 快速模式5分钟 | 📊 完整模式15分钟

---

## ⚡ 问题画像卡片（3-5分钟填写）

### 使用说明

1. **填写问题画像**（3分钟）
2. **查表获取推荐算法**（2分钟）
3. **选择Top 3候选算法**进行实验

---

### 问题画像卡片（可复制填写）

```
===========================================
算法选择决策卡片 - [项目名称]
决策日期：____年____月____日
===========================================

【问题类型】（必填）
□ [ ] 回归（预测连续数值）
□ [ ] 二分类（2个类别）
□ [ ] 多分类（3+类别，具体：____类）
□ [ ] 聚类（无标签分群）
□ [ ] 降维（高维可视化/特征压缩）
□ [ ] 异常检测（识别稀有样本）

【数据规模】（必填）
□ 样本量：____行
  [ ] 小样本（<1,000）
  [ ] 中样本（1K-100K）
  [ ] 大样本（100K-1M）
  [ ] 超大样本（>1M）

□ 特征数：____列
  [ ] 低维（<10）
  [ ] 中维（10-100）
  [ ] 高维（100-1000）
  [ ] 超高维（>1000）

【数据特征】（必填）
□ 类别平衡：
  [ ] 平衡（各类占比>20%）
  [ ] 轻度不平衡（10-20%）
  [ ] 严重不平衡（<10%）
  [ ] 极度不平衡（<1%）→ 考虑异常检测

□ 特征类型：
  [ ] 纯数值
  [ ] 纯分类
  [ ] 混合（数值+分类）
  [ ] 文本/图像（需要特殊处理）

□ 线性可分性：
  [ ] 线性可分（特征与目标线性关系）
  [ ] 非线性（复杂关系）
  [ ] 不确定（需要实验）

【业务约束】（选填）
□ 可解释性要求：
  [ ] 高（必须能向业务解释）
  [ ] 中（部分可解释）
  [ ] 低（黑盒可接受）

□ 训练时间限制：
  [ ] 严格（<5分钟）
  [ ] 宽松（<1小时）
  [ ] 无限制（可离线训练）

□ 预测速度要求：
  [ ] 实时（<100ms）
  [ ] 准实时（<1s）
  [ ] 批量（无要求）

□ 模型更新频率：
  [ ] 实时更新
  [ ] 每天更新
  [ ] 每周/月更新
  [ ] 一次性训练

===========================================
→ 推荐算法Top 3（根据上述信息）
===========================================

第1推荐：________________
理由：____________________

第2推荐：________________
理由：____________________

第3推荐：________________
理由：____________________

备选方案：________________
（用于对比实验）

===========================================
```

---

### 快速决策矩阵

#### 矩阵1：基于问题类型和样本量

| 问题类型 | 小样本(<1K) | 中样本(1K-100K) | 大样本(>100K) | 超大样本(>1M) |
|---------|------------|---------------|-------------|-------------|
| **回归** | Linear/Ridge<br>决策树 | Random Forest<br>XGBoost | XGBoost<br>LightGBM | LightGBM<br>深度学习 |
| **二分类** | Logistic<br>Naive Bayes | Random Forest<br>XGBoost | XGBoost<br>LightGBM | LightGBM<br>深度学习 |
| **多分类** | Logistic<br>决策树 | Random Forest<br>XGBoost | XGBoost<br>LightGBM | LightGBM<br>深度学习 |
| **聚类** | K-Means<br>层次聚类 | K-Means<br>DBSCAN | K-Means<br>Mini-Batch K-Means | Mini-Batch K-Means |
| **降维** | PCA<br>t-SNE | PCA<br>t-SNE | PCA<br>UMAP | PCA<br>增量PCA |
| **异常检测** | LOF<br>Isolation Forest | Isolation Forest<br>One-Class SVM | Isolation Forest | Isolation Forest |

#### 矩阵2：基于特征数和可解释性

| 特征维度 | 高可解释性 | 中可解释性 | 低可解释性（黑盒） |
|---------|----------|-----------|------------------|
| **低维(<10)** | Linear/Logistic<br>决策树 | Random Forest<br>XGBoost（浅层） | XGBoost（深层）<br>SVM |
| **中维(10-100)** | Linear/Logistic<br>Lasso/Ridge | Random Forest<br>XGBoost | XGBoost<br>深度学习 |
| **高维(100-1000)** | Lasso（L1正则）<br>Elastic Net | Random Forest<br>XGBoost | XGBoost<br>深度学习 |
| **超高维(>1000)** | Lasso<br>PCA+Linear | PCA+XGBoost | 深度学习<br>AutoEncoder |

#### 矩阵3：基于类别不平衡程度

| 不平衡程度 | 推荐算法 | 推荐策略 |
|-----------|---------|---------|
| **平衡（>20%）** | 任何算法 | 标准训练 |
| **轻度不平衡（10-20%）** | Random Forest<br>XGBoost | class_weight='balanced'<br>分层抽样 |
| **中度不平衡（5-10%）** | XGBoost<br>LightGBM | SMOTE过采样<br>scale_pos_weight调整 |
| **严重不平衡（1-5%）** | XGBoost<br>LightGBM | SMOTE+Tomek Links<br>focal loss |
| **极度不平衡（<1%）** | Isolation Forest<br>One-Class SVM | **改为异常检测框架**<br>而非分类 |

---

## 📊 完整决策树（15分钟系统化决策）

### 决策流程总览

```
问题定义
    ↓
┌──────────────────────────────────────┐
│ Step 1: 确定问题类型（2分钟）         │
│ - 有无标签？                          │
│ - 目标变量类型？                      │
│ - 监督 vs 无监督 vs 混合             │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Step 2: 评估数据特征（3分钟）         │
│ - 样本量和特征数                      │
│ - 类别平衡性                          │
│ - 线性可分性                          │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Step 3: 考虑业务约束（2分钟）         │
│ - 可解释性要求                        │
│ - 时间和性能限制                      │
│ - 部署环境                            │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Step 4: 筛选候选算法（5分钟）         │
│ - 基于决策矩阵筛选                    │
│ - 选择3-5个算法                       │
│ - 确定Baseline算法                   │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Step 5: 实验验证（3分钟规划）         │
│ - 确定对比维度                        │
│ - 设定实验顺序                        │
│ - 准备评估指标                        │
└──────────────────────────────────────┘
    ↓
候选算法列表 + 实验计划
```

---

### Step 1: 确定问题类型（2分钟）

#### 决策树

```
是否有标签（目标变量）？
├─ 有标签 → 监督学习
│   ├─ 目标变量是连续数值？
│   │   ├─ 是 → 【回归问题】
│   │   │   └─ 示例：房价预测、销售额预测、温度预测
│   │   │
│   │   └─ 否 → 【分类问题】
│   │       ├─ 2个类别 → 二分类
│   │       │   └─ 示例：客户流失、疾病诊断、垃圾邮件
│   │       │
│   │       └─ 3+类别 → 多分类
│   │           └─ 示例：手写数字识别、情感分析、物种分类
│   │
│   └─ 还有其他需求？
│       ├─ 需要排序 → 排序学习（Learning to Rank）
│       └─ 需要推荐 → 推荐系统
│
└─ 无标签 → 无监督学习
    ├─ 想要分群 → 【聚类问题】
    │   └─ 示例：客户分群、文档聚类、图像分割
    │
    ├─ 想要降维/可视化 → 【降维问题】
    │   └─ 示例：高维可视化、特征压缩、去噪
    │
    ├─ 想要找异常 → 【异常检测问题】
    │   └─ 示例：欺诈检测、设备故障、网络入侵
    │
    └─ 想要发现模式 → 【关联规则挖掘】
        └─ 示例：购物篮分析、频繁模式挖掘
```

#### 特殊情况处理

| 情况 | 传统方法 | 推荐方法 |
|------|---------|---------|
| **极度不平衡分类（<1%）** | 分类算法+SMOTE | 改为异常检测（Isolation Forest） |
| **半监督学习（部分标签）** | 只用有标签数据 | 先聚类+伪标签，再训练 |
| **多标签分类（一个样本多标签）** | 多个二分类器 | MultiOutputClassifier包装 |
| **时间序列预测** | 传统回归 | ARIMA / LSTM / Prophet |

---

### Step 2: 评估数据特征（3分钟）

#### 2.1 样本量评估

```
样本量分类：
├─ 小样本（<1,000）
│   ├─ 推荐：简单模型（Linear/Logistic/Naive Bayes）
│   ├─ 避免：深度学习、复杂集成模型
│   └─ 策略：交叉验证（k-fold或LOO）、正则化
│
├─ 中样本（1K-100K）
│   ├─ 推荐：Random Forest、XGBoost
│   ├─ 可用：SVM（中小样本）
│   └─ 策略：标准5-10折交叉验证
│
├─ 大样本（100K-1M）
│   ├─ 推荐：XGBoost、LightGBM
│   ├─ 可用：深度学习
│   └─ 策略：hold-out验证（节省时间）
│
└─ 超大样本（>1M）
    ├─ 推荐：LightGBM（速度快）、深度学习
    ├─ 避免：SVM（O(n²)复杂度）
    └─ 策略：采样验证、分布式训练
```

#### 2.2 特征维度评估

```
特征数量分类：
├─ 低维（<10）
│   ├─ 推荐：任何算法
│   └─ 注意：可能特征不足，需要特征工程
│
├─ 中维（10-100）
│   ├─ 推荐：Random Forest、XGBoost
│   └─ 策略：特征重要性分析
│
├─ 高维（100-1000）
│   ├─ 推荐：Lasso（特征选择）、XGBoost
│   ├─ 避免：KNN（维度灾难）
│   └─ 策略：特征选择或降维（PCA）
│
└─ 超高维（>1000）
    ├─ 推荐：Lasso、Elastic Net（L1正则化）
    ├─ 必须：降维或特征选择
    └─ 策略：PCA降维 + 建模
```

#### 2.3 线性可分性判断

```python
# 快速判断线性可分性（2分钟实验）

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# 分类任务
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 线性模型
linear_model = LogisticRegression(max_iter=1000)
linear_model.fit(X_train, y_train)
linear_score = accuracy_score(y_test, linear_model.predict(X_test))

# 非线性模型
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_score = accuracy_score(y_test, rf_model.predict(X_test))

# 判断
print(f"线性模型得分: {linear_score:.3f}")
print(f"随机森林得分: {rf_score:.3f}")
print(f"性能差异: {rf_score - linear_score:.3f}")

if rf_score - linear_score < 0.05:
    print("→ 结论：线性可分，优先使用线性模型")
else:
    print("→ 结论：非线性关系，使用树模型或SVM")
```

**判断标准**：
- 差异<5%：线性可分 → 优先Linear/Logistic
- 差异5-15%：轻微非线性 → Random Forest
- 差异>15%：强非线性 → XGBoost/深度学习

---

### Step 3: 考虑业务约束（2分钟）

#### 约束决策矩阵

| 约束类型 | 约束条件 | 推荐算法 | 避免算法 |
|---------|---------|---------|---------|
| **可解释性** | 必须高度可解释 | Linear/Logistic<br>决策树 | XGBoost（深层）<br>深度学习<br>SVM（RBF核） |
| | 部分可解释 | Random Forest<br>XGBoost（浅层） | 深度学习（黑盒） |
| | 可接受黑盒 | 任何算法 | - |
| **训练时间** | 极度受限（<5分钟） | Naive Bayes<br>Linear/Logistic | XGBoost（大量树）<br>SVM（大数据）<br>深度学习 |
| | 宽松（<1小时） | Random Forest<br>XGBoost | 深度学习（大数据） |
| | 无限制 | 任何算法 | - |
| **预测速度** | 实时（<100ms） | Linear/Logistic<br>决策树 | KNN（O(n)查询）<br>深度学习（大模型） |
| | 准实时（<1s） | Random Forest<br>XGBoost | - |
| | 批量处理 | 任何算法 | - |
| **模型大小** | 严格限制（<10MB） | Linear/Logistic<br>Naive Bayes | Random Forest（大量树）<br>深度学习 |
| | 宽松（<100MB） | XGBoost<br>LightGBM | - |
| | 无限制 | 任何算法 | - |

---

### Step 4: 筛选候选算法（5分钟）

#### 筛选流程

```
综合前3步信息
    ↓
查阅快速决策矩阵（矩阵1-3）
    ↓
列出符合条件的算法（5-10个）
    ↓
根据优先级排序
    ├─ P0: 必须满足所有约束
    ├─ P1: 满足大部分约束，性能可能更好
    └─ P2: 作为对比Baseline
    ↓
选择Top 3-5个算法
    ├─ 1个简单Baseline（Linear/Logistic）
    ├─ 2-3个主力算法（Random Forest/XGBoost）
    └─ 1个进阶算法（LightGBM/深度学习）
```

#### 推荐组合模板

**组合1：保守组合（强调稳定）**
```
Baseline: Logistic Regression
主力1: Random Forest
主力2: XGBoost
备选: LightGBM
```

**组合2：性能组合（追求最优）**
```
Baseline: Random Forest
主力1: XGBoost
主力2: LightGBM
进阶: Stacking（RF+XGB+LGB）
```

**组合3：快速组合（时间受限）**
```
Baseline: Logistic Regression
主力: Random Forest（默认参数）
快速调优: XGBoost（少量树）
```

---

### Step 5: 实验验证规划（3分钟）

#### 实验计划模板

```
===========================================
算法对比实验计划
===========================================

【实验目标】
主要目标：______________________
次要目标：______________________

【候选算法列表】
1. Baseline: ________（用于建立性能基线）
2. 算法A: ________（预期最优）
3. 算法B: ________（备选方案）
4. 算法C: ________（对比实验）

【评估指标】（来自model_evaluation_template）
主指标：________
辅助指标：________、________

【对比维度】
- [ ] 预测性能（主指标）
- [ ] 训练时间
- [ ] 预测速度
- [ ] 模型大小
- [ ] 可解释性
- [ ] 鲁棒性（交叉验证std）

【实验顺序】
Day 1: Baseline（快速验证思路）
Day 2: 算法A（主力算法）
Day 3: 算法B、C（对比和优化）
Day 4: 超参数调优最优算法
Day 5: 模型融合（可选）

===========================================
```

---

## 📚 实战决策案例

### 案例1：信用卡欺诈检测（极度不平衡分类）

**问题画像**：
```
【问题类型】
[✓] 二分类（欺诈 vs 正常）

【数据规模】
样本量：284,807行 → [✓] 大样本
特征数：30列 → [✓] 中维

【数据特征】
类别平衡：[✓] 极度不平衡（欺诈仅0.17%）
特征类型：[✓] 纯数值（已PCA处理）
线性可分性：[✓] 非线性

【业务约束】
可解释性：[✓] 中（需要部分解释）
训练时间：[✓] 宽松
预测速度：[✓] 准实时（<1s）
```

**决策过程**：

**Step 1: 问题类型**
- 传统：二分类问题
- **重新定义**：极度不平衡（<1%）→ 改为异常检测问题

**Step 2: 数据特征**
- 大样本（28万）+ 中维（30特征）→ 适合复杂模型
- 但极度不平衡 → SMOTE过采样效果有限

**Step 3: 业务约束**
- 需要部分可解释（识别关键特征）
- 预测速度要求不严格

**Step 4: 筛选算法**

| 方案 | 算法 | 优点 | 缺点 | 评分 |
|------|------|------|------|------|
| 方案1 | XGBoost + SMOTE | 强大性能 | SMOTE可能生成不真实样本 | 6/10 |
| 方案2 | Isolation Forest | 专为异常检测设计 | 无监督，不用标签 | 9/10 ⭐ |
| 方案3 | One-Class SVM | 异常检测 | 训练慢 | 7/10 |
| 方案4 | AutoEncoder | 深度学习 | 训练复杂，可解释性差 | 5/10 |

**最终选择**：
```
主力方案：Isolation Forest（异常检测框架）
对比方案：XGBoost + SMOTE（传统分类框架）

理由：
1. 欺诈占比0.17%，更符合"异常"定义
2. Isolation Forest无需标签，避免过采样问题
3. 可解释性：可分析异常分数分布
4. 实验证明：F1-Score提升17%（0.65→0.82）
```

**代码示例**：
```python
from sklearn.ensemble import Isolation Forest
from sklearn.metrics import classification_report

# Isolation Forest方法
iso_forest = IsolationForest(
    contamination=0.0017,  # 设置为欺诈比例
    random_state=42
)

# 训练（仅用正常交易，异常检测思想）
iso_forest.fit(X_train[y_train == 0])

# 预测
y_pred = iso_forest.predict(X_test)
y_pred = [1 if x == -1 else 0 for x in y_pred]  # -1表示异常

print(classification_report(y_test, y_pred))
```

---

### 案例2：房价预测（回归任务）

**问题画像**：
```
【问题类型】
[✓] 回归（预测房价）

【数据规模】
样本量：1,460行 → [✓] 中样本
特征数：81列 → [✓] 中维

【数据特征】
特征类型：[✓] 混合（数值+分类）
线性可分性：[✓] 部分线性

【业务约束】
可解释性：[✓] 高（房东想知道哪些因素影响价格）
训练时间：[✓] 无限制
预测速度：[✓] 批量处理
```

**决策过程**：

**Step 1-2: 中样本+中维+高可解释性**
→ 排除深度学习（黑盒）
→ 优先Linear、Ridge、Random Forest

**Step 3: 线性可分性实验**
```python
# 快速对比
linear_score = 0.72  # Ridge回归 R²
rf_score = 0.88      # Random Forest R²
差异 = 16%           # 显著非线性
```
→ 存在非线性关系，但仍需保持可解释性

**Step 4: 筛选算法**

| 算法 | 可解释性 | 性能预期 | 适用性 | 评分 |
|------|---------|---------|--------|------|
| Linear Regression | ⭐⭐⭐⭐⭐ | ⭐⭐ | Baseline | 7/10 |
| Ridge Regression | ⭐⭐⭐⭐ | ⭐⭐⭐ | 主力1 | 9/10 ⭐ |
| Random Forest | ⭐⭐⭐ | ⭐⭐⭐⭐ | 主力2 | 9/10 ⭐ |
| XGBoost | ⭐⭐ | ⭐⭐⭐⭐⭐ | 性能对比 | 8/10 |

**最终选择**：
```
Baseline: Ridge Regression（可解释+正则化）
主力1: Random Forest（非线性+特征重要性）
主力2: XGBoost（性能上限对比）

实验结果（R²）:
- Ridge: 0.89
- Random Forest: 0.91
- XGBoost: 0.92

最终部署：Random Forest
理由：性能接近XGBoost，但特征重要性更直观，可向业务解释
```

---

### 案例3：客户分群（无监督聚类）

**问题画像**：
```
【问题类型】
[✓] 聚类（客户分群）

【数据规模】
样本量：8,950行 → [✓] 中样本
特征数：8列 → [✓] 低维

【数据特征】
特征类型：[✓] 纯数值（消费行为特征）

【业务约束】
可解释性：[✓] 高（需要理解每个群体特征）
聚类数量：[✓] 不确定（需要确定最优k值）
```

**决策过程**：

**Step 1: 无监督聚类**
→ 候选：K-Means、DBSCAN、层次聚类、GMM

**Step 2: 数据特征评估**
- 中样本 + 低维 → 任何聚类算法都适用
- 纯数值 → 不需要特殊编码

**Step 3: 业务约束**
- 需要指定聚类数 → 排除DBSCAN（自动确定）
- 需要可解释性 → K-Means（聚类中心明确）

**Step 4: 算法选择**

| 算法 | 优点 | 缺点 | 适用性 |
|------|------|------|--------|
| K-Means | 简单、快速、中心点可解释 | 需要指定k、球形簇假设 | ⭐⭐⭐⭐⭐ |
| DBSCAN | 自动确定k、任意形状 | 难解释、对参数敏感 | ⭐⭐ |
| 层次聚类 | 树状图可视化 | 计算慢、难确定切分点 | ⭐⭐⭐ |
| GMM | 概率分布、软聚类 | 复杂、不够直观 | ⭐⭐⭐ |

**最终选择**：
```
主力方案：K-Means（k=3-7，用Elbow法确定）
对比方案：层次聚类（验证k值合理性）

实验结果：
- Elbow法：k=5时拐点明显
- Silhouette Score：k=5时最优（0.58）
- 业务验证：5个客户群体有明确业务含义
  1. 高价值VIP客户（10%）
  2. 中频消费客户（25%）
  3. 低频消费客户（30%）
  4. 新客户（20%）
  5. 流失风险客户（15%）

最终部署：K-Means (k=5)
```

---

### 案例4：图像降维可视化（t-SNE vs UMAP）

**问题画像**：
```
【问题类型】
[✓] 降维（高维图像可视化）

【数据规模】
样本量：70,000行（MNIST手写数字）→ [✓] 大样本
特征数：784列（28×28像素）→ [✓] 高维

【业务约束】
目标：降至2维可视化
保持：局部结构（相似图像接近）
```

**决策过程**：

**Step 1: 降维算法选择**
→ 候选：PCA、t-SNE、UMAP

**Step 2: 高维数据(784维)**
- PCA：线性降维，速度快但可能损失非线性结构
- t-SNE：非线性，保持局部结构，但慢
- UMAP：非线性，速度快，保持全局+局部结构

**Step 3: 大样本(70K)**
- t-SNE：O(n²)复杂度，70K样本会很慢
- UMAP：O(n log n)，适合大样本

**算法对比**：

| 算法 | 速度 | 局部结构 | 全局结构 | 大样本适用性 |
|------|------|---------|---------|------------|
| PCA | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| t-SNE | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| UMAP | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**最终选择**：
```
方案1：PCA降至50维 → UMAP降至2维（两步降维）
方案2：直接UMAP降至2维

实验结果：
- 方案1耗时：12分钟（PCA 2分钟 + UMAP 10分钟）
- 方案2耗时：15分钟
- 可视化质量：方案1略好（先用PCA去噪）

最终部署：PCA(50) + UMAP(2)
```

---

### 案例5：设备故障预测（时间序列异常检测）

**问题画像**：
```
【问题类型】
[✓] 异常检测 + 时间序列

【数据规模】
样本量：500,000条传感器数据 → [✓] 大样本
特征数：15列（温度、压力、振动等）→ [✓] 中维
时间跨度：6个月

【数据特征】
故障样本：< 0.5%（极度稀少）
时间依赖：强（传感器数据有时序关系）
```

**决策过程**：

**特殊之处**：
- 既有极度不平衡（<0.5%），又有时间序列特性
- 传统分类/异常检测都不完全适用

**算法筛选**：

| 方案 | 算法 | 优点 | 缺点 |
|------|------|------|------|
| 方案1 | Isolation Forest（忽略时序） | 简单、快速 | 忽略了时序信息 |
| 方案2 | LSTM AutoEncoder | 考虑时序 | 训练复杂 |
| 方案3 | Prophet + 异常检测 | 时序+异常 | 逐个传感器分析 |

**最终选择**：
```
混合方案：
1. 特征工程：构造时序特征
   - 滑动窗口统计（3小时均值、标准差）
   - 时序差分（当前值 - 1小时前）
   - 趋势特征（是否持续上升）

2. 异常检测：Isolation Forest
   - 输入：原始特征 + 时序特征（15+10=25维）
   - 优点：结合了时序信息，但仍用简单模型

3. 后处理：时序平滑
   - 如果连续3个时间点都异常 → 真异常
   - 孤立异常点 → 忽略（噪声）

实验结果：
- Precision: 0.78（减少误报）
- Recall: 0.85（提前2小时预警）
- 业务价值：减少30%意外停机
```

---

## 🎯 算法快速对比表（精简版）

### 监督学习算法对比

| 算法 | 优点 | 缺点 | 适用场景 | 不适用场景 |
|------|------|------|---------|-----------|
| **Linear/Logistic** | 快速、可解释、低维效果好 | 不能处理非线性 | 线性可分、高可解释性、快速Baseline | 复杂非线性关系 |
| **Naive Bayes** | 极快、适合高维文本 | 假设特征独立 | 文本分类、小样本 | 特征强相关 |
| **Decision Tree** | 可解释、处理非线性 | 易过拟合 | 可解释性要求、特征交互 | 需要高精度 |
| **Random Forest** | 强大、鲁棒、特征重要性 | 黑盒、大模型 | 中大样本、非线性、特征选择 | 高可解释性需求 |
| **XGBoost** | 性能最优、处理缺失 | 参数多、训练慢 | 竞赛、高性能需求、结构化数据 | 实时训练、高可解释性 |
| **LightGBM** | 超快、大数据 | 小样本易过拟合 | 超大数据集、速度要求 | 小样本(<1K) |
| **SVM** | 小样本效果好、核技巧 | 大数据慢、难调参 | 小中样本、非线性分类 | 大数据(>100K)、需要概率输出 |
| **KNN** | 简单、无训练 | 预测慢、维度灾难 | 小样本、Baseline | 高维数据、实时预测 |

### 无监督学习算法对比

| 算法 | 优点 | 缺点 | 适用场景 | 不适用场景 |
|------|------|------|---------|-----------|
| **K-Means** | 快速、简单、可扩展 | 需指定k、球形簇 | 清晰分群、大数据 | 任意形状簇、离群点多 |
| **DBSCAN** | 任意形状、自动确定k | 参数敏感 | 空间聚类、任意形状 | 密度不均 |
| **Hierarchical** | 树状图可视化 | 慢O(n³) | 小数据、层次关系 | 大数据(>10K) |
| **GMM** | 软聚类、概率模型 | 参数多 | 重叠簇、概率需求 | 大规模聚类 |
| **PCA** | 快速、线性降维 | 仅线性 | 降噪、特征压缩 | 保持非线性结构 |
| **t-SNE** | 保持局部结构 | 慢、随机性 | 可视化（2D/3D） | 大数据(>50K)、降维后建模 |
| **UMAP** | 快、保持全局+局部 | 较新、参数多 | 大数据可视化、降维 | 要求完全确定性 |
| **Isolation Forest** | 快、适合高维 | 无监督 | 异常检测、离群点检测 | 需要类别标签 |

---

## ⚠️ 常见陷阱与解决

### 陷阱1：盲目追求复杂模型

**错误做法**：
```python
# ❌ 直接上最复杂的模型
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=10,
    ...
)  # 训练3小时，过拟合
```

**正确做法**：
```python
# ✅ 先用简单Baseline
from sklearn.linear_model import LogisticRegression
baseline = LogisticRegression()
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)  # 0.85

# 再用复杂模型
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)  # 0.87

print(f"提升: {rf_score - baseline_score:.3f}")  # 仅提升0.02

# 结论：数据可能线性可分，Baseline已足够
# 节省了大量调参时间！
```

---

### 陷阱2：忽略业务约束，只看性能

**错误案例**：
```
医疗诊断系统
- XGBoost: AUC=0.95 ✓
- 但医生无法理解模型决策 ✗
- 法律要求可解释性 ✗
→ 无法部署！
```

**正确做法**：
```
方案调整：
- Logistic Regression: AUC=0.92
- 可解释每个特征的影响
- 满足法律要求
→ 虽然性能略低，但可部署
→ 后续用SHAP等工具进一步解释
```

---

### 陷阱3：数据泄漏导致虚高性能

**错误示例**：
```python
# ❌ 在全量数据上做预处理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 使用了全量数据的均值/标准差！

# 再划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# 结果：测试集性能虚高（因为用了测试集的统计量）
```

**正确做法**：
```python
# ✅ 先划分，再预处理
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 只用训练集
X_test_scaled = scaler.transform(X_test)  # 用训练集的统计量

# 或使用Pipeline自动处理
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)  # Pipeline自动处理
```

---

### 陷阱4：小样本使用复杂模型

**错误做法**：
```python
# 样本量：300行
# ❌ 使用深度学习
import tensorflow as tf
model = tf.keras.Sequential([...])  # 100万参数
model.fit(X_train, y_train)  # 严重过拟合
```

**正确做法**：
```python
# ✅ 小样本用简单模型+正则化
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    C=0.1,  # 强正则化
    penalty='l2'
)
model.fit(X_train, y_train)

# 或使用LOO交叉验证
from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
```

---

## 🔗 相关资源

### 前置学习

💡 **第一次选择算法**，建议先学习：
- 📖 03_algorithm_selection_matrix/algorithm_comparison_table.md（14个算法详细对比）
- 📖 03_algorithm_selection_matrix/algorithm_selection_decision_tree.md（完整决策树）
- 📖 ML_WORKFLOW_GUIDE.md - 第3部分：算法选择阶段

### 后续步骤

📖 **完成算法选择后**，查看：
- 📋 model_evaluation_template.md（选择评估指标）
- 📋 hyperparameter_tuning_template.md（调优选定的算法）
- 💻 08_code_templates/modeling_templates.py（获取训练代码）

### 深入学习

📚 **想深入理解算法**，参考：
- 📖 supervised_learning/（监督学习算法原理）
- 📖 unsupervised_learning/（无监督学习算法原理）
- 📖 06_comprehensive_project/phase3_supervised_solution.ipynb（完整建模流程）

---

**最后更新**：2024年11月
**适用场景**：确定问题类型后的算法选择
**建议使用频率**：每个新项目使用一次
