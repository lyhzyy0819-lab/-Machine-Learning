# 客户流失预测 - AutoML 实战项目

> 🎯 对比AutoML与传统手动方法，理解AutoML的优势与局限

---

## 📋 项目概述

本项目使用AutoML工具重新实现 `supervised_learning/projects/02_customer_churn_prediction/` 项目，并进行详细对比分析。

### 业务背景

**客户流失预测**是企业级机器学习的经典应用：
- 📊 预测哪些客户可能流失
- 💰 提前采取挽留措施（成本降低5-25倍）
- 🎯 优化营销策略，提升客户生命周期价值

### 数据集

**Telco Customer Churn Dataset**
- 样本数：7043条客户记录
- 特征数：21个特征
- 目标：预测客户是否流失（二分类）

---

## 🎯 项目目标

通过本项目，你将：

1. ✅ 使用AutoML快速建立高性能baseline（3分钟）
2. ✅ 对比AutoML vs 手动方法的性能差异
3. ✅ 分析AutoML的特征工程和模型选择逻辑
4. ✅ 学习如何从AutoML中提取insights指导手动优化
5. ✅ 理解AutoML的适用场景和局限性

---

## 📂 文件结构

```
01_customer_churn_automl/
├── README.md                              # 本文件
├── 01_pycaret_churn.ipynb                 # PyCaret实现（最易用）
├── 02_flaml_churn.ipynb                   # FLAML实现（最快速）
├── 03_comparison_with_manual.ipynb        # 详细对比AutoML vs 手动
└── data/                                  # 数据目录（与原项目共享）
    └── README.md                          # 数据说明
```

---

## 🚀 快速开始

### 1. 数据准备

```python
# 数据位置（使用原项目的数据）
original_project = "../../supervised_learning/projects/02_customer_churn_prediction/"
data_path = original_project + "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

### 2. PyCaret 3分钟建模

```python
from pycaret.classification import *

# 1. 初始化（自动特征工程）
setup(data, target='Churn', session_id=42)

# 2. 对比15+模型（自动训练+交叉验证）
best = compare_models()

# 3. 调优
tuned = tune_model(best)

# 4. 完成！
```

### 3. FLAML 快速优化

```python
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task='classification', time_budget=180)

print(f"最佳模型: {automl.best_estimator}")
print(f"准确率: {automl.best_loss}")
```

---

## 📊 预期对比结果

### 时间成本对比

| 方法 | 数据预处理 | 特征工程 | 模型选择 | 超参数调优 | 总时间 |
|------|-----------|---------|---------|-----------|--------|
| **手动方法** | 30分钟 | 1小时 | 30分钟 | 1小时 | **~3小时** |
| **PyCaret** | 自动 | 自动 | 自动 | 自动 | **3-5分钟** |
| **FLAML** | 自动 | 自动 | 自动 | 自动 | **1-3分钟** |

### 性能对比（预期）

| 方法 | 准确率 | Precision | Recall | F1 Score | AUC |
|------|--------|-----------|--------|----------|-----|
| **手动基线** | 0.78 | 0.65 | 0.54 | 0.59 | 0.82 |
| **手动优化** | 0.81 | 0.68 | 0.60 | 0.64 | 0.85 |
| **PyCaret** | 0.80-0.82 | 0.66-0.70 | 0.58-0.63 | 0.62-0.66 | 0.84-0.87 |
| **FLAML** | 0.79-0.81 | 0.65-0.68 | 0.56-0.61 | 0.60-0.64 | 0.83-0.86 |

**结论**：
- ✅ AutoML用3分钟达到手动3小时的性能水平
- ✅ 特定场景下，手动精调可能略优1-2%
- ✅ AutoML适合快速baseline和数据探索

---

## 🔍 关键分析点

### 1. AutoML的特征工程

**PyCaret自动处理**：
- ✅ 缺失值填充
- ✅ 类别编码（One-Hot/Label Encoding）
- ✅ 数值特征标准化
- ✅ 特征交互生成
- ✅ 异常值处理

**对比手动方法**：
```python
# 手动特征工程（需要领域知识）
data['tenure_group'] = pd.cut(data['tenure'], bins=[0, 12, 24, 48, 72])
data['monthly_charges_per_year'] = data['MonthlyCharges'] * data['tenure']
data['contract_value'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

# PyCaret自动生成（无需手动）
setup(..., feature_interaction=True, polynomial_features=True)
```

### 2. AutoML的模型选择

**PyCaret对比的模型**：
1. Logistic Regression
2. K Neighbors Classifier
3. Decision Tree
4. Random Forest
5. Extra Trees
6. Gradient Boosting (GBM)
7. XGBoost
8. LightGBM
9. CatBoost
10. AdaBoost
11. Naive Bayes
12. SVM
13. Ridge Classifier
14. Quadratic Discriminant
15. MLP Neural Network

**手动方法通常只尝试**：
- Logistic Regression（baseline）
- Random Forest
- XGBoost
- 最多3-5个模型

### 3. AutoML的超参数优化

**PyCaret的调优策略**：
- 使用随机搜索/贝叶斯优化
- 自动交叉验证
- 智能提前停止

**FLAML的成本优化**：
- 成本敏感的超参数搜索
- 优先搜索"性价比"高的配置
- 自适应资源分配

---

## 💡 最佳实践

### Workflow建议

```
步骤1: FLAML快速验证（1分钟）
  ↓ 验证数据质量和特征价值

步骤2: PyCaret深入分析（10分钟）
  ↓ 对比模型、分析特征重要性

步骤3: 提取Insights
  ↓ 查看AutoML选择的最佳模型和特征

步骤4: 手动精细调优（按需）
  ↓ 基于业务需求和AutoML的发现

步骤5: 模型融合
  ↓ 结合AutoML和手动模型
```

### 何时使用AutoML

✅ **适合**：
- 项目初期，快速验证数据价值
- 时间紧迫，需要快速baseline
- 非ML专家需要建模
- 对比多种模型
- 探索性分析

❌ **不适合**：
- 需要完全自定义模型架构
- 需要深入理解每个步骤
- 有严格的模型可解释性要求
- 需要极致性能（竞赛冲榜）

---

## 📈 学习要点

### 从AutoML中学到什么

1. **特征工程技巧**
   - 查看PyCaret自动生成了哪些特征
   - 分析特征重要性排序
   - 学习特征交互的逻辑

2. **模型选择经验**
   - 为什么某些模型在这个数据集上表现好
   - 不同模型的性能差异
   - 模型融合的策略

3. **调优策略**
   - AutoML找到的最优超参数
   - 超参数对性能的影响
   - 交叉验证的重要性

### 将AutoML应用到手动建模

```python
# 1. 从AutoML获取最佳模型
best_model_type = automl.best_estimator  # 例如：XGBoost

# 2. 从AutoML获取最优超参数
best_params = automl.best_config

# 3. 手动实现并进一步优化
from xgboost import XGBClassifier
manual_model = XGBClassifier(**best_params)
# 基于业务需求进一步调整...
```

---

## 🎓 练习任务

### 基础任务
1. ✅ 运行PyCaret和FLAML，对比性能
2. ✅ 分析AutoML选择的最佳模型
3. ✅ 查看AutoML的特征重要性

### 进阶任务
4. ✅ 对比AutoML vs 手动方法的详细指标
5. ✅ 分析AutoML的特征工程逻辑
6. ✅ 尝试将AutoML的insights应用到手动模型

### 挑战任务
7. ✅ 融合AutoML模型和手动模型
8. ✅ 分析AutoML的局限性
9. ✅ 设计一个AutoML + 手动的最优workflow

---

## 📚 参考资源

### 原项目
- 📂 `supervised_learning/projects/02_customer_churn_prediction/`
- 查看手动实现的完整代码和分析

### AutoML文档
- [PyCaret Classification](https://pycaret.gitbook.io/docs/get-started/quickstart#classification)
- [FLAML Classification](https://microsoft.github.io/FLAML/docs/Examples/AutoML-Classification)

### 业务背景
- [Customer Churn Prediction Guide](https://www.kaggle.com/code/bandiatindra/telecom-churn-prediction)
- [客户流失分析最佳实践](https://towardsdatascience.com/churn-prediction-3a4a36c2129a)

---

## ✅ 完成标准

完成本项目后，你应该能够：

- [ ] 使用PyCaret在3分钟内建立客户流失预测模型
- [ ] 使用FLAML快速优化模型
- [ ] 对比AutoML vs 手动方法的性能和时间成本
- [ ] 分析AutoML的特征工程和模型选择逻辑
- [ ] 从AutoML结果中提取insights
- [ ] 理解AutoML的适用场景和局限性
- [ ] 设计AutoML + 手动精调的workflow

---

**最后更新**: 2025-11-18
**难度**: ⭐⭐⭐ 中级
**预计时间**: 2-3小时（包括分析和对比）

---

**🚀 开始学习**: 从 `01_pycaret_churn.ipynb` 开始！
