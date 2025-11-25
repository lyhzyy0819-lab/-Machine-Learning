# 电商评分预测 - AutoML 多分类任务

> 🎯 使用AutoML解决电商评分预测（多分类/回归）问题

---

## 📋 项目概述

本项目使用AutoML工具重新实现 `supervised_learning/projects/04_ecommerce_rating_prediction/` 项目，展示AutoML在多分类任务中的应用。

### 业务背景

**电商评分预测**：
- 🛒 预测客户对商品的评分（1-5星）
- 📊 可以作为多分类或回归任务
- 💡 应用：推荐系统、商品质量分析

### 数据集

**Women's E-Commerce Clothing Reviews**
- 样本数：~23,000条评论
- 特征：评论文本、年龄、商品类别等
- 目标：评分（1-5星）

---

## 🎯 项目目标

1. ✅ 对比多分类 vs 回归两种建模方式
2. ✅ 使用AutoML处理文本特征
3. ✅ 分析AutoML的特征工程策略
4. ✅ 理解AutoML处理不平衡数据

---

## 📂 文件结构

```
03_ecommerce_rating_automl/
├── README.md                          # 本文件
├── 01_pycaret_multiclass.ipynb       # PyCaret多分类
├── 02_pycaret_regression.ipynb       # PyCaret回归方法
└── 03_comparison.ipynb               # 多分类vs回归对比
```

---

## 🚀 两种建模方式

### 方式1: 多分类（Classification）

将评分1-5视为5个类别

```python
from pycaret.classification import *

# 目标变量转为类别
data['Rating'] = data['Rating'].astype(str)

setup(data, target='Rating')
best = compare_models()
```

**优势**：
- ✅ 直接预测评分类别
- ✅ 可以使用分类评估指标（Accuracy、F1）

**劣势**：
- ❌ 忽略了评分的顺序关系（3星比1星好）

### 方式2: 回归（Regression）

将评分视为连续值

```python
from pycaret.regression import *

setup(data, target='Rating')
best = compare_models()

# 预测后四舍五入到1-5
predictions = np.clip(np.round(predictions), 1, 5)
```

**优势**：
- ✅ 考虑评分的顺序关系
- ✅ 可以预测中间值（如3.5星）

**劣势**：
- ❌ 需要后处理（四舍五入）

---

## 📊 预期性能对比

| 方法 | Accuracy | MAE | RMSE | 训练时间 |
|------|----------|-----|------|---------|
| **多分类AutoML** | ~0.55 | ~0.65 | ~0.85 | ~3分钟 |
| **回归AutoML** | ~0.52 | ~0.58 | ~0.78 | ~3分钟 |
| **手动优化** | ~0.58 | ~0.60 | ~0.80 | ~3小时 |

---

## 💡 AutoML处理多分类的技巧

### 1. 不平衡数据

评分分布通常不均衡（5星很多，1星较少）

```python
# PyCaret自动处理
setup(..., fix_imbalance=True)
```

### 2. 文本特征

如果有评论文本：

```python
# 需要先手动TF-IDF或词嵌入
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100)
text_features = tfidf.fit_transform(data['Review Text'])
```

### 3. 评估指标选择

- **Accuracy**: 整体准确率
- **Macro F1**: 考虑每个类别的平衡
- **Weighted F1**: 考虑类别权重

---

## 🔍 关键分析

### AutoML在多分类任务的表现

✅ **优势**：
- 自动处理类别不平衡
- 对比多种多分类算法
- 自动调优

⚠️ **注意**：
- 需要人工决定：多分类 vs 回归
- 文本特征需要预处理
- 评估指标的选择很重要

---

## 🎓 学习要点

1. **多分类 vs 回归的权衡**
   - 评分问题的特殊性
   - 业务需求决定建模方式

2. **AutoML的灵活性**
   - 同一数据可以用不同方式建模
   - 快速对比不同方法

3. **特征工程**
   - 文本特征的处理
   - 类别特征的编码

---

## ✅ 完成标准

- [ ] 理解多分类vs回归的差异
- [ ] 使用PyCaret建立多分类模型
- [ ] 使用PyCaret建立回归模型
- [ ] 对比两种方法的性能
- [ ] 分析AutoML的特征工程

---

**最后更新**: 2025-11-19
**难度**: ⭐⭐⭐ 中级
**预计时间**: 1.5-2小时
