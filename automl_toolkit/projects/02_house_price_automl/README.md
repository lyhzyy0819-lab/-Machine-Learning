# 房价预测 - AutoML 回归任务实战

> 🎯 使用AutoML快速实现房价预测，对比回归任务中AutoML的表现

---

## 📋 项目概述

本项目使用AutoML工具重新实现 `supervised_learning/projects/01_house_price_prediction/` 项目，展示AutoML在回归任务中的应用。

### 业务背景

**房价预测**是机器学习回归任务的经典案例：
- 🏠 预测房屋价格（连续值）
- 📊 多变量回归问题
- 💰 实际应用广泛（房产估值、投资决策）

### 数据集

**California Housing Dataset**
- 样本数：20,640个区块
- 特征数：8个特征
- 目标：房价中位数（单位：10万美元）
- 特征：收入、房龄、房间数、地理位置等

---

## 🎯 项目目标

1. ✅ 使用AutoML快速建立回归模型baseline
2. ✅ 对比AutoML在回归任务中的性能
3. ✅ 分析AutoML的特征重要性
4. ✅ 理解回归任务中AutoML的特点

---

## 📂 文件结构

```
02_house_price_automl/
├── README.md                          # 本文件
├── 01_pycaret_regression.ipynb       # PyCaret回归实现
├── 02_flaml_regression.ipynb         # FLAML回归实现
└── 03_automl_comparison.ipynb        # 多工具对比
```

---

## 🚀 快速开始

### PyCaret回归模型（3分钟）

```python
from pycaret.regression import *

# 1. 初始化
setup(data, target='MedHouseVal', session_id=42)

# 2. 对比模型
best = compare_models()

# 3. 调优
tuned = tune_model(best)

# 完成！
```

### FLAML回归模型（1分钟）

```python
from flaml import AutoML

automl = AutoML()
automl.fit(X_train, y_train, task='regression', time_budget=60)
```

---

## 📊 预期对比结果

### 性能指标

| 方法 | R² Score | RMSE | MAE | 训练时间 |
|------|----------|------|-----|---------|
| **手动Baseline** | 0.60 | 0.75 | 0.53 | ~2小时 |
| **手动优化** | 0.64 | 0.71 | 0.50 | ~3小时 |
| **PyCaret** | 0.63-0.65 | 0.70-0.72 | 0.49-0.51 | ~3分钟 |
| **FLAML** | 0.62-0.64 | 0.71-0.73 | 0.50-0.52 | ~1分钟 |

---

## 💡 回归任务中AutoML的特点

### 1. 评估指标不同

**分类任务**: Accuracy, AUC, F1
**回归任务**: R², RMSE, MAE, MAPE

### 2. 模型选择

**回归常用模型**:
- Linear Regression
- Ridge/Lasso
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost/LightGBM/CatBoost

### 3. 特征工程

**回归特有**:
- 目标变量的对数变换
- 多项式特征更重要
- 特征缩放影响更大

---

## 🔍 关键分析

### AutoML在回归任务中的优势

✅ **快速对比**: 自动尝试多种回归算法
✅ **特征变换**: 自动处理偏态分布
✅ **超参数优化**: 针对回归指标优化
✅ **模型融合**: Stacking提升性能

### 局限性

❌ **目标变换**: 可能需要手动对数变换
❌ **异常值**: 需要额外注意
❌ **地理特征**: 可能需要手动工程

---

## 🎓 学习要点

### 从AutoML学到的回归技巧

1. **特征重要性排序**
   - 收入（MedInc）通常最重要
   - 地理位置（Latitude/Longitude）影响大

2. **模型选择**
   - 树模型（XGBoost/LightGBM）通常表现最好
   - 线性模型作为baseline

3. **特征工程**
   - 地理位置的交叉特征
   - 房间数相关比率特征

---

## ✅ 完成标准

- [ ] 使用PyCaret建立回归模型
- [ ] 使用FLAML快速优化
- [ ] 对比AutoML与手动方法
- [ ] 分析特征重要性
- [ ] 理解回归任务的AutoML特点

---

**最后更新**: 2025-11-19
**难度**: ⭐⭐ 初-中级
**预计时间**: 1-2小时
