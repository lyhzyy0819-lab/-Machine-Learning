# 📊 评估指标计算指南

> **定位**：指标的计算方法和代码实现速查手册
>
> **前置条件**：已在 [02_问题定义指南](../02_problem_definition_guide/metrics_selection_guide.md) 中选定评估指标

---

## 📌 本文档内容

**核心功能：**
1. 指标公式与sklearn代码速查
2. 多指标批量计算
3. 自定义指标实现
4. 评估结果可视化

**不包含：** 指标选择决策（参考 02_问题定义指南）

---

## 📋 目录

1. [快速参考速查表](#快速参考速查表) ⭐
2. [回归问题指标](#回归问题指标)
3. [分类问题指标](#分类问题指标)
4. [多指标同时计算](#多指标同时计算)
5. [自定义指标实现](#自定义指标实现)
6. [评估结果可视化](#评估结果可视化)
7. [代码模块使用](#代码模块使用)

---

## 快速参考速查表

### 回归指标速查表

| 指标 | 公式 | sklearn代码 | 适用场景 | 优点 | 缺点 |
|------|------|------------|---------|------|------|
| **MAE** | mean(\|y-ŷ\|) | `mean_absolute_error()` | 默认选择、有异常值 | 易解释、对异常值稳健 | 不区分大小误差 |
| **RMSE** | √mean((y-ŷ)²) | `mean_squared_error(squared=False)` | 关注大误差、无异常值 | 惩罚大误差 | 对异常值敏感 |
| **MAPE** | mean(\|y-ŷ\|/y)×100% | `mean_absolute_percentage_error()` | 需要相对误差 | 百分比形式易理解 | y=0时无定义 |
| **R²** | 1 - SS_res/SS_tot | `r2_score()` | 评估拟合优度 | 归一化（0-1） | 特征越多越高 |

**典型阈值**（供参考）：
- MAE/RMSE：取决于目标变量的尺度，一般<10%的平均值为良好
- MAPE：<10%优秀，10-20%良好，>20%需改进
- R²：>0.7良好，>0.8优秀

### 分类指标速查表

| 指标 | 公式 | sklearn代码 | 适用场景 | 优点 | 缺点 |
|------|------|------------|---------|------|------|
| **Accuracy** | (TP+TN)/Total | `accuracy_score()` | 类别平衡 | 直观易懂 | 不平衡数据失效 |
| **Precision** | TP/(TP+FP) | `precision_score()` | 关注误报代价 | 衡量查准率 | 忽略漏报 |
| **Recall** | TP/(TP+FN) | `recall_score()` | 关注漏报代价 | 衡量查全率 | 忽略误报 |
| **F1-Score** | 2PR/(P+R) | `f1_score()` | P和R同等重要 | 平衡P和R | 等权重不灵活 |
| **AUC** | ROC曲线下面积 | `roc_auc_score()` | 排序能力、阈值优化 | 不受阈值影响 | 不平衡数据可能误导 |

**典型阈值**（供参考）：
- Accuracy：>0.85良好（平衡数据）
- Precision/Recall：取决于业务，通常>0.7为可接受
- F1-Score：>0.75良好
- AUC：>0.8良好，>0.9优秀，0.5=随机

### 指标选择快速决策（速查）

**回归问题**：
```
有异常值？
├─ 是 → MAE（稳健）
└─ 否 → RMSE（惩罚大误差）

需要相对误差？
└─ 是 → MAPE（百分比形式）

评估模型拟合度？
└─ 是 → R²（归一化指标）
```

**分类问题**：
```
数据平衡？
├─ 是 → Accuracy（直观）
└─ 否 → F1-Score 或 AUC

误报代价高？（如垃圾邮件检测）
└─ 是 → Precision（宁可漏报，不能误报）

漏报代价高？（如疾病检测）
└─ 是 → Recall（宁可误报，不能漏报）

需要阈值优化？
└─ 是 → AUC + PR曲线
```

⚠️ **注意**：完整的指标选择决策请参考 [02_问题定义指南/metrics_selection_guide.md](../02_problem_定义_guide/metrics_selection_guide.md)

---

## 回归问题指标

### MAE (Mean Absolute Error) - 平均绝对误差

**公式**：
```
MAE = (1/n) × Σ|y_true - y_pred|
```

**含义**：预测值与真实值的平均绝对差距

**sklearn 实现**：
```python
from sklearn.metrics import mean_absolute_error

# 计算MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")
```

**手动实现**：
```python
import numpy as np

def calculate_mae(y_true, y_pred):
    """
    手动计算MAE

    参数：
        y_true: 真实值，shape (n_samples,)
        y_pred: 预测值，shape (n_samples,)

    返回：
        mae: 平均绝对误差
    """
    return np.mean(np.abs(y_true - y_pred))

# 使用
mae = calculate_mae(y_true, y_pred)
```

---

### RMSE (Root Mean Squared Error) - 均方根误差

**公式**：
```
MSE = (1/n) × Σ(y_true - y_pred)²
RMSE = √MSE
```

**含义**：误差的平方平均后开方（单位与目标一致）

**sklearn 实现**：
```python
from sklearn.metrics import mean_squared_error
import numpy as np

# 计算RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.2f}")

# 也可以这样（sklearn新版本）
rmse = mean_squared_error(y_true, y_pred, squared=False)
```

**手动实现**：
```python
def calculate_rmse(y_true, y_pred):
    """手动计算RMSE"""
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)
```

---

### MAPE (Mean Absolute Percentage Error) - 平均绝对百分比误差

**公式**：
```
MAPE = (100%/n) × Σ|y_true - y_pred| / |y_true|
```

**含义**：相对误差的百分比表示

**sklearn 实现**：
```python
from sklearn.metrics import mean_absolute_percentage_error

# 计算MAPE（返回0-1范围，需乘100转为百分比）
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
print(f"MAPE: {mape:.2f}%")
```

**手动实现**：
```python
def calculate_mape(y_true, y_pred):
    """
    手动计算MAPE

    注意：y_true 不能包含0值
    """
    # 过滤掉y_true为0的样本
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
```

---

### R² (R-Squared) - 决定系数

**公式**：
```
R² = 1 - (SS_res / SS_tot)
其中：
  SS_res = Σ(y_true - y_pred)²  # 残差平方和
  SS_tot = Σ(y_true - y_mean)²  # 总平方和
```

**含义**：模型解释的方差比例

**sklearn 实现**：
```python
from sklearn.metrics import r2_score

# 计算R²
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")
```

**手动实现**：
```python
def calculate_r2(y_true, y_pred):
    """手动计算R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
```

---

### 回归指标快速对比

| 指标 | 公式 | 单位 | 对异常值 | sklearn函数 |
|------|------|------|---------|-------------|
| MAE | mean(\|y - ŷ\|) | 与y相同 | 稳健 | `mean_absolute_error()` |
| RMSE | √mean((y - ŷ)²) | 与y相同 | 敏感 | `mean_squared_error(squared=False)` |
| MAPE | mean(\|y - ŷ\|/y)×100% | 百分比 | 敏感 | `mean_absolute_percentage_error()` |
| R² | 1 - SS_res/SS_tot | 无量纲 | 中等 | `r2_score()` |

---

## 分类问题指标

### 混淆矩阵 (Confusion Matrix)

**基础概念**：
```
              预测为正    预测为负
实际为正        TP         FN
              (真正例)   (假负例)
实际为负        FP         TN
              (假正例)   (真负例)
```

**sklearn 实现**：
```python
from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[TN, FP],
#  [FN, TP]]

# 提取各值
tn, fp, fn, tp = cm.ravel()
```

---

### Accuracy - 准确率

**公式**：
```
Accuracy = (TP + TN) / (TP + FP + FN + TN)
```

**含义**：所有预测中正确的比例

**sklearn 实现**：
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

**从混淆矩阵计算**：
```python
def calculate_accuracy(cm):
    """从混淆矩阵计算准确率"""
    return (cm[0, 0] + cm[1, 1]) / cm.sum()
```

---

### Precision - 精确率

**公式**：
```
Precision = TP / (TP + FP)
```

**含义**："预测为正"的样本中，真正为正的比例

**sklearn 实现**：
```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.3f}")

# 多分类（需指定average参数）
precision = precision_score(y_true, y_pred, average='macro')  # 宏平均
precision = precision_score(y_true, y_pred, average='weighted')  # 加权平均
```

---

### Recall - 召回率

**公式**：
```
Recall = TP / (TP + FN)
```

**含义**："实际为正"的样本中，被正确预测的比例

**sklearn 实现**：
```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.3f}")
```

---

### F1-Score - F1分数

**公式**：
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**含义**：精确率和召回率的调和平均

**sklearn 实现**：
```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.3f}")

# 多分类
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```

---

### AUC (Area Under ROC Curve) - ROC曲线下面积

**含义**：ROC曲线下的面积，度量排序能力

**sklearn 实现**：
```python
from sklearn.metrics import roc_auc_score

# 需要概率预测值（不是0/1预测）
auc = roc_auc_score(y_true, y_pred_proba)
print(f"AUC: {auc:.3f}")

# 多分类（需要指定multi_class参数）
auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')  # One-vs-Rest
```

---

### 分类指标快速对比

| 指标 | 公式 | 关注点 | sklearn函数 |
|------|------|--------|-------------|
| Accuracy | (TP+TN)/Total | 整体正确率 | `accuracy_score()` |
| Precision | TP/(TP+FP) | 预测为正中的准确性 | `precision_score()` |
| Recall | TP/(TP+FN) | 真正例的覆盖率 | `recall_score()` |
| F1-Score | 2PR/(P+R) | P和R的调和平均 | `f1_score()` |
| AUC | ROC曲线下面积 | 排序能力 | `roc_auc_score()` |

---

## 多指标同时计算

### 使用 classification_report

**一次性计算多个分类指标：**
```python
from sklearn.metrics import classification_report

# 生成完整报告
report = classification_report(y_true, y_pred)
print(report)

# 输出示例：
#               precision    recall  f1-score   support
#            0       0.85      0.92      0.88      1000
#            1       0.78      0.65      0.71       500
#     accuracy                           0.83      1500
#    macro avg       0.82      0.79      0.80      1500
# weighted avg       0.83      0.83      0.83      1500

# 获取字典格式
report_dict = classification_report(y_true, y_pred, output_dict=True)
print(f"Class 1 F1-Score: {report_dict['1']['f1-score']:.3f}")
```

---

### 自定义多指标函数（回归）

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import numpy as np

def evaluate_regression(y_true, y_pred, metrics=None):
    """
    一次性计算多个回归指标

    参数：
        y_true: 真实值
        y_pred: 预测值
        metrics: 要计算的指标列表，默认计算所有

    返回：
        results: 字典，包含所有指标
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'r2', 'mape']

    results = {}

    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true, y_pred)

    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    if 'r2' in metrics:
        results['r2'] = r2_score(y_true, y_pred)

    if 'mape' in metrics:
        # 过滤0值
        mask = y_true != 0
        if mask.any():
            results['mape'] = mean_absolute_percentage_error(
                y_true[mask], y_pred[mask]
            ) * 100
        else:
            results['mape'] = np.nan

    return results

# 使用示例
metrics = evaluate_regression(y_true, y_pred)
print(f"MAE:  {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"R²:   {metrics['r2']:.3f}")
print(f"MAPE: {metrics['mape']:.2f}%")
```

---

### 自定义多指标函数（分类）

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_classification(y_true, y_pred, y_pred_proba=None, metrics=None):
    """
    一次性计算多个分类指标

    参数：
        y_true: 真实标签
        y_pred: 预测标签（0/1）
        y_pred_proba: 预测概率（计算AUC需要）
        metrics: 要计算的指标列表，默认计算所有

    返回：
        results: 字典，包含所有指标
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        if y_pred_proba is not None:
            metrics.append('auc')

    results = {}

    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_true, y_pred)

    if 'precision' in metrics:
        results['precision'] = precision_score(y_true, y_pred)

    if 'recall' in metrics:
        results['recall'] = recall_score(y_true, y_pred)

    if 'f1' in metrics:
        results['f1'] = f1_score(y_true, y_pred)

    if 'auc' in metrics and y_pred_proba is not None:
        results['auc'] = roc_auc_score(y_true, y_pred_proba)

    return results

# 使用示例
metrics = evaluate_classification(y_true, y_pred, y_pred_proba)
print(f"Accuracy:  {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1-Score:  {metrics['f1']:.3f}")
print(f"AUC:       {metrics['auc']:.3f}")
```

---

## 自定义指标实现

### 自定义回归指标

**示例：WMAPE（加权MAPE）**
```python
def weighted_mape(y_true, y_pred, weights=None):
    """
    加权MAPE - 不同样本的误差有不同权重

    参数：
        y_true: 真实值
        y_pred: 预测值
        weights: 权重（默认为None，表示等权重）

    返回：
        wmape: 加权MAPE（百分比）
    """
    if weights is None:
        weights = np.ones_like(y_true)

    # 过滤0值
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    weights = weights[mask]

    # 计算加权MAPE
    weighted_errors = weights * np.abs((y_true - y_pred) / y_true)
    wmape = np.sum(weighted_errors) / np.sum(weights) * 100

    return wmape

# 使用示例
weights = np.array([1, 1, 2, 2, 3])  # 后面的样本权重更大
wmape = weighted_mape(y_true, y_pred, weights)
print(f"Weighted MAPE: {wmape:.2f}%")
```

---

### 自定义分类指标

**示例：Precision@K**
```python
def precision_at_k(y_true, y_pred_proba, k=100):
    """
    Precision@K - Top K个预测中的精确率

    适用场景：资源有限，只能处理Top K个样本

    参数：
        y_true: 真实标签
        y_pred_proba: 预测概率
        k: 取Top K个样本

    返回：
        precision: Precision@K
    """
    # 按预测概率排序，取Top K
    top_k_indices = np.argsort(y_pred_proba)[-k:]

    # 计算Top K中的正例数量
    true_positives = np.sum(y_true[top_k_indices])

    # Precision@K
    precision = true_positives / k

    return precision

# 使用示例
k = 100  # 每月只能联系100个客户
precision_k = precision_at_k(y_true, y_pred_proba, k=k)
print(f"Precision@{k}: {precision_k:.3f}")
print(f"Top {k}中有 {int(precision_k * k)} 个真流失客户")
```

---

## 评估结果可视化

### ROC曲线

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_pred_proba, title='ROC Curve'):
    """
    绘制ROC曲线

    参数：
        y_true: 真实标签
        y_pred_proba: 预测概率（正类的概率）
        title: 图表标题
    """
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # 绘制
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

# 使用示例
plot_roc_curve(y_true, y_pred_proba)
```

---

### 混淆矩阵可视化

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """
    绘制混淆矩阵热力图

    参数：
        y_true: 真实标签
        y_pred: 预测标签
        labels: 类别标签（如['Not Churn', 'Churn']）
        title: 图表标题
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(title)
    plt.show()

# 使用示例
plot_confusion_matrix(y_true, y_pred, labels=['Not Churn', 'Churn'])
```

---

### PR曲线（Precision-Recall）

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve(y_true, y_pred_proba, title='Precision-Recall Curve'):
    """
    绘制PR曲线

    适用于不平衡数据（比ROC更敏感）
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.show()

# 使用示例
plot_pr_curve(y_true, y_pred_proba)
```

---

## 代码模块使用

### 使用 src/model_evaluation.py

本项目提供了完整的评估模块（538行），包含上述所有功能。

**回归评估示例：**
```python
from src import model_evaluation

# 方法1：使用便捷函数
metrics = model_evaluation.evaluate_regression(
    y_true, y_pred,
    metrics=['mae', 'rmse', 'r2', 'mape']
)

print("回归评估结果：")
for metric, value in metrics.items():
    print(f"  {metric.upper()}: {value:.3f}")
```

**分类评估示例：**
```python
# 方法2：使用便捷函数
metrics = model_evaluation.evaluate_classification(
    y_true, y_pred, y_pred_proba,
    metrics=['accuracy', 'precision', 'recall', 'f1', 'auc']
)

print("分类评估结果：")
for metric, value in metrics.items():
    print(f"  {metric.capitalize()}: {value:.3f}")

# 绘制可视化
model_evaluation.plot_roc_curve(y_true, y_pred_proba)
model_evaluation.plot_confusion_matrix(y_true, y_pred)
```

---

## 📚 相关文档

- **指标选择决策**：[02_问题定义指南/metrics_selection_guide.md](../02_problem_definition_guide/metrics_selection_guide.md)
- **模型比较与选择**：[model_comparison_and_selection.md](model_comparison_and_selection.md)
- **业务价值转化**：[business_value_translation.md](business_value_translation.md)
- **代码实现**：src/model_evaluation.py（538行）

**sklearn官方文档**：
- [分类指标](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [回归指标](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

---

**最后更新**：2024年11月
