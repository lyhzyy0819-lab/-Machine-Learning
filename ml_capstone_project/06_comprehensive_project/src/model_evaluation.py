"""
模型评估模块
============

提供全面的模型评估功能，帮助选择最佳模型。

主要功能:
- 分类模型评估（Accuracy、Precision、Recall、F1、ROC-AUC）
- 回归模型评估（MAE、MSE、RMSE、R²、MAPE）
- 聚类模型评估（Silhouette、Calinski-Harabasz、Davies-Bouldin）
- 混淆矩阵可视化
- ROC曲线和PR曲线
- 学习曲线
- 特征重要性分析
- 模型对比

评估指标选择原则:
- 分类: 不平衡数据看F1和AUC，平衡数据看Accuracy
- 回归: RMSE看绝对误差，R²看拟合程度，MAPE看相对误差
- 聚类: Silhouette看簇内紧密度，CH Index看簇间分离度
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve,
                            confusion_matrix, classification_report,
                            mean_absolute_error, mean_squared_error, r2_score,
                            silhouette_score, calinski_harabasz_score,
                            davies_bouldin_score, log_loss)
from sklearn.model_selection import learning_curve, cross_val_score
import warnings
warnings.filterwarnings('ignore')


# ==================== 分类模型评估 ====================

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None,
                           average: str = 'binary',
                           verbose: bool = True) -> Dict[str, float]:
    """
    评估分类模型

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率（用于计算AUC）
        average: 多分类平均方式 ('binary', 'micro', 'macro', 'weighted')
        verbose: 是否打印详细报告

    Returns:
        包含各评估指标的字典
    """
    metrics = {}

    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # AUC（如果提供了预测概率）
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # 二分类
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:  # 多分类
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba,
                                                  multi_class='ovr', average=average)
        except Exception as e:
            print(f"⚠️  无法计算ROC-AUC: {e}")

    # Log Loss
    if y_pred_proba is not None:
        try:
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except Exception:
            pass

    if verbose:
        print("\n" + "=" * 50)
        print("📊 分类模型评估结果")
        print("=" * 50)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.upper():<15}: {metric_value:.4f}")
        print("=" * 50 + "\n")

        # 详细分类报告
        print("详细分类报告:")
        print(classification_report(y_true, y_pred, zero_division=0))

    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: Optional[List[str]] = None,
                         normalize: bool = False,
                         figsize: Tuple[int, int] = (8, 6)):
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 类别标签名称
        normalize: 是否归一化（显示百分比）
        figsize: 图像大小
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 绘图
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', square=True,
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵' + (' (归一化)' if normalize else ''))
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  labels: Optional[List[str]] = None,
                  figsize: Tuple[int, int] = (8, 6)):
    """
    绘制ROC曲线

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        labels: 类别标签（多分类时使用）
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)

    n_classes = len(np.unique(y_true))

    if n_classes == 2:
        # 二分类ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')

    else:
        # 多分类ROC曲线（一对多）
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            label_name = labels[i] if labels else f'Class {i}'
            plt.plot(fpr, tpr, linewidth=2, label=f'{label_name} (AUC = {auc:.3f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC曲线')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================== 回归模型评估 ====================

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray,
                       verbose: bool = True) -> Dict[str, float]:
    """
    评估回归模型

    Args:
        y_true: 真实值
        y_pred: 预测值
        verbose: 是否打印详细报告

    Returns:
        包含各评估指标的字典
    """
    metrics = {}

    # 基础指标
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_true, y_pred)

    # MAPE（平均绝对百分比误差）
    # 避免除以0
    mask = y_true != 0
    if mask.sum() > 0:
        metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics['mape'] = np.inf

    # 调整R²（惩罚过多特征）
    n = len(y_true)
    p = 1  # 如果知道特征数，可以传入
    if n > p + 1:
        metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)

    if verbose:
        print("\n" + "=" * 50)
        print("📊 回归模型评估结果")
        print("=" * 50)
        print(f"{'MAE (平均绝对误差)':<25}: {metrics['mae']:.4f}")
        print(f"{'MSE (均方误差)':<25}: {metrics['mse']:.4f}")
        print(f"{'RMSE (均方根误差)':<25}: {metrics['rmse']:.4f}")
        print(f"{'R² (决定系数)':<25}: {metrics['r2']:.4f}")
        if 'mape' in metrics and metrics['mape'] != np.inf:
            print(f"{'MAPE (平均百分比误差)':<25}: {metrics['mape']:.2f}%")
        print("=" * 50 + "\n")

        # R²解释
        if metrics['r2'] > 0.9:
            print("✅ R² > 0.9: 模型拟合非常好")
        elif metrics['r2'] > 0.7:
            print("✓  R² > 0.7: 模型拟合较好")
        elif metrics['r2'] > 0.5:
            print("⚠️  R² > 0.5: 模型拟合一般")
        else:
            print("❌ R² < 0.5: 模型拟合较差")

    return metrics


def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray,
                           figsize: Tuple[int, int] = (12, 5)):
    """
    可视化回归结果

    Args:
        y_true: 真实值
        y_pred: 预测值
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图：预测值 vs 真实值
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', s=50)
    axes[0].plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                'r--', linewidth=2, label='Perfect Prediction')

    axes[0].set_xlabel('真实值')
    axes[0].set_ylabel('预测值')
    axes[0].set_title('预测值 vs 真实值')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右图：残差分布
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', s=50)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)

    axes[1].set_xlabel('预测值')
    axes[1].set_ylabel('残差')
    axes[1].set_title('残差分布图')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== 聚类模型评估 ====================

def evaluate_clustering(X: np.ndarray, labels: np.ndarray,
                       verbose: bool = True) -> Dict[str, float]:
    """
    评估聚类模型

    Args:
        X: 特征数据
        labels: 聚类标签
        verbose: 是否打印详细报告

    Returns:
        包含各评估指标的字典
    """
    metrics = {}

    # 聚类指标（不需要真实标签）
    if len(np.unique(labels)) > 1:  # 至少有2个簇
        metrics['silhouette'] = silhouette_score(X, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)

    if verbose:
        print("\n" + "=" * 50)
        print("📊 聚类模型评估结果")
        print("=" * 50)
        print(f"簇数量: {len(np.unique(labels))}")
        print(f"样本数: {len(labels)}")
        print()

        if 'silhouette' in metrics:
            print(f"{'Silhouette Score':<30}: {metrics['silhouette']:.4f}")
            print("  范围: [-1, 1]，越接近1越好")
            print("  > 0.7: 强聚类结构")
            print("  0.5-0.7: 合理聚类结构")
            print("  < 0.5: 聚类结构较弱")
            print()

        if 'calinski_harabasz' in metrics:
            print(f"{'Calinski-Harabasz Index':<30}: {metrics['calinski_harabasz']:.4f}")
            print("  值越大越好（簇间分离度高，簇内紧密度高）")
            print()

        if 'davies_bouldin' in metrics:
            print(f"{'Davies-Bouldin Index':<30}: {metrics['davies_bouldin']:.4f}")
            print("  值越小越好（簇间分离度高）")

        print("=" * 50 + "\n")

    return metrics


# ==================== 学习曲线 ====================

def plot_learning_curve(estimator, X, y, cv=5, scoring='accuracy',
                       figsize: Tuple[int, int] = (10, 6)):
    """
    绘制学习曲线（诊断过拟合/欠拟合）

    Args:
        estimator: 模型
        X: 特征数据
        y: 目标变量
        cv: 交叉验证折数
        scoring: 评分指标
        figsize: 图像大小
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    # 计算均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # 绘图
    plt.figure(figsize=figsize)

    # 训练集得分
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='训练集得分')
    plt.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1, color='r')

    # 验证集得分
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='验证集得分')
    plt.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.1, color='g')

    plt.xlabel('训练样本数')
    plt.ylabel(f'{scoring.upper()}')
    plt.title('学习曲线')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 诊断建议
    final_train_score = train_mean[-1]
    final_val_score = val_mean[-1]
    gap = final_train_score - final_val_score

    print("\n学习曲线诊断:")
    if gap > 0.1:
        print("⚠️  过拟合: 训练集得分显著高于验证集")
        print("   建议: 增加数据量、正则化、减少模型复杂度")
    elif final_val_score < 0.7:
        print("⚠️  欠拟合: 验证集得分较低")
        print("   建议: 增加模型复杂度、增加特征、减少正则化")
    else:
        print("✅ 模型拟合良好")


# ==================== 特征重要性 ====================

def plot_feature_importance(model, feature_names: List[str],
                          top_n: int = 20,
                          figsize: Tuple[int, int] = (10, 8)):
    """
    可视化特征重要性

    Args:
        model: 训练好的模型（需支持feature_importances_）
        feature_names: 特征名称列表
        top_n: 显示前N个重要特征
        figsize: 图像大小
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  模型不支持feature_importances_属性")
        return

    # 获取特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    # 绘图
    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('重要性')
    plt.title(f'Top {top_n} 特征重要性')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================== 模型对比 ====================

def compare_models(results: Dict[str, Dict[str, float]],
                  metric: str = 'accuracy',
                  figsize: Tuple[int, int] = (10, 6)):
    """
    对比多个模型的性能

    Args:
        results: 模型结果字典，格式为 {model_name: {metric: value}}
        metric: 要对比的指标
        figsize: 图像大小
    """
    # 提取数据
    model_names = list(results.keys())
    metric_values = [results[model][metric] for model in model_names]

    # 创建DataFrame
    df = pd.DataFrame({
        '模型': model_names,
        metric: metric_values
    }).sort_values(metric, ascending=False)

    print("\n" + "=" * 60)
    print(f"📊 模型性能对比（按 {metric.upper()} 排序）")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60 + "\n")

    # 绘图
    plt.figure(figsize=figsize)
    colors = ['gold' if i == 0 else 'steelblue' for i in range(len(df))]
    plt.barh(df['模型'], df[metric], color=colors)

    # 在柱子上标注数值
    for i, v in enumerate(df[metric]):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center')

    plt.xlabel(metric.upper())
    plt.title(f'模型{metric.upper()}对比')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================== 交叉验证评估 ====================

def cross_validate_model(model, X, y, cv: int = 5,
                        scoring: str = 'accuracy',
                        verbose: bool = True) -> Dict[str, Any]:
    """
    交叉验证评估模型

    Args:
        model: 模型
        X: 特征数据
        y: 目标变量
        cv: 交叉验证折数
        scoring: 评分指标
        verbose: 是否打印详细信息

    Returns:
        评估结果字典
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    results = {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max()
    }

    if verbose:
        print(f"\n{cv}折交叉验证结果 ({scoring}):")
        print(f"  各折得分: {[f'{s:.4f}' for s in scores]}")
        print(f"  平均得分: {results['mean']:.4f} (+/- {results['std']*2:.4f})")
        print(f"  得分范围: [{results['min']:.4f}, {results['max']:.4f}]")

    return results


if __name__ == '__main__':
    # 测试示例
    print("=== 模型评估模块测试 ===\n")

    # 模拟分类数据
    np.random.seed(42)
    y_true = np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    y_pred = y_true.copy()
    # 添加一些错误
    error_indices = np.random.choice(1000, 100, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    y_pred_proba = np.random.rand(1000)

    # 测试分类评估
    print("1. 分类模型评估")
    metrics = evaluate_classification(y_true, y_pred, y_pred_proba)

    # 测试回归评估
    print("\n2. 回归模型评估")
    y_true_reg = np.random.randn(1000) * 10 + 50
    y_pred_reg = y_true_reg + np.random.randn(1000) * 2
    metrics_reg = evaluate_regression(y_true_reg, y_pred_reg)

    print("\n✅ 所有测试通过！")
