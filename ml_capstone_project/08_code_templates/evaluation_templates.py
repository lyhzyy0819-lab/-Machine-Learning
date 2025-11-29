"""
模型评估代码模板库
=================

快速使用:
    from code_templates.evaluation_templates import (
        quick_evaluate,
        quick_cross_validate,
        compare_models,
        plot_confusion_matrix,
        plot_roc_curve
    )

    # 3行代码完成评估
    metrics = quick_evaluate(y_test, y_pred, problem_type='classification')

    # 交叉验证
    cv_scores = quick_cross_validate(model, X, y, cv=5)

    # 模型对比
    comparison = compare_models(models_dict, X_test, y_test)

对应决策模板: 07_decision_templates/model_evaluation_template.md
参考实现: 06_comprehensive_project/src/model_evaluation.py (538行)

项目定位: ML实战操作手册（非教学项目）
核心价值: 5-15分钟快速代码落地
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ==================== 1. 指标计算 ====================

def quick_evaluate(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_proba: np.ndarray = None,
                  problem_type: str = 'auto',
                  metrics: List[str] = None,
                  verbose: bool = True) -> Dict[str, float]:
    """
    快速模型评估（3分钟）

    对应决策: model_evaluation_template.md - 评估指标选择卡

    Parameters
    ----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    y_proba : array-like, optional
        预测概率（分类问题）
    problem_type : {'auto', 'classification', 'regression'}
        问题类型，auto自动识别
    metrics : list, optional
        指定评估指标，None则使用默认
    verbose : bool, default=True
        是否打印评估结果

    Returns
    -------
    Dict[str, float]
        评估指标字典

    Examples
    --------
    >>> # 分类问题
    >>> metrics = quick_evaluate(y_test, y_pred, problem_type='classification')
    >>> # ✓ 模型评估完成
    >>> #   Accuracy: 0.8500
    >>> #   Precision: 0.8300
    >>> #   Recall: 0.8700
    >>> #   F1: 0.8500

    >>> # 回归问题
    >>> metrics = quick_evaluate(y_test, y_pred, problem_type='regression')
    >>> # ✓ 模型评估完成
    >>> #   MSE: 10.25
    >>> #   RMSE: 3.20
    >>> #   MAE: 2.15
    >>> #   R2: 0.85

    Decision Logic
    --------------
    分类问题:
    - 二分类 → Accuracy, Precision, Recall, F1, AUC
    - 多分类 → Accuracy, Macro F1, Weighted F1

    回归问题:
    - MAE, MSE, RMSE, R2

    Notes
    -----
    - 适合快速评估单个模型
    - 参考06章src/model_evaluation.py:45-120
    """
    # 自动识别问题类型
    if problem_type == 'auto':
        if len(np.unique(y_true)) <= 10:
            problem_type = 'classification'
        else:
            problem_type = 'regression'

    result = {}

    if problem_type == 'classification':
        # 分类指标
        result['accuracy'] = accuracy_score(y_true, y_pred)
        result['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        result['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        result['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # AUC（二分类）
        if y_proba is not None and len(np.unique(y_true)) == 2:
            result['auc'] = roc_auc_score(y_true, y_proba if len(y_proba.shape) == 1 else y_proba[:, 1])

        if verbose:
            print("✓ 模型评估完成（分类）")
            for metric_name, value in result.items():
                print(f"   {metric_name.upper()}: {value:.4f}")
            print()

    else:
        # 回归指标
        result['mae'] = mean_absolute_error(y_true, y_pred)
        result['mse'] = mean_squared_error(y_true, y_pred)
        result['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        result['r2'] = r2_score(y_true, y_pred)

        if verbose:
            print("✓ 模型评估完成（回归）")
            for metric_name, value in result.items():
                print(f"   {metric_name.upper()}: {value:.4f}")
            print()

    return result


# ==================== 2. 交叉验证 ====================

def quick_cross_validate(model: Any,
                        X: pd.DataFrame,
                        y: pd.Series,
                        cv: int = 5,
                        scoring: str = 'auto',
                        verbose: bool = True) -> Dict[str, Any]:
    """
    快速交叉验证（5分钟）

    对应决策: model_evaluation_template.md - 交叉验证策略表

    Parameters
    ----------
    model : estimator
        模型对象
    X : DataFrame
        特征数据
    y : Series
        目标变量
    cv : int, default=5
        交叉验证折数
    scoring : str, default='auto'
        评分指标，auto自动选择
    verbose : bool, default=True
        是否打印结果

    Returns
    -------
    Dict[str, Any]
        交叉验证结果
        {
            'scores': array,  # 各折得分
            'mean': float,    # 平均得分
            'std': float      # 标准差
        }

    Examples
    --------
    >>> cv_results = quick_cross_validate(model, X, y, cv=5)
    >>> # ✓ 交叉验证完成
    >>> #   5折交叉验证得分: [0.83, 0.85, 0.84, 0.86, 0.82]
    >>> #   平均得分: 0.84 ± 0.015

    Decision Logic
    --------------
    分类问题:
    - 二分类 → roc_auc
    - 多分类 → accuracy

    回归问题:
    - r2

    Notes
    -----
    - 交叉验证更稳定可靠
    - 参考06章src/model_evaluation.py:125-170
    """
    # 自动选择评分指标
    if scoring == 'auto':
        if len(np.unique(y)) <= 10:
            if len(np.unique(y)) == 2:
                scoring = 'roc_auc'
            else:
                scoring = 'accuracy'
        else:
            scoring = 'r2'

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    result = {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std()
    }

    if verbose:
        print(f"✓ {cv}折交叉验证完成")
        print(f"   评分指标: {scoring.upper()}")
        print(f"   各折得分: {[f'{s:.4f}' for s in scores]}")
        print(f"   平均得分: {result['mean']:.4f} ± {result['std']:.4f}")
        print()

    return result


# ==================== 3. 模型对比 ====================

def compare_models(models: Dict[str, Any],
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  problem_type: str = 'auto',
                  verbose: bool = True) -> pd.DataFrame:
    """
    多模型对比（10分钟）

    对应决策: model_evaluation_template.md - 模型对比

    Parameters
    ----------
    models : dict
        模型字典 {'model_name': model_object}
    X_test : DataFrame
        测试集特征
    y_test : Series
        测试集标签
    problem_type : {'auto', 'classification', 'regression'}
        问题类型
    verbose : bool, default=True
        是否打印对比表

    Returns
    -------
    DataFrame
        对比结果表

    Examples
    --------
    >>> models = {
    ...     'Random Forest': rf_model,
    ...     'XGBoost': xgb_model,
    ...     'LightGBM': lgb_model
    ... }
    >>> comparison = compare_models(models, X_test, y_test)
    >>> # ✓ 模型对比完成
    >>> #   Model          Accuracy    F1      AUC
    >>> #   Random Forest  0.8500      0.8300  0.8700
    >>> #   XGBoost        0.8700      0.8500  0.8900  ← 最佳
    >>> #   LightGBM       0.8600      0.8400  0.8800

    Notes
    -----
    - 适合最终模型选择
    - 参考06章src/model_evaluation.py:175-235
    """
    # 自动识别问题类型
    if problem_type == 'auto':
        if len(np.unique(y_test)) <= 10:
            problem_type = 'classification'
        else:
            problem_type = 'regression'

    results = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        if problem_type == 'classification':
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics['AUC'] = roc_auc_score(y_test, y_proba)

        else:
            metrics = {
                'Model': model_name,
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred)
            }

        results.append(metrics)

    df_results = pd.DataFrame(results)

    if verbose:
        print("✓ 模型对比完成\n")
        print(df_results.to_string(index=False))
        print()

        # 标注最佳模型
        if problem_type == 'classification':
            best_metric = 'AUC' if 'AUC' in df_results.columns else 'F1'
        else:
            best_metric = 'R2'

        best_idx = df_results[best_metric].idxmax() if best_metric == 'R2' else df_results[best_metric].idxmax()
        best_model = df_results.loc[best_idx, 'Model']
        print(f"🏆 最佳模型: {best_model}\n")

    return df_results


# ==================== 4. 可视化 ====================

def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         labels: List[str] = None,
                         figsize: Tuple[int, int] = (8, 6)):
    """
    绘制混淆矩阵

    Parameters
    ----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    labels : list, optional
        类别标签名称
    figsize : tuple, default=(8, 6)
        图表大小

    Examples
    --------
    >>> plot_confusion_matrix(y_test, y_pred, labels=['Class 0', 'Class 1'])
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    print("✓ 混淆矩阵已绘制\n")


def plot_roc_curve(y_true: np.ndarray,
                  y_proba: np.ndarray,
                  model_name: str = 'Model',
                  figsize: Tuple[int, int] = (8, 6)):
    """
    绘制ROC曲线

    Parameters
    ----------
    y_true : array-like
        真实标签
    y_proba : array-like
        预测概率
    model_name : str, default='Model'
        模型名称
    figsize : tuple, default=(8, 6)
        图表大小

    Examples
    --------
    >>> plot_roc_curve(y_test, y_proba, model_name='XGBoost')
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"✓ ROC曲线已绘制 (AUC = {auc_score:.4f})\n")


def plot_feature_importance(model: Any,
                           feature_names: List[str],
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (10, 8)):
    """
    绘制特征重要性

    Parameters
    ----------
    model : estimator
        模型对象（必须有feature_importances_属性）
    feature_names : list
        特征名称列表
    top_n : int, default=20
        展示前N个重要特征
    figsize : tuple, default=(10, 8)
        图表大小

    Examples
    --------
    >>> plot_feature_importance(model, X.columns.tolist(), top_n=20)
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  模型不支持特征重要性分析")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importances[indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"✓ 特征重要性已绘制 (Top {top_n})\n")
