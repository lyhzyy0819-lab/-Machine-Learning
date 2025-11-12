"""
模型评估模块
提供交叉验证、混淆矩阵、分类报告等评估功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

from sklearn.model_selection import cross_val_score, cross_validate, learning_curve
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer


def evaluate_regression_cv(model, X: np.ndarray, y: np.ndarray,
                          cv: int = None,
                          scoring: List[str] = None) -> Dict[str, Any]:
    """
    使用交叉验证评估回归模型

    Args:
        model: 回归模型
        X: 特征数组
        y: 目标数组
        cv: 交叉验证折数
        scoring: 评分指标列表

    Returns:
        评估结果字典

    TODO 1: 设置默认参数
    # logger = logging.getLogger("Ecommerce_Rating")
    # if cv is None:
    #     cv = config.CV_FOLDS
    # if scoring is None:
    #     scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

    TODO 2: 打印开始信息
    # logger.info(f"执行 {cv}-折交叉验证...")

    TODO 3: 执行交叉验证
    # with Timer("交叉验证"):
    #     cv_results = cross_validate(
    #         model, X, y,
    #         cv=cv,
    #         scoring=scoring,
    #         return_train_score=True,
    #         n_jobs=-1
    #     )

    TODO 4: 整理结果
    #     results = {}
    #     for metric in scoring:
    #         test_key = f'test_{metric}'
    #         train_key = f'train_{metric}'
    #
    #         # 对于负指标（如neg_mean_squared_error），取反
    #         if 'neg_' in metric:
    #             test_scores = -cv_results[test_key]
    #             train_scores = -cv_results[train_key]
    #             metric_name = metric.replace('neg_', '')
    #         else:
    #             test_scores = cv_results[test_key]
    #             train_scores = cv_results[train_key]
    #             metric_name = metric
    #
    #         results[f'{metric_name}_mean'] = test_scores.mean()
    #         results[f'{metric_name}_std'] = test_scores.std()
    #         results[f'{metric_name}_train_mean'] = train_scores.mean()

    TODO 5: 打印结果并返回
    #     logger.info(f"交叉验证结果:")
    #     for key, value in results.items():
    #         logger.info(f"  {key}: {value:.4f}")
    #
    #     return results
    """
    # TODO: 实现回归交叉验证
    pass


def evaluate_classification_cv(model, X: np.ndarray, y: np.ndarray,
                              cv: int = None,
                              scoring: List[str] = None) -> Dict[str, Any]:
    """
    使用交叉验证评估分类模型

    Args:
        model: 分类模型
        X: 特征数组
        y: 目标数组
        cv: 交叉验证折数
        scoring: 评分指标列表

    Returns:
        评估结果字典

    TODO 1: 设置默认参数
    # logger = logging.getLogger("Ecommerce_Rating")
    # if cv is None:
    #     cv = config.CV_FOLDS
    # if scoring is None:
    #     scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    TODO 2: 执行交叉验证（类似evaluate_regression_cv）
    # logger.info(f"执行 {cv}-折交叉验证...")
    # with Timer("交叉验证"):
    #     cv_results = cross_validate(
    #         model, X, y,
    #         cv=cv,
    #         scoring=scoring,
    #         return_train_score=True,
    #         n_jobs=-1
    #     )

    TODO 3: 整理结果并返回
    #     results = {}
    #     for metric in scoring:
    #         test_scores = cv_results[f'test_{metric}']
    #         train_scores = cv_results[f'train_{metric}']
    #
    #         results[f'{metric}_mean'] = test_scores.mean()
    #         results[f'{metric}_std'] = test_scores.std()
    #         results[f'{metric}_train_mean'] = train_scores.mean()
    #
    #     logger.info(f"交叉验证结果:")
    #     for key, value in results.items():
    #         logger.info(f"  {key}: {value:.4f}")
    #
    #     return results
    """
    # TODO: 实现分类交叉验证
    pass


def get_confusion_matrix_stats(y_true: np.ndarray,
                               y_pred: np.ndarray) -> Dict[str, Any]:
    """
    计算混淆矩阵及相关统计

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        包含混淆矩阵和统计信息的字典

    TODO 1: 计算混淆矩阵
    # cm = confusion_matrix(y_true, y_pred)

    TODO 2: 提取TP、FP、TN、FN
    # tn, fp, fn, tp = cm.ravel()

    TODO 3: 计算额外的统计指标
    # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 等同于recall
    # false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    # false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    TODO 4: 组织结果并返回
    # results = {
    #     'confusion_matrix': cm,
    #     'true_negative': int(tn),
    #     'false_positive': int(fp),
    #     'false_negative': int(fn),
    #     'true_positive': int(tp),
    #     'specificity': specificity,
    #     'sensitivity': sensitivity,
    #     'false_positive_rate': false_positive_rate,
    #     'false_negative_rate': false_negative_rate
    # }
    #
    # return results
    """
    # TODO: 实现混淆矩阵统计
    pass


def get_classification_report_dict(y_true: np.ndarray,
                                   y_pred: np.ndarray) -> Dict[str, Any]:
    """
    获取分类报告字典

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        分类报告字典

    TODO 1: 使用sklearn的classification_report生成报告
    # report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    TODO 2: 记录日志并返回
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("\n分类报告:")
    # logger.info(classification_report(y_true, y_pred, zero_division=0))
    #
    # return report
    """
    # TODO: 实现分类报告
    pass


def calculate_roc_curve_data(y_true: np.ndarray,
                             y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    计算ROC曲线数据

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率（正类）

    Returns:
        包含ROC曲线数据的字典

    TODO 1: 计算ROC曲线
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    TODO 2: 计算AUC
    # roc_auc = auc(fpr, tpr)

    TODO 3: 返回结果
    # return {
    #     'fpr': fpr,
    #     'tpr': tpr,
    #     'thresholds': thresholds,
    #     'auc': roc_auc
    # }
    """
    # TODO: 实现ROC曲线计算
    pass


def calculate_precision_recall_curve_data(y_true: np.ndarray,
                                         y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    计算Precision-Recall曲线数据

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率（正类）

    Returns:
        包含PR曲线数据的字典

    TODO 1: 计算PR曲线
    # precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    TODO 2: 计算AUC
    # pr_auc = auc(recall, precision)

    TODO 3: 返回结果
    # return {
    #     'precision': precision,
    #     'recall': recall,
    #     'thresholds': thresholds,
    #     'auc': pr_auc
    # }
    """
    # TODO: 实现PR曲线计算
    pass


def calculate_learning_curve_data(model, X: np.ndarray, y: np.ndarray,
                                  cv: int = None,
                                  train_sizes: List[float] = None) -> Dict[str, Any]:
    """
    计算学习曲线数据

    Args:
        model: 模型
        X: 特征数组
        y: 目标数组
        cv: 交叉验证折数
        train_sizes: 训练集大小比例列表

    Returns:
        学习曲线数据字典

    TODO 1: 设置默认参数
    # logger = logging.getLogger("Ecommerce_Rating")
    # if cv is None:
    #     cv = config.CV_FOLDS
    # if train_sizes is None:
    #     train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    TODO 2: 计算学习曲线
    # logger.info("计算学习曲线...")
    # with Timer("学习曲线计算"):
    #     train_sizes_abs, train_scores, test_scores = learning_curve(
    #         model, X, y,
    #         cv=cv,
    #         train_sizes=train_sizes,
    #         scoring='r2',  # 或根据任务类型选择
    #         n_jobs=-1,
    #         shuffle=True,
    #         random_state=config.RANDOM_STATE
    #     )

    TODO 3: 计算均值和标准差
    #     train_scores_mean = train_scores.mean(axis=1)
    #     train_scores_std = train_scores.std(axis=1)
    #     test_scores_mean = test_scores.mean(axis=1)
    #     test_scores_std = test_scores.std(axis=1)

    TODO 4: 返回结果
    #     return {
    #         'train_sizes': train_sizes_abs,
    #         'train_scores_mean': train_scores_mean,
    #         'train_scores_std': train_scores_std,
    #         'test_scores_mean': test_scores_mean,
    #         'test_scores_std': test_scores_std
    #     }
    """
    # TODO: 实现学习曲线计算
    pass


def analyze_prediction_errors(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              percentiles: List[int] = None) -> Dict[str, Any]:
    """
    分析预测误差（用于回归）

    Args:
        y_true: 真实值
        y_pred: 预测值
        percentiles: 要计算的百分位数列表

    Returns:
        误差分析结果字典

    TODO 1: 设置默认百分位数
    # if percentiles is None:
    #     percentiles = [10, 25, 50, 75, 90, 95, 99]

    TODO 2: 计算误差
    # errors = y_true - y_pred
    # abs_errors = np.abs(errors)

    TODO 3: 计算误差统计
    # error_stats = {
    #     'mean_error': errors.mean(),
    #     'std_error': errors.std(),
    #     'mean_abs_error': abs_errors.mean(),
    #     'median_abs_error': np.median(abs_errors),
    #     'max_error': abs_errors.max(),
    #     'min_error': abs_errors.min()
    # }

    TODO 4: 计算误差百分位数
    # for p in percentiles:
    #     error_stats[f'abs_error_p{p}'] = np.percentile(abs_errors, p)

    TODO 5: 找出最大误差的样本索引
    # worst_predictions_idx = np.argsort(abs_errors)[-10:]  # 前10个最差预测
    # error_stats['worst_predictions_idx'] = worst_predictions_idx.tolist()

    TODO 6: 返回结果
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("\n预测误差分析:")
    # for key, value in error_stats.items():
    #     if 'idx' not in key:
    #         logger.info(f"  {key}: {value:.4f}")
    #
    # return error_stats
    """
    # TODO: 实现误差分析
    pass


def get_feature_importance(model, feature_names: List[str],
                          top_n: int = 20) -> pd.DataFrame:
    """
    获取特征重要性

    Args:
        model: 训练好的模型（需要有feature_importances_属性）
        feature_names: 特征名称列表
        top_n: 返回前N个最重要的特征

    Returns:
        特征重要性DataFrame

    TODO 1: 检查模型是否有特征重要性属性
    # if not hasattr(model, 'feature_importances_'):
    #     logger = logging.getLogger("Ecommerce_Rating")
    #     logger.warning("模型没有特征重要性属性")
    #     return None

    TODO 2: 获取特征重要性
    # importances = model.feature_importances_

    TODO 3: 创建DataFrame并排序
    # importance_df = pd.DataFrame({
    #     'feature': feature_names,
    #     'importance': importances
    # }).sort_values('importance', ascending=False)

    TODO 4: 选择前N个特征
    # if top_n is not None:
    #     importance_df = importance_df.head(top_n)

    TODO 5: 打印并返回
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info(f"\n特征重要性（前{top_n}个）:")
    # for idx, row in importance_df.iterrows():
    #     logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    #
    # return importance_df
    """
    # TODO: 实现特征重要性获取
    pass


def comprehensive_model_evaluation(model, X_train: np.ndarray, X_test: np.ndarray,
                                  y_train: np.ndarray, y_test: np.ndarray,
                                  feature_names: List[str] = None,
                                  task: str = 'regression') -> Dict[str, Any]:
    """
    综合模型评估（包含多种评估方法）

    Args:
        model: 训练好的模型
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练目标
        y_test: 测试目标
        feature_names: 特征名称列表
        task: 任务类型 ('regression' 或 'classification')

    Returns:
        综合评估结果字典

    TODO 1: 初始化结果字典
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("=" * 60)
    # logger.info("开始综合模型评估")
    # logger.info("=" * 60)
    #
    # results = {}

    TODO 2: 执行交叉验证
    # if task == 'regression':
    #     cv_results = evaluate_regression_cv(model, X_train, y_train)
    # else:
    #     cv_results = evaluate_classification_cv(model, X_train, y_train)
    # results['cross_validation'] = cv_results

    TODO 3: 如果是分类任务，计算混淆矩阵和分类报告
    # if task == 'classification':
    #     y_pred = model.predict(X_test)
    #     results['confusion_matrix'] = get_confusion_matrix_stats(y_test, y_pred)
    #     results['classification_report'] = get_classification_report_dict(y_test, y_pred)
    #
    #     # 如果模型支持概率预测，计算ROC和PR曲线
    #     if hasattr(model, 'predict_proba'):
    #         y_pred_proba = model.predict_proba(X_test)[:, 1]
    #         results['roc_curve'] = calculate_roc_curve_data(y_test, y_pred_proba)
    #         results['pr_curve'] = calculate_precision_recall_curve_data(y_test, y_pred_proba)

    TODO 4: 如果是回归任务，分析预测误差
    # if task == 'regression':
    #     y_pred = model.predict(X_test)
    #     results['error_analysis'] = analyze_prediction_errors(y_test, y_pred)

    TODO 5: 获取特征重要性（如果支持）
    # if feature_names is not None:
    #     importance_df = get_feature_importance(model, feature_names)
    #     if importance_df is not None:
    #         results['feature_importance'] = importance_df

    TODO 6: 计算学习曲线
    # learning_curve_data = calculate_learning_curve_data(model, X_train, y_train)
    # results['learning_curve'] = learning_curve_data

    TODO 7: 返回结果
    # logger.info("=" * 60)
    # logger.info("综合评估完成")
    # logger.info("=" * 60)
    #
    # return results
    """
    # TODO: 实现综合评估
    pass


if __name__ == '__main__':
    # 测试模型评估
    from src.utils import setup_logger
    from src.data_loader import load_raw_data
    from src.data_preprocessing import preprocess_data, split_features_target
    from src.feature_engineering import engineer_features
    from src.model_training import RegressionTrainer, ClassificationTrainer, create_classification_target

    # TODO: 设置日志
    # logger = setup_logger("Ecommerce_Rating", config.LOG_DIR / "model_evaluation_test.log", "INFO")

    print("=" * 60)
    print("模型评估模块测试")
    print("=" * 60)

    # TODO 1: 准备数据并训练模型
    # try:
    #     df = load_raw_data(use_sample=True, sample_size=300)
    #     df_clean = preprocess_data(df)
    #     df_feat, _ = engineer_features(df_clean, include_text_features=False)
    #     X, y = split_features_target(df_feat, target_col='rating')
    #
    #     # 训练回归模型
    #     reg_trainer = RegressionTrainer()
    #     reg_trainer.prepare_data(X, y)
    #     reg_trainer.train_model('ridge', tune_hyperparameters=False)
    #     best_model = reg_trainer.models['ridge']['model']

    # TODO 2: 测试综合评估
    #     print("\n测试综合模型评估:")
    #     eval_results = comprehensive_model_evaluation(
    #         best_model,
    #         reg_trainer.X_train_scaled,
    #         reg_trainer.X_test_scaled,
    #         reg_trainer.y_train,
    #         reg_trainer.y_test,
    #         feature_names=list(X.columns),
    #         task='regression'
    #     )

    # TODO 3: 测试分类评估
    #     print("\n测试分类模型评估:")
    #     y_binary = create_classification_target(y)
    #     clf_trainer = ClassificationTrainer()
    #     clf_trainer.prepare_data(X, y_binary)
    #     clf_trainer.train_model('logistic', tune_hyperparameters=False)
    #     clf_model = clf_trainer.models['logistic']['model']
    #
    #     clf_eval_results = comprehensive_model_evaluation(
    #         clf_model,
    #         clf_trainer.X_train_scaled,
    #         clf_trainer.X_test_scaled,
    #         clf_trainer.y_train,
    #         clf_trainer.y_test,
    #         feature_names=list(X.columns),
    #         task='classification'
    #     )

    # except Exception as e:
    #     print(f"\n错误: {str(e)}")

    print("\n提示：实现上述TODO后运行此文件进行测试")
