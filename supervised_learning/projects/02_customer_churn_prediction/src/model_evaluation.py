"""
模型评估模块 - 客户流失预测
提供全面的分类模型评估、对比和报告生成功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, save_json


class ModelEvaluator:
    """
    模型评估器类
    提供全面的分类模型评估功能
    """

    def __init__(self):
        """初始化评估器"""
        self.logger = logging.getLogger("ChurnPrediction")
        self.evaluation_results = {}

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: np.ndarray,
                         model_name: str = "Model") -> Dict[str, float]:
        """
        计算全面的分类评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率
            model_name: 模型名称

        Returns:
            指标字典
        """
        self.logger.info(f"计算 {model_name} 的评估指标...")

        metrics = {
            # 基础指标
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),

            # ROC-AUC
            'roc_auc': roc_auc_score(y_true, y_pred_proba),

            # PR-AUC
            'pr_auc': average_precision_score(y_true, y_pred_proba),

            # 特异性（Specificity）
            'specificity': self._calculate_specificity(y_true, y_pred),
        }

        # 混淆矩阵元素
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positive'] = int(tp)
        metrics['false_positive'] = int(fp)
        metrics['true_negative'] = int(tn)
        metrics['false_negative'] = int(fn)

        # 假阳性率和假阴性率
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        self.logger.info(f"\n{model_name} 评估结果:")
        self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        self.logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        self.logger.info(f"  PR AUC:    {metrics['pr_auc']:.4f}")
        self.logger.info(f"  Specificity: {metrics['specificity']:.4f}")

        return metrics

    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算特异性（真阴性率）

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            特异性值
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return specificity

    def confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str = "Model") -> Dict[str, Any]:
        """
        混淆矩阵分析

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称

        Returns:
            混淆矩阵分析结果
        """
        self.logger.info(f"\n{model_name} 混淆矩阵分析...")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        total = len(y_true)
        positive = np.sum(y_true == 1)
        negative = np.sum(y_true == 0)

        analysis = {
            'confusion_matrix': cm.tolist(),
            'true_positive': int(tp),
            'false_positive': int(fp),
            'true_negative': int(tn),
            'false_negative': int(fn),
            'total_samples': int(total),
            'positive_samples': int(positive),
            'negative_samples': int(negative),
            'tp_rate': tp / positive if positive > 0 else 0,
            'fp_rate': fp / negative if negative > 0 else 0,
            'tn_rate': tn / negative if negative > 0 else 0,
            'fn_rate': fn / positive if positive > 0 else 0
        }

        self.logger.info(f"\n混淆矩阵:")
        self.logger.info(f"              预测负类    预测正类")
        self.logger.info(f"实际负类:      {tn:6d}      {fp:6d}")
        self.logger.info(f"实际正类:      {fn:6d}      {tp:6d}")

        self.logger.info(f"\n详细分析:")
        self.logger.info(f"  真阳性 (TP): {tp} ({analysis['tp_rate']:.2%})")
        self.logger.info(f"  假阳性 (FP): {fp} ({analysis['fp_rate']:.2%})")
        self.logger.info(f"  真阴性 (TN): {tn} ({analysis['tn_rate']:.2%})")
        self.logger.info(f"  假阴性 (FN): {fn} ({analysis['fn_rate']:.2%})")

        return analysis

    def classification_report_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      model_name: str = "Model") -> Dict[str, Any]:
        """
        分类报告分析

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称

        Returns:
            分类报告字典
        """
        self.logger.info(f"\n{model_name} 分类报告...")

        # 获取分类报告
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        self.logger.info(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

        return report

    def roc_curve_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          model_name: str = "Model") -> Dict[str, Any]:
        """
        ROC曲线分析

        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            model_name: 模型名称

        Returns:
            ROC曲线数据
        """
        self.logger.info(f"\n{model_name} ROC曲线分析...")

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        # 找到最优阈值（Youden's J statistic）
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc),
            'optimal_threshold': float(optimal_threshold),
            'optimal_tpr': float(tpr[optimal_idx]),
            'optimal_fpr': float(fpr[optimal_idx])
        }

        self.logger.info(f"  ROC AUC: {roc_auc:.4f}")
        self.logger.info(f"  最优阈值: {optimal_threshold:.4f}")
        self.logger.info(f"  最优点 TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")

        return roc_data

    def precision_recall_curve_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                       model_name: str = "Model") -> Dict[str, Any]:
        """
        精确率-召回率曲线分析

        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            model_name: 模型名称

        Returns:
            PR曲线数据
        """
        self.logger.info(f"\n{model_name} PR曲线分析...")

        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)

        # 找到最优阈值（F1最大）
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5

        pr_data = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': np.append(thresholds, 0).tolist(),  # 补齐长度
            'auc': float(pr_auc),
            'optimal_threshold': float(optimal_threshold),
            'optimal_precision': float(precision[optimal_idx]),
            'optimal_recall': float(recall[optimal_idx]),
            'optimal_f1': float(f1_scores[optimal_idx])
        }

        self.logger.info(f"  PR AUC: {pr_auc:.4f}")
        self.logger.info(f"  最优阈值: {optimal_threshold:.4f}")
        self.logger.info(f"  最优点 Precision: {precision[optimal_idx]:.4f}, Recall: {recall[optimal_idx]:.4f}")
        self.logger.info(f"  最优 F1: {f1_scores[optimal_idx]:.4f}")

        return pr_data

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                      cv: int = None,
                      scoring: str = None) -> Dict[str, Any]:
        """
        交叉验证评估

        Args:
            model: 模型对象
            X: 特征数据
            y: 目标数据
            cv: 交叉验证折数
            scoring: 评分指标

        Returns:
            交叉验证结果字典
        """
        if cv is None:
            cv = config.CV_FOLDS

        if scoring is None:
            scoring = config.PRIMARY_METRIC

        self.logger.info(f"\n执行 {cv} 折交叉验证 (评分指标: {scoring})...")

        with Timer("交叉验证"):
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        cv_results = {
            'scores': scores.tolist(),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }

        self.logger.info(f"  交叉验证得分: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
        self.logger.info(f"  分数范围: [{cv_results['min']:.4f}, {cv_results['max']:.4f}]")

        return cv_results

    def learning_curve_analysis(self, model, X: np.ndarray, y: np.ndarray,
                                train_sizes: Optional[List[float]] = None,
                                cv: int = None,
                                scoring: str = None,
                                model_name: str = "Model") -> Dict[str, Any]:
        """
        学习曲线分析

        Args:
            model: 模型对象
            X: 特征数据
            y: 目标数据
            train_sizes: 训练集大小比例列表
            cv: 交叉验证折数
            scoring: 评分指标
            model_name: 模型名称

        Returns:
            学习曲线结果字典
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        if cv is None:
            cv = config.CV_FOLDS

        if scoring is None:
            scoring = config.PRIMARY_METRIC

        self.logger.info(f"\n生成 {model_name} 的学习曲线...")

        with Timer("学习曲线分析"):
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=config.RANDOM_STATE
            )

        # 计算均值和标准差
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        lc_results = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores_mean.tolist(),
            'train_scores_std': train_scores_std.tolist(),
            'val_scores_mean': val_scores_mean.tolist(),
            'val_scores_std': val_scores_std.tolist()
        }

        # 判断过拟合/欠拟合
        final_gap = train_scores_mean[-1] - val_scores_mean[-1]
        if final_gap > 0.1:
            self.logger.warning(f"  可能存在过拟合 (训练-验证差距: {final_gap:.4f})")
        elif val_scores_mean[-1] < 0.7:
            self.logger.warning(f"  可能存在欠拟合 (验证得分: {val_scores_mean[-1]:.4f})")
        else:
            self.logger.info(f"  模型拟合良好 (训练-验证差距: {final_gap:.4f})")

        return lc_results

    def feature_importance_analysis(self, model, feature_names: List[str],
                                   model_name: str = "Model",
                                   top_n: int = 20) -> pd.DataFrame:
        """
        特征重要性分析

        Args:
            model: 模型对象
            feature_names: 特征名称列表
            model_name: 模型名称
            top_n: 显示前N个重要特征

        Returns:
            特征重要性DataFrame
        """
        self.logger.info(f"\n分析 {model_name} 的特征重要性...")

        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 对于逻辑回归，使用系数的绝对值
            importances = np.abs(model.coef_[0])
        else:
            self.logger.warning(f"{model_name} 不支持特征重要性分析")
            return None

        # 创建DataFrame
        importance_df = pd.DataFrame({
            '特征': feature_names,
            '重要性': importances
        }).sort_values('重要性', ascending=False).reset_index(drop=True)

        # 计算累积重要性
        importance_df['累积重要性'] = importance_df['重要性'].cumsum() / importance_df['重要性'].sum()

        # 显示top特征
        self.logger.info(f"\nTop {min(top_n, len(importance_df))} 重要特征:")
        for idx, row in importance_df.head(top_n).iterrows():
            self.logger.info(f"  {idx+1}. {row['特征']}: {row['重要性']:.4f} (累积: {row['累积重要性']:.2%})")

        return importance_df

    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        对比多个模型

        Args:
            models_results: 模型结果字典

        Returns:
            对比结果DataFrame
        """
        self.logger.info("=" * 60)
        self.logger.info("模型性能对比")
        self.logger.info("=" * 60)

        comparison = []
        for name, result in models_results.items():
            metrics = result.get('metrics', {})
            comparison.append({
                '模型': name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1', 0),
                'ROC_AUC': metrics.get('roc_auc', 0),
                'PR_AUC': metrics.get('pr_auc', 0)
            })

        df = pd.DataFrame(comparison)
        df = df.sort_values('ROC_AUC', ascending=False).reset_index(drop=True)

        # 打印对比表格
        self.logger.info("\n" + str(df.to_string(index=False)))

        # 标记达到目标的指标
        self.logger.info("\n目标达成情况:")
        for idx, row in df.iterrows():
            self.logger.info(f"\n{row['模型']}:")
            for metric, target in config.TARGET_METRICS.items():
                actual = row[metric.capitalize()] if metric != 'roc_auc' else row['ROC_AUC']
                status = "✓" if actual >= target else "✗"
                self.logger.info(f"  {metric.upper()}: {actual:.4f} (目标: {target:.4f}) {status}")

        return df

    def generate_evaluation_report(self, model, X_train, X_test, y_train, y_test,
                                  feature_names: List[str],
                                  model_name: str = "Model",
                                  save_dir: Path = None) -> Dict[str, Any]:
        """
        生成完整的评估报告

        Args:
            model: 模型对象
            X_train: 训练特征
            X_test: 测试特征
            y_train: 训练目标
            y_test: 测试目标
            feature_names: 特征名称列表
            model_name: 模型名称
            save_dir: 保存目录

        Returns:
            完整的评估报告字典
        """
        self.logger.info("=" * 60)
        self.logger.info(f"生成 {model_name} 评估报告")
        self.logger.info("=" * 60)

        if save_dir is None:
            save_dir = config.FIGURES_DIR / "evaluation"
        save_dir.mkdir(parents=True, exist_ok=True)

        report = {}

        with Timer("评估报告生成"):
            # 1. 基本预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_proba_train = model.predict_proba(X_train)[:, 1]
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]

            # 2. 评估指标
            self.logger.info("\n===== 训练集评估 =====")
            train_metrics = self.calculate_metrics(
                y_train, y_pred_train, y_pred_proba_train,
                f"{model_name} (训练集)"
            )

            self.logger.info("\n===== 测试集评估 =====")
            test_metrics = self.calculate_metrics(
                y_test, y_pred_test, y_pred_proba_test,
                f"{model_name} (测试集)"
            )

            report['train_metrics'] = train_metrics
            report['test_metrics'] = test_metrics

            # 3. 混淆矩阵分析
            cm_analysis = self.confusion_matrix_analysis(
                y_test, y_pred_test, model_name
            )
            report['confusion_matrix'] = cm_analysis

            # 4. 分类报告
            class_report = self.classification_report_analysis(
                y_test, y_pred_test, model_name
            )
            report['classification_report'] = class_report

            # 5. ROC曲线分析
            roc_data = self.roc_curve_analysis(
                y_test, y_pred_proba_test, model_name
            )
            report['roc_curve'] = roc_data

            # 6. PR曲线分析
            pr_data = self.precision_recall_curve_analysis(
                y_test, y_pred_proba_test, model_name
            )
            report['pr_curve'] = pr_data

            # 7. 特征重要性
            importance_df = self.feature_importance_analysis(
                model, feature_names, model_name, top_n=20
            )
            if importance_df is not None:
                report['feature_importance'] = importance_df.head(20).to_dict('records')

            # 8. 交叉验证（可选，较耗时）
            # cv_results = self.cross_validate(model, X_train, y_train)
            # report['cross_validation'] = cv_results

        # 保存报告
        report_path = save_dir / f"{model_name}_evaluation_report.json"
        save_json(report, report_path)

        self.logger.info(f"\n评估报告已保存: {report_path}")

        return report

    def evaluate_all_models(self, models_dict: Dict[str, Dict],
                          X_train, X_test, y_train, y_test,
                          feature_names: List[str]) -> Dict[str, Any]:
        """
        评估所有模型并生成对比报告

        Args:
            models_dict: 模型字典
            X_train: 训练特征
            X_test: 测试特征
            y_train: 训练目标
            y_test: 测试目标
            feature_names: 特征名称列表

        Returns:
            所有模型的评估结果
        """
        self.logger.info("=" * 60)
        self.logger.info("评估所有模型")
        self.logger.info("=" * 60)

        all_results = {}

        for model_name, model_info in models_dict.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"评估模型: {model_name}")
            self.logger.info(f"{'='*60}")

            model = model_info['model']

            # 生成单个模型的评估报告
            report = self.generate_evaluation_report(
                model, X_train, X_test, y_train, y_test,
                feature_names, model_name
            )

            all_results[model_name] = report

        # 生成对比报告
        self.logger.info("\n" + "=" * 60)
        self.logger.info("所有模型对比")
        self.logger.info("=" * 60)

        comparison_df = self.compare_models(models_dict)

        # 保存对比结果
        comparison_path = config.FIGURES_DIR / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"\n对比结果已保存: {comparison_path}")

        return all_results

    def plot_all_models_comparison(self, models_dict: Dict[str, Dict],
                                   X_test, y_test,
                                   save_dir: Path = None) -> None:
        """
        生成所有模型的可视化对比图表

        Args:
            models_dict: 模型字典
            X_test: 测试特征
            y_test: 测试目标
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = config.FIGURES_DIR / "evaluation"

        from src.utils import ensure_dir
        ensure_dir(save_dir)

        self.logger.info("=" * 60)
        self.logger.info("生成模型对比可视化")
        self.logger.info("=" * 60)

        # 导入可视化模块
        from src.visualization import (
            plot_confusion_matrix,
            plot_roc_curve,
            plot_precision_recall_curve,
            plot_model_comparison
        )

        # 1. 为每个模型生成混淆矩阵、ROC曲线和PR曲线
        for model_name, model_info in models_dict.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # 混淆矩阵
            plot_confusion_matrix(
                y_test, y_pred,
                title=f"{model_name} - 混淆矩阵",
                save_path=save_dir / f"{model_name}_confusion_matrix.png"
            )

            # ROC曲线
            plot_roc_curve(
                y_test, y_pred_proba,
                title=f"{model_name} - ROC曲线",
                save_path=save_dir / f"{model_name}_roc_curve.png"
            )

            # PR曲线
            plot_precision_recall_curve(
                y_test, y_pred_proba,
                title=f"{model_name} - Precision-Recall曲线",
                save_path=save_dir / f"{model_name}_pr_curve.png"
            )

        # 2. 生成模型对比图
        plot_model_comparison(
            models_dict,
            title="模型性能对比",
            save_path=save_dir / "models_comparison.png"
        )

        self.logger.info(f"\n所有模型对比图表已保存到: {save_dir}")


if __name__ == '__main__':
    # 测试评估模块
    from src.utils import setup_logger
    from src.data_loader import load_data
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import FeatureEngineer
    from src.model_training import ModelTrainer

    logger = setup_logger("Churn_Prediction", config.LOG_FILE, "INFO")

    print("=" * 60)
    print("模型评估模块测试")
    print("=" * 60)

    # 准备数据
    print("\n1. 准备数据")
    df = load_data()
    df_clean = preprocess_data(df)

    engineer = FeatureEngineer()
    df_feat = engineer.engineer_features(df_clean)

    X = df_feat.drop(columns=[config.TARGET_COL, config.ID_COL], errors='ignore')
    y = df_feat[config.TARGET_COL]

    # 训练模型
    print("\n2. 训练模型")
    trainer = ModelTrainer()
    trainer.prepare_data(X, y, use_smote=True)

    # 只训练几个快速模型用于测试
    trainer.train_logistic_regression()
    trainer.train_decision_tree()

    # 评估模型
    print("\n3. 评估模型")
    evaluator = ModelEvaluator()

    # 评估所有模型
    results = evaluator.evaluate_all_models(
        trainer.models,
        trainer.X_train_resampled,
        trainer.X_test,
        trainer.y_train_resampled,
        trainer.y_test,
        list(X.columns)
    )

    print("\n模型评估模块测试完成！")
    print(f"评估结果已保存到: {config.FIGURES_DIR}")
