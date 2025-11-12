"""
模型评估模块
提供全面的模型评估、对比和报告生成功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, learning_curve

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, save_json
from src.visualization import (
    plot_predictions, plot_residuals, plot_feature_importance,
    plot_learning_curve, plot_model_comparison
)


class ModelEvaluator:
    """
    模型评估器类
    提供全面的模型评估功能
    """

    def __init__(self):
        """初始化评估器"""
        self.logger = logging.getLogger("NYC_Taxi")
        self.evaluation_results = {}

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model") -> Dict[str, float]:
        """
        计算评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称

        Returns:
            指标字典
        """
        self.logger.info(f"计算 {model_name} 的评估指标...")

        # R² Score
        r2 = r2_score(y_true, y_pred)

        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAE
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        # 避免除零错误
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        # MSE
        mse = mean_squared_error(y_true, y_pred)

        # 相对误差
        relative_error = np.mean(np.abs(y_true - y_pred) / y_true) * 100

        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'mse': mse,
            'relative_error': relative_error
        }

        self.logger.info(f"  R² Score: {r2:.4f}")
        self.logger.info(f"  RMSE: {rmse:.2f} 秒 ({rmse/60:.2f} 分钟)")
        self.logger.info(f"  MAE: {mae:.2f} 秒 ({mae/60:.2f} 分钟)")
        self.logger.info(f"  MAPE: {mape:.2f}%")

        return metrics

    def residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model",
                         save_plot: bool = True) -> Dict[str, Any]:
        """
        残差分析

        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            save_plot: 是否保存图表

        Returns:
            残差分析结果字典
        """
        self.logger.info(f"执行 {model_name} 的残差分析...")

        residuals = y_true - y_pred

        # 残差统计
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75)
        }

        # 检查正态性 (Shapiro-Wilk test)
        from scipy import stats
        if len(residuals) <= 5000:
            statistic, p_value = stats.shapiro(residuals[:5000])
            is_normal = p_value > 0.05
            residual_stats['normality_test'] = {
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': is_normal
            }

        # 异常残差 (超过3个标准差)
        threshold = 3 * residual_stats['std']
        outliers = np.abs(residuals) > threshold
        outlier_count = np.sum(outliers)
        outlier_ratio = outlier_count / len(residuals) * 100

        residual_stats['outlier_count'] = int(outlier_count)
        residual_stats['outlier_ratio'] = outlier_ratio

        self.logger.info(f"  残差均值: {residual_stats['mean']:.2f}")
        self.logger.info(f"  残差标准差: {residual_stats['std']:.2f}")
        self.logger.info(f"  异常残差数量: {outlier_count} ({outlier_ratio:.2f}%)")

        # 绘制残差图
        if save_plot:
            plot_residuals(
                y_true, y_pred,
                title=f"{model_name}_残差分析",
                save_path=config.FIGURES_DIR / f"{model_name}_residuals.png"
            )

        return residual_stats

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                      cv: int = None, scoring: str = 'r2') -> Dict[str, Any]:
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

        self.logger.info(f"执行 {cv} 折交叉验证...")

        with Timer("交叉验证"):
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        cv_results = {
            'scores': scores.tolist(),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }

        self.logger.info(f"  交叉验证得分: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")

        return cv_results

    def learning_curve_analysis(self, model, X: np.ndarray, y: np.ndarray,
                                train_sizes: Optional[List[float]] = None,
                                cv: int = None,
                                save_plot: bool = True,
                                model_name: str = "Model") -> Dict[str, Any]:
        """
        学习曲线分析

        Args:
            model: 模型对象
            X: 特征数据
            y: 目标数据
            train_sizes: 训练集大小比例列表
            cv: 交叉验证折数
            save_plot: 是否保存图表
            model_name: 模型名称

        Returns:
            学习曲线结果字典
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        if cv is None:
            cv = config.CV_FOLDS

        self.logger.info(f"生成 {model_name} 的学习曲线...")

        with Timer("学习曲线分析"):
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=cv,
                scoring='r2',
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

        # 判断是否过拟合/欠拟合
        final_gap = train_scores_mean[-1] - val_scores_mean[-1]
        if final_gap > 0.1:
            self.logger.warning(f"  可能存在过拟合 (训练-验证差距: {final_gap:.4f})")
        elif val_scores_mean[-1] < 0.7:
            self.logger.warning(f"  可能存在欠拟合 (验证得分: {val_scores_mean[-1]:.4f})")

        # 绘制学习曲线
        if save_plot:
            plot_learning_curve(
                train_scores_mean.tolist(),
                val_scores_mean.tolist(),
                train_sizes_abs.tolist(),
                metric_name="R²",
                title=f"{model_name}_学习曲线",
                save_path=config.FIGURES_DIR / f"{model_name}_learning_curve.png"
            )

        return lc_results

    def feature_importance_analysis(self, model, feature_names: List[str],
                                   model_name: str = "Model",
                                   top_n: int = 20,
                                   save_plot: bool = True) -> pd.DataFrame:
        """
        特征重要性分析

        Args:
            model: 模型对象
            feature_names: 特征名称列表
            model_name: 模型名称
            top_n: 显示前N个重要特征
            save_plot: 是否保存图表

        Returns:
            特征重要性DataFrame
        """
        self.logger.info(f"分析 {model_name} 的特征重要性...")

        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            self.logger.warning(f"{model_name} 不支持特征重要性分析")
            return None

        # 创建DataFrame
        importance_df = pd.DataFrame({
            '特征': feature_names,
            '重要性': importances
        }).sort_values('重要性', ascending=False).reset_index(drop=True)

        # 显示top特征
        self.logger.info(f"\nTop {min(top_n, len(importance_df))} 重要特征:")
        for idx, row in importance_df.head(top_n).iterrows():
            self.logger.info(f"  {idx+1}. {row['特征']}: {row['重要性']:.4f}")

        # 绘制特征重要性图
        if save_plot:
            plot_feature_importance(
                feature_names,
                importances,
                title=f"{model_name}_特征重要性",
                top_n=top_n,
                save_path=config.FIGURES_DIR / f"{model_name}_feature_importance.png"
            )

        return importance_df

    def compare_models(self, models_results: Dict[str, Dict],
                      save_plot: bool = True) -> pd.DataFrame:
        """
        对比多个模型

        Args:
            models_results: 模型结果字典
            save_plot: 是否保存图表

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
                'R²': metrics.get('r2', 0),
                'RMSE': metrics.get('rmse', 0),
                'MAE': metrics.get('mae', 0),
                'MAPE': metrics.get('mape', 0)
            })

        df = pd.DataFrame(comparison)
        df = df.sort_values('R²', ascending=False).reset_index(drop=True)

        # 打印对比表格
        self.logger.info("\n" + str(df.to_string(index=False)))

        # 绘制对比图
        if save_plot:
            # 转换为字典格式用于绘图
            plot_data = {}
            for _, row in df.iterrows():
                plot_data[row['模型']] = {
                    'r2': row['R²'],
                    'rmse': row['RMSE'],
                    'mae': row['MAE'],
                    'mape': row['MAPE']
                }

            from src.visualization import plot_model_comparison
            plot_model_comparison(
                plot_data,
                metric='rmse',
                save_name='model_comparison.png'
            )

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

            # 2. 评估指标
            self.logger.info("\n训练集评估:")
            train_metrics = self.calculate_metrics(y_train, y_pred_train, f"{model_name} (训练集)")

            self.logger.info("\n测试集评估:")
            test_metrics = self.calculate_metrics(y_test, y_pred_test, f"{model_name} (测试集)")

            report['train_metrics'] = train_metrics
            report['test_metrics'] = test_metrics

            # 3. 残差分析
            self.logger.info("\n残差分析:")
            residual_stats = self.residual_analysis(
                y_test, y_pred_test,
                model_name=model_name,
                save_plot=True
            )
            report['residual_stats'] = residual_stats

            # 4. 绘制预测对比图
            plot_predictions(
                y_test, y_pred_test,
                title=f"{model_name}_预测对比",
                save_path=save_dir / f"{model_name}_predictions.png"
            )

            # 5. 特征重要性
            importance_df = self.feature_importance_analysis(
                model, feature_names,
                model_name=model_name,
                save_plot=True
            )
            if importance_df is not None:
                report['feature_importance'] = importance_df.head(20).to_dict('records')

            # 6. 学习曲线 (可选，比较耗时)
            # lc_results = self.learning_curve_analysis(
            #     model, X_train, y_train,
            #     model_name=model_name,
            #     save_plot=True
            # )
            # report['learning_curve'] = lc_results

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

        comparison_df = self.compare_models(models_dict, save_plot=True)

        # 保存对比结果
        comparison_path = config.FIGURES_DIR / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"\n对比结果已保存: {comparison_path}")

        return all_results


if __name__ == '__main__':
    # 测试评估模块
    from src.utils import setup_logger
    from src.data_loader import load_train_data
    from src.data_preprocessing import preprocess_data, split_features_target
    from src.feature_engineering import engineer_features, select_features
    from src.model_training import ModelTrainer

    logger = setup_logger("NYC_Taxi", config.LOG_FILE, "INFO")

    print("=" * 60)
    print("模型评估模块测试")
    print("=" * 60)

    # 准备数据
    print("\n1. 准备数据")
    df = load_train_data(use_sample=True, sample_size=1000)
    df_clean = preprocess_data(df)
    df_feat = engineer_features(df_clean)
    X_selected = select_features(df_feat)
    y = df_feat['trip_duration']

    # 训练模型
    print("\n2. 训练模型")
    trainer = ModelTrainer()
    trainer.prepare_data(X_selected, y)
    models = trainer.train_all_models(include_xgboost=False, include_stacking=False)

    # 评估模型
    print("\n3. 评估模型")
    evaluator = ModelEvaluator()

    # 评估所有模型
    results = evaluator.evaluate_all_models(
        models,
        trainer.X_train_scaled,
        trainer.X_test_scaled,
        trainer.y_train,
        trainer.y_test,
        list(X_selected.columns)
    )

    print("\n模型评估模块测试完成！")
    print(f"评估结果已保存到: {config.FIGURES_DIR}")
