"""
可视化模块 - 客户流失预测
提供数据分析和模型评估的可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import logging
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, ensure_dir


# 设置matplotlib支持中文

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_class_distribution(y: pd.Series,
                           title: str = "类别分布",
                           save_path: Optional[Path] = None,
                           figsize: Tuple[int, int] = None) -> None:
    """
    绘制类别分布图

    Args:
        y: 目标变量Series
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info(f"绘制类别分布图: {title}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

        # 统计类别分布
        counts = y.value_counts()
        percentages = y.value_counts(normalize=True) * 100

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 柱状图
        colors = ['#2ecc71', '#e74c3c']
        ax1.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('类别', fontsize=12)
        ax1.set_ylabel('数量', fontsize=12)
        ax1.set_title('类别数量分布', fontsize=12)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['未流失 (0)', '流失 (1)'])
        ax1.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, v in enumerate(counts.values):
            ax1.text(i, v + max(counts) * 0.02, str(v),
                    ha='center', va='bottom', fontweight='bold')

        # 饼图
        ax2.pie(counts.values, labels=['未流失 (0)', '流失 (1)'],
               colors=colors, autopct='%1.1f%%', startangle=90,
               explode=(0, 0.1), shadow=True)
        ax2.set_title('类别比例分布', fontsize=12)

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制类别分布图时出错: {str(e)}")
        raise


def plot_feature_distribution(data: pd.DataFrame,
                             features: List[str],
                             target_col: str = None,
                             title: str = "特征分布",
                             save_path: Optional[Path] = None,
                             figsize: Tuple[int, int] = None) -> None:
    """
    绘制特征分布图（按目标变量分组）

    Args:
        data: 输入数据
        features: 要绘制的特征列表
        target_col: 目标变量列名
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info(f"绘制特征分布图: {title}")

    try:
        if figsize is None:
            n_features = len(features)
            figsize = (15, 5 * ((n_features + 2) // 3))

        # 计算子图布局
        n_cols = 3
        n_rows = (len(features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 展平axes数组
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, feature in enumerate(features):
            if feature not in data.columns:
                logger.warning(f"特征 '{feature}' 不存在，跳过")
                continue

            ax = axes[idx]

            # 根据特征类型选择绘图方式
            if target_col and target_col in data.columns:
                # 按目标变量分组
                if data[feature].dtype in ['float64', 'int64'] and data[feature].nunique() > 10:
                    # 数值型特征：直方图
                    data[data[target_col] == 0][feature].hist(
                        bins=30, alpha=0.6, label='未流失', color='#2ecc71', ax=ax
                    )
                    data[data[target_col] == 1][feature].hist(
                        bins=30, alpha=0.6, label='流失', color='#e74c3c', ax=ax
                    )
                    ax.set_xlabel(feature)
                    ax.set_ylabel('频数')
                    ax.legend()
                else:
                    # 分类型特征：柱状图
                    crosstab = pd.crosstab(data[feature], data[target_col], normalize='index') * 100
                    crosstab.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
                    ax.set_xlabel(feature)
                    ax.set_ylabel('百分比 (%)')
                    ax.legend(['未流失', '流失'])
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                # 不分组：单独分布
                if data[feature].dtype in ['float64', 'int64'] and data[feature].nunique() > 10:
                    data[feature].hist(bins=30, alpha=0.7, color='skyblue', ax=ax, edgecolor='black')
                else:
                    data[feature].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            ax.set_title(feature, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制特征分布图时出错: {str(e)}")
        raise


def plot_correlation_heatmap(data: pd.DataFrame,
                            features: Optional[List[str]] = None,
                            title: str = "特征相关性热力图",
                            save_path: Optional[Path] = None,
                            figsize: Tuple[int, int] = None,
                            annot: bool = False) -> None:
    """
    绘制相关性热力图

    Args:
        data: 输入数据
        features: 要分析的特征列表（None表示所有数值列）
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
        annot: 是否显示相关系数数值
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info(f"绘制相关性热力图: {title}")

    try:
        if figsize is None:
            figsize = config.LARGE_FIGURE_SIZE

        # 选择数值列
        if features is None:
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            # 确保只选择数值型列
            numeric_data = data[features].select_dtypes(include=[np.number])

        # 计算相关系数矩阵
        corr_matrix = numeric_data.corr()

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
        sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制相关性热力图时出错: {str(e)}")
        raise


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         title: str = "混淆矩阵",
                         save_path: Optional[Path] = None,
                         figsize: Tuple[int, int] = None) -> None:
    """
    绘制混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info(f"绘制混淆矩阵: {title}")

    try:
        if figsize is None:
            figsize = (10, 8)

        # 确保y_true和y_pred是数值类型
        y_true_numeric = np.asarray(y_true, dtype=int)
        y_pred_numeric = np.asarray(y_pred, dtype=int)

        # 计算混淆矩阵
        cm = confusion_matrix(y_true_numeric, y_pred_numeric)

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   cbar_kws={'label': '数量'},
                   square=True, linewidths=2, linecolor='black',
                   ax=ax)

        ax.set_xlabel('预测标签', fontsize=14, fontweight='bold')
        ax.set_ylabel('真实标签', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # 设置刻度标签
        ax.set_xticklabels(['未流失 (0)', '流失 (1)'], fontsize=12)
        ax.set_yticklabels(['未流失 (0)', '流失 (1)'], fontsize=12, rotation=0)

        # 添加统计信息
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        textstr = f'Accuracy:  {accuracy:.4f}\n'
        textstr += f'Precision: {precision:.4f}\n'
        textstr += f'Recall:    {recall:.4f}\n'
        textstr += f'F1 Score:  {f1:.4f}'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', bbox=props)

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制混淆矩阵时出错: {str(e)}")
        raise


def plot_roc_curve(y_true: np.ndarray,
                  y_pred_proba: np.ndarray,
                  title: str = "ROC曲线",
                  save_path: Optional[Path] = None,
                  figsize: Tuple[int, int] = None) -> None:
    """
    绘制ROC曲线

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info(f"绘制ROC曲线: {title}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

        # 确保y_true是数值类型
        y_true_numeric = np.asarray(y_true, dtype=int)

        # 计算ROC曲线
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thresholds = roc_curve(y_true_numeric, y_pred_proba)
        auc = roc_auc_score(y_true_numeric, y_pred_proba)

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制ROC曲线
        ax.plot(fpr, tpr, color='#e74c3c', linewidth=3,
               label=f'ROC曲线 (AUC = {auc:.4f})')

        # 绘制对角线（随机分类器）
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--',
               linewidth=2, label='随机分类器 (AUC = 0.5)')

        # 标记最优阈值点
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='blue',
                  s=200, marker='o', edgecolors='black', linewidth=2,
                  label=f'最优阈值 = {optimal_threshold:.4f}', zorder=3)

        ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
        ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制ROC曲线时出错: {str(e)}")
        raise


def plot_precision_recall_curve(y_true: np.ndarray,
                                y_pred_proba: np.ndarray,
                                title: str = "精确率-召回率曲线",
                                save_path: Optional[Path] = None,
                                figsize: Tuple[int, int] = None) -> None:
    """
    绘制精确率-召回率曲线

    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info(f"绘制PR曲线: {title}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

        # 确保y_true是数值类型
        y_true_numeric = np.asarray(y_true, dtype=int)

        # 计算PR曲线
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, thresholds = precision_recall_curve(y_true_numeric, y_pred_proba)
        ap = average_precision_score(y_true_numeric, y_pred_proba)

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制PR曲线
        ax.plot(recall, precision, color='#3498db', linewidth=3,
               label=f'PR曲线 (AP = {ap:.4f})')

        # 绘制基线（正类比例）
        baseline = np.sum(y_true_numeric) / len(y_true_numeric)
        ax.axhline(y=baseline, color='gray', linestyle='--',
                  linewidth=2, label=f'基线 (正类比例 = {baseline:.4f})')

        # 标记最优F1点
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        ax.scatter(recall[optimal_idx], precision[optimal_idx],
                  color='red', s=200, marker='o',
                  edgecolors='black', linewidth=2,
                  label=f'最优F1 = {f1_scores[optimal_idx]:.4f}', zorder=3)

        ax.set_xlabel('召回率 (Recall)', fontsize=12)
        ax.set_ylabel('精确率 (Precision)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制PR曲线时出错: {str(e)}")
        raise


def plot_feature_importance(feature_names: List[str],
                           importance_values: np.ndarray,
                           title: str = "特征重要性",
                           save_path: Optional[Path] = None,
                           figsize: Tuple[int, int] = None,
                           top_n: int = 20) -> None:
    """
    绘制特征重要性图

    Args:
        feature_names: 特征名称列表
        importance_values: 特征重要性值
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
        top_n: 显示前N个最重要的特征
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info(f"绘制特征重要性图: {title}")

    try:
        if figsize is None:
            figsize = (12, max(8, top_n * 0.3))

        # 创建DataFrame并排序
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)

        # 选择top_n个特征
        if len(importance_df) > top_n:
            importance_df = importance_df.head(top_n)
            logger.info(f"显示前 {top_n} 个最重要的特征")

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制水平条形图
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = ax.barh(range(len(importance_df)), importance_df['importance'],
                      color=colors, edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(val, i, f' {val:.4f}',
                   va='center', ha='left', fontsize=9, fontweight='bold')

        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # 最重要的特征在顶部
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制特征重要性图时出错: {str(e)}")
        raise


def plot_model_comparison(models_results: Dict[str, Dict],
                         metrics: List[str] = None,
                         title: str = "模型性能对比",
                         save_path: Optional[Path] = None,
                         figsize: Tuple[int, int] = None) -> None:
    """
    绘制模型对比图

    Args:
        models_results: 模型结果字典
        metrics: 要对比的指标列表
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info(f"绘制模型对比图: {title}")

    try:
        if figsize is None:
            figsize = (14, 8)

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        # 准备数据
        comparison_data = []
        for model_name, result in models_results.items():
            model_metrics = result.get('metrics', {})
            row = {'模型': model_name}
            for metric in metrics:
                row[metric] = model_metrics.get(metric, 0)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 设置柱状图参数
        x = np.arange(len(df))
        width = 0.15
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))

        # 绘制分组柱状图
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (i - len(metrics) / 2)
            ax.bar(x + offset, df[metric], width,
                  label=metric.upper(), color=color,
                  edgecolor='black', linewidth=1)

        ax.set_xlabel('模型', fontsize=12, fontweight='bold')
        ax.set_ylabel('分数', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['模型'], rotation=45, ha='right')
        ax.legend(loc='lower right', ncol=len(metrics))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])

        # 添加目标线
        if config.TARGET_METRICS:
            for metric, target in config.TARGET_METRICS.items():
                if metric in metrics:
                    ax.axhline(y=target, color='red', linestyle='--',
                             linewidth=1, alpha=0.5)

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制模型对比图时出错: {str(e)}")
        raise


def plot_learning_curve(train_scores: List[float],
                       val_scores: List[float],
                       train_sizes: List[int],
                       metric_name: str = "ROC AUC",
                       title: str = "学习曲线",
                       save_path: Optional[Path] = None,
                       figsize: Tuple[int, int] = None) -> None:
    """
    绘制学习曲线

    Args:
        train_scores: 训练集得分列表
        val_scores: 验证集得分列表
        train_sizes: 训练样本数量列表
        metric_name: 评估指标名称
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info(f"绘制学习曲线: {title}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制训练集和验证集曲线
        ax.plot(train_sizes, train_scores, 'o-', color='#3498db',
               linewidth=3, markersize=8, label='训练集')
        ax.plot(train_sizes, val_scores, 'o-', color='#e74c3c',
               linewidth=3, markersize=8, label='验证集')

        # 填充区域
        ax.fill_between(train_sizes, train_scores, val_scores,
                       alpha=0.2, color='gray')

        ax.set_xlabel('训练样本数量', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制学习曲线时出错: {str(e)}")
        raise


def create_eda_report(data: pd.DataFrame,
                     target_col: str = None,
                     save_dir: Optional[Path] = None) -> None:
    """
    创建探索性数据分析报告

    Args:
        data: 输入数据
        target_col: 目标变量列名
        save_dir: 保存目录（可选）
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info("=" * 60)
    logger.info("创建EDA可视化报告")
    logger.info("=" * 60)

    try:
        if save_dir is None:
            save_dir = config.FIGURES_DIR / "eda"

        ensure_dir(save_dir)

        with Timer("EDA报告生成"):
            # 1. 目标变量分布
            if target_col and target_col in data.columns:
                plot_class_distribution(
                    data[target_col],
                    title="客户流失分布",
                    save_path=save_dir / "class_distribution.png"
                )

            # 2. 数值特征分布
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_features:
                numeric_features.remove(target_col)
            if config.ID_COL in numeric_features:
                numeric_features.remove(config.ID_COL)

            if numeric_features:
                # 分批绘制（每批6个特征）
                batch_size = 6
                for i in range(0, len(numeric_features), batch_size):
                    batch_features = numeric_features[i:i+batch_size]
                    plot_feature_distribution(
                        data,
                        batch_features,
                        target_col=target_col,
                        title=f"数值特征分布_batch{i//batch_size + 1}",
                        save_path=save_dir / f"numeric_features_batch{i//batch_size + 1}.png"
                    )

            # 3. 分类特征分布
            categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if config.ID_COL in categorical_features:
                categorical_features.remove(config.ID_COL)

            if categorical_features:
                # 分批绘制
                batch_size = 6
                for i in range(0, len(categorical_features), batch_size):
                    batch_features = categorical_features[i:i+batch_size]
                    plot_feature_distribution(
                        data,
                        batch_features,
                        target_col=target_col,
                        title=f"分类特征分布_batch{i//batch_size + 1}",
                        save_path=save_dir / f"categorical_features_batch{i//batch_size + 1}.png"
                    )

            # 4. 相关性热力图
            if len(numeric_features) > 1:
                # 选择前15个数值特征
                selected_features = numeric_features[:15]
                # 只在目标列是数值型时才添加
                if target_col and target_col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[target_col]):
                        selected_features.append(target_col)

                plot_correlation_heatmap(
                    data,
                    features=selected_features,
                    title="特征相关性分析",
                    save_path=save_dir / "correlation_heatmap.png",
                    annot=False
                )

            logger.info(f"\nEDA报告已生成，保存至: {save_dir}")

    except Exception as e:
        logger.error(f"创建EDA报告时出错: {str(e)}")
        raise


# 别名函数，用于向后兼容
def create_visualization_report(data: pd.DataFrame,
                                target_col: str = None,
                                save_dir: Optional[Path] = None) -> None:
    """
    创建可视化报告（create_eda_report的别名）

    Args:
        data: 输入数据
        target_col: 目标变量列名
        save_dir: 保存目录（可选）
    """
    return create_eda_report(data, target_col, save_dir)


if __name__ == '__main__':
    # 测试可视化模块
    from src.utils import setup_logger
    from src.data_loader import load_data
    from src.data_preprocessing import preprocess_data

    logger = setup_logger("Churn_Prediction", config.LOG_FILE, "INFO")

    print("=" * 60)
    print("可视化模块测试")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据")
    df = load_data()
    df_clean = preprocess_data(df)

    # 2. 测试类别分布图
    print("\n2. 测试类别分布图")
    plot_class_distribution(
        df_clean[config.TARGET_COL],
        title="测试_类别分布"
    )

    # 3. 测试特征分布图
    print("\n3. 测试特征分布图")
    test_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    plot_feature_distribution(
        df_clean,
        test_features,
        target_col=config.TARGET_COL,
        title="测试_特征分布"
    )

    # 4. 测试相关性热力图
    print("\n4. 测试相关性热力图")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()[:8]
    plot_correlation_heatmap(
        df_clean,
        features=numeric_cols,
        title="测试_相关性热力图"
    )

    # 5. 测试混淆矩阵
    print("\n5. 测试混淆矩阵")
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1] * 50)
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1] * 50)
    plot_confusion_matrix(y_true, y_pred, title="测试_混淆矩阵")

    # 6. 测试ROC曲线
    print("\n6. 测试ROC曲线")
    y_true = np.array([0, 0, 1, 1] * 100)
    y_pred_proba = np.array([0.1, 0.4, 0.35, 0.8] * 100)
    plot_roc_curve(y_true, y_pred_proba, title="测试_ROC曲线")

    # 7. 测试PR曲线
    print("\n7. 测试PR曲线")
    plot_precision_recall_curve(y_true, y_pred_proba, title="测试_PR曲线")

    # 8. 测试特征重要性图
    print("\n8. 测试特征重要性图")
    feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
    importance_values = np.array([0.35, 0.25, 0.20, 0.15, 0.05])
    plot_feature_importance(feature_names, importance_values, title="测试_特征重要性")

    # 9. 创建完整EDA报告
    print("\n9. 创建EDA报告")
    create_eda_report(df_clean, target_col=config.TARGET_COL)

    print("\n可视化模块测试完成！")
    print(f"所有图表已保存至: {config.FIGURES_DIR}")
