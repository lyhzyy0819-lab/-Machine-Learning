"""
可视化模块
提供数据分析和模型评估的可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import logging
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, ensure_dir


# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = [config.CHINESE_FONT]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.style.use(config.PLOT_STYLE)


def plot_distribution(data: pd.DataFrame,
                     columns: List[str],
                     title: str = "数据分布",
                     save_path: Optional[Path] = None,
                     figsize: Tuple[int, int] = None) -> None:
    """
    绘制数据分布图

    Args:
        data: 输入数据
        columns: 要绘制的列名列表
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"绘制数据分布图: {title}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

        # 计算子图布局
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3  # 每行3个子图
        n_subplot_cols = min(3, n_cols)

        fig, axes = plt.subplots(n_rows, n_subplot_cols,
                                figsize=(figsize[0], figsize[1] * n_rows // 2))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 展平axes数组以便迭代
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()

        for idx, col in enumerate(columns):
            if col not in data.columns:
                logger.warning(f"列 '{col}' 不存在，跳过")
                continue

            ax = axes[idx]

            # 移除缺失值
            col_data = data[col].dropna()

            # 绘制直方图和KDE
            ax.hist(col_data, bins=50, alpha=0.6, color='skyblue',
                   edgecolor='black', density=True, label='直方图')

            # 绘制KDE曲线
            col_data.plot.kde(ax=ax, color='red', linewidth=2, label='密度曲线')

            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('密度', fontsize=10)
            ax.set_title(f'{col} 分布', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            mean_val = col_data.mean()
            median_val = col_data.median()
            ax.axvline(mean_val, color='green', linestyle='--',
                      linewidth=1.5, label=f'均值: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='--',
                      linewidth=1.5, label=f'中位数: {median_val:.2f}')
            ax.legend(fontsize=8)

        # 隐藏多余的子图
        for idx in range(len(columns), len(axes)):
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
        logger.error(f"绘制分布图时出错: {str(e)}")
        raise


def plot_correlation_heatmap(data: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            title: str = "特征相关性热力图",
                            save_path: Optional[Path] = None,
                            figsize: Tuple[int, int] = None,
                            annot: bool = True) -> None:
    """
    绘制相关性热力图

    Args:
        data: 输入数据
        columns: 要分析的列名列表（None表示使用所有数值列）
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
        annot: 是否显示相关系数数值
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"绘制相关性热力图: {title}")

    try:
        if figsize is None:
            figsize = config.LARGE_FIGURE_SIZE

        # 选择数值列
        if columns is None:
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data[columns]

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


def plot_geo_scatter(data: pd.DataFrame,
                    lon_col: str = 'pickup_longitude',
                    lat_col: str = 'pickup_latitude',
                    color_col: Optional[str] = None,
                    title: str = "地理位置散点图",
                    save_path: Optional[Path] = None,
                    figsize: Tuple[int, int] = None,
                    sample_size: int = 10000,
                    alpha: float = 0.3) -> None:
    """
    绘制地理散点图

    Args:
        data: 输入数据
        lon_col: 经度列名
        lat_col: 纬度列名
        color_col: 用于着色的列名（可选）
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
        sample_size: 采样数量（用于大数据集）
        alpha: 透明度
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"绘制地理散点图: {title}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

        # 采样数据
        if len(data) > sample_size:
            plot_data = data.sample(n=sample_size, random_state=config.RANDOM_STATE)
            logger.info(f"数据已采样至 {sample_size} 条")
        else:
            plot_data = data.copy()

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制散点图
        if color_col is not None and color_col in plot_data.columns:
            scatter = ax.scatter(plot_data[lon_col], plot_data[lat_col],
                               c=plot_data[color_col], cmap='viridis',
                               alpha=alpha, s=10)
            plt.colorbar(scatter, ax=ax, label=color_col)
        else:
            ax.scatter(plot_data[lon_col], plot_data[lat_col],
                      color='blue', alpha=alpha, s=10)

        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 设置坐标轴范围为NYC边界
        ax.set_xlim(config.NYC_BOUNDS['min_longitude'],
                   config.NYC_BOUNDS['max_longitude'])
        ax.set_ylim(config.NYC_BOUNDS['min_latitude'],
                   config.NYC_BOUNDS['max_latitude'])

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制地理散点图时出错: {str(e)}")
        raise


def plot_time_series(data: pd.DataFrame,
                    datetime_col: str = 'pickup_datetime',
                    value_col: str = 'trip_duration',
                    agg_func: str = 'mean',
                    freq: str = 'D',
                    title: str = "时间序列分析",
                    save_path: Optional[Path] = None,
                    figsize: Tuple[int, int] = None) -> None:
    """
    绘制时间序列图

    Args:
        data: 输入数据
        datetime_col: 日期时间列名
        value_col: 数值列名
        agg_func: 聚合函数 ('mean', 'sum', 'count', 'median')
        freq: 时间频率 ('H'=小时, 'D'=天, 'W'=周, 'M'=月)
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"绘制时间序列图: {title}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

        # 确保datetime列是datetime类型
        if data[datetime_col].dtype != 'datetime64[ns]':
            plot_data = data.copy()
            plot_data[datetime_col] = pd.to_datetime(plot_data[datetime_col])
        else:
            plot_data = data.copy()

        # 设置索引为日期时间
        plot_data = plot_data.set_index(datetime_col)

        # 按时间频率聚合
        if agg_func == 'mean':
            ts_data = plot_data[value_col].resample(freq).mean()
        elif agg_func == 'sum':
            ts_data = plot_data[value_col].resample(freq).sum()
        elif agg_func == 'count':
            ts_data = plot_data[value_col].resample(freq).count()
        elif agg_func == 'median':
            ts_data = plot_data[value_col].resample(freq).median()
        else:
            raise ValueError(f"不支持的聚合函数: {agg_func}")

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制时间序列
        ax.plot(ts_data.index, ts_data.values, linewidth=2, color='blue')
        ax.fill_between(ts_data.index, ts_data.values, alpha=0.3, color='blue')

        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel(f'{value_col} ({agg_func})', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制时间序列图时出错: {str(e)}")
        raise


def plot_predictions(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    title: str = "预测值 vs 实际值",
                    save_path: Optional[Path] = None,
                    figsize: Tuple[int, int] = None,
                    sample_size: int = 5000) -> None:
    """
    绘制预测值vs实际值对比图

    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
        sample_size: 采样数量（用于大数据集）
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"绘制预测对比图: {title}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

        # 采样数据
        if len(y_true) > sample_size:
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_plot = y_true[indices]
            y_pred_plot = y_pred[indices]
            logger.info(f"数据已采样至 {sample_size} 条")
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred

        # 计算评估指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制散点图
        ax.scatter(y_true_plot, y_pred_plot, alpha=0.5, s=10, color='blue')

        # 绘制理想预测线（y=x）
        min_val = min(y_true_plot.min(), y_pred_plot.min())
        max_val = max(y_true_plot.max(), y_pred_plot.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, label='理想预测')

        ax.set_xlabel('实际值', fontsize=12)
        ax.set_ylabel('预测值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加评估指标文本
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制预测对比图时出错: {str(e)}")
        raise


def plot_residuals(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  title: str = "残差分析",
                  save_path: Optional[Path] = None,
                  figsize: Tuple[int, int] = None,
                  sample_size: int = 5000) -> None:
    """
    绘制残差分析图

    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
        sample_size: 采样数量（用于大数据集）
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"绘制残差分析图: {title}")

    try:
        if figsize is None:
            figsize = (15, 5)

        # 计算残差
        residuals = y_true - y_pred

        # 采样数据
        if len(residuals) > sample_size:
            indices = np.random.choice(len(residuals), sample_size, replace=False)
            residuals_plot = residuals[indices]
            y_pred_plot = y_pred[indices]
        else:
            residuals_plot = residuals
            y_pred_plot = y_pred

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 1. 残差散点图
        axes[0].scatter(y_pred_plot, residuals_plot, alpha=0.5, s=10, color='blue')
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('预测值', fontsize=10)
        axes[0].set_ylabel('残差', fontsize=10)
        axes[0].set_title('残差散点图', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # 2. 残差分布直方图
        axes[1].hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('残差', fontsize=10)
        axes[1].set_ylabel('频数', fontsize=10)
        axes[1].set_title('残差分布', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # 3. Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q图', fontsize=12)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / f"{title.replace(' ', '_')}.png"

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制残差分析图时出错: {str(e)}")
        raise


def plot_learning_curve(train_scores: List[float],
                       val_scores: List[float],
                       train_sizes: List[int],
                       metric_name: str = "R²",
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
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"绘制学习曲线: {title}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制训练集和验证集曲线
        ax.plot(train_sizes, train_scores, 'o-', color='blue',
               linewidth=2, markersize=8, label='训练集')
        ax.plot(train_sizes, val_scores, 'o-', color='red',
               linewidth=2, markersize=8, label='验证集')

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
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"绘制特征重要性图: {title}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

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
        ax.barh(range(len(importance_df)), importance_df['importance'],
               color=colors, edgecolor='black')

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


def plot_model_comparison(models_results: Dict[str, Dict[str, float]],
                         metric: str = 'rmse',
                         title: Optional[str] = None,
                         save_name: str = 'model_comparison.png',
                         save_path: Optional[Path] = None,
                         figsize: Tuple[int, int] = None) -> None:
    """
    绘制模型对比图

    Args:
        models_results: 模型结果字典，格式为 {模型名: {r2: ..., rmse: ..., mae: ..., mape: ...}}
        metric: 用于对比的指标名称 ('r2', 'rmse', 'mae', 'mape')
        title: 图表标题（可选）
        save_name: 保存文件名
        save_path: 保存路径（可选）
        figsize: 图表尺寸（可选）
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"绘制模型对比图: {metric}")

    try:
        if figsize is None:
            figsize = config.FIGURE_SIZE

        # 提取模型名称和指标值
        model_names = list(models_results.keys())
        metric_values = [models_results[name].get(metric, 0) for name in model_names]

        # 根据指标排序（R²降序，其他升序）
        if metric == 'r2':
            sorted_indices = np.argsort(metric_values)[::-1]
        else:
            sorted_indices = np.argsort(metric_values)

        model_names = [model_names[i] for i in sorted_indices]
        metric_values = [metric_values[i] for i in sorted_indices]

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制柱状图
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(range(len(model_names)), metric_values,
                     color=colors, edgecolor='black', linewidth=1.5)

        # 在柱子上方添加数值标签
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}' if metric == 'r2' else f'{value:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 设置标题和标签
        if title is None:
            metric_names = {
                'r2': 'R² Score',
                'rmse': 'RMSE (Root Mean Squared Error)',
                'mae': 'MAE (Mean Absolute Error)',
                'mape': 'MAPE (Mean Absolute Percentage Error)'
            }
            title = f"模型对比 - {metric_names.get(metric, metric.upper())}"

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = config.FIGURES_DIR / save_name

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=config.SAVE_DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制模型对比图时出错: {str(e)}")
        raise


def create_visualization_report(data: pd.DataFrame,
                               target_col: str = 'trip_duration',
                               save_dir: Optional[Path] = None) -> None:
    """
    创建完整的可视化报告

    Args:
        data: 输入数据
        target_col: 目标变量列名
        save_dir: 保存目录（可选）
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("=" * 60)
    logger.info("创建可视化报告")
    logger.info("=" * 60)

    try:
        if save_dir is None:
            save_dir = config.FIGURES_DIR / "report"

        ensure_dir(save_dir)

        with Timer("可视化报告生成"):
            # 1. 目标变量分布
            if target_col in data.columns:
                plot_distribution(
                    data,
                    [target_col],
                    title=f"{target_col}分布",
                    save_path=save_dir / f"{target_col}_distribution.png"
                )

            # 2. 数值特征分布
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)

            if numeric_cols:
                # 分批绘制
                batch_size = 6
                for i in range(0, len(numeric_cols), batch_size):
                    batch_cols = numeric_cols[i:i+batch_size]
                    plot_distribution(
                        data,
                        batch_cols,
                        title=f"数值特征分布_batch{i//batch_size + 1}",
                        save_path=save_dir / f"numeric_features_batch{i//batch_size + 1}.png"
                    )

            # 3. 相关性热力图
            if len(numeric_cols) > 0:
                plot_correlation_heatmap(
                    data,
                    columns=numeric_cols[:15] if len(numeric_cols) > 15 else numeric_cols,
                    title="特征相关性分析",
                    save_path=save_dir / "correlation_heatmap.png"
                )

            # 4. 地理位置分布
            if all(col in data.columns for col in ['pickup_longitude', 'pickup_latitude']):
                plot_geo_scatter(
                    data,
                    lon_col='pickup_longitude',
                    lat_col='pickup_latitude',
                    color_col=target_col if target_col in data.columns else None,
                    title="起点地理位置分布",
                    save_path=save_dir / "pickup_geo_distribution.png"
                )

            if all(col in data.columns for col in ['dropoff_longitude', 'dropoff_latitude']):
                plot_geo_scatter(
                    data,
                    lon_col='dropoff_longitude',
                    lat_col='dropoff_latitude',
                    color_col=target_col if target_col in data.columns else None,
                    title="终点地理位置分布",
                    save_path=save_dir / "dropoff_geo_distribution.png"
                )

            # 5. 时间序列分析
            if 'pickup_datetime' in data.columns and target_col in data.columns:
                plot_time_series(
                    data,
                    datetime_col='pickup_datetime',
                    value_col=target_col,
                    agg_func='mean',
                    freq='D',
                    title="每日平均行程时长",
                    save_path=save_dir / "daily_avg_duration.png"
                )

                plot_time_series(
                    data,
                    datetime_col='pickup_datetime',
                    value_col=target_col,
                    agg_func='count',
                    freq='H',
                    title="每小时行程数量",
                    save_path=save_dir / "hourly_trip_count.png"
                )

            logger.info(f"可视化报告已生成，保存至: {save_dir}")

    except Exception as e:
        logger.error(f"创建可视化报告时出错: {str(e)}")
        raise


if __name__ == '__main__':
    # 测试可视化模块
    from src.utils import setup_logger
    from src.data_loader import generate_sample_data
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import engineer_features

    logger = setup_logger("NYC_Taxi", config.LOG_FILE, "INFO")

    print("=" * 60)
    print("可视化模块测试")
    print("=" * 60)

    # 生成测试数据
    print("\n1. 生成测试数据")
    df = generate_sample_data(2000)
    df_clean = preprocess_data(df, remove_outliers_flag=False)
    df_feat = engineer_features(df_clean)

    # 测试各种可视化函数
    print("\n2. 测试分布图")
    plot_distribution(
        df_feat,
        ['trip_duration', 'manhattan_distance', 'euclidean_distance'],
        title="测试_特征分布"
    )

    print("\n3. 测试相关性热力图")
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()[:10]
    plot_correlation_heatmap(
        df_feat,
        columns=numeric_cols,
        title="测试_相关性热力图"
    )

    print("\n4. 测试地理散点图")
    plot_geo_scatter(
        df_feat,
        lon_col='pickup_longitude',
        lat_col='pickup_latitude',
        color_col='trip_duration',
        title="测试_地理位置分布"
    )

    print("\n5. 测试时间序列图")
    plot_time_series(
        df_feat,
        datetime_col='pickup_datetime',
        value_col='trip_duration',
        agg_func='mean',
        freq='D',
        title="测试_时间序列"
    )

    print("\n6. 测试预测对比图")
    y_true = df_feat['trip_duration'].values
    y_pred = y_true + np.random.normal(0, 100, len(y_true))  # 模拟预测值
    plot_predictions(y_true, y_pred, title="测试_预测对比")

    print("\n7. 测试残差分析图")
    plot_residuals(y_true, y_pred, title="测试_残差分析")

    print("\n8. 测试学习曲线")
    train_sizes = [100, 500, 1000, 1500, 2000]
    train_scores = [0.65, 0.72, 0.78, 0.82, 0.85]
    val_scores = [0.63, 0.70, 0.75, 0.77, 0.78]
    plot_learning_curve(train_scores, val_scores, train_sizes, title="测试_学习曲线")

    print("\n9. 测试特征重要性图")
    feature_names = ['distance', 'hour', 'weekday', 'passenger_count', 'is_rush_hour']
    importance_values = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
    plot_feature_importance(feature_names, importance_values, title="测试_特征重要性")

    print("\n10. 测试完整可视化报告")
    create_visualization_report(df_feat, target_col='trip_duration')

    print("\n可视化模块测试完成！")
    print(f"所有图表已保存至: {config.FIGURES_DIR}")
