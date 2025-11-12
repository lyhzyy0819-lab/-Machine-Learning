"""
可视化模块
提供数据分析和模型评估的可视化功能
包含20+种可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, ensure_dir


# 设置matplotlib样式
plt.style.use(config.PLOT_STYLE)
sns.set_palette(config.COLOR_PALETTE)


def plot_rating_distribution(df: pd.DataFrame,
                            rating_col: str = 'rating',
                            title: str = "评分分布",
                            save_path: Optional[Path] = None,
                            figsize: Tuple[int, int] = None) -> None:
    """
    绘制评分分布图（直方图+KDE）

    Args:
        df: 输入数据
        rating_col: 评分列名
        title: 图表标题
        save_path: 保存路径
        figsize: 图表尺寸

    TODO 1: 设置图表尺寸
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info(f"绘制评分分布图: {title}")
    #
    # if figsize is None:
    #     figsize = config.FIGURE_SIZE

    TODO 2: 创建图表
    # fig, ax = plt.subplots(figsize=figsize)

    TODO 3: 绘制直方图和KDE
    # ax.hist(df[rating_col].dropna(), bins=50, alpha=0.6,
    #        color=config.PRIMARY_COLOR, edgecolor='black',
    #        density=True, label='直方图')
    # df[rating_col].plot.kde(ax=ax, color='red', linewidth=2, label='密度曲线')

    TODO 4: 添加均值和中位数线
    # mean_val = df[rating_col].mean()
    # median_val = df[rating_col].median()
    # ax.axvline(mean_val, color='green', linestyle='--',
    #           linewidth=2, label=f'均值: {mean_val:.2f}')
    # ax.axvline(median_val, color='orange', linestyle='--',
    #           linewidth=2, label=f'中位数: {median_val:.2f}')

    TODO 5: 设置标签和标题
    # ax.set_xlabel('评分', fontsize=12)
    # ax.set_ylabel('密度', fontsize=12)
    # ax.set_title(title, fontsize=14, fontweight='bold')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()

    TODO 6: 保存图表
    # if save_path is None:
    #     save_path = config.FIGURE_DIR / f"{title}.png"
    # ensure_dir(save_path.parent)
    # plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    # logger.info(f"图表已保存: {save_path}")
    # plt.close()
    """
    # TODO: 实现评分分布图
    pass


def plot_category_distribution(df: pd.DataFrame,
                              category_col: str = 'category',
                              title: str = "商品类别分布",
                              save_path: Optional[Path] = None,
                              top_n: int = 15) -> None:
    """
    绘制类别分布柱状图（显示前N个）

    Args:
        df: 输入数据
        category_col: 类别列名
        title: 图表标题
        save_path: 保存路径
        top_n: 显示前N个类别

    TODO: 实现类别分布柱状图
    提示：
    1. 使用value_counts()获取类别计数
    2. 选择前top_n个类别
    3. 使用ax.barh()绘制水平柱状图
    4. 设置标签、标题、保存
    """
    # TODO: 实现类别分布图
    pass


def plot_price_distribution(df: pd.DataFrame,
                           price_col: str = 'discounted_price',
                           title: str = "价格分布",
                           save_path: Optional[Path] = None,
                           log_scale: bool = True) -> None:
    """
    绘制价格分布图（通常使用对数刻度）

    Args:
        df: 输入数据
        price_col: 价格列名
        title: 图表标题
        save_path: 保存路径
        log_scale: 是否使用对数刻度

    TODO: 实现价格分布图
    提示：价格数据通常呈现长尾分布，使用对数刻度可以更好地展示
    """
    # TODO: 实现价格分布图
    pass


def plot_rating_vs_price(df: pd.DataFrame,
                        rating_col: str = 'rating',
                        price_col: str = 'discounted_price',
                        title: str = "评分与价格关系",
                        save_path: Optional[Path] = None,
                        sample_size: int = 5000) -> None:
    """
    绘制评分与价格的散点图

    Args:
        df: 输入数据
        rating_col: 评分列名
        price_col: 价格列名
        title: 图表标题
        save_path: 保存路径
        sample_size: 采样数量

    TODO: 实现评分与价格散点图
    提示：
    1. 如果数据量大，先采样
    2. 使用scatter绘制散点图
    3. 可以添加回归线（使用sns.regplot）
    """
    # TODO: 实现评分价格散点图
    pass


def plot_discount_analysis(df: pd.DataFrame,
                          discount_col: str = 'discount_percentage',
                          rating_col: str = 'rating',
                          title: str = "折扣力度分析",
                          save_path: Optional[Path] = None) -> None:
    """
    绘制折扣力度与评分的关系（箱线图）

    Args:
        df: 输入数据
        discount_col: 折扣列名
        rating_col: 评分列名
        title: 图表标题
        save_path: 保存路径

    TODO: 实现折扣分析图
    提示：
    1. 将折扣百分比分组（如：0-20%, 20-40%, 40-60%, 60-80%, 80-100%）
    2. 使用sns.boxplot绘制每组的评分分布
    """
    # TODO: 实现折扣分析图
    pass


def plot_rating_count_distribution(df: pd.DataFrame,
                                   rating_count_col: str = 'rating_count',
                                   title: str = "评分数量分布",
                                   save_path: Optional[Path] = None) -> None:
    """
    绘制评分数量分布（对数刻度直方图）

    TODO: 实现评分数量分布图
    """
    # TODO: 实现评分数量分布图
    pass


def plot_correlation_heatmap(df: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            title: str = "特征相关性热力图",
                            save_path: Optional[Path] = None,
                            annot: bool = True) -> None:
    """
    绘制相关性热力图

    Args:
        df: 输入数据
        columns: 要分析的列（None表示所有数值列）
        title: 图表标题
        save_path: 保存路径
        annot: 是否显示数值

    TODO: 实现相关性热力图
    提示：
    1. 选择数值列
    2. 计算相关系数矩阵
    3. 使用sns.heatmap绘制热力图
    4. 可以只显示下三角（使用mask）
    """
    # TODO: 实现相关性热力图
    pass


def plot_predictions_vs_actual(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               title: str = "预测值 vs 实际值",
                               save_path: Optional[Path] = None,
                               sample_size: int = 3000) -> None:
    """
    绘制预测值vs实际值散点图

    TODO: 实现预测对比图
    提示：
    1. 采样数据（如果数据量大）
    2. 绘制散点图
    3. 添加y=x参考线（完美预测线）
    4. 添加R²、RMSE、MAE等指标文本框
    """
    # TODO: 实现预测对比图
    pass


def plot_residuals(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  title: str = "残差分析",
                  save_path: Optional[Path] = None) -> None:
    """
    绘制残差分析图（3个子图：残差散点图、残差分布、Q-Q图）

    TODO: 实现残差分析图
    提示：
    1. 创建1x3子图布局
    2. 子图1：残差散点图（y_pred vs residuals）
    3. 子图2：残差分布直方图
    4. 子图3：Q-Q图（使用scipy.stats.probplot）
    """
    # TODO: 实现残差分析图
    pass


def plot_confusion_matrix(cm: np.ndarray,
                         title: str = "混淆矩阵",
                         save_path: Optional[Path] = None,
                         labels: List[str] = None) -> None:
    """
    绘制混淆矩阵热力图

    Args:
        cm: 混淆矩阵
        title: 图表标题
        save_path: 保存路径
        labels: 类别标签

    TODO: 实现混淆矩阵图
    提示：使用sns.heatmap，设置annot=True显示数值
    """
    # TODO: 实现混淆矩阵图
    pass


def plot_roc_curve(fpr: np.ndarray,
                  tpr: np.ndarray,
                  auc_score: float,
                  title: str = "ROC曲线",
                  save_path: Optional[Path] = None) -> None:
    """
    绘制ROC曲线

    Args:
        fpr: 假正率
        tpr: 真正率
        auc_score: AUC分数
        title: 图表标题
        save_path: 保存路径

    TODO: 实现ROC曲线图
    提示：
    1. 绘制ROC曲线
    2. 添加对角线（随机分类器）
    3. 在图例中显示AUC分数
    """
    # TODO: 实现ROC曲线图
    pass


def plot_precision_recall_curve(precision: np.ndarray,
                                recall: np.ndarray,
                                auc_score: float,
                                title: str = "Precision-Recall曲线",
                                save_path: Optional[Path] = None) -> None:
    """
    绘制Precision-Recall曲线

    TODO: 实现PR曲线图
    """
    # TODO: 实现PR曲线图
    pass


def plot_feature_importance(feature_names: List[str],
                           importance_values: np.ndarray,
                           title: str = "特征重要性",
                           save_path: Optional[Path] = None,
                           top_n: int = 20) -> None:
    """
    绘制特征重要性图

    Args:
        feature_names: 特征名称列表
        importance_values: 重要性值
        title: 图表标题
        save_path: 保存路径
        top_n: 显示前N个特征

    TODO: 实现特征重要性图
    提示：
    1. 创建DataFrame并排序
    2. 选择前top_n个特征
    3. 使用水平柱状图（barh）
    4. 颜色渐变效果（使用colormap）
    """
    # TODO: 实现特征重要性图
    pass


def plot_learning_curve(train_sizes: np.ndarray,
                       train_scores: np.ndarray,
                       test_scores: np.ndarray,
                       title: str = "学习曲线",
                       save_path: Optional[Path] = None) -> None:
    """
    绘制学习曲线

    Args:
        train_sizes: 训练样本数量
        train_scores: 训练集得分
        test_scores: 测试集得分
        title: 图表标题
        save_path: 保存路径

    TODO: 实现学习曲线图
    提示：
    1. 绘制训练集和验证集曲线
    2. 可以添加标准差阴影（fill_between）
    """
    # TODO: 实现学习曲线图
    pass


def plot_rating_by_category(df: pd.DataFrame,
                           category_col: str = 'category',
                           rating_col: str = 'rating',
                           title: str = "不同类别的评分分布",
                           save_path: Optional[Path] = None,
                           top_n: int = 10) -> None:
    """
    绘制不同类别的评分分布箱线图

    TODO: 实现类别评分箱线图
    提示：
    1. 选择评论数最多的前N个类别
    2. 使用sns.boxplot或violinplot
    """
    # TODO: 实现类别评分箱线图
    pass


def plot_price_range_analysis(df: pd.DataFrame,
                             price_col: str = 'discounted_price',
                             rating_col: str = 'rating',
                             title: str = "不同价格区间的评分分布",
                             save_path: Optional[Path] = None) -> None:
    """
    绘制不同价格区间的评分分布

    TODO: 实现价格区间评分分析图
    提示：
    1. 使用pd.cut将价格分桶
    2. 绘制每个价格区间的评分分布
    """
    # TODO: 实现价格区间分析图
    pass


def plot_text_length_analysis(df: pd.DataFrame,
                             text_col: str = 'review_content',
                             rating_col: str = 'rating',
                             title: str = "评论长度与评分关系",
                             save_path: Optional[Path] = None) -> None:
    """
    绘制文本长度与评分的关系

    TODO: 实现文本长度分析图
    提示：
    1. 计算文本长度
    2. 将长度分组
    3. 绘制每组的平均评分
    """
    # TODO: 实现文本长度分析图
    pass


def plot_model_comparison(comparison_df: pd.DataFrame,
                         metric_col: str = 'r2',
                         title: str = "模型性能对比",
                         save_path: Optional[Path] = None) -> None:
    """
    绘制模型性能对比柱状图

    Args:
        comparison_df: 模型对比DataFrame（包含模型名称和指标）
        metric_col: 要对比的指标列名
        title: 图表标题
        save_path: 保存路径

    TODO: 实现模型对比图
    提示：使用柱状图，按性能排序
    """
    # TODO: 实现模型对比图
    pass


def plot_error_distribution(errors: np.ndarray,
                           title: str = "预测误差分布",
                           save_path: Optional[Path] = None) -> None:
    """
    绘制预测误差分布图

    TODO: 实现误差分布图
    """
    # TODO: 实现误差分布图
    pass


def plot_rating_trends(df: pd.DataFrame,
                      date_col: str = None,
                      rating_col: str = 'rating',
                      title: str = "评分趋势",
                      save_path: Optional[Path] = None) -> None:
    """
    绘制评分随时间变化趋势（如果有时间列）

    TODO: 实现评分趋势图
    注意：只有当数据包含时间列时才绘制
    """
    # TODO: 实现评分趋势图
    pass


def create_comprehensive_report(df: pd.DataFrame,
                               save_dir: Optional[Path] = None) -> None:
    """
    创建完整的数据可视化报告（生成所有适用的图表）

    Args:
        df: 输入数据
        save_dir: 保存目录

    TODO 1: 设置保存目录
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("=" * 60)
    # logger.info("创建综合可视化报告")
    # logger.info("=" * 60)
    #
    # if save_dir is None:
    #     save_dir = config.FIGURE_DIR / "report"
    # ensure_dir(save_dir)

    TODO 2: 使用Timer计时
    # with Timer("可视化报告生成"):

    TODO 3: 绘制基础分布图
    #     # 1. 评分分布
    #     if 'rating' in df.columns:
    #         plot_rating_distribution(df, save_path=save_dir / "01_rating_distribution.png")
    #
    #     # 2. 类别分布
    #     if 'category' in df.columns:
    #         plot_category_distribution(df, save_path=save_dir / "02_category_distribution.png")
    #
    #     # 3. 价格分布
    #     if 'discounted_price' in df.columns:
    #         plot_price_distribution(df, save_path=save_dir / "03_price_distribution.png")

    TODO 4: 绘制关系图
    #     # 4. 评分与价格关系
    #     if 'rating' in df.columns and 'discounted_price' in df.columns:
    #         plot_rating_vs_price(df, save_path=save_dir / "04_rating_vs_price.png")
    #
    #     # 5. 折扣分析
    #     if 'discount_percentage' in df.columns and 'rating' in df.columns:
    #         plot_discount_analysis(df, save_path=save_dir / "05_discount_analysis.png")

    TODO 5: 绘制其他分析图
    #     # 继续添加其他图表...

    TODO 6: 打印完成信息
    #     logger.info(f"可视化报告已生成，保存至: {save_dir}")
    """
    # TODO: 实现综合报告生成
    pass


if __name__ == '__main__':
    # 测试可视化模块
    from src.utils import setup_logger
    from src.data_loader import load_raw_data
    from src.data_preprocessing import preprocess_data

    # TODO: 设置日志
    # logger = setup_logger("Ecommerce_Rating", config.LOG_DIR / "visualization_test.log", "INFO")

    print("=" * 60)
    print("可视化模块测试")
    print("=" * 60)

    # TODO 1: 加载测试数据
    # try:
    #     df = load_raw_data(use_sample=True, sample_size=500)
    #     df_clean = preprocess_data(df, remove_outliers_flag=False)

    # TODO 2: 测试各种可视化函数
    #     print("\n测试评分分布图...")
    #     plot_rating_distribution(df_clean, save_path=config.FIGURE_DIR / "test_rating_dist.png")
    #
    #     print("\n测试类别分布图...")
    #     plot_category_distribution(df_clean, save_path=config.FIGURE_DIR / "test_category_dist.png")
    #
    #     print("\n测试价格分布图...")
    #     plot_price_distribution(df_clean, save_path=config.FIGURE_DIR / "test_price_dist.png")

    # TODO 3: 测试完整报告生成
    #     print("\n生成完整可视化报告...")
    #     create_comprehensive_report(df_clean)
    #
    #     print(f"\n所有图表已保存至: {config.FIGURE_DIR}")

    # except Exception as e:
    #     print(f"\n错误: {str(e)}")

    print("\n提示：实现上述TODO后运行此文件进行测试")
    print("\n可视化函数列表：")
    print("1. plot_rating_distribution - 评分分布")
    print("2. plot_category_distribution - 类别分布")
    print("3. plot_price_distribution - 价格分布")
    print("4. plot_rating_vs_price - 评分与价格关系")
    print("5. plot_discount_analysis - 折扣分析")
    print("6. plot_rating_count_distribution - 评分数量分布")
    print("7. plot_correlation_heatmap - 相关性热力图")
    print("8. plot_predictions_vs_actual - 预测对比")
    print("9. plot_residuals - 残差分析")
    print("10. plot_confusion_matrix - 混淆矩阵")
    print("11. plot_roc_curve - ROC曲线")
    print("12. plot_precision_recall_curve - PR曲线")
    print("13. plot_feature_importance - 特征重要性")
    print("14. plot_learning_curve - 学习曲线")
    print("15. plot_rating_by_category - 类别评分箱线图")
    print("16. plot_price_range_analysis - 价格区间分析")
    print("17. plot_text_length_analysis - 文本长度分析")
    print("18. plot_model_comparison - 模型对比")
    print("19. plot_error_distribution - 误差分布")
    print("20. plot_rating_trends - 评分趋势")
    print("21. create_comprehensive_report - 综合报告")
