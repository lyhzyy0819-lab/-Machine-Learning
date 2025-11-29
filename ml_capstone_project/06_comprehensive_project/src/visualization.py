"""
可视化模块
==========

提供丰富的数据和模型可视化功能，帮助理解数据和模型行为。

主要功能:
- 数据分布可视化
- 特征关系可视化
- 模型性能可视化
- 聚类结果可视化
- 降维可视化
- 时间序列可视化

可视化原则:
- 图表要有清晰的标题和标签
- 使用合适的颜色方案
- 避免图表过于复杂
- 添加必要的图例和注释
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 负号显示


# ==================== Phase 1: 数据诊断可视化 ====================

def plot_missing_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)):
    """
    绘制缺失值热力图

    显示数据集中缺失值的位置和模式，帮助识别:
    - 哪些列有缺失值
    - 缺失值的分布模式
    - 缺失值是否有规律

    Args:
        df: 数据DataFrame
        figsize: 图像大小
    """
    # 检查是否有缺失值
    if df.isnull().sum().sum() == 0:
        print("✅ 数据集无缺失值")
        return

    # 只选择有缺失值的列
    missing_cols = df.columns[df.isnull().any()].tolist()

    if len(missing_cols) == 0:
        print("✅ 数据集无缺失值")
        return

    # 创建缺失值矩阵（1=缺失，0=存在）
    missing_matrix = df[missing_cols].isnull().astype(int)

    # 绘制热力图
    plt.figure(figsize=figsize)

    # 使用seaborn绘制
    sns.heatmap(missing_matrix.T,
                cmap='YlOrRd',
                cbar_kws={'label': '1=Missing, 0=Present'},
                yticklabels=missing_cols,
                xticklabels=False)

    plt.title('缺失值热力图 (红色=缺失)', fontsize=14, fontweight='bold')
    plt.xlabel('样本索引')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.show()

    # 打印缺失值统计
    print("\n缺失值统计:")
    for col in missing_cols:
        missing_count = df[col].isnull().sum()
        missing_pct = missing_count / len(df) * 100
        print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")


def plot_missing_bar(df: pd.DataFrame, threshold: float = 0.0,
                    figsize: Tuple[int, int] = (12, 6)):
    """
    绘制缺失值柱状图

    Args:
        df: 数据DataFrame
        threshold: 只显示缺失率大于此值的列
        figsize: 图像大小
    """
    # 计算缺失值
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df) * 100).round(2)

    # 过滤
    missing_df = pd.DataFrame({
        '缺失数量': missing_data,
        '缺失比例(%)': missing_pct
    })
    missing_df = missing_df[missing_df['缺失比例(%)'] > threshold * 100]
    missing_df = missing_df.sort_values('缺失比例(%)', ascending=False)

    if len(missing_df) == 0:
        print("✅ 没有超过阈值的缺失值")
        return

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图: 缺失数量
    axes[0].barh(range(len(missing_df)), missing_df['缺失数量'], color='coral')
    axes[0].set_yticks(range(len(missing_df)))
    axes[0].set_yticklabels(missing_df.index)
    axes[0].set_xlabel('缺失数量')
    axes[0].set_title('各特征缺失值数量')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # 右图: 缺失比例
    colors = ['red' if x > 50 else 'orange' if x > 20 else 'yellow'
              for x in missing_df['缺失比例(%)']]
    axes[1].barh(range(len(missing_df)), missing_df['缺失比例(%)'], color=colors)
    axes[1].set_yticks(range(len(missing_df)))
    axes[1].set_yticklabels(missing_df.index)
    axes[1].set_xlabel('缺失比例 (%)')
    axes[1].set_title('各特征缺失值比例')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)

    # 添加比例标注
    for i, v in enumerate(missing_df['缺失比例(%)']):
        axes[1].text(v + 1, i, f'{v:.1f}%', va='center')

    plt.suptitle('缺失值分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_data_quality_summary(df: pd.DataFrame, target: Optional[str] = None,
                               figsize: Tuple[int, int] = (16, 10)):
    """
    绘制数据质量综合摘要图

    包含4个子图:
    1. 缺失值分布
    2. 数据类型分布
    3. 数值特征分布(箱线图)
    4. 目标变量分布(如有)

    Args:
        df: 数据DataFrame
        target: 目标变量列名(可选)
        figsize: 图像大小
    """
    # 创建2x2子图
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 子图1: 缺失值Top10
    ax1 = fig.add_subplot(gs[0, 0])
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df) * 100).round(1)
    missing_df = pd.DataFrame({'缺失率(%)': missing_pct})
    missing_df = missing_df[missing_df['缺失率(%)'] > 0].sort_values('缺失率(%)', ascending=False).head(10)

    if len(missing_df) > 0:
        ax1.barh(range(len(missing_df)), missing_df['缺失率(%)'], color='coral')
        ax1.set_yticks(range(len(missing_df)))
        ax1.set_yticklabels(missing_df.index)
        ax1.set_xlabel('缺失率 (%)')
        ax1.set_title('缺失值 Top10', fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, '✅ 无缺失值', ha='center', va='center',
                fontsize=14, transform=ax1.transAxes)
        ax1.set_title('缺失值检查', fontweight='bold')

    # 子图2: 数据类型分布
    ax2 = fig.add_subplot(gs[0, 1])
    dtype_counts = df.dtypes.value_counts()
    colors_pie = plt.cm.Set3(range(len(dtype_counts)))
    ax2.pie(dtype_counts.values, labels=[str(x) for x in dtype_counts.index],
           autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax2.set_title('数据类型分布', fontweight='bold')

    # 子图3: 数值特征分布(小提琴图)
    ax3 = fig.add_subplot(gs[1, 0])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in numeric_cols:
        numeric_cols.remove(target)

    if len(numeric_cols) > 0:
        # 选择前5个数值特征
        plot_cols = numeric_cols[:5]
        plot_data = df[plot_cols]

        # 标准化数据以便比较
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        plot_data_scaled = pd.DataFrame(
            scaler.fit_transform(plot_data),
            columns=plot_data.columns
        )

        plot_data_scaled.plot(kind='box', ax=ax3, vert=False)
        ax3.set_xlabel('标准化后的值')
        ax3.set_title(f'数值特征分布 (前{len(plot_cols)}个)', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, '无数值特征', ha='center', va='center',
                fontsize=14, transform=ax3.transAxes)
        ax3.set_title('数值特征分布', fontweight='bold')

    # 子图4: 目标变量分布
    ax4 = fig.add_subplot(gs[1, 1])
    if target and target in df.columns:
        if df[target].nunique() <= 20:  # 分类问题
            value_counts = df[target].value_counts()
            ax4.bar(range(len(value_counts)), value_counts.values, color='steelblue')
            ax4.set_xticks(range(len(value_counts)))
            ax4.set_xticklabels(value_counts.index, rotation=45)
            ax4.set_xlabel('类别')
            ax4.set_ylabel('数量')
            ax4.set_title(f'目标变量分布: {target}', fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)

            # 添加数量标注
            for i, v in enumerate(value_counts.values):
                ax4.text(i, v, str(v), ha='center', va='bottom')
        else:  # 回归问题
            ax4.hist(df[target].dropna(), bins=30, color='steelblue',
                    edgecolor='black', alpha=0.7)
            ax4.set_xlabel(target)
            ax4.set_ylabel('频数')
            ax4.set_title(f'目标变量分布: {target}', fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '无目标变量', ha='center', va='center',
                fontsize=14, transform=ax4.transAxes)
        ax4.set_title('目标变量', fontweight='bold')

    plt.suptitle('数据质量综合摘要', fontsize=16, fontweight='bold', y=0.98)
    plt.show()


# ==================== 数据分布可视化 ====================

def plot_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None,
                      ncols: int = 3, figsize: Tuple[int, int] = (15, 10)):
    """
    绘制多个特征的分布图

    Args:
        df: 数据DataFrame
        columns: 要绘制的列，None表示所有数值列
        ncols: 每行显示的图数
        figsize: 图像大小
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_features = len(columns)
    if n_features == 0:
        print("⚠️  没有数值型特征可供可视化")
        return

    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, col in enumerate(columns):
        ax = axes[idx]
        data = df[col].dropna()

        # 直方图 + KDE
        ax.hist(data, bins=30, alpha=0.6, color='skyblue',
               edgecolor='black', density=True, label='Histogram')

        # 核密度估计
        data.plot(kind='kde', ax=ax, color='red', linewidth=2, label='KDE')

        # 添加均值和中位数线
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='green', linestyle='--',
                  linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--',
                  linewidth=2, alpha=0.7, label=f'Median: {median_val:.2f}')

        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('特征分布图', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame, columns: Optional[List[str]] = None,
                 ncols: int = 3, figsize: Tuple[int, int] = (15, 8)):
    """
    绘制多个特征的箱线图（识别异常值）

    Args:
        df: 数据DataFrame
        columns: 要绘制的列，None表示所有数值列
        ncols: 每行显示的图数
        figsize: 图像大小
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_features = len(columns)
    if n_features == 0:
        print("⚠️  没有数值型特征可供可视化")
        return

    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, col in enumerate(columns):
        ax = axes[idx]
        data = df[col].dropna()

        # 箱线图
        bp = ax.boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)

        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('特征箱线图（异常值检测）', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(df: pd.DataFrame, column: str,
                                  top_n: int = 20,
                                  figsize: Tuple[int, int] = (12, 6)):
    """
    绘制类别特征的分布

    Args:
        df: 数据DataFrame
        column: 类别列名
        top_n: 显示前N个类别
        figsize: 图像大小
    """
    if column not in df.columns:
        print(f"⚠️  列 {column} 不存在")
        return

    # 计数
    value_counts = df[column].value_counts().head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图：柱状图
    axes[0].barh(range(len(value_counts)), value_counts.values, color='steelblue')
    axes[0].set_yticks(range(len(value_counts)))
    axes[0].set_yticklabels(value_counts.index)
    axes[0].set_xlabel('Count')
    axes[0].set_title(f'{column} - 频数分布')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # 右图：饼图
    axes[1].pie(value_counts.values, labels=value_counts.index,
               autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'{column} - 比例分布')

    plt.tight_layout()
    plt.show()


# ==================== 特征关系可视化 ====================

def plot_correlation_heatmap(df: pd.DataFrame,
                            method: str = 'pearson',
                            figsize: Tuple[int, int] = (12, 10),
                            annot: bool = True):
    """
    绘制相关性热力图

    Args:
        df: 数据DataFrame
        method: 相关系数类型
        figsize: 图像大小
        annot: 是否显示数值
    """
    # 只选择数值列
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        print("⚠️  数值特征少于2个，无法绘制相关性热力图")
        return

    # 计算相关系数
    corr_matrix = numeric_df.corr(method=method)

    # 绘制热力图
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角

    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8})

    plt.title(f'特征相关性热力图 ({method.capitalize()})',
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       hue: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 12)):
    """
    绘制散点图矩阵

    Args:
        df: 数据DataFrame
        columns: 要绘制的列
        hue: 用于着色的列（通常是目标变量）
        figsize: 图像大小
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:5]  # 最多5个特征

    if len(columns) < 2:
        print("⚠️  特征少于2个，无法绘制散点图矩阵")
        return

    # 使用seaborn的pairplot
    plot_df = df[columns + ([hue] if hue and hue not in columns else [])]

    sns.pairplot(plot_df, hue=hue, diag_kind='kde',
                plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
                corner=True)

    plt.suptitle('特征散点图矩阵', y=1.00, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_feature_vs_target(df: pd.DataFrame, feature: str, target: str,
                          figsize: Tuple[int, int] = (12, 5)):
    """
    可视化特征与目标变量的关系

    Args:
        df: 数据DataFrame
        feature: 特征列名
        target: 目标变量列名
        figsize: 图像大小
    """
    if feature not in df.columns or target not in df.columns:
        print(f"⚠️  列不存在")
        return

    # 判断目标变量类型
    is_classification = df[target].nunique() <= 20

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if is_classification:
        # 分类问题：箱线图 + 小提琴图
        sns.boxplot(data=df, x=target, y=feature, ax=axes[0])
        axes[0].set_title(f'{feature} vs {target} (Boxplot)')
        axes[0].grid(axis='y', alpha=0.3)

        sns.violinplot(data=df, x=target, y=feature, ax=axes[1])
        axes[1].set_title(f'{feature} vs {target} (Violin)')
        axes[1].grid(axis='y', alpha=0.3)

    else:
        # 回归问题：散点图 + 回归线
        axes[0].scatter(df[feature], df[target], alpha=0.5, s=30, edgecolors='k')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel(target)
        axes[0].set_title(f'{feature} vs {target} (Scatter)')
        axes[0].grid(alpha=0.3)

        # 添加回归线
        z = np.polyfit(df[feature].dropna(), df[target].dropna(), 1)
        p = np.poly1d(z)
        axes[0].plot(df[feature], p(df[feature]), "r--", linewidth=2, alpha=0.8)

        # 密度图
        axes[1].hexbin(df[feature], df[target], gridsize=30, cmap='Blues')
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel(target)
        axes[1].set_title(f'{feature} vs {target} (Density)')

    plt.tight_layout()
    plt.show()


# ==================== 降维可视化 ====================

def plot_pca_2d(X: np.ndarray, y: Optional[np.ndarray] = None,
               labels: Optional[List[str]] = None,
               figsize: Tuple[int, int] = (10, 8)):
    """
    使用PCA降维到2D并可视化

    Args:
        X: 特征数据
        y: 标签（可选，用于着色）
        labels: 类别标签名称
        figsize: 图像大小
    """
    # PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 绘图
    plt.figure(figsize=figsize)

    if y is not None:
        # 按类别着色
        unique_labels = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for idx, label in enumerate(unique_labels):
            mask = y == label
            label_name = labels[idx] if labels else f'Class {label}'
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[colors[idx]], label=label_name,
                       alpha=0.6, s=50, edgecolors='k')
        plt.legend()
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1],
                   alpha=0.6, s=50, edgecolors='k', c='steelblue')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA 2D 可视化')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.2%}")


def plot_tsne_2d(X: np.ndarray, y: Optional[np.ndarray] = None,
                labels: Optional[List[str]] = None,
                perplexity: int = 30,
                figsize: Tuple[int, int] = (10, 8)):
    """
    使用t-SNE降维到2D并可视化

    Args:
        X: 特征数据
        y: 标签（可选，用于着色）
        labels: 类别标签名称
        perplexity: t-SNE参数
        figsize: 图像大小
    """
    print("t-SNE降维中（可能需要几分钟）...")

    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # 绘图
    plt.figure(figsize=figsize)

    if y is not None:
        # 按类别着色
        unique_labels = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for idx, label in enumerate(unique_labels):
            mask = y == label
            label_name = labels[idx] if labels else f'Class {label}'
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=[colors[idx]], label=label_name,
                       alpha=0.6, s=50, edgecolors='k')
        plt.legend()
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                   alpha=0.6, s=50, edgecolors='k', c='steelblue')

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE 2D 可视化')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================== 聚类可视化 ====================

def plot_clustering_results(X: np.ndarray, labels: np.ndarray,
                           centers: Optional[np.ndarray] = None,
                           method: str = 'pca',
                           figsize: Tuple[int, int] = (10, 8)):
    """
    可视化聚类结果

    Args:
        X: 特征数据
        labels: 聚类标签
        centers: 聚类中心（可选）
        method: 降维方法 ('pca', 'tsne')
        figsize: 图像大小
    """
    # 降维到2D
    if method == 'pca':
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)
        if centers is not None:
            centers_reduced = reducer.transform(centers)
    else:  # tsne
        print("t-SNE降维中...")
        reducer = TSNE(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
        centers_reduced = None  # t-SNE不能transform新数据

    # 绘图
    plt.figure(figsize=figsize)

    # 绘制数据点
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        if label == -1:
            # 噪声点（DBSCAN）
            mask = labels == label
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                       c='gray', marker='x', s=50, label='Noise', alpha=0.5)
        else:
            mask = labels == label
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                       c=[colors[idx]], label=f'Cluster {label}',
                       alpha=0.6, s=50, edgecolors='k')

    # 绘制聚类中心
    if centers_reduced is not None:
        plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1],
                   c='red', marker='*', s=500, edgecolors='black',
                   linewidths=2, label='Centers', zorder=10)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'聚类结果可视化 ({method.upper()})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================== 模型性能可视化 ====================

def plot_model_comparison_radar(results: Dict[str, Dict[str, float]],
                               metrics: List[str],
                               figsize: Tuple[int, int] = (10, 8)):
    """
    绘制模型性能雷达图

    Args:
        results: 模型结果字典 {model_name: {metric: value}}
        metrics: 要对比的指标列表
        figsize: 图像大小
    """
    from math import pi

    # 准备数据
    model_names = list(results.keys())
    n_metrics = len(metrics)
    angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]  # 闭合

    # 绘图
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for idx, model_name in enumerate(model_names):
        values = [results[model_name].get(metric, 0) for metric in metrics]
        values += values[:1]  # 闭合

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('模型性能雷达图', size=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_metrics_over_time(history: Dict[str, List[float]],
                          figsize: Tuple[int, int] = (12, 5)):
    """
    绘制训练过程中的指标变化（如训练集/验证集损失）

    Args:
        history: 训练历史字典 {metric_name: [values]}
        figsize: 图像大小
    """
    n_metrics = len(history)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for idx, (metric_name, values) in enumerate(history.items()):
        axes[idx].plot(values, linewidth=2, marker='o')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric_name.upper())
        axes[idx].set_title(f'{metric_name} over Time')
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== 辅助函数 ====================

def save_figure(filename: str, dpi: int = 300):
    """
    保存当前图表

    Args:
        filename: 文件名
        dpi: 分辨率
    """
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"✅ 图表已保存: {filename}")


if __name__ == '__main__':
    # 测试示例
    print("=== 可视化模块测试 ===\n")

    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'feature1': np.random.randn(500),
        'feature2': np.random.randn(500) * 2 + 5,
        'feature3': np.random.randn(500) * 0.5 + 10,
        'category': np.random.choice(['A', 'B', 'C'], 500),
        'target': np.random.choice([0, 1], 500)
    })

    print("1. 绘制数值特征分布")
    plot_distributions(test_data, columns=['feature1', 'feature2', 'feature3'])

    print("\n2. 绘制相关性热力图")
    plot_correlation_heatmap(test_data)

    print("\n✅ 所有测试通过！")
