"""
数据诊断模块
============

提供全面的数据质量诊断功能，帮助快速了解数据特征和潜在问题。

主要功能:
- 基础统计信息分析
- 缺失值检测与可视化
- 异常值识别
- 数据分布分析
- 特征相关性分析
- 数据类型推断
- 数据质量报告生成

这是机器学习项目的第一步，也是最关键的一步。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ==================== 基础统计分析 ====================

def basic_info(df: pd.DataFrame, show: bool = True) -> Dict[str, Any]:
    """
    获取数据集的基础信息

    Args:
        df: 数据DataFrame
        show: 是否打印信息

    Returns:
        包含基础信息的字典
    """
    info_dict = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicated_rows': df.duplicated().sum(),
        'column_types': df.dtypes.value_counts().to_dict()
    }

    if show:
        print("=" * 60)
        print("📊 数据集基础信息")
        print("=" * 60)
        print(f"样本数量: {info_dict['n_samples']:,}")
        print(f"特征数量: {info_dict['n_features']}")
        print(f"内存占用: {info_dict['memory_usage_mb']:.2f} MB")
        print(f"重复行数: {info_dict['duplicated_rows']}")
        print(f"\n数据类型分布:")
        for dtype, count in info_dict['column_types'].items():
            print(f"  {dtype}: {count}")
        print("=" * 60 + "\n")

    return info_dict


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成每列的详细摘要信息

    Args:
        df: 数据DataFrame

    Returns:
        包含每列统计信息的DataFrame
    """
    summary = pd.DataFrame({
        '数据类型': df.dtypes,
        '缺失值数量': df.isnull().sum(),
        '缺失值比例(%)': (df.isnull().sum() / len(df) * 100).round(2),
        '唯一值数量': df.nunique(),
        '唯一值比例(%)': (df.nunique() / len(df) * 100).round(2),
    })

    # 添加数值型特征的统计信息
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            summary.loc[col, '最小值'] = df[col].min()
            summary.loc[col, '最大值'] = df[col].max()
            summary.loc[col, '均值'] = df[col].mean()
            summary.loc[col, '标准差'] = df[col].std()

    # 添加类别型特征的信息
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col in df.columns:
            # 获取最常见的值
            if df[col].nunique() > 0:
                summary.loc[col, '最常见值'] = df[col].mode()[0] if len(df[col].mode()) > 0 else None
                summary.loc[col, '最常见值频数'] = df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0

    return summary


# ==================== 缺失值分析 ====================

def missing_value_analysis(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    分析数据集中的缺失值情况

    Args:
        df: 数据DataFrame
        threshold: 缺失值比例阈值，只显示缺失率高于此值的列

    Returns:
        缺失值统计DataFrame
    """
    # 计算缺失值统计
    missing_stats = pd.DataFrame({
        '缺失数量': df.isnull().sum(),
        '缺失比例(%)': (df.isnull().sum() / len(df) * 100).round(2),
        '数据类型': df.dtypes
    })

    # 筛选缺失值大于阈值的列
    missing_stats = missing_stats[missing_stats['缺失比例(%)'] > threshold * 100]

    # 按缺失比例降序排序
    missing_stats = missing_stats.sort_values('缺失比例(%)', ascending=False)

    if len(missing_stats) > 0:
        print(f"\n🔍 发现 {len(missing_stats)} 个特征存在缺失值（阈值: {threshold*100}%）\n")
        print(missing_stats)

        # 缺失值严重程度分类
        severe = missing_stats[missing_stats['缺失比例(%)'] > 50]
        moderate = missing_stats[(missing_stats['缺失比例(%)'] > 20) &
                                (missing_stats['缺失比例(%)'] <= 50)]
        mild = missing_stats[missing_stats['缺失比例(%)'] <= 20]

        print(f"\n📈 缺失值严重程度分类:")
        print(f"  严重缺失 (>50%): {len(severe)} 个特征")
        print(f"  中度缺失 (20%-50%): {len(moderate)} 个特征")
        print(f"  轻度缺失 (<=20%): {len(mild)} 个特征")
    else:
        print("\n✅ 没有发现缺失值！")

    return missing_stats


def visualize_missing_values(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)):
    """
    可视化缺失值分布

    Args:
        df: 数据DataFrame
        figsize: 图像大小
    """
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if len(missing_cols) == 0:
        print("✅ 数据集无缺失值，无需可视化")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图：缺失值数量柱状图
    missing_cols.plot(kind='barh', ax=axes[0], color='coral')
    axes[0].set_xlabel('缺失值数量')
    axes[0].set_title('各特征缺失值数量')
    axes[0].grid(axis='x', alpha=0.3)

    # 右图：缺失值比例
    missing_ratio = (missing_cols / len(df) * 100).round(2)
    missing_ratio.plot(kind='barh', ax=axes[1], color='lightblue')
    axes[1].set_xlabel('缺失比例 (%)')
    axes[1].set_title('各特征缺失值比例')
    axes[1].grid(axis='x', alpha=0.3)

    # 在右图上标注百分比
    for i, v in enumerate(missing_ratio):
        axes[1].text(v + 0.5, i, f'{v:.1f}%', va='center')

    plt.tight_layout()
    plt.show()


# ==================== 异常值检测 ====================

def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> Tuple[np.ndarray, float, float]:
    """
    使用IQR方法检测异常值

    原理：
        IQR = Q3 - Q1（四分位距）
        下界 = Q1 - k * IQR
        上界 = Q3 + k * IQR
        超出上下界的值被视为异常值

    Args:
        series: 数据序列
        k: IQR倍数，通常取1.5（温和异常值）或3.0（极端异常值）

    Returns:
        (异常值索引数组, 下界, 上界)
    """
    # 计算四分位数
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    # 计算上下界
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    # 找出异常值的索引
    outliers = ((series < lower_bound) | (series > upper_bound)).values

    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> np.ndarray:
    """
    使用Z-Score方法检测异常值

    原理：
        Z-Score = (x - mean) / std
        |Z-Score| > threshold 的值被视为异常值

    Args:
        series: 数据序列
        threshold: Z-Score阈值，通常取3.0

    Returns:
        异常值索引数组
    """
    z_scores = np.abs(stats.zscore(series.dropna()))
    outliers = np.zeros(len(series), dtype=bool)
    outliers[series.notna()] = z_scores > threshold

    return outliers


def outlier_analysis(df: pd.DataFrame, method: str = 'iqr',
                    columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    对数值型特征进行异常值分析

    Args:
        df: 数据DataFrame
        method: 检测方法 ('iqr' 或 'zscore')
        columns: 要分析的列名列表，None表示所有数值列

    Returns:
        包含异常值信息的字典
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_report = {}

    print(f"\n🔍 异常值检测（方法: {method.upper()}）\n")
    print(f"{'特征名称':<20} {'异常值数量':<12} {'异常值比例':<12} {'异常值范围'}")
    print("-" * 70)

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()

        if len(series) == 0:
            continue

        # 检测异常值
        if method == 'iqr':
            outliers, lower, upper = detect_outliers_iqr(series)
            outlier_range = f"<{lower:.2f} or >{upper:.2f}"
        else:  # zscore
            outliers = detect_outliers_zscore(series)
            outlier_range = "|Z-Score| > 3.0"

        n_outliers = outliers.sum()
        outlier_ratio = n_outliers / len(df) * 100

        outlier_report[col] = {
            'n_outliers': n_outliers,
            'outlier_ratio': outlier_ratio,
            'outlier_indices': np.where(outliers)[0].tolist()
        }

        print(f"{col:<20} {n_outliers:<12} {outlier_ratio:>6.2f}%      {outlier_range}")

    print("-" * 70)

    return outlier_report


# ==================== 数据分布分析 ====================

def distribution_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (15, 10)):
    """
    分析并可视化数值型特征的分布

    Args:
        df: 数据DataFrame
        columns: 要分析的列名列表，None表示所有数值列
        figsize: 图像大小
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = len(columns)
    if n_cols == 0:
        print("⚠️  没有数值型特征可供分析")
        return

    # 动态计算子图布局
    n_rows = (n_cols + 2) // 3
    n_plot_cols = min(n_cols, 3)

    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=figsize)
    if n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, col in enumerate(columns):
        ax = axes[idx]
        data = df[col].dropna()

        # 绘制直方图 + KDE
        ax.hist(data, bins=30, alpha=0.6, color='skyblue', edgecolor='black', density=True)
        data.plot(kind='kde', ax=ax, color='red', linewidth=2)

        # 添加统计信息
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

        # 设置标题和标签
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('密度')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    # 打印分布统计信息
    print("\n📊 分布统计摘要\n")
    for col in columns:
        data = df[col].dropna()
        skewness = data.skew()
        kurtosis = data.kurtosis()

        print(f"{col}:")
        print(f"  偏度 (Skewness): {skewness:.3f}", end="")
        if abs(skewness) < 0.5:
            print(" - 近似对称分布")
        elif skewness > 0:
            print(" - 右偏（正偏）")
        else:
            print(" - 左偏（负偏）")

        print(f"  峰度 (Kurtosis): {kurtosis:.3f}", end="")
        if abs(kurtosis) < 0.5:
            print(" - 近似正态分布")
        elif kurtosis > 0:
            print(" - 尖峰分布")
        else:
            print(" - 平峰分布")
        print()


# ==================== 相关性分析 ====================

def correlation_analysis(df: pd.DataFrame, method: str = 'pearson',
                        threshold: float = 0.5, figsize: Tuple[int, int] = (12, 10)):
    """
    分析数值型特征之间的相关性

    Args:
        df: 数据DataFrame
        method: 相关系数类型 ('pearson', 'spearman', 'kendall')
        threshold: 强相关阈值
        figsize: 图像大小

    Returns:
        相关系数矩阵
    """
    # 只选择数值型特征
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        print("⚠️  数值型特征少于2个，无法进行相关性分析")
        return None

    # 计算相关系数矩阵
    corr_matrix = numeric_df.corr(method=method)

    # 可视化热力图
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(f'特征相关性热力图 ({method.capitalize()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 找出强相关特征对
    print(f"\n🔗 强相关特征对（|相关系数| > {threshold}）\n")
    strong_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                strong_corr.append({
                    '特征1': corr_matrix.columns[i],
                    '特征2': corr_matrix.columns[j],
                    '相关系数': corr_value
                })

    if strong_corr:
        strong_corr_df = pd.DataFrame(strong_corr).sort_values('相关系数',
                                                                key=abs,
                                                                ascending=False)
        print(strong_corr_df.to_string(index=False))
        print(f"\n💡 建议: 考虑移除{len(strong_corr)}对强相关特征中的一个，以避免多重共线性问题")
    else:
        print(f"✅ 未发现强相关特征对（阈值: {threshold}）")

    return corr_matrix


# ==================== 数据类型推断 ====================

def infer_column_types(df: pd.DataFrame,
                      categorical_threshold: int = 20) -> Dict[str, List[str]]:
    """
    智能推断每列的数据类型（数值型、类别型、ID型、日期型等）

    Args:
        df: 数据DataFrame
        categorical_threshold: 唯一值数量阈值，低于此值视为类别型

    Returns:
        分类后的列名字典
    """
    column_types = {
        'numeric': [],        # 数值型
        'categorical': [],    # 类别型
        'binary': [],         # 二元型
        'id': [],            # ID型（唯一标识）
        'datetime': [],      # 日期时间型
        'text': [],          # 文本型
        'constant': []       # 常量（只有一个值）
    }

    for col in df.columns:
        # 唯一值比例
        unique_ratio = df[col].nunique() / len(df)
        n_unique = df[col].nunique()

        # 常量检测
        if n_unique == 1:
            column_types['constant'].append(col)
            continue

        # ID检测（唯一值比例 > 95%）
        if unique_ratio > 0.95:
            column_types['id'].append(col)
            continue

        # 日期时间检测
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types['datetime'].append(col)
            continue

        # 数值型检测
        if pd.api.types.is_numeric_dtype(df[col]):
            # 二元型检测
            if n_unique == 2:
                column_types['binary'].append(col)
            else:
                column_types['numeric'].append(col)
            continue

        # 类别型 vs 文本型
        if n_unique <= categorical_threshold:
            column_types['categorical'].append(col)
        else:
            column_types['text'].append(col)

    # 打印结果
    print("\n🏷️  数据类型智能推断结果\n")
    for dtype, cols in column_types.items():
        if cols:
            print(f"{dtype.upper():.<20} {len(cols)} 个特征")
            print(f"  {', '.join(cols[:5])}" + (" ..." if len(cols) > 5 else ""))
            print()

    return column_types


# ==================== 数据质量评分 ====================

def calculate_data_quality_score(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """
    计算数据质量评分（0-100分）

    评分维度:
    1. 完整性 (30分): 缺失值情况
    2. 一致性 (20分): 数据类型、异常值
    3. 准确性 (20分): 重复值、常量列
    4. 平衡性 (15分): 目标变量分布（如有）
    5. 多样性 (15分): 特征数量、唯一值

    Args:
        df: 数据DataFrame
        target: 目标变量列名（可选）

    Returns:
        评分详情字典
    """
    scores = {}

    # 1. 完整性评分 (30分)
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    completeness_score = max(0, 30 * (1 - missing_ratio))
    scores['completeness'] = {
        'score': round(completeness_score, 2),
        'max': 30,
        'missing_ratio': round(missing_ratio * 100, 2)
    }

    # 2. 一致性评分 (20分)
    # 数据类型一致性
    type_consistency = 20
    # 检查数值列是否有字符串混入等问题（这里简化处理）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 检查是否有异常多的唯一值（可能是ID）
        if df[col].nunique() / len(df) > 0.95:
            type_consistency -= 2
    scores['consistency'] = {
        'score': max(0, round(type_consistency, 2)),
        'max': 20
    }

    # 3. 准确性评分 (20分)
    accuracy_score = 20

    # 重复行扣分
    dup_ratio = df.duplicated().sum() / len(df)
    accuracy_score -= min(10, dup_ratio * 50)

    # 常量列扣分（无信息特征）
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    accuracy_score -= min(10, len(constant_cols) * 2)

    scores['accuracy'] = {
        'score': max(0, round(accuracy_score, 2)),
        'max': 20,
        'duplicate_ratio': round(dup_ratio * 100, 2),
        'constant_columns': len(constant_cols)
    }

    # 4. 平衡性评分 (15分) - 仅当有目标变量时
    if target and target in df.columns:
        balance_score = 15
        if df[target].nunique() <= 20:  # 分类问题
            value_counts = df[target].value_counts()
            max_class_ratio = value_counts.max() / len(df)

            # 极度不平衡扣分
            if max_class_ratio > 0.9:
                balance_score = 5
            elif max_class_ratio > 0.8:
                balance_score = 8
            elif max_class_ratio > 0.7:
                balance_score = 11

            scores['balance'] = {
                'score': round(balance_score, 2),
                'max': 15,
                'max_class_ratio': round(max_class_ratio * 100, 2)
            }
        else:  # 回归问题
            scores['balance'] = {
                'score': 15,
                'max': 15,
                'note': '回归问题，无需平衡性检查'
            }
    else:
        scores['balance'] = {
            'score': 15,
            'max': 15,
            'note': '无目标变量，默认满分'
        }

    # 5. 多样性评分 (15分)
    diversity_score = 15

    # 特征数量太少扣分
    n_features = df.shape[1] - (1 if target else 0)
    if n_features < 5:
        diversity_score -= 5
    elif n_features < 10:
        diversity_score -= 2

    # 样本数量太少扣分
    if len(df) < 100:
        diversity_score -= 5
    elif len(df) < 500:
        diversity_score -= 2

    scores['diversity'] = {
        'score': max(0, round(diversity_score, 2)),
        'max': 15,
        'n_samples': len(df),
        'n_features': n_features
    }

    # 计算总分
    total_score = sum(s['score'] for s in scores.values())

    # 评级
    if total_score >= 90:
        grade = 'A (优秀)'
    elif total_score >= 80:
        grade = 'B (良好)'
    elif total_score >= 70:
        grade = 'C (中等)'
    elif total_score >= 60:
        grade = 'D (及格)'
    else:
        grade = 'F (需要改进)'

    result = {
        'total_score': round(total_score, 2),
        'grade': grade,
        'scores': scores
    }

    # 打印评分
    print("\n" + "=" * 60)
    print("📊 数据质量评分")
    print("=" * 60)
    print(f"\n总分: {total_score:.1f} / 100  -  等级: {grade}\n")
    print(f"{'维度':<15} {'得分':<10} {'满分':<10} {'详情'}")
    print("-" * 60)
    print(f"{'完整性':<15} {scores['completeness']['score']:<10.1f} {scores['completeness']['max']:<10} 缺失率: {scores['completeness']['missing_ratio']:.1f}%")
    print(f"{'一致性':<15} {scores['consistency']['score']:<10.1f} {scores['consistency']['max']:<10} 类型一致性")
    print(f"{'准确性':<15} {scores['accuracy']['score']:<10.1f} {scores['accuracy']['max']:<10} 重复率: {scores['accuracy']['duplicate_ratio']:.1f}%")

    if target and target in df.columns and 'max_class_ratio' in scores['balance']:
        print(f"{'平衡性':<15} {scores['balance']['score']:<10.1f} {scores['balance']['max']:<10} 最大类占比: {scores['balance']['max_class_ratio']:.1f}%")
    else:
        print(f"{'平衡性':<15} {scores['balance']['score']:<10.1f} {scores['balance']['max']:<10} {scores['balance'].get('note', '')}")

    print(f"{'多样性':<15} {scores['diversity']['score']:<10.1f} {scores['diversity']['max']:<10} {scores['diversity']['n_samples']}样本, {scores['diversity']['n_features']}特征")
    print("=" * 60 + "\n")

    return result


def detect_missing_pattern(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, str]:
    """
    检测缺失值模式（MCAR/MAR/MNAR）

    MCAR (Missing Completely At Random): 缺失完全随机
    MAR (Missing At Random): 缺失随机，但与其他变量相关
    MNAR (Missing Not At Random): 缺失不随机，与缺失值本身相关

    注: 这是简化的检测方法，真实的MCAR/MAR/MNAR检测需要更复杂的统计检验

    Args:
        df: 数据DataFrame
        columns: 要检测的列名列表，None表示所有有缺失值的列

    Returns:
        每列的缺失模式字典
    """
    if columns is None:
        # 只分析有缺失值的列
        columns = df.columns[df.isnull().any()].tolist()

    if len(columns) == 0:
        print("✅ 数据无缺失值，无需检测缺失模式")
        return {}

    patterns = {}

    print("\n🔍 缺失值模式检测\n")
    print(f"{'列名':<20} {'缺失率':<10} {'模式':<10} {'建议处理方法'}")
    print("-" * 70)

    for col in columns:
        if col not in df.columns:
            continue

        missing_rate = df[col].isnull().sum() / len(df)

        # 简化的模式判断逻辑
        # 1. 如果缺失率很低(<5%)，假设为MCAR
        if missing_rate < 0.05:
            pattern = 'MCAR'
            suggestion = '删除或简单填充'
        # 2. 如果缺失率中等(5%-30%)，假设为MAR
        elif missing_rate < 0.30:
            pattern = 'MAR'
            suggestion = 'KNN/迭代填充'
        # 3. 如果缺失率很高(>30%)，可能为MNAR
        else:
            pattern = 'MNAR'
            suggestion = '建模处理或删除'

        patterns[col] = pattern
        print(f"{col:<20} {missing_rate*100:>6.1f}%    {pattern:<10} {suggestion}")

    print("-" * 70 + "\n")

    return patterns


def save_diagnosis_report(report: Dict[str, Any], output_path: str, format: str = 'json'):
    """
    保存诊断报告为文件

    Args:
        report: 诊断报告字典
        output_path: 输出路径
        format: 文件格式 ('json' 或 'html')
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False, default=str)
        print(f"✅ 诊断报告已保存为JSON: {output_path}")

    elif format == 'html':
        # 保存为HTML
        html_content = _generate_html_report(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ 诊断报告已保存为HTML: {output_path}")

    else:
        raise ValueError(f"不支持的格式: {format}，请使用'json'或'html'")


def _generate_html_report(report: Dict[str, Any]) -> str:
    """
    生成HTML格式的诊断报告

    Args:
        report: 诊断报告字典

    Returns:
        HTML字符串
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>数据诊断报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .score {{ font-size: 48px; font-weight: bold; color: #4CAF50; text-align: center; margin: 20px 0; }}
            .warning {{ color: #ff9800; font-weight: bold; }}
            .error {{ color: #f44336; font-weight: bold; }}
            .success {{ color: #4CAF50; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📋 数据诊断报告</h1>

            <h2>基础信息</h2>
            <table>
                <tr><th>指标</th><th>值</th></tr>
                <tr><td>样本数量</td><td>{n_samples}</td></tr>
                <tr><td>特征数量</td><td>{n_features}</td></tr>
                <tr><td>内存占用</td><td>{memory_mb:.2f} MB</td></tr>
                <tr><td>重复行数</td><td>{duplicated_rows}</td></tr>
            </table>

            <h2>数据质量评分</h2>
            <div class="score">{quality_score:.1f} / 100</div>
            <p style="text-align: center; font-size: 24px;">{grade}</p>

            <h2>诊断建议</h2>
            <ul>
                {suggestions}
            </ul>
        </div>
    </body>
    </html>
    """

    # 提取信息
    basic_info = report.get('basic_info', {})
    quality_score = report.get('quality_score', {}).get('total_score', 0)
    grade = report.get('quality_score', {}).get('grade', 'N/A')

    suggestions_html = ""
    if 'suggestions' in report and report['suggestions']:
        for suggestion in report['suggestions']:
            suggestions_html += f"<li>{suggestion}</li>"
    else:
        suggestions_html = "<li>数据质量良好，无明显问题</li>"

    # 填充HTML模板
    html = html.format(
        n_samples=basic_info.get('n_samples', 0),
        n_features=basic_info.get('n_features', 0),
        memory_mb=basic_info.get('memory_usage_mb', 0),
        duplicated_rows=basic_info.get('duplicated_rows', 0),
        quality_score=quality_score,
        grade=grade,
        suggestions=suggestions_html
    )

    return html


# ==================== 综合诊断报告 ====================

def generate_diagnosis_report(df: pd.DataFrame, target: Optional[str] = None,
                             save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    生成完整的数据诊断报告

    Args:
        df: 数据DataFrame
        target: 目标变量列名（可选）
        save_path: 保存路径（可选），如提供则保存为JSON和HTML格式

    Returns:
        包含所有诊断信息的字典
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "📋 数据诊断报告")
    print("=" * 70 + "\n")

    report = {}

    # 1. 基础信息
    report['basic_info'] = basic_info(df, show=True)

    # 2. 数据类型推断
    report['column_types'] = infer_column_types(df)

    # 3. 缺失值分析
    report['missing_values'] = missing_value_analysis(df, threshold=0.0)

    # 4. 缺失值模式检测（新增）
    if len(report['missing_values']) > 0:
        report['missing_patterns'] = detect_missing_pattern(df)

    # 5. 目标变量分析（如果提供）
    if target and target in df.columns:
        print(f"\n🎯 目标变量分析: {target}\n")
        if df[target].dtype in [np.int64, np.float64] and df[target].nunique() > 10:
            # 回归问题
            print(f"  类型: 回归问题")
            print(f"  范围: [{df[target].min():.2f}, {df[target].max():.2f}]")
            print(f"  均值: {df[target].mean():.2f}")
            print(f"  标准差: {df[target].std():.2f}")
        else:
            # 分类问题
            print(f"  类型: 分类问题")
            print(f"  类别数: {df[target].nunique()}")
            print(f"\n  类别分布:")
            value_counts = df[target].value_counts()
            for val, count in value_counts.items():
                ratio = count / len(df) * 100
                print(f"    {val}: {count} ({ratio:.1f}%)")

            # 检查类别不平衡
            max_ratio = value_counts.max() / len(df)
            if max_ratio > 0.8:
                print(f"\n  ⚠️  警告: 检测到严重的类别不平衡（最大类占比 {max_ratio*100:.1f}%）")
                print(f"     建议: 考虑使用SMOTE、调整类别权重或使用特殊的评估指标")

    # 6. 数值特征统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(f"\n📊 数值特征描述统计\n")
        print(df[numeric_cols].describe().round(2))

    # 7. 数据质量评分（新增）
    report['quality_score'] = calculate_data_quality_score(df, target)

    # 8. 诊断建议
    print("\n" + "=" * 70)
    print("💡 诊断建议")
    print("=" * 70)

    suggestions = []

    # 缺失值建议
    if len(report['missing_values']) > 0:
        suggestions.append("✓ 处理缺失值: 建议查看缺失模式，选择删除、填充或建模方法")

    # 重复行建议
    if report['basic_info']['duplicated_rows'] > 0:
        suggestions.append(f"✓ 移除 {report['basic_info']['duplicated_rows']} 行重复数据")

    # ID列建议
    if report['column_types']['id']:
        suggestions.append(f"✓ 移除ID列: {', '.join(report['column_types']['id'][:3])}")

    # 常量列建议
    if report['column_types']['constant']:
        suggestions.append(f"✓ 移除常量列: {', '.join(report['column_types']['constant'])}")

    # 类别型特征建议
    if report['column_types']['categorical']:
        suggestions.append(f"✓ 编码类别特征: 考虑使用One-Hot或Label Encoding")

    report['suggestions'] = suggestions

    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    else:
        print("✅ 数据质量良好，无明显问题")

    print("\n" + "=" * 70 + "\n")

    # 9. 保存报告（如果指定路径）
    if save_path:
        from pathlib import Path
        save_path = Path(save_path)

        # 保存JSON格式
        json_path = save_path.parent / f"{save_path.stem}.json"
        save_diagnosis_report(report, str(json_path), format='json')

        # 保存HTML格式
        html_path = save_path.parent / f"{save_path.stem}.html"
        save_diagnosis_report(report, str(html_path), format='html')

    return report


if __name__ == '__main__':
    # 测试示例
    print("=== 数据诊断模块测试 ===\n")

    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'id': range(1000),
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000) * 10 + 50,
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })

    # 添加一些缺失值
    test_data.loc[np.random.choice(1000, 50, replace=False), 'feature1'] = np.nan

    # 添加一些异常值
    test_data.loc[np.random.choice(1000, 10, replace=False), 'feature2'] = 1000

    # 生成诊断报告
    report = generate_diagnosis_report(test_data, target='target')

    print("\n✅ 测试完成！")
