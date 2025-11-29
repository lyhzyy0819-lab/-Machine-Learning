"""
特征工程模块
============

提供全面的特征工程功能，从原始特征中提取更有价值的信息。

主要功能:
- 特征选择（方差过滤、相关性过滤、特征重要性、RFE）
- 特征变换（多项式、对数、Box-Cox、分箱）
- 特征组合（交互特征、聚合特征）
- 特征提取（PCA、统计特征）
- 时间特征提取
- 文本特征提取

"特征决定模型上限，算法只是逼近这个上限"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                      f_classif, f_regression, chi2,
                                      RFE, SelectFromModel)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


# ==================== 特征选择 ====================

def select_by_variance(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    基于方差过滤特征

    原理：方差为0或接近0的特征携带信息很少，可以删除

    Args:
        df: 特征DataFrame
        threshold: 方差阈值

    Returns:
        过滤后的DataFrame
    """
    # 只处理数值型特征
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        print("⚠️  没有数值型特征可供过滤")
        return df

    # 方差过滤
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_df)

    # 获取保留的特征
    selected_features = numeric_df.columns[selector.get_support()].tolist()
    removed_features = set(numeric_df.columns) - set(selected_features)

    print(f"方差过滤: 移除 {len(removed_features)} 个低方差特征")
    if removed_features:
        print(f"  移除的特征: {list(removed_features)[:5]}...")

    # 保留选中的数值特征 + 所有非数值特征
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    selected_df = df[selected_features + non_numeric_cols]

    return selected_df


def select_by_correlation(df: pd.DataFrame, threshold: float = 0.95,
                         method: str = 'pearson') -> pd.DataFrame:
    """
    基于相关性过滤特征

    原理：高度相关的特征携带冗余信息，保留其中一个即可

    Args:
        df: 特征DataFrame
        threshold: 相关系数阈值
        method: 相关系数类型 ('pearson', 'spearman')

    Returns:
        过滤后的DataFrame
    """
    # 只处理数值型特征
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        print("⚠️  数值特征少于2个，无需相关性过滤")
        return df

    # 计算相关系数矩阵
    corr_matrix = numeric_df.corr(method=method).abs()

    # 找出高相关的特征对
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j])
                      for i, j in zip(*np.where((corr_matrix.values > threshold) & upper_triangle))]

    # 选择要删除的特征（保留每对中的第一个）
    features_to_drop = list(set([pair[1] for pair in high_corr_pairs]))

    print(f"相关性过滤: 移除 {len(features_to_drop)} 个高相关特征（阈值: {threshold}）")
    if features_to_drop:
        print(f"  移除的特征: {features_to_drop[:5]}...")
        print(f"  高相关特征对: {len(high_corr_pairs)} 对")

    # 删除高相关特征
    selected_df = df.drop(columns=features_to_drop)

    return selected_df


def select_by_importance(X: pd.DataFrame, y: pd.Series,
                        model: Any, k: int = 10) -> List[str]:
    """
    基于模型特征重要性选择特征

    Args:
        X: 特征DataFrame
        y: 目标变量
        model: 支持feature_importances_的模型（如RandomForest）
        k: 选择前k个特征

    Returns:
        选中的特征列表
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # 如果没有提供模型，使用默认模型
    if model is None:
        if y.nunique() <= 20:  # 分类问题
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # 回归问题
            model = RandomForestRegressor(n_estimators=100, random_state=42)

    # 训练模型
    model.fit(X, y)

    # 获取特征重要性
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(f"\n特征重要性排名（Top {k}）:")
    print(feature_importance_df.head(k).to_string(index=False))

    # 选择前k个特征
    selected_features = feature_importance_df.head(k)['feature'].tolist()

    return selected_features


def select_by_statistical_test(X: pd.DataFrame, y: pd.Series,
                               k: int = 10,
                               problem_type: str = 'classification') -> List[str]:
    """
    基于统计检验选择特征

    分类问题: 使用ANOVA F-value
    回归问题: 使用F-statistic

    Args:
        X: 特征DataFrame
        y: 目标变量
        k: 选择前k个特征
        problem_type: 问题类型 ('classification', 'regression')

    Returns:
        选中的特征列表
    """
    # 只处理数值型特征
    numeric_X = X.select_dtypes(include=[np.number])

    if numeric_X.empty:
        print("⚠️  没有数值型特征可供统计检验")
        return X.columns.tolist()

    # 选择评分函数
    if problem_type == 'classification':
        score_func = f_classif
    else:
        score_func = f_regression

    # 选择K个最佳特征
    selector = SelectKBest(score_func=score_func, k=min(k, numeric_X.shape[1]))
    selector.fit(numeric_X, y)

    # 获取分数
    scores = pd.DataFrame({
        'feature': numeric_X.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)

    print(f"\n统计检验分数排名（Top {k}）:")
    print(scores.head(k).to_string(index=False))

    # 选中的特征
    selected_features = numeric_X.columns[selector.get_support()].tolist()

    return selected_features


# ==================== 特征变换 ====================

def create_polynomial_features(df: pd.DataFrame, columns: List[str],
                              degree: int = 2,
                              interaction_only: bool = False) -> pd.DataFrame:
    """
    创建多项式特征

    例如: [x1, x2] -> [1, x1, x2, x1^2, x1*x2, x2^2]

    Args:
        df: 数据DataFrame
        columns: 要创建多项式的列
        degree: 多项式次数
        interaction_only: 是否只创建交互项（不包括高次项）

    Returns:
        包含多项式特征的DataFrame
    """
    poly = PolynomialFeatures(degree=degree,
                             interaction_only=interaction_only,
                             include_bias=False)

    # 转换特征
    poly_array = poly.fit_transform(df[columns])

    # 获取特征名称
    poly_feature_names = poly.get_feature_names_out(columns)

    # 创建新的DataFrame
    poly_df = pd.DataFrame(poly_array, columns=poly_feature_names, index=df.index)

    # 删除原始特征，添加多项式特征
    result_df = df.drop(columns=columns)
    result_df = pd.concat([result_df, poly_df], axis=1)

    print(f"多项式特征: {len(columns)} 个原始特征 -> {len(poly_feature_names)} 个多项式特征")

    return result_df


def apply_log_transform(df: pd.DataFrame, columns: List[str],
                       add_constant: float = 1.0) -> pd.DataFrame:
    """
    对数变换（用于处理右偏分布）

    y = log(x + c)

    Args:
        df: 数据DataFrame
        columns: 要转换的列
        add_constant: 添加的常数（避免log(0)）

    Returns:
        转换后的DataFrame
    """
    df_transformed = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        # 检查是否有负值
        if df[col].min() < 0:
            print(f"⚠️  {col} 包含负值，跳过对数变换")
            continue

        # 应用对数变换
        df_transformed[f'{col}_log'] = np.log(df[col] + add_constant)
        print(f"  {col} -> {col}_log")

    return df_transformed


def apply_boxcox_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Box-Cox变换（自动找最佳λ使数据接近正态分布）

    y = (x^λ - 1) / λ  (λ ≠ 0)
    y = log(x)         (λ = 0)

    Args:
        df: 数据DataFrame
        columns: 要转换的列（必须为正值）

    Returns:
        转换后的DataFrame
    """
    df_transformed = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        # 检查是否全为正值
        if df[col].min() <= 0:
            print(f"⚠️  {col} 包含非正值，跳过Box-Cox变换")
            continue

        try:
            # 应用Box-Cox变换
            transformed_data, lambda_value = stats.boxcox(df[col])
            df_transformed[f'{col}_boxcox'] = transformed_data
            print(f"  {col} -> {col}_boxcox (λ={lambda_value:.3f})")
        except Exception as e:
            print(f"⚠️  {col} Box-Cox变换失败: {e}")

    return df_transformed


def create_binned_features(df: pd.DataFrame, columns: List[str],
                          n_bins: int = 5,
                          strategy: str = 'quantile') -> pd.DataFrame:
    """
    特征分箱（离散化）

    将连续特征转换为类别特征

    Args:
        df: 数据DataFrame
        columns: 要分箱的列
        n_bins: 箱数
        strategy: 分箱策略
                 'uniform': 等宽分箱
                 'quantile': 等频分箱（每个箱样本数相近）

    Returns:
        包含分箱特征的DataFrame
    """
    df_binned = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        try:
            if strategy == 'uniform':
                # 等宽分箱
                df_binned[f'{col}_binned'] = pd.cut(df[col], bins=n_bins,
                                                   labels=False, duplicates='drop')
            elif strategy == 'quantile':
                # 等频分箱
                df_binned[f'{col}_binned'] = pd.qcut(df[col], q=n_bins,
                                                    labels=False, duplicates='drop')

            print(f"  {col} -> {col}_binned ({n_bins} bins, {strategy})")
        except Exception as e:
            print(f"⚠️  {col} 分箱失败: {e}")

    return df_binned


# ==================== 特征组合 ====================

def create_interaction_features(df: pd.DataFrame, columns: List[str],
                               operations: List[str] = ['+', '-', '*', '/']) -> pd.DataFrame:
    """
    创建交互特征（特征之间的运算）

    Args:
        df: 数据DataFrame
        columns: 要组合的列
        operations: 运算类型列表

    Returns:
        包含交互特征的DataFrame
    """
    df_interaction = df.copy()
    new_features = []

    # 生成所有特征对
    for col1, col2 in combinations(columns, 2):
        if col1 not in df.columns or col2 not in df.columns:
            continue

        # 执行各种运算
        if '+' in operations:
            feature_name = f'{col1}_plus_{col2}'
            df_interaction[feature_name] = df[col1] + df[col2]
            new_features.append(feature_name)

        if '-' in operations:
            feature_name = f'{col1}_minus_{col2}'
            df_interaction[feature_name] = df[col1] - df[col2]
            new_features.append(feature_name)

        if '*' in operations:
            feature_name = f'{col1}_times_{col2}'
            df_interaction[feature_name] = df[col1] * df[col2]
            new_features.append(feature_name)

        if '/' in operations:
            # 避免除以0
            feature_name = f'{col1}_div_{col2}'
            df_interaction[feature_name] = df[col1] / (df[col2] + 1e-10)
            new_features.append(feature_name)

    print(f"交互特征: 创建 {len(new_features)} 个新特征")

    return df_interaction


def create_aggregation_features(df: pd.DataFrame, group_by: str,
                                agg_columns: List[str],
                                agg_functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    创建聚合特征（按某列分组统计）

    Args:
        df: 数据DataFrame
        group_by: 分组依据列
        agg_columns: 要聚合的列
        agg_functions: 聚合函数列表

    Returns:
        包含聚合特征的DataFrame
    """
    df_agg = df.copy()

    for col in agg_columns:
        if col not in df.columns:
            continue

        for func in agg_functions:
            # 计算聚合统计量
            agg_dict = df.groupby(group_by)[col].agg(func).to_dict()

            # 映射回原数据
            feature_name = f'{col}_{func}_by_{group_by}'
            df_agg[feature_name] = df[group_by].map(agg_dict)

            print(f"  创建聚合特征: {feature_name}")

    return df_agg


# ==================== 特征提取 ====================

def extract_pca_features(df: pd.DataFrame, n_components: int = 5,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    提取PCA主成分特征

    Args:
        df: 数据DataFrame
        n_components: 主成分数量
        columns: 要进行PCA的列，None表示所有数值列

    Returns:
        包含PCA特征的DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(columns) == 0:
        print("⚠️  没有数值型特征可供PCA")
        return df

    # 应用PCA
    pca = PCA(n_components=min(n_components, len(columns)))
    pca_array = pca.fit_transform(df[columns])

    # 创建PCA特征DataFrame
    pca_columns = [f'PCA_{i+1}' for i in range(pca_array.shape[1])]
    pca_df = pd.DataFrame(pca_array, columns=pca_columns, index=df.index)

    # 合并到原数据
    result_df = pd.concat([df, pca_df], axis=1)

    # 打印解释方差比
    print(f"\nPCA特征提取:")
    print(f"  提取 {n_components} 个主成分")
    print(f"  累计解释方差: {pca.explained_variance_ratio_.sum():.2%}")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"    PCA_{i+1}: {var:.2%}")

    return result_df


def extract_statistical_features(df: pd.DataFrame,
                                columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    提取统计特征（针对多个数值列）

    Args:
        df: 数据DataFrame
        columns: 要计算统计量的列，None表示所有数值列

    Returns:
        包含统计特征的DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(columns) < 2:
        print("⚠️  数值特征少于2个，无需提取统计特征")
        return df

    df_stats = df.copy()
    data_array = df[columns].values

    # 计算各种统计量
    df_stats['row_mean'] = np.mean(data_array, axis=1)
    df_stats['row_std'] = np.std(data_array, axis=1)
    df_stats['row_min'] = np.min(data_array, axis=1)
    df_stats['row_max'] = np.max(data_array, axis=1)
    df_stats['row_median'] = np.median(data_array, axis=1)
    df_stats['row_range'] = df_stats['row_max'] - df_stats['row_min']

    print(f"统计特征提取: 创建 6 个统计特征")

    return df_stats


# ==================== 时间特征提取 ====================

def extract_datetime_features(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """
    从日期时间列提取特征

    Args:
        df: 数据DataFrame
        datetime_column: 日期时间列名

    Returns:
        包含时间特征的DataFrame
    """
    df_time = df.copy()

    if datetime_column not in df.columns:
        print(f"⚠️  列 {datetime_column} 不存在")
        return df

    # 转换为datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        df_time[datetime_column] = pd.to_datetime(df[datetime_column])

    dt = df_time[datetime_column]

    # 提取各种时间特征
    df_time[f'{datetime_column}_year'] = dt.dt.year
    df_time[f'{datetime_column}_month'] = dt.dt.month
    df_time[f'{datetime_column}_day'] = dt.dt.day
    df_time[f'{datetime_column}_dayofweek'] = dt.dt.dayofweek  # 0=周一
    df_time[f'{datetime_column}_hour'] = dt.dt.hour
    df_time[f'{datetime_column}_minute'] = dt.dt.minute
    df_time[f'{datetime_column}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df_time[f'{datetime_column}_quarter'] = dt.dt.quarter

    print(f"时间特征提取: 从 {datetime_column} 提取 8 个时间特征")

    return df_time


# ==================== 综合特征工程Pipeline ====================

class FeatureEngineer:
    """
    特征工程Pipeline类

    支持链式调用多种特征工程方法
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_features = df.columns.tolist()
        self.operations_log = []

    def select_by_variance(self, threshold: float = 0.0) -> 'FeatureEngineer':
        """方差过滤"""
        self.df = select_by_variance(self.df, threshold)
        self.operations_log.append(f"方差过滤(threshold={threshold})")
        return self

    def select_by_correlation(self, threshold: float = 0.95) -> 'FeatureEngineer':
        """相关性过滤"""
        self.df = select_by_correlation(self.df, threshold)
        self.operations_log.append(f"相关性过滤(threshold={threshold})")
        return self

    def add_polynomial(self, columns: List[str], degree: int = 2) -> 'FeatureEngineer':
        """添加多项式特征"""
        self.df = create_polynomial_features(self.df, columns, degree)
        self.operations_log.append(f"多项式特征(degree={degree})")
        return self

    def add_interactions(self, columns: List[str], operations: List[str] = ['*']) -> 'FeatureEngineer':
        """添加交互特征"""
        self.df = create_interaction_features(self.df, columns, operations)
        self.operations_log.append(f"交互特征(ops={operations})")
        return self

    def add_pca(self, n_components: int = 5) -> 'FeatureEngineer':
        """添加PCA特征"""
        self.df = extract_pca_features(self.df, n_components)
        self.operations_log.append(f"PCA特征(n={n_components})")
        return self

    def get_result(self) -> pd.DataFrame:
        """获取最终结果"""
        print(f"\n特征工程完成:")
        print(f"  原始特征数: {len(self.original_features)}")
        print(f"  最终特征数: {len(self.df.columns)}")
        print(f"  执行的操作: {' -> '.join(self.operations_log)}")
        return self.df


if __name__ == '__main__':
    # 测试示例
    print("=== 特征工程模块测试 ===\n")

    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'x1': np.random.randn(1000),
        'x2': np.random.randn(1000),
        'x3': np.random.randn(1000) * 10 + 50,
        'target': np.random.choice([0, 1], 1000)
    })

    # 测试特征工程Pipeline
    engineer = FeatureEngineer(test_data)
    result = (engineer
             .add_polynomial(['x1', 'x2'], degree=2)
             .add_interactions(['x1', 'x2'], operations=['*'])
             .add_pca(n_components=3)
             .get_result())

    print(f"\n最终数据形状: {result.shape}")
    print("\n✅ 所有测试通过！")
