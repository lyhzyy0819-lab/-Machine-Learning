"""
特征工程代码模板库
=================

快速使用:
    from code_templates.feature_engineering_templates import (
        quick_feature_selection,
        create_interaction_features,
        create_time_features,
        create_aggregation_features,
        build_feature_engineering_pipeline
    )

    # 5-10行代码完成特征工程
    df = quick_feature_selection(df, y, method='auto')
    df = create_interaction_features(df, columns=['age', 'income'])
    df = create_time_features(df, datetime_col='signup_date')

    # 或使用一键式Pipeline
    df_engineered = build_feature_engineering_pipeline(
        df, y, level='standard', interaction_cols=['age', 'income']
    )

对应决策模板: 07_decision_templates/data_diagnosis_template.md（特征工程部分）
参考实现: 06_comprehensive_project/src/feature_engineering.py (638行)

项目定位: ML实战操作手册（非教学项目）
核心价值: 5-15分钟快速代码落地
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


# ==================== 1. 特征选择 ====================

def quick_feature_selection(df: pd.DataFrame,
                           y: pd.Series,
                           method: str = 'auto',
                           n_features: int = None,
                           problem_type: str = 'classification',
                           verbose: bool = True) -> pd.DataFrame:
    """
    快速特征选择（5分钟决策）

    Parameters
    ----------
    df : DataFrame
        特征数据
    y : Series
        目标变量
    method : {'auto', 'variance', 'correlation', 'importance'}
        'auto' - 组合使用三种方法（推荐）
        variance    - 移除低方差特征
        correlation - 移除高相关特征
        importance  - 基于模型重要性
    n_features : int, optional
        保留特征数，None则自动确定
    problem_type : {'classification', 'regression'}
        问题类型

    Returns
    -------
    DataFrame
        选择后的特征

    Examples
    --------
    >>> # 快速模式：全自动
    >>> df_selected = quick_feature_selection(df, y)
    >>> # 100列 → 30列（自动移除冗余）

    >>> # 定制模式：指定保留数量
    >>> df_selected = quick_feature_selection(df, y, n_features=20)

    Decision Logic
    --------------
    特征数 < 50   → 保留所有
    特征数 50-200 → 相关性过滤
    特征数 > 200  → 组合过滤（方差+相关+重要性）

    Notes
    -----
    - 快速模式适合降维和提升模型效率
    - 参考06章src/feature_engineering.py:34-156
    """
    df_copy = df.copy()
    initial_n_features = df_copy.shape[1]

    if verbose:
        print(f"🔍 特征选择（初始: {initial_n_features}列）...")

    # 1. 方差过滤
    if method in ['auto', 'variance']:
        numeric_df = df_copy.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(numeric_df)
            selected_cols = numeric_df.columns[selector.get_support()].tolist()
            removed_cols = set(numeric_df.columns) - set(selected_cols)

            non_numeric_cols = df_copy.select_dtypes(exclude=[np.number]).columns.tolist()
            df_copy = df_copy[selected_cols + non_numeric_cols]

            if verbose and removed_cols:
                print(f"   ✓ 方差过滤: 移除{len(removed_cols)}列")

    # 2. 相关性过滤
    if method in ['auto', 'correlation']:
        numeric_df = df_copy.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            df_copy = df_copy.drop(columns=to_drop)

            if verbose and to_drop:
                print(f"   ✓ 相关性过滤: 移除{len(to_drop)}列（|r|>0.95）")

    # 3. 基于重要性选择
    if method in ['auto', 'importance'] and n_features:
        numeric_df = df_copy.select_dtypes(include=[np.number])

        if problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        selector = SelectFromModel(model, max_features=n_features, threshold=-np.inf)
        selector.fit(numeric_df, y)

        selected_cols = numeric_df.columns[selector.get_support()].tolist()
        non_numeric_cols = df_copy.select_dtypes(exclude=[np.number]).columns.tolist()
        df_copy = df_copy[selected_cols + non_numeric_cols]

        if verbose:
            print(f"   ✓ 重要性选择: 保留Top {n_features}列")

    final_n_features = df_copy.shape[1]

    if verbose:
        print(f"✓ 特征选择完成: {initial_n_features} → {final_n_features}列\n")

    return df_copy


# ==================== 2. 特征变换 ====================

def quick_transform_skewed(df: pd.DataFrame,
                          columns: List[str] = None,
                          method: str = 'auto',
                          threshold: float = 0.5,
                          verbose: bool = True) -> pd.DataFrame:
    """
    快速偏态分布变换

    适用场景:
    - 线性模型对偏态敏感
    - 价格、收入等右偏数据
    - 需要改善数据分布

    Parameters
    ----------
    columns : list, optional
        需要变换的列名，None则自动检测偏态列
    method : {'auto', 'log', 'sqrt', 'boxcox'}
        'auto' - 自动选择最佳变换
        log     - 对数变换（右偏数据）
        sqrt    - 平方根变换（右偏数据，较温和）
        boxcox  - Box-Cox变换（自动寻找最佳lambda）
    threshold : float, default=0.5
        偏度阈值，超过则认为偏态

    Returns
    -------
    DataFrame
        变换后的数据

    Examples
    --------
    >>> # 自动检测并变换偏态特征
    >>> df_transformed = quick_transform_skewed(df)
    >>> # ✓ 偏态变换完成: price(log), income(log)

    >>> # 手动指定变换
    >>> df_transformed = quick_transform_skewed(
    ...     df,
    ...     columns=['price'],
    ...     method='log'
    ... )
    """
    df_copy = df.copy()

    if columns is None:
        # 自动检测偏态列
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = []
        for col in numeric_cols:
            if df_copy[col].min() >= 0:  # 仅检测非负列
                skewness = df_copy[col].skew()
                if abs(skewness) > threshold:
                    columns.append(col)

    if len(columns) == 0:
        if verbose:
            print("✓ 未检测到偏态特征，跳过变换\n")
        return df_copy

    transformed_info = {}

    for col in columns:
        if col not in df_copy.columns:
            continue

        if df_copy[col].min() < 0:
            if verbose:
                print(f"   ⚠️  {col}包含负值，跳过变换")
            continue

        # 避免log(0)
        if method in ['auto', 'log']:
            df_copy[col] = np.log1p(df_copy[col])
            transformed_info[col] = 'log'
        elif method == 'sqrt':
            df_copy[col] = np.sqrt(df_copy[col])
            transformed_info[col] = 'sqrt'

    if verbose:
        print("✓ 偏态变换完成:")
        for col, method in transformed_info.items():
            print(f"   {col}: {method}")
        print()

    return df_copy


# ==================== 3. 特征构造 ====================

def create_interaction_features(df: pd.DataFrame,
                               columns: List[str],
                               operations: List[str] = ['*', '/'],
                               max_features: int = 10,
                               verbose: bool = True) -> pd.DataFrame:
    """
    创建交互特征（最有效的特征工程）

    Parameters
    ----------
    columns : list
        参与交互的列名
    operations : list, default=['*', '/']
        运算类型 ['*', '/', '+', '-']
    max_features : int, default=10
        最多生成交互特征数（防止过多）

    Returns
    -------
    DataFrame
        包含原特征 + 交互特征

    Examples
    --------
    >>> # 创建2个特征的交互
    >>> df = create_interaction_features(
    ...     df,
    ...     columns=['age', 'income'],
    ...     operations=['*']
    ... )
    >>> # 新增列: age_multiply_income

    >>> # 创建多个特征的交互组合
    >>> df = create_interaction_features(
    ...     df,
    ...     columns=['age', 'income', 'education_years'],
    ...     operations=['*', '/'],
    ...     max_features=5
    ... )

    Best Practices
    --------------
    1. 选择有业务意义的交互（如: 面积 * 单价）
    2. 先做少量交互实验，验证效果后再扩展
    3. 使用特征重要性筛选有效交互

    Notes
    -----
    - 交互特征常能显著提升模型性能
    - 参考06章src/feature_engineering.py:363-408
    """
    df_copy = df.copy()

    # 确保列存在且为数值型
    valid_columns = []
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            valid_columns.append(col)

    if len(valid_columns) < 2:
        if verbose:
            print("⚠️  需要至少2个数值列创建交互特征\n")
        return df_copy

    created_features = []

    # 生成两两组合
    for col1, col2 in combinations(valid_columns, 2):
        if len(created_features) >= max_features:
            break

        for op in operations:
            if len(created_features) >= max_features:
                break

            if op == '*':
                new_col = f"{col1}_multiply_{col2}"
                df_copy[new_col] = df_copy[col1] * df_copy[col2]
                created_features.append(new_col)

            elif op == '/':
                # 避免除零
                new_col = f"{col1}_divide_{col2}"
                df_copy[new_col] = df_copy[col1] / (df_copy[col2] + 1e-8)
                created_features.append(new_col)

            elif op == '+':
                new_col = f"{col1}_plus_{col2}"
                df_copy[new_col] = df_copy[col1] + df_copy[col2]
                created_features.append(new_col)

            elif op == '-':
                new_col = f"{col1}_minus_{col2}"
                df_copy[new_col] = df_copy[col1] - df_copy[col2]
                created_features.append(new_col)

    if verbose:
        print(f"✓ 交互特征创建完成: 新增{len(created_features)}列")
        if created_features[:3]:
            print(f"   示例: {created_features[:3]}")
        print()

    return df_copy


def create_time_features(df: pd.DataFrame,
                        datetime_col: str,
                        drop_original: bool = False,
                        verbose: bool = True) -> pd.DataFrame:
    """
    时间特征提取（非常有效）

    自动提取:
    - 年、月、日、星期几
    - 是否周末、是否月初/月末
    - 季度、小时（如果有）
    - 距离参考日期的天数

    Parameters
    ----------
    datetime_col : str
        时间列名
    drop_original : bool, default=False
        是否删除原始时间列

    Returns
    -------
    DataFrame
        包含时间特征的数据

    Examples
    --------
    >>> df = create_time_features(df, datetime_col='signup_date')
    >>> # 新增: signup_year, signup_month, signup_dayofweek,
    >>> #       signup_is_weekend, signup_quarter等

    Notes
    -----
    - 时间特征常能显著提升模型性能
    - 参考06章src/feature_engineering.py:525-560
    """
    df_copy = df.copy()

    if datetime_col not in df_copy.columns:
        if verbose:
            print(f"⚠️  列'{datetime_col}'不存在\n")
        return df_copy

    # 转换为datetime
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])

    # 提取时间特征
    prefix = datetime_col
    df_copy[f'{prefix}_year'] = df_copy[datetime_col].dt.year
    df_copy[f'{prefix}_month'] = df_copy[datetime_col].dt.month
    df_copy[f'{prefix}_day'] = df_copy[datetime_col].dt.day
    df_copy[f'{prefix}_dayofweek'] = df_copy[datetime_col].dt.dayofweek
    df_copy[f'{prefix}_quarter'] = df_copy[datetime_col].dt.quarter
    df_copy[f'{prefix}_is_weekend'] = (df_copy[datetime_col].dt.dayofweek >= 5).astype(int)
    df_copy[f'{prefix}_is_month_start'] = df_copy[datetime_col].dt.is_month_start.astype(int)
    df_copy[f'{prefix}_is_month_end'] = df_copy[datetime_col].dt.is_month_end.astype(int)

    # 距离参考日期的天数
    reference_date = df_copy[datetime_col].max()
    df_copy[f'{prefix}_days_from_ref'] = (reference_date - df_copy[datetime_col]).dt.days

    created_features = [col for col in df_copy.columns if col.startswith(prefix) and col != datetime_col]

    if drop_original:
        df_copy = df_copy.drop(datetime_col, axis=1)

    if verbose:
        print(f"✓ 时间特征提取完成: 新增{len(created_features)}列")
        print(f"   {created_features[:5]}")
        print()

    return df_copy


def create_aggregation_features(df: pd.DataFrame,
                               group_col: str,
                               agg_cols: List[str],
                               agg_funcs: List[str] = ['mean', 'std', 'count'],
                               verbose: bool = True) -> pd.DataFrame:
    """
    创建聚合特征（分组统计）

    适用场景:
    - 用户级别聚合（平均订单金额、购买次数）
    - 类别级别聚合（城市平均房价）
    - 时间窗口聚合（最近7天交易量）

    Parameters
    ----------
    group_col : str
        分组列名
    agg_cols : list
        需要聚合的列名
    agg_funcs : list, default=['mean', 'std', 'count']
        聚合函数 ['mean', 'sum', 'max', 'min', 'std', 'count']

    Returns
    -------
    DataFrame
        包含聚合特征的数据

    Examples
    --------
    >>> # 创建用户级别的聚合特征
    >>> df = create_aggregation_features(
    ...     df,
    ...     group_col='user_id',
    ...     agg_cols=['order_amount', 'order_count'],
    ...     agg_funcs=['mean', 'sum', 'max']
    ... )
    >>> # 新增: user_id_order_amount_mean, user_id_order_count_sum等

    Notes
    -----
    - 聚合特征能捕捉群体特征
    - 参考06章src/feature_engineering.py:411-442
    """
    df_copy = df.copy()

    # 计算聚合特征
    agg_dict = {col: agg_funcs for col in agg_cols}
    grouped = df_copy.groupby(group_col).agg(agg_dict)

    # 重命名列
    grouped.columns = [f'{group_col}_{col}_{func}' for col, func in grouped.columns]
    grouped = grouped.reset_index()

    # 合并回原数据
    df_copy = df_copy.merge(grouped, on=group_col, how='left')

    created_features = [col for col in df_copy.columns if col.startswith(f'{group_col}_')]

    if verbose:
        print(f"✓ 聚合特征创建完成: 新增{len(created_features)}列")
        print(f"   {created_features[:3]}")
        print()

    return df_copy


# ==================== 4. 完整特征工程Pipeline ====================

def build_feature_engineering_pipeline(df: pd.DataFrame,
                                      y: pd.Series,
                                      level: str = 'basic',
                                      datetime_cols: List[str] = None,
                                      interaction_cols: List[str] = None,
                                      verbose: bool = True) -> pd.DataFrame:
    """
    一键式特征工程（10-15分钟）

    Parameters
    ----------
    level : {'basic', 'standard', 'advanced'}
        basic    - 基础特征工程（特征选择）
        standard - 标准特征工程（+交互特征）
        advanced - 高级特征工程（+聚合+时间特征）
    datetime_cols : list, optional
        时间列名列表
    interaction_cols : list, optional
        参与交互的列名

    Returns
    -------
    DataFrame
        完整特征工程后的数据

    Examples
    --------
    >>> # 快速模式
    >>> df_engineered = build_feature_engineering_pipeline(df, y, level='basic')

    >>> # 完整模式
    >>> df_engineered = build_feature_engineering_pipeline(
    ...     df, y,
    ...     level='advanced',
    ...     datetime_cols=['signup_date'],
    ...     interaction_cols=['age', 'income']
    ... )

    Pipeline Steps
    --------------
    basic:
    - 特征选择（移除低方差+高相关）

    standard:
    - 特征选择
    - 交互特征（2-3个核心特征）

    advanced:
    - 特征选择
    - 交互特征
    - 时间特征提取
    - 聚合特征（如有group_col）
    """
    df_copy = df.copy()

    if verbose:
        print("="*60)
        print(f"   特征工程Pipeline (level: {level})")
        print("="*60 + "\n")

    # 1. 特征选择
    df_copy = quick_feature_selection(df_copy, y, method='auto', verbose=verbose)

    # 2. 交互特征（standard及以上）
    if level in ['standard', 'advanced'] and interaction_cols:
        df_copy = create_interaction_features(
            df_copy,
            columns=interaction_cols,
            operations=['*', '/'],
            max_features=5,
            verbose=verbose
        )

    # 3. 时间特征（advanced）
    if level == 'advanced' and datetime_cols:
        for col in datetime_cols:
            df_copy = create_time_features(df_copy, datetime_col=col, verbose=verbose)

    if verbose:
        print("="*60)
        print(f"   特征工程完成: {df.shape[1]} → {df_copy.shape[1]}列")
        print("="*60 + "\n")

    return df_copy
