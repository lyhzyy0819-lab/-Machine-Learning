"""
数据预处理模块
负责数据清洗、异常值处理、缺失值填充等
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer


def remove_outliers(df: pd.DataFrame,
                    columns: list,
                    method: str = 'iqr',
                    threshold: float = 3.0) -> pd.DataFrame:
    """
    移除异常值

    Args:
        df: 输入DataFrame
        columns: 要检测异常值的列名列表
        method: 异常值检测方法 ('iqr' 或 'zscore')
        threshold: 阈值（IQR方法为倍数，Z-score方法为标准差倍数）

    Returns:
        移除异常值后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    df_clean = df.copy()
    initial_len = len(df_clean)

    for col in columns:
        if col not in df_clean.columns:
            logger.warning(f"列 '{col}' 不存在，跳过")
            continue

        before_len = len(df_clean)

        if method == 'iqr':
            # IQR方法
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            df_clean = df_clean[
                (df_clean[col] >= lower_bound) &
                (df_clean[col] <= upper_bound)
            ]

        elif method == 'zscore':
            # Z-score方法
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            z_scores = np.abs((df_clean[col] - mean) / std)
            df_clean = df_clean[z_scores < threshold]

        after_len = len(df_clean)
        removed = before_len - after_len

        if removed > 0:
            logger.info(f"  列 '{col}': 移除 {removed} 个异常值 "
                       f"({removed/before_len*100:.2f}%)")

    total_removed = initial_len - len(df_clean)
    logger.info(f"总共移除 {total_removed} 行异常值 "
               f"({total_removed/initial_len*100:.2f}%)")

    return df_clean


def filter_by_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据配置的边界过滤数据

    Args:
        df: 输入DataFrame

    Returns:
        过滤后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("根据边界条件过滤数据...")

    df_filtered = df.copy()
    initial_len = len(df_filtered)

    # 地理坐标过滤
    logger.info("  过滤地理坐标...")
    df_filtered = df_filtered[
        (df_filtered['pickup_longitude'] >= config.NYC_BOUNDS['min_longitude']) &
        (df_filtered['pickup_longitude'] <= config.NYC_BOUNDS['max_longitude']) &
        (df_filtered['pickup_latitude'] >= config.NYC_BOUNDS['min_latitude']) &
        (df_filtered['pickup_latitude'] <= config.NYC_BOUNDS['max_latitude']) &
        (df_filtered['dropoff_longitude'] >= config.NYC_BOUNDS['min_longitude']) &
        (df_filtered['dropoff_longitude'] <= config.NYC_BOUNDS['max_longitude']) &
        (df_filtered['dropoff_latitude'] >= config.NYC_BOUNDS['min_latitude']) &
        (df_filtered['dropoff_latitude'] <= config.NYC_BOUNDS['max_latitude'])
    ]

    geo_removed = initial_len - len(df_filtered)
    logger.info(f"    移除 {geo_removed} 行超出NYC范围的数据")

    # 行程时长过滤
    if 'trip_duration' in df_filtered.columns:
        logger.info("  过滤行程时长...")
        before_len = len(df_filtered)
        df_filtered = df_filtered[
            (df_filtered['trip_duration'] >= config.TRIP_DURATION_LIMITS['min']) &
            (df_filtered['trip_duration'] <= config.TRIP_DURATION_LIMITS['max'])
        ]
        duration_removed = before_len - len(df_filtered)
        logger.info(f"    移除 {duration_removed} 行异常行程时长")

    # 乘客数量过滤
    if 'passenger_count' in df_filtered.columns:
        logger.info("  过滤乘客数量...")
        before_len = len(df_filtered)
        df_filtered = df_filtered[
            (df_filtered['passenger_count'] >= config.PASSENGER_COUNT_LIMITS['min']) &
            (df_filtered['passenger_count'] <= config.PASSENGER_COUNT_LIMITS['max'])
        ]
        passenger_removed = before_len - len(df_filtered)
        logger.info(f"    移除 {passenger_removed} 行异常乘客数量")

    total_removed = initial_len - len(df_filtered)
    logger.info(f"总共过滤 {total_removed} 行数据 "
               f"({total_removed/initial_len*100:.2f}%)")

    return df_filtered


def handle_missing_values(df: pd.DataFrame,
                         strategy: str = 'drop') -> pd.DataFrame:
    """
    处理缺失值

    Args:
        df: 输入DataFrame
        strategy: 处理策略 ('drop', 'fill_mean', 'fill_median', 'fill_mode')

    Returns:
        处理后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("处理缺失值...")

    df_clean = df.copy()
    missing = df_clean.isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        logger.info("  无缺失值")
        return df_clean

    logger.info(f"  发现 {total_missing} 个缺失值")

    if strategy == 'drop':
        # 删除含有缺失值的行
        before_len = len(df_clean)
        df_clean = df_clean.dropna()
        after_len = len(df_clean)
        logger.info(f"  删除 {before_len - after_len} 行含缺失值的数据")

    elif strategy in ['fill_mean', 'fill_median', 'fill_mode']:
        # 填充缺失值
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    if strategy == 'fill_mean':
                        fill_value = df_clean[col].mean()
                    elif strategy == 'fill_median':
                        fill_value = df_clean[col].median()
                    else:  # fill_mode
                        fill_value = df_clean[col].mode()[0]

                    df_clean[col].fillna(fill_value, inplace=True)
                    logger.info(f"    列 '{col}': 使用 {strategy} 填充 ({fill_value:.2f})")
                else:
                    # 分类变量使用众数填充
                    fill_value = df_clean[col].mode()[0]
                    df_clean[col].fillna(fill_value, inplace=True)
                    logger.info(f"    列 '{col}': 使用众数填充 ({fill_value})")

    return df_clean


def convert_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    转换数据类型以优化内存

    Args:
        df: 输入DataFrame

    Returns:
        转换后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("转换数据类型...")

    df_converted = df.copy()

    # 转换日期时间类型
    datetime_cols = ['pickup_datetime', 'dropoff_datetime']
    for col in datetime_cols:
        if col in df_converted.columns and df_converted[col].dtype != 'datetime64[ns]':
            df_converted[col] = pd.to_datetime(df_converted[col])
            logger.info(f"  '{col}' 转换为 datetime64")

    # 转换分类变量
    categorical_cols = ['vendor_id', 'store_and_fwd_flag']
    for col in categorical_cols:
        if col in df_converted.columns:
            df_converted[col] = df_converted[col].astype('category')
            logger.info(f"  '{col}' 转换为 category")

    # 转换数值类型
    if 'passenger_count' in df_converted.columns:
        df_converted['passenger_count'] = df_converted['passenger_count'].astype('int8')
        logger.info("  'passenger_count' 转换为 int8")

    # 转换浮点数精度
    float_cols = [
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
    ]
    for col in float_cols:
        if col in df_converted.columns and df_converted[col].dtype == 'float64':
            df_converted[col] = df_converted[col].astype('float32')
            logger.info(f"  '{col}' 转换为 float32")

    return df_converted


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    移除重复行

    Args:
        df: 输入DataFrame

    Returns:
        移除重复后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("检查重复数据...")

    before_len = len(df)
    df_dedup = df.drop_duplicates()
    after_len = len(df_dedup)
    removed = before_len - after_len

    if removed > 0:
        logger.info(f"  移除 {removed} 行重复数据 ({removed/before_len*100:.2f}%)")
    else:
        logger.info("  无重复数据")

    return df_dedup


def preprocess_data(df: pd.DataFrame,
                   remove_outliers_flag: bool = True,
                   outlier_method: str = 'iqr') -> pd.DataFrame:
    """
    完整的数据预处理流程

    Args:
        df: 输入DataFrame
        remove_outliers_flag: 是否移除异常值
        outlier_method: 异常值检测方法

    Returns:
        预处理后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("=" * 60)
    logger.info("开始数据预处理")
    logger.info("=" * 60)
    logger.info(f"输入数据形状: {df.shape}")

    with Timer("数据预处理"):
        # 1. 转换数据类型
        df_clean = convert_datatypes(df)

        # 2. 移除重复
        df_clean = remove_duplicates(df_clean)

        # 3. 处理缺失值
        df_clean = handle_missing_values(df_clean, strategy='drop')

        # 4. 边界过滤
        df_clean = filter_by_bounds(df_clean)

        # 5. 移除异常值（可选）
        if remove_outliers_flag and 'trip_duration' in df_clean.columns:
            logger.info("移除异常值...")
            df_clean = remove_outliers(
                df_clean,
                columns=['trip_duration'],
                method=outlier_method,
                threshold=3.0
            )

        # 重置索引
        df_clean = df_clean.reset_index(drop=True)

        logger.info(f"\n预处理完成！")
        logger.info(f"输出数据形状: {df_clean.shape}")
        logger.info(f"保留比例: {len(df_clean)/len(df)*100:.2f}%")

    return df_clean


def split_features_target(df: pd.DataFrame,
                         target_col: str = 'trip_duration') -> Tuple[pd.DataFrame, pd.Series]:
    """
    分离特征和目标变量

    Args:
        df: 输入DataFrame
        target_col: 目标变量列名

    Returns:
        (特征DataFrame, 目标Series)
    """
    logger = logging.getLogger("NYC_Taxi")

    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在")

    # 排除的列（ID、目标变量、时间戳等）
    exclude_cols = [
        'id', target_col, 'pickup_datetime', 'dropoff_datetime'
    ]

    # 选择特征列
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    logger.info(f"特征数量: {X.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    logger.info(f"目标变量: {target_col}")

    return X, y


if __name__ == '__main__':
    # 测试数据预处理
    from src.utils import setup_logger
    from src.data_loader import generate_sample_data

    logger = setup_logger("NYC_Taxi", config.LOG_FILE, "INFO")

    print("=" * 60)
    print("数据预处理模块测试")
    print("=" * 60)

    # 生成测试数据
    print("\n1. 生成测试数据")
    df = generate_sample_data(1000)

    # 添加一些异常值和缺失值用于测试
    df.loc[0:10, 'trip_duration'] = 10000  # 异常值
    df.loc[20:25, 'passenger_count'] = np.nan  # 缺失值
    df.loc[30:35, 'pickup_longitude'] = -80  # 超出范围

    print(f"原始数据形状: {df.shape}")
    print(f"缺失值数量: {df.isnull().sum().sum()}")

    # 数据预处理
    print("\n2. 执行预处理")
    df_clean = preprocess_data(df, remove_outliers_flag=True)

    print(f"\n清洗后数据形状: {df_clean.shape}")
    print(f"缺失值数量: {df_clean.isnull().sum().sum()}")

    # 分离特征和目标
    print("\n3. 分离特征和目标")
    X, y = split_features_target(df_clean)
    print(f"特征形状: {X.shape}")
    print(f"目标形状: {y.shape}")

    print("\n数据预处理模块测试完成！")
