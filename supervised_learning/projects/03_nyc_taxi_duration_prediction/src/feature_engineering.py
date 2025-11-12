"""
特征工程模块
创建时间特征、地理特征和交互特征
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer


def extract_datetime_features(df: pd.DataFrame,
                              datetime_col: str = 'pickup_datetime') -> pd.DataFrame:
    """
    从日期时间列提取时间特征

    Args:
        df: 输入DataFrame
        datetime_col: 日期时间列名

    Returns:
        添加时间特征后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"从 '{datetime_col}' 提取时间特征...")

    df_feat = df.copy()

    # 确保是datetime类型
    if df_feat[datetime_col].dtype != 'datetime64[ns]':
        df_feat[datetime_col] = pd.to_datetime(df_feat[datetime_col])

    # 提取基础时间特征
    df_feat['year'] = df_feat[datetime_col].dt.year.astype('int16')
    df_feat['month'] = df_feat[datetime_col].dt.month.astype('int8')
    df_feat['day'] = df_feat[datetime_col].dt.day.astype('int8')
    df_feat['weekday'] = df_feat[datetime_col].dt.weekday.astype('int8')  # 0=Monday, 6=Sunday
    df_feat['hour'] = df_feat[datetime_col].dt.hour.astype('int8')

    logger.info("  提取基础时间特征: year, month, day, weekday, hour")

    # 周末标记
    df_feat['is_weekend'] = (df_feat['weekday'] >= 5).astype('int8')
    logger.info("  创建周末标记: is_weekend")

    # 高峰时段标记
    morning_rush = (df_feat['hour'] >= config.RUSH_HOURS['morning'][0]) & \
                   (df_feat['hour'] < config.RUSH_HOURS['morning'][1])
    evening_rush = (df_feat['hour'] >= config.RUSH_HOURS['evening'][0]) & \
                   (df_feat['hour'] < config.RUSH_HOURS['evening'][1])
    df_feat['is_rush_hour'] = (morning_rush | evening_rush).astype('int8')
    logger.info("  创建高峰时段标记: is_rush_hour")

    # 深夜时段标记
    df_feat['is_night'] = (
        (df_feat['hour'] >= config.NIGHT_HOURS[0]) &
        (df_feat['hour'] < config.NIGHT_HOURS[1])
    ).astype('int8')
    logger.info("  创建深夜标记: is_night")

    return df_feat


def calculate_haversine_distance(lon1: np.ndarray, lat1: np.ndarray,
                                 lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    使用Haversine公式计算两点间的球面距离

    Args:
        lon1, lat1: 起点经纬度
        lon2, lat2: 终点经纬度

    Returns:
        距离数组（单位：公里）
    """
    # 转换为弧度
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # 地球半径（公里）
    r = 6371

    return c * r


def calculate_manhattan_distance(lon1: np.ndarray, lat1: np.ndarray,
                                lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    计算曼哈顿距离（近似）

    Args:
        lon1, lat1: 起点经纬度
        lon2, lat2: 终点经纬度

    Returns:
        距离数组（单位：公里）
    """
    # 每度约111公里
    return (np.abs(lon2 - lon1) + np.abs(lat2 - lat1)) * 111


def calculate_bearing(lon1: np.ndarray, lat1: np.ndarray,
                     lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    计算两点间的方位角

    Args:
        lon1, lat1: 起点经纬度
        lon2, lat2: 终点经纬度

    Returns:
        方位角数组（单位：度，0-360）
    """
    # 转换为弧度
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    initial_bearing = np.arctan2(x, y)

    # 转换为度数（0-360）
    bearing = (np.degrees(initial_bearing) + 360) % 360

    return bearing


def create_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建地理特征

    Args:
        df: 输入DataFrame

    Returns:
        添加地理特征后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("创建地理特征...")

    df_feat = df.copy()

    # 提取坐标
    pickup_lon = df_feat['pickup_longitude'].values
    pickup_lat = df_feat['pickup_latitude'].values
    dropoff_lon = df_feat['dropoff_longitude'].values
    dropoff_lat = df_feat['dropoff_latitude'].values

    # 曼哈顿距离
    df_feat['manhattan_distance'] = calculate_manhattan_distance(
        pickup_lon, pickup_lat, dropoff_lon, dropoff_lat
    )
    logger.info("  计算曼哈顿距离: manhattan_distance")

    # 欧氏距离（Haversine）
    df_feat['euclidean_distance'] = calculate_haversine_distance(
        pickup_lon, pickup_lat, dropoff_lon, dropoff_lat
    )
    logger.info("  计算欧氏距离: euclidean_distance")

    # 方位角
    df_feat['bearing'] = calculate_bearing(
        pickup_lon, pickup_lat, dropoff_lon, dropoff_lat
    )
    logger.info("  计算方位角: bearing")

    # 中心坐标
    df_feat['center_latitude'] = (pickup_lat + dropoff_lat) / 2
    df_feat['center_longitude'] = (pickup_lon + dropoff_lon) / 2
    logger.info("  计算中心坐标: center_latitude, center_longitude")

    # 转换为float32以节省内存
    geo_cols = ['manhattan_distance', 'euclidean_distance', 'bearing',
                'center_latitude', 'center_longitude']
    for col in geo_cols:
        df_feat[col] = df_feat[col].astype('float32')

    return df_feat


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建交互特征

    Args:
        df: 输入DataFrame

    Returns:
        添加交互特征后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("创建交互特征...")

    df_feat = df.copy()

    # 距离 × 乘客数
    if 'euclidean_distance' in df_feat.columns and 'passenger_count' in df_feat.columns:
        df_feat['distance_per_passenger'] = (
            df_feat['euclidean_distance'] / df_feat['passenger_count']
        ).astype('float32')
        logger.info("  创建特征: distance_per_passenger")

    # 距离 × 高峰时段
    if 'euclidean_distance' in df_feat.columns and 'is_rush_hour' in df_feat.columns:
        df_feat['distance_rush_hour'] = (
            df_feat['euclidean_distance'] * df_feat['is_rush_hour']
        ).astype('float32')
        logger.info("  创建特征: distance_rush_hour")

    # 小时 × 距离
    if 'hour' in df_feat.columns and 'euclidean_distance' in df_feat.columns:
        df_feat['hour_distance'] = (
            df_feat['hour'] * df_feat['euclidean_distance']
        ).astype('float32')
        logger.info("  创建特征: hour_distance")

    return df_feat


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    完整的特征工程流程

    Args:
        df: 输入DataFrame

    Returns:
        完成特征工程后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("=" * 60)
    logger.info("开始特征工程")
    logger.info("=" * 60)

    with Timer("特征工程"):
        # 1. 时间特征
        df_feat = extract_datetime_features(df, 'pickup_datetime')

        # 2. 地理特征
        df_feat = create_geo_features(df_feat)

        # 3. 交互特征
        df_feat = create_interaction_features(df_feat)

        logger.info(f"\n特征工程完成！")
        logger.info(f"最终特征数量: {df_feat.shape[1]}")

    return df_feat


def select_features(df: pd.DataFrame,
                   feature_list: list = None) -> pd.DataFrame:
    """
    选择特定特征

    Args:
        df: 输入DataFrame
        feature_list: 要选择的特征列表（None表示使用配置中的特征）

    Returns:
        选择后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")

    if feature_list is None:
        # 使用配置中定义的特征
        feature_list = config.ALL_FEATURES + ['passenger_count']

    # 检查特征是否存在
    available_features = [f for f in feature_list if f in df.columns]
    missing_features = [f for f in feature_list if f not in df.columns]

    if missing_features:
        logger.warning(f"缺少以下特征: {missing_features}")

    logger.info(f"选择 {len(available_features)} 个特征用于建模")

    return df[available_features]


def get_feature_importance_names() -> list:
    """
    获取特征名称列表（用于特征重要性分析）

    Returns:
        特征名称列表
    """
    return config.ALL_FEATURES + ['passenger_count']


if __name__ == '__main__':
    # 测试特征工程
    from src.utils import setup_logger
    from src.data_loader import generate_sample_data
    from src.data_preprocessing import preprocess_data

    logger = setup_logger("NYC_Taxi", config.LOG_FILE, "INFO")

    print("=" * 60)
    print("特征工程模块测试")
    print("=" * 60)

    # 生成测试数据
    print("\n1. 生成测试数据")
    df = generate_sample_data(1000)
    df_clean = preprocess_data(df, remove_outliers_flag=False)

    print(f"预处理后数据形状: {df_clean.shape}")

    # 特征工程
    print("\n2. 执行特征工程")
    df_feat = engineer_features(df_clean)

    print(f"\n特征工程后数据形状: {df_feat.shape}")
    print(f"\n新增特征:")
    new_features = set(df_feat.columns) - set(df_clean.columns)
    for feat in sorted(new_features):
        print(f"  - {feat}")

    # 查看部分特征
    print("\n3. 查看地理特征统计")
    geo_features = ['manhattan_distance', 'euclidean_distance', 'bearing']
    print(df_feat[geo_features].describe())

    # 选择特征
    print("\n4. 选择建模特征")
    X = select_features(df_feat)
    print(f"建模特征数量: {X.shape[1]}")
    print(f"特征列表: {list(X.columns)}")

    print("\n特征工程模块测试完成！")
