"""
数据加载模块
负责从各种来源加载NYC出租车数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, print_dataframe_info, reduce_mem_usage


def generate_sample_data(n_samples: int = 10000,
                         save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    生成模拟的NYC出租车数据（用于演示和测试）

    Args:
        n_samples: 样本数量
        save_path: 保存路径（可选）

    Returns:
        生成的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info(f"生成 {n_samples} 条模拟数据...")

    np.random.seed(config.RANDOM_STATE)

    # 生成时间戳（2016年1-6月）
    start_date = pd.Timestamp('2016-01-01')
    end_date = pd.Timestamp('2016-06-30')
    date_range = (end_date - start_date).days

    pickup_datetimes = [
        start_date + pd.Timedelta(days=np.random.randint(0, date_range),
                                   hours=np.random.randint(0, 24),
                                   minutes=np.random.randint(0, 60),
                                   seconds=np.random.randint(0, 60))
        for _ in range(n_samples)
    ]

    # 生成地理坐标（纽约市范围）
    pickup_longitude = np.random.uniform(
        config.NYC_BOUNDS['min_longitude'],
        config.NYC_BOUNDS['max_longitude'],
        n_samples
    )
    pickup_latitude = np.random.uniform(
        config.NYC_BOUNDS['min_latitude'],
        config.NYC_BOUNDS['max_latitude'],
        n_samples
    )

    # 生成终点坐标（相对起点有一定偏移）
    max_distance = 0.1  # 约11公里
    dropoff_longitude = pickup_longitude + np.random.uniform(-max_distance, max_distance, n_samples)
    dropoff_latitude = pickup_latitude + np.random.uniform(-max_distance, max_distance, n_samples)

    # 限制在NYC范围内
    dropoff_longitude = np.clip(
        dropoff_longitude,
        config.NYC_BOUNDS['min_longitude'],
        config.NYC_BOUNDS['max_longitude']
    )
    dropoff_latitude = np.clip(
        dropoff_latitude,
        config.NYC_BOUNDS['min_latitude'],
        config.NYC_BOUNDS['max_latitude']
    )

    # 乘客数量（1-6人）
    passenger_count = np.random.randint(1, 7, n_samples)

    # 计算距离（简化的曼哈顿距离）
    distance = (
        np.abs(dropoff_longitude - pickup_longitude) +
        np.abs(dropoff_latitude - pickup_latitude)
    ) * 111  # 转换为大致的公里数

    # 生成行程时长（基于距离和时间）
    # 基础时长：与距离成正比
    base_duration = distance * 180 + 300  # 每公里约3分钟 + 基础300秒

    # 时间因素影响
    hours = [dt.hour for dt in pickup_datetimes]
    time_factor = np.array([
        1.5 if 7 <= h <= 10 or 17 <= h <= 20 else 1.0  # 高峰时段更慢
        for h in hours
    ])

    # 随机噪声
    noise = np.random.normal(0, 0.15, n_samples)

    trip_duration = base_duration * time_factor * (1 + noise)

    # 限制在合理范围内
    trip_duration = np.clip(
        trip_duration,
        config.TRIP_DURATION_LIMITS['min'],
        config.TRIP_DURATION_LIMITS['max']
    )

    # 创建DataFrame
    df = pd.DataFrame({
        'id': [f'id{i:07d}' for i in range(n_samples)],
        'vendor_id': np.random.choice([1, 2], n_samples),
        'pickup_datetime': pickup_datetimes,
        'dropoff_datetime': [pickup_datetimes[i] + pd.Timedelta(seconds=trip_duration[i])
                             for i in range(n_samples)],
        'passenger_count': passenger_count,
        'pickup_longitude': pickup_longitude,
        'pickup_latitude': pickup_latitude,
        'dropoff_longitude': dropoff_longitude,
        'dropoff_latitude': dropoff_latitude,
        'store_and_fwd_flag': np.random.choice(['N', 'Y'], n_samples, p=[0.95, 0.05]),
        'trip_duration': trip_duration.astype(int)
    })

    logger.info(f"模拟数据生成完成: {df.shape}")

    # 保存数据
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"数据已保存到: {save_path}")

    return df


def load_data_from_csv(filepath: Path,
                       nrows: Optional[int] = None,
                       optimize_memory: bool = True) -> pd.DataFrame:
    """
    从CSV文件加载数据

    Args:
        filepath: CSV文件路径
        nrows: 读取的行数（None表示读取全部）
        optimize_memory: 是否优化内存使用

    Returns:
        加载的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")

    with Timer(f"加载数据: {filepath.name}"):
        # 检查文件是否存在
        if not filepath.exists():
            logger.warning(f"文件不存在: {filepath}")
            logger.info("将生成模拟数据...")
            df = generate_sample_data(
                n_samples=nrows if nrows else config.SAMPLE_SIZE,
                save_path=filepath
            )
            return df

        # 定义数据类型（优化内存）
        dtypes = {
            'id': str,
            'vendor_id': 'int8',
            'passenger_count': 'int8',
            'pickup_longitude': 'float32',
            'pickup_latitude': 'float32',
            'dropoff_longitude': 'float32',
            'dropoff_latitude': 'float32',
            'store_and_fwd_flag': 'category'
        }

        # 解析日期的列
        parse_dates = ['pickup_datetime', 'dropoff_datetime']

        # 读取数据
        df = pd.read_csv(
            filepath,
            dtype=dtypes,
            parse_dates=parse_dates,
            nrows=nrows
        )

        logger.info(f"数据加载成功: {df.shape[0]} 行 × {df.shape[1]} 列")

        # 优化内存
        if optimize_memory:
            df = reduce_mem_usage(df, verbose=True)

    return df


def load_train_data(use_sample: bool = False,
                    sample_size: int = 10000) -> pd.DataFrame:
    """
    加载训练数据

    Args:
        use_sample: 是否使用样本数据
        sample_size: 样本大小

    Returns:
        训练数据DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("=" * 60)
    logger.info("加载训练数据")
    logger.info("=" * 60)

    nrows = sample_size if use_sample else None

    df = load_data_from_csv(
        config.TRAIN_DATA_PATH,
        nrows=nrows,
        optimize_memory=True
    )

    # 打印数据信息
    print_dataframe_info(df, "训练数据")

    return df


def load_test_data(use_sample: bool = False,
                   sample_size: int = 5000) -> pd.DataFrame:
    """
    加载测试数据

    Args:
        use_sample: 是否使用样本数据
        sample_size: 样本大小

    Returns:
        测试数据DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("=" * 60)
    logger.info("加载测试数据")
    logger.info("=" * 60)

    nrows = sample_size if use_sample else None

    df = load_data_from_csv(
        config.TEST_DATA_PATH,
        nrows=nrows,
        optimize_memory=True
    )

    # 打印数据信息
    print_dataframe_info(df, "测试数据")

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    获取数据摘要统计

    Args:
        df: 输入DataFrame

    Returns:
        包含统计信息的字典
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }

    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()

    # 分类列统计
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        summary['categorical_stats'] = {
            col: df[col].value_counts().to_dict()
            for col in categorical_cols
        }

    return summary


def validate_data(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    验证数据完整性和合理性

    Args:
        df: 输入DataFrame

    Returns:
        (是否通过验证, 问题列表)
    """
    logger = logging.getLogger("NYC_Taxi")
    issues = []

    # 检查必需列
    required_columns = [
        'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
    ]

    for col in required_columns:
        if col not in df.columns:
            issues.append(f"缺少必需列: {col}")

    # 检查缺失值
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col, count in missing[missing > 0].items():
            issues.append(f"列 '{col}' 有 {count} 个缺失值")

    # 检查地理坐标范围
    if 'pickup_longitude' in df.columns:
        invalid_lon = (
            (df['pickup_longitude'] < config.NYC_BOUNDS['min_longitude']) |
            (df['pickup_longitude'] > config.NYC_BOUNDS['max_longitude'])
        ).sum()
        if invalid_lon > 0:
            issues.append(f"{invalid_lon} 个起点经度超出NYC范围")

    if 'pickup_latitude' in df.columns:
        invalid_lat = (
            (df['pickup_latitude'] < config.NYC_BOUNDS['min_latitude']) |
            (df['pickup_latitude'] > config.NYC_BOUNDS['max_latitude'])
        ).sum()
        if invalid_lat > 0:
            issues.append(f"{invalid_lat} 个起点纬度超出NYC范围")

    # 检查行程时长
    if 'trip_duration' in df.columns:
        invalid_duration = (
            (df['trip_duration'] < config.TRIP_DURATION_LIMITS['min']) |
            (df['trip_duration'] > config.TRIP_DURATION_LIMITS['max'])
        ).sum()
        if invalid_duration > 0:
            issues.append(f"{invalid_duration} 个行程时长超出合理范围")

    is_valid = len(issues) == 0

    if is_valid:
        logger.info("✓ 数据验证通过")
    else:
        logger.warning(f"✗ 发现 {len(issues)} 个问题:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return is_valid, issues


if __name__ == '__main__':
    # 测试数据加载
    from src.utils import setup_logger

    logger = setup_logger("NYC_Taxi", config.LOG_FILE, "INFO")

    print("=" * 60)
    print("数据加载模块测试")
    print("=" * 60)

    # 生成测试数据
    print("\n1. 生成模拟数据")
    df = generate_sample_data(1000, config.TRAIN_DATA_PATH)
    print(f"生成数据形状: {df.shape}")
    print(f"\n前5行数据:")
    print(df.head())

    # 验证数据
    print("\n2. 验证数据")
    is_valid, issues = validate_data(df)

    # 数据摘要
    print("\n3. 数据摘要")
    summary = get_data_summary(df)
    print(f"数据维度: {summary['shape']}")
    print(f"内存使用: {summary['memory_usage_mb']:.2f} MB")

    print("\n数据加载模块测试完成！")
