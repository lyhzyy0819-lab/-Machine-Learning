"""
数据加载模块
负责从各种来源加载客户流失数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, optimize_dataframe_memory

logger = logging.getLogger("ChurnPrediction")


def load_data_from_csv(
    filepath: Path,
    nrows: Optional[int] = None,
    optimize_memory: bool = True
) -> pd.DataFrame:
    """
    从CSV文件加载数据

    Args:
        filepath: CSV文件路径
        nrows: 读取的行数（None表示读取全部）
        optimize_memory: 是否优化内存使用

    Returns:
        加载的DataFrame
    """
    with Timer(f"加载数据: {filepath.name}"):
        # 检查文件是否存在
        if not filepath.exists():
            logger.warning(f"文件不存在: {filepath}")
            logger.info("将尝试从在线源加载数据...")
            df = load_data_online(config.ONLINE_DATA_URL, nrows=nrows)
            # 保存到本地
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"数据已保存到: {filepath}")
            return df

        # 读取数据
        df = pd.read_csv(filepath, nrows=nrows)

        logger.info(f"数据加载成功: {df.shape[0]} 行 × {df.shape[1]} 列")

        # 优化内存
        if optimize_memory:
            df = optimize_dataframe_memory(df, verbose=True)

    return df


def load_data_online(url: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    从在线URL加载数据

    Args:
        url: 数据URL
        nrows: 读取的行数

    Returns:
        DataFrame
    """
    logger.info(f"从在线源加载数据: {url}")

    try:
        df = pd.read_csv(url, nrows=nrows)
        logger.info(f"在线数据加载成功: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"在线加载失败: {e}")
        raise FileNotFoundError(
            "无法加载数据。请:\n"
            "1. 检查网络连接\n"
            "2. 手动下载数据集: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "3. 将文件放置在 data/raw/ 目录并命名为 WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )


def load_train_data(
    use_sample: bool = False,
    sample_size: int = 5000
) -> pd.DataFrame:
    """
    加载训练数据

    Args:
        use_sample: 是否使用样本数据
        sample_size: 样本大小

    Returns:
        训练数据DataFrame
    """
    logger.info("=" * 80)
    logger.info("加载训练数据")
    logger.info("=" * 80)

    nrows = sample_size if use_sample else None

    df = load_data_from_csv(
        config.DATA_PATH,
        nrows=nrows,
        optimize_memory=True
    )

    # 打印数据信息
    print_dataframe_info(df, "训练数据")

    # 验证数据
    validate_data(df)

    return df


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    打印DataFrame详细信息

    Args:
        df: DataFrame
        name: 数据集名称
    """
    print(f"\n{name} 信息:")
    print("=" * 80)
    print(f"形状: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"缺失值总数: {df.isnull().sum().sum():,}")
    print(f"重复行数: {df.duplicated().sum():,}")

    print(f"\n列信息:")
    print("-" * 80)
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        unique = df[col].nunique()
        print(f"{i:2d}. {col:25s} | {str(dtype):10s} | 缺失: {missing:6,} | 唯一值: {unique:6,}")

    print("=" * 80)


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
        'n_samples': len(df),
        'n_features': df.shape[1],
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_rows': df.duplicated().sum()
    }

    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_columns'] = numeric_cols.tolist()
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()

    # 分类列统计
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary['categorical_columns'] = categorical_cols.tolist()
        summary['categorical_stats'] = {
            col: {
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                'value_counts': df[col].value_counts().to_dict()
            }
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
    issues = []

    logger.info("\n验证数据质量...")

    # 1. 检查必需列
    required_columns = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]

    for col in required_columns:
        if col not in df.columns:
            issues.append(f"缺少必需列: {col}")

    # 2. 检查缺失值
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"发现 {missing.sum()} 个缺失值:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"  - {col}: {count} ({count/len(df)*100:.2f}%)")
            issues.append(f"列 '{col}' 有 {count} 个缺失值")

    # 3. 检查重复行
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"发现 {n_duplicates} 行重复数据")
        issues.append(f"{n_duplicates} 行重复数据")

    # 4. 检查数值范围
    if 'tenure' in df.columns:
        invalid_tenure = ((df['tenure'] < 0) | (df['tenure'] > 100)).sum()
        if invalid_tenure > 0:
            issues.append(f"{invalid_tenure} 个异常的在网时长值")

    if 'MonthlyCharges' in df.columns:
        invalid_monthly = ((df['MonthlyCharges'] < 0) | (df['MonthlyCharges'] > 200)).sum()
        if invalid_monthly > 0:
            issues.append(f"{invalid_monthly} 个异常的月费用值")

    # 5. 检查目标变量
    if 'Churn' in df.columns:
        unique_values = df['Churn'].unique()
        expected_values = ['Yes', 'No']
        if not all(val in expected_values for val in unique_values if pd.notna(val)):
            issues.append(f"目标变量Churn包含异常值: {unique_values}")

        # 检查类别分布
        churn_counts = df['Churn'].value_counts()
        if len(churn_counts) > 0:
            churn_rate = churn_counts.get('Yes', 0) / len(df) * 100
            logger.info(f"  流失率: {churn_rate:.2f}%")

            # 检查类别不平衡
            if churn_rate < 5 or churn_rate > 95:
                logger.warning(f"  ⚠️  严重的类别不平衡: {churn_rate:.2f}%")
                issues.append(f"严重的类别不平衡（流失率: {churn_rate:.2f}%）")
            elif churn_rate < 20 or churn_rate > 80:
                logger.warning(f"  ⚠️  存在类别不平衡: {churn_rate:.2f}%")

    is_valid = len(issues) == 0

    if is_valid:
        logger.info("✓ 数据验证通过")
    else:
        logger.warning(f"✗ 发现 {len(issues)} 个问题")

    return is_valid, issues


def check_data_types(df: pd.DataFrame) -> dict:
    """
    检查并分类数据类型

    Args:
        df: DataFrame

    Returns:
        数据类型分类字典
    """
    type_info = {
        'numerical': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'binary': []
    }

    for col in df.columns:
        dtype = df[col].dtype

        # 数值型
        if dtype in ['int64', 'int32', 'int16', 'int8',
                     'float64', 'float32', 'float16']:
            type_info['numerical'].append(col)

        # 日期时间型
        elif dtype == 'datetime64[ns]':
            type_info['datetime'].append(col)

        # 对象型（需进一步判断）
        elif dtype == 'object':
            unique_count = df[col].nunique()

            # 二值特征
            if unique_count == 2:
                type_info['binary'].append(col)
            # 分类特征（唯一值较少）
            elif unique_count < 50:
                type_info['categorical'].append(col)
            # 文本特征（唯一值很多）
            else:
                type_info['text'].append(col)

    return type_info


def split_features_target(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    id_col: str = 'customerID'
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    分离特征、目标变量和ID

    Args:
        df: 完整DataFrame
        target_col: 目标变量列名
        id_col: ID列名

    Returns:
        (特征DataFrame, 目标Series, ID Series)
    """
    logger.info("\n分离特征和目标变量...")

    # 检查目标列是否存在
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在")

    # 提取ID（如果存在）
    ids = None
    if id_col in df.columns:
        ids = df[id_col].copy()
        logger.info(f"  ID列: {id_col}")

    # 提取目标变量
    y = df[target_col].copy()
    logger.info(f"  目标变量: {target_col}")
    logger.info(f"    - 样本数: {len(y):,}")
    logger.info(f"    - 类别分布: {y.value_counts().to_dict()}")

    # 提取特征
    drop_cols = [target_col]
    if id_col in df.columns:
        drop_cols.append(id_col)

    X = df.drop(drop_cols, axis=1)
    logger.info(f"  特征矩阵: {X.shape}")

    return X, y, ids


def sample_data(df: pd.DataFrame, n: int, stratify: Optional[pd.Series] = None,
                random_state: int = 42) -> pd.DataFrame:
    """
    从DataFrame中抽样

    Args:
        df: 原始DataFrame
        n: 抽样数量
        stratify: 分层抽样的列
        random_state: 随机种子

    Returns:
        抽样后的DataFrame
    """
    if n >= len(df):
        logger.warning(f"抽样数量({n})大于等于数据总数({len(df)})，返回全部数据")
        return df.copy()

    if stratify is not None:
        # 分层抽样
        from sklearn.model_selection import train_test_split
        sampled_df, _ = train_test_split(
            df,
            train_size=n,
            stratify=stratify,
            random_state=random_state
        )
        logger.info(f"分层抽样 {n} 条记录（基于 {stratify.name}）")
    else:
        # 随机抽样
        sampled_df = df.sample(n=n, random_state=random_state)
        logger.info(f"随机抽样 {n} 条记录")

    return sampled_df.reset_index(drop=True)


def save_data(df: pd.DataFrame, filepath: Path, format: str = 'csv') -> None:
    """
    保存DataFrame到文件

    Args:
        df: 要保存的DataFrame
        filepath: 保存路径
        format: 文件格式 ('csv', 'parquet', 'pickle')
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"不支持的格式: {format}")

    logger.info(f"数据已保存到: {filepath}")
    logger.info(f"  格式: {format}")
    logger.info(f"  大小: {filepath.stat().st_size / 1024**2:.2f} MB")


if __name__ == '__main__':
    # 测试数据加载
    from src.utils import setup_logger

    setup_logger("ChurnPrediction", config.LOG_FILE, "INFO")

    print("=" * 80)
    print("数据加载模块测试")
    print("=" * 80)

    # 1. 加载数据
    print("\n1. 加载数据")
    df = load_train_data(use_sample=True, sample_size=1000)

    # 2. 数据摘要
    print("\n2. 数据摘要")
    summary = get_data_summary(df)
    print(f"形状: {summary['shape']}")
    print(f"内存使用: {summary['memory_usage_mb']:.2f} MB")
    print(f"数值特征数: {len(summary.get('numeric_columns', []))}")
    print(f"分类特征数: {len(summary.get('categorical_columns', []))}")

    # 3. 数据类型检查
    print("\n3. 数据类型分类")
    type_info = check_data_types(df)
    for dtype, cols in type_info.items():
        if cols:
            print(f"  {dtype}: {len(cols)} 个 - {cols[:3]}...")

    # 4. 分离特征和目标
    print("\n4. 分离特征和目标")
    X, y, ids = split_features_target(df)

    print("\n数据加载模块测试完成！")
