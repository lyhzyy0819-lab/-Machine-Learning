"""
数据加载模块
负责从CSV文件加载Amazon销售数据
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


def download_data_instructions() -> None:
    """
    打印下载数据的说明

    TODO: 打印下载指南
    提示用户如何从Kaggle下载数据集
    数据集: karkavelrajaj/amazon-sales-dataset
    下载后放置在: config.RAW_DATA_DIR / 'amazon.csv'
    """
    print("=" * 60)
    print("📥 数据下载说明")
    print("=" * 60)
    print(f"1. 访问Kaggle数据集: {config.KAGGLE_DATASET}")
    print("2. 下载 'amazon.csv' 文件")
    print(f"3. 将文件放置到: {config.RAW_DATA_FILE}")
    print("=" * 60)


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

    TODO 1: 检查文件是否存在
    # logger = logging.getLogger("Ecommerce_Rating")
    # if not filepath.exists():
    #     logger.error(f"文件不存在: {filepath}")
    #     download_data_instructions()
    #     raise FileNotFoundError(f"请先下载数据文件到: {filepath}")

    TODO 2: 使用Timer计时，读取CSV文件
    # with Timer(f"加载数据: {filepath.name}"):
    #     df = pd.read_csv(filepath, nrows=nrows)
    #     logger.info(f"数据加载成功: {df.shape[0]} 行 × {df.shape[1]} 列")

    TODO 3: 如果需要，优化内存使用
    #     if optimize_memory:
    #         df = reduce_mem_usage(df, verbose=True)

    TODO 4: 返回DataFrame
    # return df
    """
    # TODO: 实现CSV数据加载
    pass


def load_raw_data(use_sample: bool = False,
                  sample_size: int = 500) -> pd.DataFrame:
    """
    加载原始数据

    Args:
        use_sample: 是否使用样本数据（用于快速测试）
        sample_size: 样本大小

    Returns:
        原始数据DataFrame

    TODO 1: 打印加载信息
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("=" * 60)
    # logger.info("加载Amazon销售数据")
    # logger.info("=" * 60)

    TODO 2: 根据use_sample决定读取行数
    # nrows = sample_size if use_sample else None

    TODO 3: 调用load_data_from_csv加载数据
    # df = load_data_from_csv(
    #     config.RAW_DATA_FILE,
    #     nrows=nrows,
    #     optimize_memory=True
    # )

    TODO 4: 打印数据信息并返回
    # print_dataframe_info(df, "原始数据")
    # return df
    """
    # TODO: 实现原始数据加载
    pass


def validate_data(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    验证数据完整性和合理性

    Args:
        df: 输入DataFrame

    Returns:
        (是否通过验证, 问题列表)

    TODO 1: 初始化问题列表
    # logger = logging.getLogger("Ecommerce_Rating")
    # issues = []

    TODO 2: 检查必需列是否存在
    # required_columns = ['rating', 'discounted_price', 'category']
    # for col in required_columns:
    #     if col not in df.columns:
    #         issues.append(f"缺少必需列: {col}")

    TODO 3: 检查评分范围（应该在1.0-5.0之间）
    # if 'rating' in df.columns:
    #     invalid_rating = ((df['rating'] < 1.0) | (df['rating'] > 5.0)).sum()
    #     if invalid_rating > 0:
    #         issues.append(f"{invalid_rating} 个评分超出范围 [1.0, 5.0]")

    TODO 4: 检查价格列（应该大于0）
    # if 'discounted_price' in df.columns:
    #     invalid_price = (df['discounted_price'] <= 0).sum()
    #     if invalid_price > 0:
    #         issues.append(f"{invalid_price} 个价格小于等于0")

    TODO 5: 检查缺失值比例
    # missing = df.isnull().sum()
    # high_missing_cols = missing[missing / len(df) > 0.5]
    # if len(high_missing_cols) > 0:
    #     issues.append(f"以下列缺失值超过50%: {list(high_missing_cols.index)}")

    TODO 6: 返回验证结果
    # is_valid = len(issues) == 0
    # if is_valid:
    #     logger.info("✓ 数据验证通过")
    # else:
    #     logger.warning(f"✗ 发现 {len(issues)} 个问题:")
    #     for issue in issues:
    #         logger.warning(f"  - {issue}")
    # return is_valid, issues
    """
    # TODO: 实现数据验证
    pass


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    获取数据摘要统计

    Args:
        df: 输入DataFrame

    Returns:
        包含统计信息的字典

    TODO 1: 创建基础摘要信息
    # summary = {
    #     'shape': df.shape,
    #     'columns': df.columns.tolist(),
    #     'dtypes': df.dtypes.astype(str).to_dict(),
    #     'missing_values': df.isnull().sum().to_dict(),
    #     'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    # }

    TODO 2: 添加数值列统计
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    # if len(numeric_cols) > 0:
    #     summary['numeric_stats'] = df[numeric_cols].describe().to_dict()

    TODO 3: 添加分类列统计（每个分类列的唯一值数量）
    # categorical_cols = df.select_dtypes(include=['object']).columns
    # if len(categorical_cols) > 0:
    #     summary['categorical_stats'] = {
    #         col: {
    #             'unique_count': df[col].nunique(),
    #             'top_5_values': df[col].value_counts().head(5).to_dict()
    #         }
    #         for col in categorical_cols
    #     }

    TODO 4: 返回摘要字典
    # return summary
    """
    # TODO: 实现数据摘要
    pass


def parse_price(price_str: str) -> float:
    """
    解析价格字符串（可能包含货币符号和逗号）

    Args:
        price_str: 价格字符串（例如："₹1,299", "$99.99"）

    Returns:
        浮点数价格

    TODO 1: 处理空值
    # if pd.isna(price_str) or price_str == '':
    #     return np.nan

    TODO 2: 移除货币符号和逗号，转换为浮点数
    # try:
    #     # 移除货币符号（₹, $等）和逗号
    #     clean_str = ''.join(c for c in str(price_str) if c.isdigit() or c == '.')
    #     return float(clean_str) if clean_str else np.nan
    # except:
    #     return np.nan
    """
    # TODO: 实现价格解析
    pass


def preprocess_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理列名和数据类型

    Args:
        df: 输入DataFrame

    Returns:
        预处理后的DataFrame

    TODO 1: 复制DataFrame
    # df_clean = df.copy()
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("预处理列名和数据类型...")

    TODO 2: 标准化列名（转小写，替换空格为下划线）
    # df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    # logger.info("  列名已标准化")

    TODO 3: 解析价格列（如果存在特殊格式）
    # if 'discounted_price' in df_clean.columns:
    #     df_clean['discounted_price'] = df_clean['discounted_price'].apply(parse_price)
    # if 'actual_price' in df_clean.columns:
    #     df_clean['actual_price'] = df_clean['actual_price'].apply(parse_price)

    TODO 4: 转换评分为浮点数
    # if 'rating' in df_clean.columns:
    #     df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')

    TODO 5: 转换评分数量为整数
    # if 'rating_count' in df_clean.columns:
    #     df_clean['rating_count'] = pd.to_numeric(df_clean['rating_count'], errors='coerce').astype('Int64')

    TODO 6: 返回清洗后的DataFrame
    # logger.info("  数据类型转换完成")
    # return df_clean
    """
    # TODO: 实现列预处理
    pass


if __name__ == '__main__':
    # 测试数据加载
    from src.utils import setup_logger

    # TODO: 设置日志
    # logger = setup_logger("Ecommerce_Rating", config.LOG_DIR / "data_loader_test.log", "INFO")

    print("=" * 60)
    print("数据加载模块测试")
    print("=" * 60)

    # TODO 1: 尝试加载数据（如果文件不存在，打印下载说明）
    # try:
    #     df = load_raw_data(use_sample=True, sample_size=100)
    #     print(f"\n加载数据形状: {df.shape}")
    #     print(f"\n前5行数据:\n{df.head()}")

    # TODO 2: 验证数据
    #     is_valid, issues = validate_data(df)

    # TODO 3: 获取数据摘要
    #     summary = get_data_summary(df)
    #     print(f"\n数据维度: {summary['shape']}")
    #     print(f"内存使用: {summary['memory_usage_mb']:.2f} MB")

    # except FileNotFoundError:
    #     print("\n请先下载数据文件！")

    print("\n提示：实现上述TODO后运行此文件进行测试")
