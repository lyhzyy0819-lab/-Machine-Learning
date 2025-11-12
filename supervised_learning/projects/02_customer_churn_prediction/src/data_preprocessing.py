"""
数据预处理模块
负责数据清洗、异常值处理、缺失值填充等
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer

logger = logging.getLogger("ChurnPrediction")


def preprocess_data(
    df: pd.DataFrame,
    remove_outliers_flag: bool = True,
    outlier_method: str = 'iqr',
    handle_missing: str = 'auto'
) -> pd.DataFrame:
    """
    数据预处理主函数 - 完整的数据清洗流程

    Args:
        df: 原始DataFrame
        remove_outliers_flag: 是否移除异常值
        outlier_method: 异常值检测方法 ('iqr' 或 'zscore')
        handle_missing: 缺失值处理策略 ('auto', 'drop', 'fill')

    Returns:
        清洗后的DataFrame
    """
    logger.info("=" * 80)
    logger.info("数据预处理")
    logger.info("=" * 80)

    df_clean = df.copy()
    initial_shape = df_clean.shape

    with Timer("数据预处理总耗时"):
        # 1. 删除ID列
        df_clean = remove_id_columns(df_clean)

        # 2. 处理数据类型
        df_clean = convert_datatypes(df_clean)

        # 3. 处理TotalCharges列（特殊情况）
        df_clean = handle_total_charges(df_clean)

        # 4. 处理缺失值
        df_clean = handle_missing_values(df_clean, strategy=handle_missing)

        # 5. 检查并移除重复行
        df_clean = remove_duplicates(df_clean)

        # 6. 异常值处理
        if remove_outliers_flag:
            df_clean = detect_and_remove_outliers(
                df_clean,
                method=outlier_method
            )

        # 7. 数据验证
        df_clean = validate_processed_data(df_clean)

    final_shape = df_clean.shape
    logger.info(f"\n预处理完成:")
    logger.info(f"  原始数据: {initial_shape[0]:,} 行 × {initial_shape[1]} 列")
    logger.info(f"  清洗后: {final_shape[0]:,} 行 × {final_shape[1]} 列")
    logger.info(f"  保留率: {final_shape[0]/initial_shape[0]*100:.2f}%")

    return df_clean


def remove_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除ID列等无用特征

    Args:
        df: DataFrame

    Returns:
        删除ID列后的DataFrame
    """
    logger.info("\n1. 删除ID列...")

    df_clean = df.copy()
    id_columns = []

    # 检查配置的ID列
    if hasattr(config, 'ID_COL') and config.ID_COL in df_clean.columns:
        id_columns.append(config.ID_COL)

    # 自动检测可能的ID列
    for col in df_clean.columns:
        # ID列通常唯一值数量等于总行数
        if df_clean[col].nunique() == len(df_clean):
            # 排除数值型特征
            if df_clean[col].dtype == 'object' or 'id' in col.lower():
                if col not in id_columns:
                    id_columns.append(col)

    if id_columns:
        df_clean = df_clean.drop(id_columns, axis=1)
        logger.info(f"  删除列: {id_columns}")
    else:
        logger.info("  未发现ID列")

    return df_clean


def convert_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    转换数据类型以优化内存并确保正确性

    Args:
        df: DataFrame

    Returns:
        转换后的DataFrame
    """
    logger.info("\n2. 转换数据类型...")

    df_clean = df.copy()
    before_mem = df_clean.memory_usage(deep=True).sum() / 1024**2

    # 转换二值分类为0/1
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            unique_vals = df_clean[col].unique()

            # Yes/No -> 1/0
            if set(unique_vals).issubset({'Yes', 'No', np.nan}):
                df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
                logger.info(f"  {col}: Yes/No -> 1/0")

            # Male/Female -> 1/0
            elif set(unique_vals).issubset({'Male', 'Female', np.nan}):
                df_clean[col] = df_clean[col].map({'Male': 1, 'Female': 0})
                logger.info(f"  {col}: Male/Female -> 1/0")

    # 优化整型
    for col in df_clean.select_dtypes(include=['int64']).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], downcast='integer')

    # 优化浮点型
    for col in df_clean.select_dtypes(include=['float64']).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], downcast='float')

    after_mem = df_clean.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"  内存优化: {before_mem:.2f}MB -> {after_mem:.2f}MB "
                f"(减少{(before_mem-after_mem)/before_mem*100:.1f}%)")

    return df_clean


def handle_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理TotalCharges列的特殊情况

    TotalCharges可能包含空字符串,需要转换为数值并填充

    Args:
        df: DataFrame

    Returns:
        处理后的DataFrame
    """
    logger.info("\n3. 处理TotalCharges列...")

    if 'TotalCharges' not in df.columns:
        logger.info("  未找到TotalCharges列，跳过")
        return df

    df_clean = df.copy()

    # 检查数据类型
    if df_clean['TotalCharges'].dtype == 'object':
        logger.info("  检测到object类型，转换为数值...")

        # 转换为数值（空字符串会变成NaN）
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'],
            errors='coerce'
        )

        # 检查转换后的缺失值
        missing_count = df_clean['TotalCharges'].isnull().sum()
        if missing_count > 0:
            logger.info(f"  发现 {missing_count} 个缺失值")

            # 分析缺失值的特征
            missing_rows = df_clean[df_clean['TotalCharges'].isnull()]
            if 'tenure' in df_clean.columns:
                tenure_values = missing_rows['tenure'].value_counts()
                logger.info(f"  缺失值对应的tenure: {tenure_values.to_dict()}")

            # 填充策略：新客户(tenure=0)用MonthlyCharges填充，其他用中位数
            if 'MonthlyCharges' in df_clean.columns and 'tenure' in df_clean.columns:
                # 新客户
                mask_new = (df_clean['TotalCharges'].isnull()) & (df_clean['tenure'] == 0)
                df_clean.loc[mask_new, 'TotalCharges'] = df_clean.loc[mask_new, 'MonthlyCharges']
                logger.info(f"  新客户({mask_new.sum()}个): 用MonthlyCharges填充")

                # 老客户
                mask_old = (df_clean['TotalCharges'].isnull()) & (df_clean['tenure'] > 0)
                if mask_old.sum() > 0:
                    median_value = df_clean['TotalCharges'].median()
                    df_clean.loc[mask_old, 'TotalCharges'] = median_value
                    logger.info(f"  老客户({mask_old.sum()}个): 用中位数{median_value:.2f}填充")

    return df_clean


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'auto'
) -> pd.DataFrame:
    """
    处理缺失值

    Args:
        df: DataFrame
        strategy: 处理策略
            - 'auto': 自动选择策略
            - 'drop': 删除含缺失值的行
            - 'fill': 填充缺失值

    Returns:
        处理后的DataFrame
    """
    logger.info("\n4. 处理缺失值...")

    df_clean = df.copy()
    missing = df_clean.isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        logger.info("  ✓ 无缺失值")
        return df_clean

    logger.info(f"  发现 {total_missing} 个缺失值:")
    for col in missing[missing > 0].index:
        count = missing[col]
        percent = count / len(df_clean) * 100
        logger.info(f"    - {col}: {count} ({percent:.2f}%)")

    # 自动选择策略
    if strategy == 'auto':
        max_missing_percent = (missing / len(df_clean) * 100).max()
        if max_missing_percent < 5:
            strategy = 'drop'
            logger.info(f"  策略: 删除（最大缺失率{max_missing_percent:.2f}% < 5%）")
        else:
            strategy = 'fill'
            logger.info(f"  策略: 填充（最大缺失率{max_missing_percent:.2f}% >= 5%）")

    if strategy == 'drop':
        # 删除含有缺失值的行
        before_len = len(df_clean)
        df_clean = df_clean.dropna()
        after_len = len(df_clean)
        removed = before_len - after_len
        logger.info(f"  删除 {removed} 行 ({removed/before_len*100:.2f}%)")

    elif strategy == 'fill':
        # 填充缺失值
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['float64', 'float32', 'float16',
                                           'int64', 'int32', 'int16', 'int8']:
                    # 数值型：用中位数填充
                    fill_value = df_clean[col].median()
                    df_clean[col].fillna(fill_value, inplace=True)
                    logger.info(f"    {col}: 中位数填充 ({fill_value:.2f})")
                else:
                    # 分类型：用众数填充
                    fill_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    df_clean[col].fillna(fill_value, inplace=True)
                    logger.info(f"    {col}: 众数填充 ({fill_value})")

    # 验证
    remaining_missing = df_clean.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"  ⚠️  仍有 {remaining_missing} 个缺失值未处理")
    else:
        logger.info("  ✓ 所有缺失值已处理")

    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    检查并移除重复行

    Args:
        df: DataFrame

    Returns:
        去重后的DataFrame
    """
    logger.info("\n5. 检查重复行...")

    n_duplicates = df.duplicated().sum()

    if n_duplicates == 0:
        logger.info("  ✓ 无重复行")
        return df

    logger.info(f"  发现 {n_duplicates} 行重复数据 ({n_duplicates/len(df)*100:.2f}%)")

    df_clean = df.drop_duplicates()
    logger.info(f"  删除后: {len(df_clean):,} 行")

    return df_clean


def detect_and_remove_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    检测并移除异常值

    Args:
        df: DataFrame
        method: 检测方法 ('iqr' 或 'zscore')
        threshold: 阈值（IQR倍数或标准差倍数）

    Returns:
        移除异常值后的DataFrame
    """
    logger.info(f"\n6. 异常值检测与移除 (方法: {method})...")

    df_clean = df.copy()
    initial_len = len(df_clean)

    # 获取数值型列
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    # 排除目标变量
    if 'Churn' in numeric_cols:
        numeric_cols.remove('Churn')

    if not numeric_cols:
        logger.info("  未找到数值型特征，跳过")
        return df_clean

    logger.info(f"  检测列: {numeric_cols}")

    outliers_info = {}

    for col in numeric_cols:
        before_len = len(df_clean)

        if method == 'iqr':
            # IQR方法
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # 统计异常值
            n_outliers = ((df_clean[col] < lower_bound) |
                         (df_clean[col] > upper_bound)).sum()

            if n_outliers > 0:
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) &
                    (df_clean[col] <= upper_bound)
                ]

                outliers_info[col] = {
                    'count': n_outliers,
                    'percent': n_outliers / before_len * 100,
                    'bounds': (lower_bound, upper_bound)
                }

        elif method == 'zscore':
            # Z-score方法
            mean = df_clean[col].mean()
            std = df_clean[col].std()

            if std == 0:
                continue

            z_scores = np.abs((df_clean[col] - mean) / std)
            n_outliers = (z_scores >= threshold).sum()

            if n_outliers > 0:
                df_clean = df_clean[z_scores < threshold]

                outliers_info[col] = {
                    'count': n_outliers,
                    'percent': n_outliers / before_len * 100,
                    'threshold': threshold
                }

    # 输出异常值统计
    if outliers_info:
        logger.info("  异常值统计:")
        for col, info in outliers_info.items():
            logger.info(f"    {col}: {info['count']} 个 ({info['percent']:.2f}%)")
    else:
        logger.info("  未发现异常值")

    total_removed = initial_len - len(df_clean)
    if total_removed > 0:
        logger.info(f"  总计移除 {total_removed} 行 ({total_removed/initial_len*100:.2f}%)")

    return df_clean


def validate_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    验证处理后的数据质量

    Args:
        df: DataFrame

    Returns:
        验证后的DataFrame
    """
    logger.info("\n7. 数据验证...")

    issues = []

    # 检查缺失值
    missing = df.isnull().sum().sum()
    if missing > 0:
        issues.append(f"仍有{missing}个缺失值")

    # 检查数据范围
    if 'tenure' in df.columns:
        invalid_tenure = ((df['tenure'] < 0) | (df['tenure'] > 100)).sum()
        if invalid_tenure > 0:
            issues.append(f"{invalid_tenure}个异常tenure值")

    if 'MonthlyCharges' in df.columns:
        invalid_charges = ((df['MonthlyCharges'] <= 0) |
                          (df['MonthlyCharges'] > 200)).sum()
        if invalid_charges > 0:
            issues.append(f"{invalid_charges}个异常MonthlyCharges值")

    # 检查数据完整性
    if len(df) == 0:
        issues.append("数据为空")

    if issues:
        logger.warning("  发现问题:")
        for issue in issues:
            logger.warning(f"    - {issue}")
    else:
        logger.info("  ✓ 数据验证通过")

    return df


def get_preprocessing_report(df_before: pd.DataFrame,
                            df_after: pd.DataFrame) -> dict:
    """
    生成预处理报告

    Args:
        df_before: 预处理前的DataFrame
        df_after: 预处理后的DataFrame

    Returns:
        预处理报告字典
    """
    report = {
        'before': {
            'shape': df_before.shape,
            'missing_values': df_before.isnull().sum().sum(),
            'duplicates': df_before.duplicated().sum(),
            'memory_mb': df_before.memory_usage(deep=True).sum() / 1024**2
        },
        'after': {
            'shape': df_after.shape,
            'missing_values': df_after.isnull().sum().sum(),
            'duplicates': df_after.duplicated().sum(),
            'memory_mb': df_after.memory_usage(deep=True).sum() / 1024**2
        },
        'changes': {
            'rows_removed': df_before.shape[0] - df_after.shape[0],
            'retention_rate': df_after.shape[0] / df_before.shape[0] * 100,
            'columns_removed': df_before.shape[1] - df_after.shape[1],
            'memory_reduction': (
                (df_before.memory_usage(deep=True).sum() -
                 df_after.memory_usage(deep=True).sum()) /
                df_before.memory_usage(deep=True).sum() * 100
            )
        }
    }

    return report


def print_preprocessing_report(report: dict) -> None:
    """
    打印预处理报告

    Args:
        report: 预处理报告字典
    """
    print("\n" + "=" * 80)
    print("数据预处理报告")
    print("=" * 80)

    print("\n处理前:")
    print(f"  形状: {report['before']['shape'][0]:,} 行 × {report['before']['shape'][1]} 列")
    print(f"  缺失值: {report['before']['missing_values']:,}")
    print(f"  重复行: {report['before']['duplicates']:,}")
    print(f"  内存: {report['before']['memory_mb']:.2f} MB")

    print("\n处理后:")
    print(f"  形状: {report['after']['shape'][0]:,} 行 × {report['after']['shape'][1]} 列")
    print(f"  缺失值: {report['after']['missing_values']:,}")
    print(f"  重复行: {report['after']['duplicates']:,}")
    print(f"  内存: {report['after']['memory_mb']:.2f} MB")

    print("\n变化:")
    print(f"  删除行数: {report['changes']['rows_removed']:,}")
    print(f"  数据保留率: {report['changes']['retention_rate']:.2f}%")
    print(f"  删除列数: {report['changes']['columns_removed']}")
    print(f"  内存减少: {report['changes']['memory_reduction']:.2f}%")

    print("=" * 80)


if __name__ == '__main__':
    # 测试数据预处理
    from src.utils import setup_logger
    from src.data_loader import load_train_data

    setup_logger("ChurnPrediction", config.LOG_FILE, "INFO")

    print("=" * 80)
    print("数据预处理模块测试")
    print("=" * 80)

    # 加载数据
    df = load_train_data(use_sample=True, sample_size=1000)

    # 预处理
    df_clean = preprocess_data(df, remove_outliers_flag=True, outlier_method='iqr')

    # 生成报告
    report = get_preprocessing_report(df, df_clean)
    print_preprocessing_report(report)

    print("\n数据预处理模块测试完成！")
