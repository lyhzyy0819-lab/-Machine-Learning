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


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    移除重复行

    Args:
        df: 输入DataFrame

    Returns:
        移除重复后的DataFrame

    TODO 1: 记录日志并计算重复数量
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("检查重复数据...")
    # before_len = len(df)

    TODO 2: 移除重复行（保留第一次出现的）
    # df_dedup = df.drop_duplicates()
    # after_len = len(df_dedup)
    # removed = before_len - after_len

    TODO 3: 打印移除信息
    # if removed > 0:
    #     logger.info(f"  移除 {removed} 行重复数据 ({removed/before_len*100:.2f}%)")
    # else:
    #     logger.info("  无重复数据")

    TODO 4: 返回去重后的DataFrame
    # return df_dedup
    """
    # TODO: 实现重复行移除
    pass


def handle_missing_values(df: pd.DataFrame,
                         strategy: dict = None) -> pd.DataFrame:
    """
    处理缺失值

    Args:
        df: 输入DataFrame
        strategy: 处理策略字典，格式：
                 {
                     'rating': 'drop',           # 删除缺失行
                     'rating_count': 'fill_0',   # 填充0
                     'category': 'fill_unknown', # 填充'Unknown'
                     'review_content': 'fill_empty' # 填充空字符串
                 }

    Returns:
        处理后的DataFrame

    TODO 1: 如果没有提供策略，使用默认策略
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("处理缺失值...")
    #
    # if strategy is None:
    #     strategy = {
    #         'rating': 'drop',  # 评分缺失则删除
    #         'discounted_price': 'drop',  # 价格缺失则删除
    #         'category': 'fill_unknown',  # 类别缺失填充Unknown
    #         'rating_count': 'fill_0',  # 评分数缺失填充0
    #         'review_content': 'fill_empty',  # 评论缺失填充空字符串
    #         'review_title': 'fill_empty'
    #     }

    TODO 2: 复制DataFrame
    # df_clean = df.copy()

    TODO 3: 遍历策略字典，处理每一列的缺失值
    # for col, method in strategy.items():
    #     if col not in df_clean.columns:
    #         continue
    #
    #     missing_count = df_clean[col].isnull().sum()
    #     if missing_count == 0:
    #         continue

    TODO 4: 根据不同策略处理缺失值
    #     if method == 'drop':
    #         # 删除该列有缺失值的行
    #         df_clean = df_clean.dropna(subset=[col])
    #         logger.info(f"  列 '{col}': 删除 {missing_count} 行缺失数据")
    #
    #     elif method == 'fill_0':
    #         # 填充0
    #         df_clean[col].fillna(0, inplace=True)
    #         logger.info(f"  列 '{col}': 用0填充 {missing_count} 个缺失值")
    #
    #     elif method == 'fill_unknown':
    #         # 填充'Unknown'
    #         df_clean[col].fillna('Unknown', inplace=True)
    #         logger.info(f"  列 '{col}': 用'Unknown'填充 {missing_count} 个缺失值")
    #
    #     elif method == 'fill_empty':
    #         # 填充空字符串
    #         df_clean[col].fillna('', inplace=True)
    #         logger.info(f"  列 '{col}': 用空字符串填充 {missing_count} 个缺失值")
    #
    #     elif method == 'fill_mean':
    #         # 用均值填充（数值列）
    #         mean_val = df_clean[col].mean()
    #         df_clean[col].fillna(mean_val, inplace=True)
    #         logger.info(f"  列 '{col}': 用均值 {mean_val:.2f} 填充 {missing_count} 个缺失值")
    #
    #     elif method == 'fill_median':
    #         # 用中位数填充（数值列）
    #         median_val = df_clean[col].median()
    #         df_clean[col].fillna(median_val, inplace=True)
    #         logger.info(f"  列 '{col}': 用中位数 {median_val:.2f} 填充 {missing_count} 个缺失值")

    TODO 5: 返回处理后的DataFrame
    # return df_clean
    """
    # TODO: 实现缺失值处理
    pass


def remove_outliers(df: pd.DataFrame,
                    columns: List[str],
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

    TODO 1: 初始化日志和计数
    # logger = logging.getLogger("Ecommerce_Rating")
    # df_clean = df.copy()
    # initial_len = len(df_clean)

    TODO 2: 对每一列进行异常值检测
    # for col in columns:
    #     if col not in df_clean.columns:
    #         logger.warning(f"列 '{col}' 不存在，跳过")
    #         continue
    #
    #     before_len = len(df_clean)

    TODO 3: 使用IQR方法检测异常值
    #     if method == 'iqr':
    #         # 计算四分位数
    #         Q1 = df_clean[col].quantile(0.25)
    #         Q3 = df_clean[col].quantile(0.75)
    #         IQR = Q3 - Q1
    #
    #         # 计算上下界
    #         lower_bound = Q1 - threshold * IQR
    #         upper_bound = Q3 + threshold * IQR
    #
    #         # 过滤异常值
    #         df_clean = df_clean[
    #             (df_clean[col] >= lower_bound) &
    #             (df_clean[col] <= upper_bound)
    #         ]

    TODO 4: 使用Z-score方法检测异常值
    #     elif method == 'zscore':
    #         # 计算z-score
    #         mean = df_clean[col].mean()
    #         std = df_clean[col].std()
    #         z_scores = np.abs((df_clean[col] - mean) / std)
    #
    #         # 过滤异常值
    #         df_clean = df_clean[z_scores < threshold]

    TODO 5: 记录移除的异常值数量
    #     after_len = len(df_clean)
    #     removed = before_len - after_len
    #     if removed > 0:
    #         logger.info(f"  列 '{col}': 移除 {removed} 个异常值 ({removed/before_len*100:.2f}%)")

    TODO 6: 记录总移除数量并返回
    # total_removed = initial_len - len(df_clean)
    # logger.info(f"总共移除 {total_removed} 行异常值 ({total_removed/initial_len*100:.2f}%)")
    # return df_clean
    """
    # TODO: 实现异常值移除
    pass


def filter_by_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据业务规则过滤数据

    Args:
        df: 输入DataFrame

    Returns:
        过滤后的DataFrame

    TODO 1: 初始化
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("根据业务规则过滤数据...")
    # df_filtered = df.copy()
    # initial_len = len(df_filtered)

    TODO 2: 过滤评分范围（1.0-5.0）
    # if 'rating' in df_filtered.columns:
    #     logger.info("  过滤评分范围...")
    #     before_len = len(df_filtered)
    #     df_filtered = df_filtered[
    #         (df_filtered['rating'] >= 1.0) &
    #         (df_filtered['rating'] <= 5.0)
    #     ]
    #     removed = before_len - len(df_filtered)
    #     logger.info(f"    移除 {removed} 行评分超出范围的数据")

    TODO 3: 过滤价格（必须大于0）
    # if 'discounted_price' in df_filtered.columns:
    #     logger.info("  过滤价格...")
    #     before_len = len(df_filtered)
    #     df_filtered = df_filtered[df_filtered['discounted_price'] > 0]
    #     removed = before_len - len(df_filtered)
    #     logger.info(f"    移除 {removed} 行价格异常的数据")

    TODO 4: 过滤折扣百分比（0-100之间）
    # if 'discount_percentage' in df_filtered.columns:
    #     logger.info("  过滤折扣百分比...")
    #     before_len = len(df_filtered)
    #     df_filtered = df_filtered[
    #         (df_filtered['discount_percentage'] >= 0) &
    #         (df_filtered['discount_percentage'] <= 100)
    #     ]
    #     removed = before_len - len(df_filtered)
    #     logger.info(f"    移除 {removed} 行折扣异常的数据")

    TODO 5: 过滤评分数量（必须大于等于0）
    # if 'rating_count' in df_filtered.columns:
    #     logger.info("  过滤评分数量...")
    #     before_len = len(df_filtered)
    #     df_filtered = df_filtered[df_filtered['rating_count'] >= 0]
    #     removed = before_len - len(df_filtered)
    #     logger.info(f"    移除 {removed} 行评分数量异常的数据")

    TODO 6: 记录总过滤数量
    # total_removed = initial_len - len(df_filtered)
    # logger.info(f"总共过滤 {total_removed} 行数据 ({total_removed/initial_len*100:.2f}%)")

    TODO 7: 返回过滤后的DataFrame
    # return df_filtered
    """
    # TODO: 实现条件过滤
    pass


def clean_text_columns(df: pd.DataFrame,
                       text_columns: List[str] = None) -> pd.DataFrame:
    """
    清洗文本列

    Args:
        df: 输入DataFrame
        text_columns: 要清洗的文本列列表

    Returns:
        清洗后的DataFrame

    TODO 1: 设置默认文本列
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("清洗文本列...")
    #
    # if text_columns is None:
    #     text_columns = ['review_content', 'review_title', 'about_product', 'product_name']

    TODO 2: 复制DataFrame
    # df_clean = df.copy()

    TODO 3: 对每个文本列进行清洗
    # for col in text_columns:
    #     if col not in df_clean.columns:
    #         continue
    #
    #     logger.info(f"  清洗列: {col}")

    TODO 4: 转换为字符串类型
    #     df_clean[col] = df_clean[col].astype(str)

    TODO 5: 移除多余的空格
    #     df_clean[col] = df_clean[col].str.strip()

    TODO 6: 转换为小写（可选，用于文本分析）
    #     # df_clean[col] = df_clean[col].str.lower()  # 可选

    TODO 7: 替换'nan'字符串为空字符串
    #     df_clean[col] = df_clean[col].replace('nan', '')

    TODO 8: 返回清洗后的DataFrame
    # logger.info("  文本清洗完成")
    # return df_clean
    """
    # TODO: 实现文本清洗
    pass


def preprocess_data(df: pd.DataFrame,
                   remove_duplicates_flag: bool = True,
                   remove_outliers_flag: bool = True,
                   outlier_columns: List[str] = None) -> pd.DataFrame:
    """
    完整的数据预处理流程

    Args:
        df: 输入DataFrame
        remove_duplicates_flag: 是否移除重复
        remove_outliers_flag: 是否移除异常值
        outlier_columns: 要检测异常值的列

    Returns:
        预处理后的DataFrame

    TODO 1: 打印预处理开始信息
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("=" * 60)
    # logger.info("开始数据预处理")
    # logger.info("=" * 60)
    # logger.info(f"输入数据形状: {df.shape}")

    TODO 2: 使用Timer计时整个预处理流程
    # with Timer("数据预处理"):

    TODO 3: 1. 移除重复（如果启用）
    #     if remove_duplicates_flag:
    #         df_clean = remove_duplicates(df)
    #     else:
    #         df_clean = df.copy()

    TODO 4: 2. 处理缺失值
    #     df_clean = handle_missing_values(df_clean)

    TODO 5: 3. 清洗文本列
    #     df_clean = clean_text_columns(df_clean)

    TODO 6: 4. 根据业务规则过滤
    #     df_clean = filter_by_conditions(df_clean)

    TODO 7: 5. 移除异常值（如果启用）
    #     if remove_outliers_flag:
    #         if outlier_columns is None:
    #             outlier_columns = ['rating', 'rating_count', 'discount_percentage']
    #         df_clean = remove_outliers(df_clean, outlier_columns, method='iqr', threshold=3.0)

    TODO 8: 6. 重置索引
    #     df_clean = df_clean.reset_index(drop=True)

    TODO 9: 打印预处理完成信息
    #     logger.info(f"\n预处理完成！")
    #     logger.info(f"输出数据形状: {df_clean.shape}")
    #     logger.info(f"保留比例: {len(df_clean)/len(df)*100:.2f}%")

    TODO 10: 返回清洗后的DataFrame
    # return df_clean
    """
    # TODO: 实现完整预处理流程
    pass


def split_features_target(df: pd.DataFrame,
                         target_col: str = 'rating',
                         exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    分离特征和目标变量

    Args:
        df: 输入DataFrame
        target_col: 目标变量列名
        exclude_cols: 要排除的列（除了目标列）

    Returns:
        (特征DataFrame, 目标Series)

    TODO 1: 检查目标列是否存在
    # logger = logging.getLogger("Ecommerce_Rating")
    # if target_col not in df.columns:
    #     raise ValueError(f"目标列 '{target_col}' 不存在")

    TODO 2: 定义要排除的列（ID列、链接列、目标列等）
    # if exclude_cols is None:
    #     exclude_cols = [
    #         'product_id', 'user_id', 'review_id',
    #         'product_link', 'img_link',
    #         target_col
    #     ]
    # else:
    #     exclude_cols = exclude_cols + [target_col]

    TODO 3: 选择特征列（排除不需要的列）
    # feature_cols = [col for col in df.columns if col not in exclude_cols]

    TODO 4: 分离特征和目标
    # X = df[feature_cols].copy()
    # y = df[target_col].copy()

    TODO 5: 打印信息并返回
    # logger.info(f"特征数量: {X.shape[1]}")
    # logger.info(f"样本数量: {X.shape[0]}")
    # logger.info(f"目标变量: {target_col}")
    # logger.info(f"特征列: {feature_cols}")
    #
    # return X, y
    """
    # TODO: 实现特征目标分离
    pass


if __name__ == '__main__':
    # 测试数据预处理
    from src.utils import setup_logger
    from src.data_loader import load_raw_data

    # TODO: 设置日志
    # logger = setup_logger("Ecommerce_Rating", config.LOG_DIR / "preprocessing_test.log", "INFO")

    print("=" * 60)
    print("数据预处理模块测试")
    print("=" * 60)

    # TODO 1: 加载测试数据
    # try:
    #     df = load_raw_data(use_sample=True, sample_size=200)
    #     print(f"\n原始数据形状: {df.shape}")

    # TODO 2: 执行预处理
    #     df_clean = preprocess_data(df, remove_outliers_flag=True)
    #     print(f"\n清洗后数据形状: {df_clean.shape}")

    # TODO 3: 分离特征和目标
    #     X, y = split_features_target(df_clean, target_col='rating')
    #     print(f"\n特征形状: {X.shape}")
    #     print(f"目标形状: {y.shape}")

    # except Exception as e:
    #     print(f"\n错误: {str(e)}")

    print("\n提示：实现上述TODO后运行此文件进行测试")
