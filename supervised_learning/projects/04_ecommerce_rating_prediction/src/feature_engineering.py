"""
特征工程模块
创建文本特征、价格特征、类别编码和交互特征
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer


def create_text_features(df: pd.DataFrame,
                        text_column: str,
                        max_features: int = 100,
                        prefix: str = 'tfidf') -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    从文本列创建TF-IDF特征

    Args:
        df: 输入DataFrame
        text_column: 文本列名
        max_features: 提取的最大特征数
        prefix: 特征前缀

    Returns:
        (包含TF-IDF特征的DataFrame, TF-IDF向量化器)

    TODO 1: 检查文本列是否存在
    # logger = logging.getLogger("Ecommerce_Rating")
    # if text_column not in df.columns:
    #     logger.warning(f"文本列 '{text_column}' 不存在，跳过")
    #     return df, None

    TODO 2: 填充空文本（避免TF-IDF错误）
    # texts = df[text_column].fillna('').astype(str)
    # logger.info(f"从 '{text_column}' 提取TF-IDF特征（max_features={max_features}）...")

    TODO 3: 创建TF-IDF向量化器
    # vectorizer = TfidfVectorizer(
    #     max_features=max_features,
    #     min_df=config.TFIDF_MIN_DF,
    #     max_df=config.TFIDF_MAX_DF,
    #     stop_words='english',  # 移除英文停用词
    #     lowercase=True,
    #     ngram_range=(1, 2)  # 使用1-gram和2-gram
    # )

    TODO 4: 拟合并转换文本
    # try:
    #     tfidf_matrix = vectorizer.fit_transform(texts)
    #     logger.info(f"  成功提取 {tfidf_matrix.shape[1]} 个TF-IDF特征")

    TODO 5: 转换为DataFrame
    #     feature_names = [f"{prefix}_{i}" for i in range(tfidf_matrix.shape[1])]
    #     tfidf_df = pd.DataFrame(
    #         tfidf_matrix.toarray(),
    #         columns=feature_names,
    #         index=df.index
    #     )

    TODO 6: 合并到原DataFrame
    #     df_with_features = pd.concat([df, tfidf_df], axis=1)
    #     return df_with_features, vectorizer
    #
    # except Exception as e:
    #     logger.error(f"TF-IDF提取失败: {str(e)}")
    #     return df, None
    """
    # TODO: 实现TF-IDF特征提取
    pass


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建价格相关特征

    Args:
        df: 输入DataFrame

    Returns:
        添加价格特征后的DataFrame

    TODO 1: 复制DataFrame并记录日志
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("创建价格特征...")
    # df_feat = df.copy()

    TODO 2: 创建价格区间特征（使用pd.cut分桶）
    # if 'discounted_price' in df_feat.columns:
    #     df_feat['price_range'] = pd.cut(
    #         df_feat['discounted_price'],
    #         bins=config.PRICE_BINS,
    #         labels=config.PRICE_LABELS,
    #         include_lowest=True
    #     )
    #     logger.info("  创建特征: price_range (价格区间)")

    TODO 3: 创建价格对数特征（处理价格的偏态分布）
    #     df_feat['log_price'] = np.log1p(df_feat['discounted_price'])
    #     logger.info("  创建特征: log_price (价格对数)")

    TODO 4: 创建折扣金额特征（如果有actual_price）
    # if 'actual_price' in df_feat.columns and 'discounted_price' in df_feat.columns:
    #     df_feat['discount_amount'] = df_feat['actual_price'] - df_feat['discounted_price']
    #     logger.info("  创建特征: discount_amount (折扣金额)")

    TODO 5: 创建折扣力度区间特征
    # if 'discount_percentage' in df_feat.columns:
    #     df_feat['discount_range'] = pd.cut(
    #         df_feat['discount_percentage'],
    #         bins=config.DISCOUNT_BINS,
    #         labels=config.DISCOUNT_LABELS,
    #         include_lowest=True
    #     )
    #     logger.info("  创建特征: discount_range (折扣力度区间)")

    TODO 6: 创建是否打折特征（二值特征）
    #     df_feat['is_discounted'] = (df_feat['discount_percentage'] > 0).astype(int)
    #     logger.info("  创建特征: is_discounted (是否打折)")

    TODO 7: 创建高折扣特征（折扣超过50%）
    #     df_feat['high_discount'] = (df_feat['discount_percentage'] > 50).astype(int)
    #     logger.info("  创建特征: high_discount (高折扣)")

    TODO 8: 返回添加特征后的DataFrame
    # return df_feat
    """
    # TODO: 实现价格特征创建
    pass


def encode_categorical_features(df: pd.DataFrame,
                               categorical_columns: List[str] = None,
                               encoding_method: str = 'onehot') -> Tuple[pd.DataFrame, dict]:
    """
    编码分类特征

    Args:
        df: 输入DataFrame
        categorical_columns: 要编码的分类列列表
        encoding_method: 编码方法 ('onehot' 或 'label')

    Returns:
        (编码后的DataFrame, 编码器字典)

    TODO 1: 设置默认分类列
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info(f"编码分类特征（方法: {encoding_method}）...")
    #
    # if categorical_columns is None:
    #     categorical_columns = ['category', 'price_range', 'discount_range']

    TODO 2: 复制DataFrame和创建编码器字典
    # df_encoded = df.copy()
    # encoders = {}

    TODO 3: 对每个分类列进行编码
    # for col in categorical_columns:
    #     if col not in df_encoded.columns:
    #         logger.warning(f"列 '{col}' 不存在，跳过")
    #         continue

    TODO 4: 使用One-Hot编码
    #     if encoding_method == 'onehot':
    #         # 使用pd.get_dummies进行One-Hot编码
    #         dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
    #         df_encoded = pd.concat([df_encoded, dummies], axis=1)
    #         df_encoded.drop(col, axis=1, inplace=True)
    #         logger.info(f"  列 '{col}': One-Hot编码，生成 {len(dummies.columns)} 个特征")

    TODO 5: 使用Label编码
    #     elif encoding_method == 'label':
    #         # 使用LabelEncoder
    #         le = LabelEncoder()
    #         df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    #         encoders[col] = le
    #         logger.info(f"  列 '{col}': Label编码，{len(le.classes_)} 个类别")

    TODO 6: 返回编码后的DataFrame和编码器
    # return df_encoded, encoders
    """
    # TODO: 实现分类特征编码
    pass


def create_rating_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建评分数量相关特征

    Args:
        df: 输入DataFrame

    Returns:
        添加评分数量特征后的DataFrame

    TODO 1: 初始化
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("创建评分数量特征...")
    # df_feat = df.copy()

    TODO 2: 创建评分数量对数特征（处理长尾分布）
    # if 'rating_count' in df_feat.columns:
    #     df_feat['log_rating_count'] = np.log1p(df_feat['rating_count'])
    #     logger.info("  创建特征: log_rating_count (评分数量对数)")

    TODO 3: 创建评分数量分桶特征
    #     # 分为：低评分量、中评分量、高评分量
    #     rating_count_bins = [0, 10, 100, 1000, float('inf')]
    #     rating_count_labels = ['very_low', 'low', 'medium', 'high']
    #     df_feat['rating_count_range'] = pd.cut(
    #         df_feat['rating_count'],
    #         bins=rating_count_bins,
    #         labels=rating_count_labels,
    #         include_lowest=True
    #     )
    #     logger.info("  创建特征: rating_count_range (评分数量区间)")

    TODO 4: 创建是否为热门商品特征（评分数超过某阈值）
    #     popular_threshold = df_feat['rating_count'].quantile(0.75)  # 前25%
    #     df_feat['is_popular'] = (df_feat['rating_count'] > popular_threshold).astype(int)
    #     logger.info(f"  创建特征: is_popular (是否热门，阈值={popular_threshold:.0f})")

    TODO 5: 返回DataFrame
    # return df_feat
    """
    # TODO: 实现评分数量特征创建
    pass


def create_text_length_features(df: pd.DataFrame,
                               text_columns: List[str] = None) -> pd.DataFrame:
    """
    创建文本长度特征

    Args:
        df: 输入DataFrame
        text_columns: 文本列列表

    Returns:
        添加文本长度特征后的DataFrame

    TODO 1: 设置默认文本列
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("创建文本长度特征...")
    #
    # if text_columns is None:
    #     text_columns = ['review_content', 'review_title', 'about_product']

    TODO 2: 复制DataFrame
    # df_feat = df.copy()

    TODO 3: 对每个文本列创建长度特征
    # for col in text_columns:
    #     if col not in df_feat.columns:
    #         continue
    #
    #     # 字符数
    #     df_feat[f'{col}_length'] = df_feat[col].astype(str).str.len()
    #
    #     # 单词数（按空格分割）
    #     df_feat[f'{col}_word_count'] = df_feat[col].astype(str).str.split().str.len()
    #
    #     logger.info(f"  列 '{col}': 创建长度和单词数特征")

    TODO 4: 返回DataFrame
    # return df_feat
    """
    # TODO: 实现文本长度特征创建
    pass


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建交互特征（特征之间的组合）

    Args:
        df: 输入DataFrame

    Returns:
        添加交互特征后的DataFrame

    TODO 1: 初始化
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("创建交互特征...")
    # df_feat = df.copy()

    TODO 2: 价格 × 折扣百分比 交互
    # if 'discounted_price' in df_feat.columns and 'discount_percentage' in df_feat.columns:
    #     df_feat['price_discount_interaction'] = (
    #         df_feat['discounted_price'] * df_feat['discount_percentage']
    #     )
    #     logger.info("  创建特征: price_discount_interaction")

    TODO 3: 价格 × 评分数量 交互（价格与受欢迎程度）
    # if 'discounted_price' in df_feat.columns and 'rating_count' in df_feat.columns:
    #     df_feat['price_popularity_interaction'] = (
    #         df_feat['discounted_price'] * np.log1p(df_feat['rating_count'])
    #     )
    #     logger.info("  创建特征: price_popularity_interaction")

    TODO 4: 折扣 × 评分数量 交互（折扣力度与受欢迎程度）
    # if 'discount_percentage' in df_feat.columns and 'rating_count' in df_feat.columns:
    #     df_feat['discount_popularity_interaction'] = (
    #         df_feat['discount_percentage'] * np.log1p(df_feat['rating_count'])
    #     )
    #     logger.info("  创建特征: discount_popularity_interaction")

    TODO 5: 价格 / 评分数量比率（性价比指标）
    # if 'discounted_price' in df_feat.columns and 'rating_count' in df_feat.columns:
    #     df_feat['price_per_rating'] = df_feat['discounted_price'] / (df_feat['rating_count'] + 1)
    #     logger.info("  创建特征: price_per_rating")

    TODO 6: 返回DataFrame
    # return df_feat
    """
    # TODO: 实现交互特征创建
    pass


def engineer_features(df: pd.DataFrame,
                     include_text_features: bool = True,
                     text_column: str = 'review_content',
                     tfidf_max_features: int = None) -> Tuple[pd.DataFrame, dict]:
    """
    完整的特征工程流程

    Args:
        df: 输入DataFrame
        include_text_features: 是否包含文本TF-IDF特征
        text_column: 用于TF-IDF的文本列
        tfidf_max_features: TF-IDF最大特征数

    Returns:
        (完成特征工程后的DataFrame, 特征工程组件字典)

    TODO 1: 打印开始信息
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("=" * 60)
    # logger.info("开始特征工程")
    # logger.info("=" * 60)

    TODO 2: 初始化特征工程组件字典（保存编码器、向量化器等）
    # components = {}

    TODO 3: 使用Timer计时整个流程
    # with Timer("特征工程"):
    #     df_feat = df.copy()

    TODO 4: 1. 创建价格特征
    #     df_feat = create_price_features(df_feat)

    TODO 5: 2. 创建评分数量特征
    #     df_feat = create_rating_count_features(df_feat)

    TODO 6: 3. 创建文本长度特征
    #     df_feat = create_text_length_features(df_feat)

    TODO 7: 4. 创建TF-IDF文本特征（如果启用）
    #     if include_text_features and text_column in df_feat.columns:
    #         if tfidf_max_features is None:
    #             tfidf_max_features = config.TFIDF_MAX_FEATURES
    #
    #         df_feat, tfidf_vectorizer = create_text_features(
    #             df_feat,
    #             text_column=text_column,
    #             max_features=tfidf_max_features,
    #             prefix='tfidf'
    #         )
    #         components['tfidf_vectorizer'] = tfidf_vectorizer

    TODO 8: 5. 创建交互特征
    #     df_feat = create_interaction_features(df_feat)

    TODO 9: 6. 编码分类特征
    #     df_feat, encoders = encode_categorical_features(
    #         df_feat,
    #         encoding_method=config.CATEGORY_ENCODING
    #     )
    #     components['encoders'] = encoders

    TODO 10: 打印完成信息
    #     logger.info(f"\n特征工程完成！")
    #     logger.info(f"最终特征数量: {df_feat.shape[1]}")
    #     logger.info(f"样本数量: {df_feat.shape[0]}")

    TODO 11: 返回结果
    # return df_feat, components
    """
    # TODO: 实现完整特征工程流程
    pass


def select_features(df: pd.DataFrame,
                   feature_list: List[str] = None,
                   exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    选择特定特征用于建模

    Args:
        df: 输入DataFrame
        feature_list: 要选择的特征列表（None表示使用所有特征）
        exclude_cols: 要排除的列

    Returns:
        选择后的DataFrame

    TODO 1: 如果没有指定feature_list，使用所有列
    # logger = logging.getLogger("Ecommerce_Rating")
    #
    # if feature_list is None:
    #     feature_list = df.columns.tolist()

    TODO 2: 如果指定了排除列，从feature_list中移除
    # if exclude_cols is not None:
    #     feature_list = [f for f in feature_list if f not in exclude_cols]

    TODO 3: 检查特征是否存在
    # available_features = [f for f in feature_list if f in df.columns]
    # missing_features = [f for f in feature_list if f not in df.columns]
    #
    # if missing_features:
    #     logger.warning(f"以下特征不存在: {missing_features}")

    TODO 4: 返回选择的特征
    # logger.info(f"选择 {len(available_features)} 个特征用于建模")
    # return df[available_features]
    """
    # TODO: 实现特征选择
    pass


if __name__ == '__main__':
    # 测试特征工程
    from src.utils import setup_logger
    from src.data_loader import load_raw_data
    from src.data_preprocessing import preprocess_data

    # TODO: 设置日志
    # logger = setup_logger("Ecommerce_Rating", config.LOG_DIR / "feature_engineering_test.log", "INFO")

    print("=" * 60)
    print("特征工程模块测试")
    print("=" * 60)

    # TODO 1: 加载和预处理数据
    # try:
    #     df = load_raw_data(use_sample=True, sample_size=200)
    #     df_clean = preprocess_data(df, remove_outliers_flag=False)
    #     print(f"\n预处理后数据形状: {df_clean.shape}")

    # TODO 2: 执行特征工程
    #     df_feat, components = engineer_features(
    #         df_clean,
    #         include_text_features=True,
    #         text_column='review_content'
    #     )
    #     print(f"\n特征工程后数据形状: {df_feat.shape}")

    # TODO 3: 查看新增特征
    #     new_features = set(df_feat.columns) - set(df_clean.columns)
    #     print(f"\n新增特征数量: {len(new_features)}")
    #     print(f"新增特征示例: {list(new_features)[:10]}")

    # except Exception as e:
    #     print(f"\n错误: {str(e)}")

    print("\n提示：实现上述TODO后运行此文件进行测试")
