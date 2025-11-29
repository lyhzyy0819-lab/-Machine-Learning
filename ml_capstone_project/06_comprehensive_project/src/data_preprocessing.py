"""
数据预处理模块
==============

提供全面的数据预处理功能，将原始数据转换为适合建模的格式。

主要功能:
- 缺失值处理（删除、填充、建模）
- 异常值处理（截断、删除、转换）
- 特征编码（Label Encoding、One-Hot Encoding、Target Encoding）
- 特征缩放（标准化、归一化）
- 数据平衡（SMOTE、类权重调整）
- 数据清洗（重复值、常量特征）
- Pipeline构建

遵循sklearn的fit-transform模式，支持训练集和测试集的一致性处理。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   LabelEncoder, OneHotEncoder)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')


# ==================== 缺失值处理 ====================

class MissingValueHandler:
    """
    缺失值处理器

    支持多种填充策略:
    - 'drop': 删除含有缺失值的样本
    - 'mean': 均值填充（数值型）
    - 'median': 中位数填充（数值型）
    - 'mode': 众数填充（类别型）
    - 'constant': 常数填充
    - 'knn': KNN填充（根据相似样本填充）
    """

    def __init__(self, strategy: Dict[str, str] = None):
        """
        Args:
            strategy: 字典，键为列名，值为填充策略
                     如 {'age': 'median', 'gender': 'mode'}
                     如果为None，则对所有列使用默认策略
        """
        self.strategy = strategy or {}
        self.imputers = {}  # 存储每列的imputer
        self.fill_values = {}  # 存储填充值

    def fit(self, df: pd.DataFrame) -> 'MissingValueHandler':
        """
        学习填充策略

        Args:
            df: 训练数据

        Returns:
            self
        """
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            strategy = self.strategy.get(col, None)

            # 数值型列的默认策略
            if strategy is None:
                if pd.api.types.is_numeric_dtype(df[col]):
                    strategy = 'median'
                else:
                    strategy = 'mode'

            # 根据策略创建imputer
            if strategy in ['mean', 'median', 'most_frequent']:
                imputer = SimpleImputer(strategy=strategy)
                imputer.fit(df[[col]])
                self.imputers[col] = imputer
                self.fill_values[col] = imputer.statistics_[0]

            elif strategy == 'mode':
                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else None
                self.fill_values[col] = mode_value

            elif strategy == 'constant':
                # 常数填充，需要在strategy字典中指定具体值
                # 如: {'col_name': ('constant', 0)}
                if isinstance(strategy, tuple):
                    _, fill_value = strategy
                    self.fill_values[col] = fill_value

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用填充策略

        Args:
            df: 要转换的数据

        Returns:
            填充后的数据
        """
        df_filled = df.copy()

        for col, fill_value in self.fill_values.items():
            if col in df_filled.columns:
                if col in self.imputers:
                    # 使用sklearn imputer
                    df_filled[col] = self.imputers[col].transform(df_filled[[col]])
                else:
                    # 直接填充
                    df_filled[col].fillna(fill_value, inplace=True)

        return df_filled

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(df).transform(df)


def handle_missing_values(df: pd.DataFrame,
                         numeric_strategy: str = 'median',
                         categorical_strategy: str = 'mode',
                         drop_threshold: float = 0.5) -> pd.DataFrame:
    """
    智能处理缺失值

    处理逻辑:
    1. 缺失率 > drop_threshold 的列直接删除
    2. 剩余数值型列使用指定策略填充
    3. 剩余类别型列使用指定策略填充

    Args:
        df: 数据DataFrame
        numeric_strategy: 数值型填充策略 ('mean', 'median')
        categorical_strategy: 类别型填充策略 ('mode', 'constant')
        drop_threshold: 缺失率阈值，超过则删除列

    Returns:
        处理后的DataFrame
    """
    df_processed = df.copy()

    # 计算缺失率
    missing_ratio = df_processed.isnull().sum() / len(df_processed)

    # 删除缺失率过高的列
    cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    if cols_to_drop:
        print(f"删除缺失率>{drop_threshold*100}%的列: {cols_to_drop}")
        df_processed.drop(columns=cols_to_drop, inplace=True)

    # 数值型列填充
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            if numeric_strategy == 'mean':
                fill_value = df_processed[col].mean()
            elif numeric_strategy == 'median':
                fill_value = df_processed[col].median()
            else:
                fill_value = 0

            df_processed[col].fillna(fill_value, inplace=True)
            print(f"  {col}: 使用{numeric_strategy}填充 ({fill_value:.2f})")

    # 类别型列填充
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            if categorical_strategy == 'mode':
                fill_value = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
            else:
                fill_value = 'Unknown'

            df_processed[col].fillna(fill_value, inplace=True)
            print(f"  {col}: 使用{categorical_strategy}填充 ({fill_value})")

    return df_processed


# ==================== 异常值处理 ====================

def handle_outliers_iqr(df: pd.DataFrame, columns: List[str],
                       method: str = 'clip', k: float = 1.5) -> pd.DataFrame:
    """
    使用IQR方法处理异常值

    Args:
        df: 数据DataFrame
        columns: 要处理的列名列表
        method: 处理方法
                'clip': 截断到上下界
                'remove': 删除异常值样本
                'nan': 将异常值替换为NaN
        k: IQR倍数

    Returns:
        处理后的DataFrame
    """
    df_processed = df.copy()

    for col in columns:
        if col not in df_processed.columns:
            continue

        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        # 识别异常值
        outliers = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound))
        n_outliers = outliers.sum()

        if n_outliers > 0:
            if method == 'clip':
                # 截断
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"  {col}: 截断 {n_outliers} 个异常值到 [{lower_bound:.2f}, {upper_bound:.2f}]")

            elif method == 'remove':
                # 删除样本
                df_processed = df_processed[~outliers]
                print(f"  {col}: 删除 {n_outliers} 个异常样本")

            elif method == 'nan':
                # 替换为NaN
                df_processed.loc[outliers, col] = np.nan
                print(f"  {col}: 将 {n_outliers} 个异常值替换为NaN")

    return df_processed


# ==================== 特征编码 ====================

class CategoricalEncoder:
    """
    类别特征编码器

    支持多种编码方式:
    - Label Encoding: 适用于有序类别特征
    - One-Hot Encoding: 适用于无序类别特征（类别数较少）
    - Target Encoding: 使用目标变量的统计量编码
    """

    def __init__(self, encoding_type: str = 'onehot',
                 handle_unknown: str = 'ignore'):
        """
        Args:
            encoding_type: 编码类型 ('label', 'onehot', 'target')
            handle_unknown: 如何处理未知类别 ('ignore', 'error')
        """
        self.encoding_type = encoding_type
        self.handle_unknown = handle_unknown
        self.encoders = {}
        self.encoded_columns = []

    def fit(self, df: pd.DataFrame, columns: List[str],
           target: Optional[pd.Series] = None) -> 'CategoricalEncoder':
        """
        学习编码映射

        Args:
            df: 训练数据
            columns: 要编码的列名列表
            target: 目标变量（Target Encoding需要）

        Returns:
            self
        """
        for col in columns:
            if col not in df.columns:
                continue

            if self.encoding_type == 'label':
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                self.encoders[col] = encoder

            elif self.encoding_type == 'onehot':
                encoder = OneHotEncoder(sparse_output=False,
                                       handle_unknown=self.handle_unknown)
                encoder.fit(df[[col]])
                self.encoders[col] = encoder

            elif self.encoding_type == 'target':
                if target is None:
                    raise ValueError("Target Encoding 需要提供目标变量")

                # 计算每个类别的目标均值
                target_means = df.groupby(col)[target.name].mean().to_dict()
                global_mean = target.mean()
                self.encoders[col] = (target_means, global_mean)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用编码

        Args:
            df: 要转换的数据

        Returns:
            编码后的数据
        """
        df_encoded = df.copy()

        for col, encoder in self.encoders.items():
            if col not in df_encoded.columns:
                continue

            if self.encoding_type == 'label':
                # Label Encoding
                df_encoded[col] = encoder.transform(df_encoded[col].astype(str))

            elif self.encoding_type == 'onehot':
                # One-Hot Encoding
                encoded_array = encoder.transform(df_encoded[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

                # 创建新的DataFrame
                encoded_df = pd.DataFrame(encoded_array,
                                         columns=feature_names,
                                         index=df_encoded.index)

                # 删除原列，添加编码后的列
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

            elif self.encoding_type == 'target':
                # Target Encoding
                target_means, global_mean = encoder
                df_encoded[col] = df_encoded[col].map(target_means).fillna(global_mean)

        return df_encoded

    def fit_transform(self, df: pd.DataFrame, columns: List[str],
                     target: Optional[pd.Series] = None) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(df, columns, target).transform(df)


def encode_categorical_features(df: pd.DataFrame,
                                columns: List[str],
                                method: str = 'onehot',
                                drop_first: bool = True) -> pd.DataFrame:
    """
    快速编码类别特征

    Args:
        df: 数据DataFrame
        columns: 要编码的列名列表
        method: 编码方法 ('onehot', 'label')
        drop_first: One-Hot时是否删除第一列（避免多重共线性）

    Returns:
        编码后的DataFrame
    """
    df_encoded = df.copy()

    if method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns,
                                   drop_first=drop_first, dtype=int)
        print(f"One-Hot编码: {len(columns)}个特征 -> {df_encoded.shape[1]-df.shape[1]+len(columns)}个特征")

    elif method == 'label':
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        print(f"Label编码: {len(columns)}个特征")

    return df_encoded


# ==================== 特征缩放 ====================

class FeatureScaler:
    """
    特征缩放器

    支持多种缩放方法:
    - StandardScaler: 标准化 (mean=0, std=1)
    - MinMaxScaler: 归一化 [0, 1]
    - RobustScaler: 鲁棒缩放（对异常值不敏感）
    """

    def __init__(self, method: str = 'standard'):
        """
        Args:
            method: 缩放方法 ('standard', 'minmax', 'robust')
        """
        self.method = method

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的缩放方法: {method}")

        self.feature_names = None

    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> 'FeatureScaler':
        """
        学习缩放参数

        Args:
            df: 训练数据
            columns: 要缩放的列名列表，None表示所有数值列

        Returns:
            self
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_names = columns
        self.scaler.fit(df[columns])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用缩放

        Args:
            df: 要转换的数据

        Returns:
            缩放后的数据
        """
        df_scaled = df.copy()

        if self.feature_names:
            scaled_array = self.scaler.transform(df[self.feature_names])
            df_scaled[self.feature_names] = scaled_array

        return df_scaled

    def fit_transform(self, df: pd.DataFrame,
                     columns: Optional[List[str]] = None) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(df, columns).transform(df)


# ==================== 数据平衡 ====================

def balance_data(X: pd.DataFrame, y: pd.Series,
                method: str = 'smote',
                sampling_strategy: Union[str, float] = 'auto',
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    处理类别不平衡问题

    Args:
        X: 特征数据
        y: 目标变量
        method: 平衡方法
                'oversample': 随机过采样
                'undersample': 随机欠采样
                'smote': SMOTE过采样
        sampling_strategy: 采样策略
                          'auto': 自动平衡到相等数量
                          float: 目标比例（少数类/多数类）
        random_state: 随机种子

    Returns:
        (平衡后的X, 平衡后的y)
    """
    print(f"\n原始类别分布:")
    print(y.value_counts())

    if method == 'oversample':
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy,
                                   random_state=random_state)
    elif method == 'undersample':
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy,
                                    random_state=random_state)
    elif method == 'smote':
        # SMOTE要求数据为数值型
        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            raise ValueError("SMOTE要求所有特征为数值型，请先进行编码")

        sampler = SMOTE(sampling_strategy=sampling_strategy,
                       random_state=random_state)
    else:
        raise ValueError(f"不支持的平衡方法: {method}")

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    print(f"\n{method.upper()}后的类别分布:")
    print(pd.Series(y_resampled).value_counts())

    # 转换回DataFrame
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)

    return X_resampled, y_resampled


# ==================== 数据清洗 ====================

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None,
                     keep: str = 'first') -> pd.DataFrame:
    """
    删除重复行

    Args:
        df: 数据DataFrame
        subset: 用于判断重复的列，None表示所有列
        keep: 保留策略 ('first', 'last', False)

    Returns:
        去重后的DataFrame
    """
    n_before = len(df)
    df_dedup = df.drop_duplicates(subset=subset, keep=keep)
    n_removed = n_before - len(df_dedup)

    print(f"删除 {n_removed} 行重复数据 ({n_removed/n_before*100:.1f}%)")

    return df_dedup


def remove_constant_features(df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """
    删除常量特征或接近常量的特征

    Args:
        df: 数据DataFrame
        threshold: 阈值，如果某个值的频率超过此值则删除该特征

    Returns:
        删除常量特征后的DataFrame
    """
    constant_cols = []

    for col in df.columns:
        # 检查唯一值比例
        if df[col].nunique() == 1:
            constant_cols.append(col)
        else:
            # 检查最常见值的频率
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > threshold:
                constant_cols.append(col)

    if constant_cols:
        print(f"删除 {len(constant_cols)} 个常量特征: {constant_cols}")
        df_cleaned = df.drop(columns=constant_cols)
    else:
        print("未发现常量特征")
        df_cleaned = df.copy()

    return df_cleaned


# ==================== Pipeline构建 ====================

def build_preprocessing_pipeline(numeric_features: List[str],
                               categorical_features: List[str],
                               numeric_strategy: str = 'median',
                               categorical_strategy: str = 'most_frequent',
                               scaling_method: str = 'standard') -> ColumnTransformer:
    """
    构建sklearn预处理Pipeline

    自动处理数值型和类别型特征的填充、编码、缩放

    Args:
        numeric_features: 数值型特征列表
        categorical_features: 类别型特征列表
        numeric_strategy: 数值型填充策略
        categorical_strategy: 类别型填充策略
        scaling_method: 缩放方法

    Returns:
        ColumnTransformer对象
    """
    # 数值型特征pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_strategy)),
        ('scaler', StandardScaler() if scaling_method == 'standard' else MinMaxScaler())
    ])

    # 类别型特征pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_strategy)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 组合
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


# ==================== 高级缺失值填充 ====================

def knn_impute(df: pd.DataFrame, columns: Optional[List[str]] = None,
              n_neighbors: int = 5) -> pd.DataFrame:
    """
    使用KNN方法填充缺失值

    原理: 基于相似样本(K个最近邻居)的值来填充缺失值

    优点: 比简单填充更准确,考虑了特征间的关系
    缺点: 计算开销大,需要所有特征为数值型

    Args:
        df: 数据DataFrame
        columns: 要填充的列,None表示所有数值列
        n_neighbors: K值,邻居数量

    Returns:
        填充后的DataFrame
    """
    df_filled = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # KNN填充
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_filled[columns] = imputer.fit_transform(df[columns])

    print(f"KNN填充完成: {len(columns)}个特征, K={n_neighbors}")

    return df_filled


def iterative_impute(df: pd.DataFrame, columns: Optional[List[str]] = None,
                    max_iter: int = 10) -> pd.DataFrame:
    """
    使用迭代方法填充缺失值(MICE算法)

    原理: 将每个有缺失值的特征作为目标,用其他特征预测并填充,
         多次迭代直到收敛

    优点: 更准确,可以捕捉特征间的复杂关系
    缺点: 计算开销大,需要多次迭代

    Args:
        df: 数据DataFrame
        columns: 要填充的列,None表示所有数值列
        max_iter: 最大迭代次数

    Returns:
        填充后的DataFrame
    """
    df_filled = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # 迭代填充
    imputer = IterativeImputer(max_iter=max_iter, random_state=42)
    df_filled[columns] = imputer.fit_transform(df[columns])

    print(f"迭代填充完成: {len(columns)}个特征, 最大迭代{max_iter}次")

    return df_filled


# ==================== 高级特征编码 ====================

def frequency_encoding(df: pd.DataFrame, columns: List[str],
                      normalize: bool = True) -> pd.DataFrame:
    """
    频率编码: 用类别出现的频率替换类别值

    适用场景: 高基数类别特征(类别数>50)

    优点:
    - 不增加特征维度
    - 保留了类别的统计信息
    - 适合高基数特征

    Args:
        df: 数据DataFrame
        columns: 要编码的列名列表
        normalize: 是否归一化频率到[0,1]

    Returns:
        编码后的DataFrame
    """
    df_encoded = df.copy()

    for col in columns:
        if col not in df_encoded.columns:
            continue

        # 计算频率
        freq = df_encoded[col].value_counts(normalize=normalize)

        # 映射
        df_encoded[col] = df_encoded[col].map(freq)

        # 处理未见过的类别(填充为0或最小频率)
        min_freq = freq.min() if len(freq) > 0 else 0
        df_encoded[col].fillna(min_freq, inplace=True)

    print(f"频率编码完成: {len(columns)}个特征")

    return df_encoded


def target_encoding_with_smoothing(df: pd.DataFrame, columns: List[str],
                                   target: pd.Series, smoothing: float = 10.0) -> pd.DataFrame:
    """
    带平滑的目标编码(Target Encoding)

    原理: 用目标变量的条件均值替换类别值,并添加平滑避免过拟合

    平滑公式: (n * cat_mean + m * global_mean) / (n + m)
        n: 该类别的样本数
        m: 平滑参数
        cat_mean: 该类别的目标均值
        global_mean: 全局目标均值

    适用场景: 高基数类别特征,且与目标变量有关联

    Args:
        df: 数据DataFrame
        columns: 要编码的列名列表
        target: 目标变量Series
        smoothing: 平滑参数,越大越趋向全局均值

    Returns:
        编码后的DataFrame
    """
    df_encoded = df.copy()
    global_mean = target.mean()

    for col in columns:
        if col not in df_encoded.columns:
            continue

        # 计算每个类别的统计量
        stats = df_encoded.groupby(col).agg({
            col: 'count'  # 计数
        }).rename(columns={col: 'count'})

        stats['target_mean'] = df_encoded.groupby(col)[target.name].mean()

        # 应用平滑
        stats['smoothed_mean'] = (
            (stats['count'] * stats['target_mean'] + smoothing * global_mean) /
            (stats['count'] + smoothing)
        )

        # 映射
        encoding_map = stats['smoothed_mean'].to_dict()
        df_encoded[col] = df_encoded[col].map(encoding_map).fillna(global_mean)

    print(f"目标编码(带平滑)完成: {len(columns)}个特征, smoothing={smoothing}")

    return df_encoded


# ==================== 数据分箱 ====================

def binning_numeric_features(df: pd.DataFrame, columns: List[str],
                            n_bins: int = 5, strategy: str = 'quantile',
                            encode: str = 'ordinal') -> pd.DataFrame:
    """
    数值特征分箱

    将连续数值特征离散化为分类特征

    Args:
        df: 数据DataFrame
        columns: 要分箱的列名列表
        n_bins: 箱数
        strategy: 分箱策略
                  'uniform': 等宽分箱
                  'quantile': 等频分箱
                  'kmeans': 基于KMeans的分箱
        encode: 编码方式
                'ordinal': 序数编码 (0, 1, 2, ...)
                'onehot': One-Hot编码

    Returns:
        分箱后的DataFrame
    """
    from sklearn.preprocessing import KBinsDiscretizer

    df_binned = df.copy()

    for col in columns:
        if col not in df_binned.columns:
            continue

        # 创建分箱器
        binning = KBinsDiscretizer(n_bins=n_bins,
                                   encode=encode,
                                   strategy=strategy)

        # 分箱
        binned_values = binning.fit_transform(df_binned[[col]])

        if encode == 'ordinal':
            df_binned[f'{col}_binned'] = binned_values.astype(int)
        else:  # onehot
            # 创建多个列
            for i in range(n_bins):
                df_binned[f'{col}_bin_{i}'] = binned_values[:, i]

        # 可选: 删除原始列
        # df_binned.drop(columns=[col], inplace=True)

    print(f"数值分箱完成: {len(columns)}个特征, {n_bins}个箱, 策略={strategy}")

    return df_binned


# ==================== 快速预处理 (Phase 2专用) ====================

def quick_preprocess(df: pd.DataFrame, target_col: Optional[str] = None,
                    drop_missing_threshold: float = 0.5,
                    handle_categorical: bool = True,
                    scale_features: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    快速预处理数据(用于Phase 2 Baseline)

    执行最基本的预处理步骤,快速准备数据用于建模:
    1. 分离目标变量
    2. 删除ID列和常量列
    3. 删除重复行
    4. 删除缺失率过高的列
    5. 简单填充剩余缺失值
    6. 类别特征One-Hot编码
    7. (可选)标准化数值特征

    Args:
        df: 原始数据DataFrame
        target_col: 目标变量列名
        drop_missing_threshold: 缺失率阈值(删除超过此值的列)
        handle_categorical: 是否编码类别特征
        scale_features: 是否缩放数值特征

    Returns:
        (处理后的特征DataFrame, 目标变量Series)
    """
    print("=" * 60)
    print("快速预处理开始...")
    print("=" * 60)

    df_processed = df.copy()

    # 1. 分离目标变量
    if target_col and target_col in df_processed.columns:
        y = df_processed[target_col].copy()
        df_processed = df_processed.drop(columns=[target_col])
        print(f"1. 目标变量分离: {target_col}")
    else:
        y = None
        print(f"1. 无目标变量")

    # 2. 删除ID列(唯一值比例>95%)
    id_cols = []
    for col in df_processed.columns:
        unique_ratio = df_processed[col].nunique() / len(df_processed)
        if unique_ratio > 0.95:
            id_cols.append(col)

    if id_cols:
        df_processed = df_processed.drop(columns=id_cols)
        print(f"2. 删除ID列: {id_cols}")
    else:
        print(f"2. 未发现ID列")

    # 3. 删除常量列
    constant_cols = [col for col in df_processed.columns
                    if df_processed[col].nunique() == 1]
    if constant_cols:
        df_processed = df_processed.drop(columns=constant_cols)
        print(f"3. 删除常量列: {constant_cols}")
    else:
        print(f"3. 未发现常量列")

    # 4. 删除重复行
    n_before = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    n_removed = n_before - len(df_processed)
    if n_removed > 0:
        print(f"4. 删除重复行: {n_removed}行")
        # 同步删除目标变量中的对应行
        if y is not None:
            y = y[df_processed.index]
    else:
        print(f"4. 无重复行")

    # 5. 删除缺失率过高的列
    missing_ratio = df_processed.isnull().sum() / len(df_processed)
    high_missing_cols = missing_ratio[missing_ratio > drop_missing_threshold].index.tolist()
    if high_missing_cols:
        df_processed = df_processed.drop(columns=high_missing_cols)
        print(f"5. 删除高缺失列(>{drop_missing_threshold*100}%): {high_missing_cols}")
    else:
        print(f"5. 无高缺失列")

    # 6. 填充剩余缺失值
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()

    missing_count = df_processed.isnull().sum().sum()
    if missing_count > 0:
        # 数值列用中位数填充
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)

        # 类别列用众数填充
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
                df_processed[col].fillna(mode_val, inplace=True)

        print(f"6. 填充缺失值: {missing_count}个缺失值")
    else:
        print(f"6. 无缺失值")

    # 7. 类别特征编码
    if handle_categorical and len(categorical_cols) > 0:
        # 对于基线模型,使用简单的One-Hot编码
        # 限制:最多20个类别,否则删除该特征
        low_cardinality_cols = []
        high_cardinality_cols = []

        for col in categorical_cols:
            n_unique = df_processed[col].nunique()
            if n_unique <= 20:
                low_cardinality_cols.append(col)
            else:
                high_cardinality_cols.append(col)

        # 删除高基数类别特征
        if high_cardinality_cols:
            df_processed = df_processed.drop(columns=high_cardinality_cols)
            print(f"7a. 删除高基数类别特征(>20类): {high_cardinality_cols}")

        # One-Hot编码低基数特征
        if low_cardinality_cols:
            df_processed = pd.get_dummies(df_processed, columns=low_cardinality_cols,
                                         drop_first=True, dtype=int)
            print(f"7b. One-Hot编码: {len(low_cardinality_cols)}个特征 → {df_processed.shape[1] - len(numeric_cols)}个特征")
        else:
            print(f"7. 无需编码的类别特征")

    # 8. (可选)标准化数值特征
    if scale_features and len(numeric_cols) > 0:
        # 只标准化原有的数值列(不包括One-Hot编码后的0/1列)
        existing_numeric = [col for col in numeric_cols if col in df_processed.columns]
        if existing_numeric:
            scaler = StandardScaler()
            df_processed[existing_numeric] = scaler.fit_transform(df_processed[existing_numeric])
            print(f"8. 标准化数值特征: {len(existing_numeric)}个特征")
    else:
        print(f"8. 跳过标准化")

    print("=" * 60)
    print(f"✅ 快速预处理完成!")
    print(f"   输入: {df.shape}")
    print(f"   输出: {df_processed.shape}")
    print("=" * 60)

    return df_processed, y


# ==================== 数据集划分 ====================

def split_data(X: pd.DataFrame, y: pd.Series,
              train_size: float = 0.7,
              val_size: float = 0.15,
              test_size: float = 0.15,
              random_state: int = 42,
              stratify: bool = True) -> Tuple:
    """
    将数据划分为训练集、验证集、测试集

    Args:
        X: 特征数据
        y: 目标变量
        train_size: 训练集比例
        val_size: 验证集比例
        test_size: 测试集比例
        random_state: 随机种子
        stratify: 是否分层采样(保持类别比例)

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split

    # 检查比例
    if abs(train_size + val_size + test_size - 1.0) > 0.01:
        raise ValueError(f"train_size + val_size + test_size 必须等于1.0")

    # 是否分层
    stratify_param = y if stratify and y.nunique() <= 20 else None

    # 第一次划分: 分离测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    # 第二次划分: 从剩余数据中分离验证集
    val_ratio = val_size / (train_size + val_size)
    stratify_param2 = y_temp if stratify and y_temp.nunique() <= 20 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_param2
    )

    print(f"数据集划分完成:")
    print(f"  训练集: {X_train.shape[0]} ({train_size*100:.0f}%)")
    print(f"  验证集: {X_val.shape[0]} ({val_size*100:.0f}%)")
    print(f"  测试集: {X_test.shape[0]} ({test_size*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    # 测试示例
    print("=== 数据预处理模块测试 ===\n")

    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'numeric1': np.random.randn(1000) * 10 + 50,
        'numeric2': np.random.randn(1000),
        'category1': np.random.choice(['A', 'B', 'C'], 1000),
        'category2': np.random.choice(['X', 'Y'], 1000),
        'target': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })

    # 添加缺失值
    test_data.loc[np.random.choice(1000, 50, replace=False), 'numeric1'] = np.nan
    test_data.loc[np.random.choice(1000, 30, replace=False), 'category1'] = np.nan

    print("1. 缺失值处理")
    handler = MissingValueHandler()
    data_filled = handler.fit_transform(test_data)
    print(f"✓ 缺失值填充完成\n")

    print("2. 特征编码")
    data_encoded = encode_categorical_features(data_filled,
                                              columns=['category1', 'category2'],
                                              method='onehot')
    print(f"✓ 特征编码完成\n")

    print("3. 特征缩放")
    scaler = FeatureScaler(method='standard')
    numeric_cols = ['numeric1', 'numeric2']
    data_scaled = scaler.fit_transform(data_encoded, columns=numeric_cols)
    print(f"✓ 特征缩放完成\n")

    print("✅ 所有测试通过！")
