"""
数据预处理代码模板库
===================

快速使用:
    from code_templates.preprocessing_templates import (
        quick_impute,
        quick_outlier_clip,
        quick_encode,
        quick_scale,
        build_quick_pipeline
    )

    # 5行代码完成预处理
    df = quick_impute(df, strategy='median')
    df = quick_outlier_clip(df, columns=['price'])
    df = quick_encode(df, method='auto')
    df = quick_scale(df, method='standard')

    # 或使用一键式Pipeline
    X_train, y_train, X_test, y_test = build_quick_pipeline(
        df, target_col='price', algorithm_type='xgboost'
    )

对应决策模板: 07_decision_templates/data_diagnosis_template.md
参考实现: 06_comprehensive_project/src/data_preprocessing.py (653行)

项目定位: ML实战操作手册（非教学项目）
核心价值: 5-15分钟快速代码落地
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
import warnings
warnings.filterwarnings('ignore')


# ==================== 1. 缺失值处理 ====================

def quick_impute(df: pd.DataFrame,
                strategy: str = 'auto',
                numeric_strategy: str = 'median',
                categorical_strategy: str = 'mode',
                drop_threshold: float = 0.5,
                verbose: bool = True) -> pd.DataFrame:
    """
    快速缺失值处理（5分钟决策）

    对应决策: data_diagnosis_template.md - Step 1: 缺失值快速处理

    Parameters
    ----------
    df : DataFrame
        输入数据
    strategy : {'auto', 'median', 'mean', 'mode', 'drop'}
        'auto' - 根据缺失率自动选择（推荐）
        其他 - 手动指定策略
    numeric_strategy : str, default='median'
        数值列填充策略（当strategy='auto'时）
        'median' - 中位数填充（推荐，对异常值鲁棒）
        'mean' - 均值填充
    categorical_strategy : str, default='mode'
        类别列填充策略（当strategy='auto'时）
    drop_threshold : float, default=0.5
        缺失率超过此值的列直接删除（0.5即50%）
    verbose : bool, default=True
        是否打印处理信息

    Returns
    -------
    DataFrame
        处理后的数据

    Examples
    --------
    >>> # 快速模式：全自动
    >>> df_clean = quick_impute(df)
    >>> # ✓ 缺失值处理完成: 15个缺失值已填充, 2列已删除(缺失率>50%)

    >>> # 定制模式：指定策略
    >>> df_clean = quick_impute(
    ...     df,
    ...     numeric_strategy='mean',
    ...     drop_threshold=0.3
    ... )

    Decision Logic (对应07章决策)
    -----------------------------
    缺失率 < 5%  → 删除行（样本充足时）
    缺失率 5-20% → 中位数/众数填充
    缺失率 20-50% → KNN填充（可选，用advanced_impute）
    缺失率 > 50% → 删除列

    Notes
    -----
    - 快速模式适合Baseline建立
    - 重要项目建议使用advanced_impute()
    - 参考06章src/data_preprocessing.py:130-188
    """
    df_copy = df.copy()

    if verbose:
        print("🔍 缺失值诊断...")
        missing_stats = df_copy.isnull().sum()
        missing_stats = missing_stats[missing_stats > 0]
        if len(missing_stats) > 0:
            print(f"   发现 {len(missing_stats)} 列有缺失值")
        else:
            print("   ✓ 无缺失值")
            return df_copy

    # 1. 删除缺失率过高的列
    cols_to_drop = []
    for col in df_copy.columns:
        missing_rate = df_copy[col].isnull().sum() / len(df_copy)
        if missing_rate > drop_threshold:
            cols_to_drop.append(col)

    if cols_to_drop:
        df_copy = df_copy.drop(columns=cols_to_drop)
        if verbose:
            print(f"   ✓ 删除{len(cols_to_drop)}列(缺失率>{drop_threshold*100}%): {cols_to_drop[:3]}...")

    # 2. 填充缺失值
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns

    # 数值列填充
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            if df_copy[col].isnull().sum() > 0:
                if numeric_strategy == 'median':
                    fill_value = df_copy[col].median()
                elif numeric_strategy == 'mean':
                    fill_value = df_copy[col].mean()
                else:
                    fill_value = 0
                df_copy[col].fillna(fill_value, inplace=True)

        if verbose:
            print(f"   ✓ 数值列填充完成（策略: {numeric_strategy}）")

    # 类别列填充
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            if df_copy[col].isnull().sum() > 0:
                fill_value = df_copy[col].mode()[0] if len(df_copy[col].mode()) > 0 else 'Unknown'
                df_copy[col].fillna(fill_value, inplace=True)

        if verbose:
            print(f"   ✓ 类别列填充完成（策略: {categorical_strategy}）")

    if verbose:
        print(f"✓ 缺失值处理完成\n")

    return df_copy


# ==================== 2. 异常值处理 ====================

def quick_outlier_clip(df: pd.DataFrame,
                      columns: List[str] = None,
                      method: str = 'iqr',
                      k: float = 1.5,
                      verbose: bool = True) -> pd.DataFrame:
    """
    快速异常值截断（IQR方法，保留所有样本）

    对应决策: data_diagnosis_template.md - Step 2: 异常值快速处理

    Parameters
    ----------
    columns : list, optional
        需要处理的列名，None则处理所有数值列
    method : {'iqr', 'percentile'}
        iqr - IQR方法（推荐）
        percentile - 百分位数方法
    k : float, default=1.5
        IQR倍数
        1.5 - 标准值（检测温和异常）
        3.0 - 宽松值（仅检测极端异常）

    Returns
    -------
    DataFrame
        异常值截断后的数据（保留所有样本）

    Examples
    --------
    >>> # 快速模式
    >>> df_clean = quick_outlier_clip(df, columns=['price', 'age'])
    >>> # ✓ 异常值截断完成: price(15个), age(8个)

    >>> # 宽松模式（仅处理极端异常）
    >>> df_clean = quick_outlier_clip(df, columns=['price'], k=3.0)

    Decision Logic (对应07章决策)
    -----------------------------
    真实极值 + 线性模型 → 截断（clip）
    真实极值 + 树模型   → 保留（不处理）
    数据错误           → 删除（用quick_outlier_remove）

    Notes
    -----
    - 截断保留样本数量，适合大部分场景
    - 树模型对异常值鲁棒，可跳过此步骤
    - 参考06章src/data_preprocessing.py:193-243
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

    outlier_counts = {}

    for col in columns:
        if col not in df_copy.columns:
            continue

        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
        elif method == 'percentile':
            lower_bound = df_copy[col].quantile(0.01)
            upper_bound = df_copy[col].quantile(0.99)

        # 统计异常值数量
        n_outliers = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).sum()
        if n_outliers > 0:
            outlier_counts[col] = n_outliers

        # 截断
        df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)

    if verbose:
        if outlier_counts:
            print("✓ 异常值截断完成:")
            for col, count in outlier_counts.items():
                print(f"   {col}: {count}个异常值已截断")
        else:
            print("✓ 未检测到异常值")
        print()

    return df_copy


# ==================== 3. 特征编码 ====================

def quick_encode(df: pd.DataFrame,
                columns: List[str] = None,
                method: str = 'auto',
                algorithm_type: str = 'tree',
                verbose: bool = True) -> pd.DataFrame:
    """
    快速特征编码（自动识别最佳方法）

    对应决策: preprocessing_quick_reference.md - Step 3: 特征编码

    Parameters
    ----------
    columns : list, optional
        需要编码的列名，None则处理所有object/category列
    method : {'auto', 'onehot', 'label', 'target'}
        'auto' - 根据基数自动选择（推荐）
            基数 < 10  → One-Hot
            基数 10-50 → Label Encoding
            基数 > 50  → Label Encoding
    algorithm_type : {'tree', 'linear', 'nn'}
        算法类型，影响编码选择
        tree   - 树模型: Label Encoding
        linear - 线性模型: One-Hot
        nn     - 神经网络: One-Hot

    Returns
    -------
    DataFrame
        编码后的数据

    Examples
    --------
    >>> # 快速模式：自动编码
    >>> df_encoded = quick_encode(df)
    >>> # ✓ 特征编码完成: gender(onehot), city(label)

    >>> # 指定算法类型
    >>> df_encoded = quick_encode(df, algorithm_type='linear')
    >>> # 线性模型 → 优先使用One-Hot

    Decision Logic
    --------------
    无序分类 + 基数<10 + 线性模型 → One-Hot
    无序分类 + 基数>10            → Label/Target
    有序分类                      → Label Encoding

    Notes
    -----
    - 树模型对编码方式不敏感，Label Encoding即可
    - 线性模型建议One-Hot
    - 参考06章src/data_preprocessing.py:248-386
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(columns) == 0:
        if verbose:
            print("✓ 无需编码（无类别特征）\n")
        return df_copy

    encoding_info = {}

    for col in columns:
        if col not in df_copy.columns:
            continue

        cardinality = df_copy[col].nunique()

        # 自动选择编码方式
        if method == 'auto':
            if algorithm_type == 'linear' and cardinality < 10:
                chosen_method = 'onehot'
            else:
                chosen_method = 'label'
        else:
            chosen_method = method

        # 执行编码
        if chosen_method == 'onehot':
            dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
            df_copy = df_copy.drop(col, axis=1)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            encoding_info[col] = f"onehot({cardinality} → {len(dummies.columns)}列)"

        elif chosen_method == 'label':
            df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))
            encoding_info[col] = f"label({cardinality}类)"

    if verbose:
        print("✓ 特征编码完成:")
        for col, info in encoding_info.items():
            print(f"   {col}: {info}")
        print()

    return df_copy


# ==================== 4. 特征缩放 ====================

def quick_scale(df: pd.DataFrame,
               columns: List[str] = None,
               method: str = 'auto',
               algorithm_type: str = 'tree',
               verbose: bool = True) -> pd.DataFrame:
    """
    快速特征缩放

    Parameters
    ----------
    columns : list, optional
        需要缩放的列名，None则处理所有数值列
    method : {'auto', 'standard', 'minmax', 'robust'}
        'auto' - 根据算法类型自动选择
            线性模型/SVM → Standard
            神经网络     → MinMax
            树模型       → 不缩放
            有异常值     → Robust
    algorithm_type : {'tree', 'linear', 'nn'}
        算法类型

    Returns
    -------
    DataFrame
        缩放后的数据

    Examples
    --------
    >>> # 线性模型：需要标准化
    >>> df_scaled = quick_scale(df, algorithm_type='linear')

    >>> # 树模型：跳过缩放
    >>> df = df  # 不需要缩放

    Decision Logic
    --------------
    树模型（XGBoost/RF） → 不需要缩放
    线性模型/SVM      → Standard Scaler
    神经网络          → MinMax Scaler
    有异常值          → Robust Scaler
    """
    # 树模型不需要缩放
    if algorithm_type == 'tree':
        if verbose:
            print("✓ 树模型不需要特征缩放，跳过\n")
        return df

    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

    # 自动选择缩放方法
    if method == 'auto':
        if algorithm_type == 'linear':
            method = 'standard'
        elif algorithm_type == 'nn':
            method = 'minmax'

    # 执行缩放
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()

    df_copy[columns] = scaler.fit_transform(df_copy[columns])

    if verbose:
        print(f"✓ 特征缩放完成（方法: {method}, {len(columns)}列）\n")

    return df_copy


# ==================== 5. 完整Pipeline构建 ====================

def build_quick_pipeline(df: pd.DataFrame,
                        target_col: str,
                        algorithm_type: str = 'tree',
                        test_size: float = 0.2,
                        random_state: int = 42,
                        verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    5分钟构建完整预处理Pipeline

    对应决策: 完成07章所有决策后，一键生成预处理流程

    Parameters
    ----------
    df : DataFrame
        原始数据
    target_col : str
        目标变量列名
    algorithm_type : {'tree', 'linear', 'nn'}
        算法类型（影响预处理策略）
        tree   - 树模型: 最简预处理（仅缺失值+编码）
        linear - 线性模型: 完整预处理（+缩放）
        nn     - 神经网络: 完整预处理（+归一化）
    test_size : float, default=0.2
        测试集比例
    random_state : int, default=42
        随机种子

    Returns
    -------
    X_train, y_train, X_test, y_test : tuple
        预处理后的训练集和测试集

    Examples
    --------
    >>> # 5行代码完成完整预处理
    >>> X_train, y_train, X_test, y_test = build_quick_pipeline(
    ...     df,
    ...     target_col='price',
    ...     algorithm_type='xgboost'
    ... )
    >>> print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    Pipeline Steps
    --------------
    1. 缺失值处理
    2. 异常值处理（可选）
    3. 特征编码
    4. 特征缩放（根据algorithm_type）
    5. 划分训练/测试集

    Notes
    -----
    - 适合快速Baseline建立
    - 重要项目建议手动控制每个步骤
    """
    from sklearn.model_selection import train_test_split

    if verbose:
        print("="*60)
        print("   快速预处理Pipeline")
        print("="*60 + "\n")

    df_processed = df.copy()

    # 1. 缺失值处理
    df_processed = quick_impute(df_processed, verbose=verbose)

    # 2. 异常值处理（可选）
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    # df_processed = quick_outlier_clip(df_processed, columns=numeric_cols, verbose=verbose)

    # 3. 特征编码
    df_processed = quick_encode(df_processed, algorithm_type=algorithm_type, verbose=verbose)

    # 4. 分离X和y
    y = df_processed[target_col]
    X = df_processed.drop(target_col, axis=1)

    # 5. 特征缩放
    X = quick_scale(X, algorithm_type=algorithm_type, verbose=verbose)

    # 6. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if df_processed[target_col].nunique() < 10 else None
    )

    if verbose:
        print("✓ 数据划分完成")
        print(f"   训练集: {X_train.shape}, 测试集: {X_test.shape}")
        print("="*60 + "\n")

    return X_train, y_train, X_test, y_test
