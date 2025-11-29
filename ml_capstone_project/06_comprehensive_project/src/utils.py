"""
工具函数模块
=============

提供通用的工具函数，供其他模块使用。

主要功能:
- 数据加载与保存
- 时间和日志工具
- 数据切分
- 随机种子设置
"""

import os
import pickle
import joblib
import json
import logging
from pathlib import Path
from typing import Tuple, Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split


# ==================== 日志工具 ====================

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    配置日志器

    Args:
        name: 日志器名称
        log_file: 日志文件路径（可选）
        level: 日志级别

    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有handlers
    logger.handlers = []

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ==================== 随机种子 ====================

def set_seed(seed: int = 42):
    """
    设置所有随机种子，保证可复现性

    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ==================== 数据加载与保存 ====================

def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    加载数据文件（支持csv、xlsx、json、pkl）

    Args:
        file_path: 文件路径
        **kwargs: 传递给pandas读取函数的参数

    Returns:
        数据DataFrame
    """
    file_ext = Path(file_path).suffix.lower()

    if file_ext == '.csv':
        return pd.read_csv(file_path, **kwargs)
    elif file_ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path, **kwargs)
    elif file_ext == '.json':
        return pd.read_json(file_path, **kwargs)
    elif file_ext in ['.pkl', '.pickle']:
        return pd.read_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")


def save_data(df: pd.DataFrame, file_path: str, **kwargs):
    """
    保存数据（根据扩展名自动选择格式）

    Args:
        df: 数据DataFrame
        file_path: 保存路径
        **kwargs: 传递给pandas保存函数的参数
    """
    file_ext = Path(file_path).suffix.lower()

    # 确保目录存在
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    if file_ext == '.csv':
        df.to_csv(file_path, index=False, **kwargs)
    elif file_ext in ['.xlsx', '.xls']:
        df.to_excel(file_path, index=False, **kwargs)
    elif file_ext == '.json':
        df.to_json(file_path, **kwargs)
    elif file_ext in ['.pkl', '.pickle']:
        df.to_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")


# ==================== 模型保存与加载 ====================

def save_model(model: Any, file_path: str, method: str = 'joblib'):
    """
    保存模型

    Args:
        model: 模型对象
        file_path: 保存路径
        method: 保存方法 ('joblib' 或 'pickle')
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    if method == 'joblib':
        joblib.dump(model, file_path)
    elif method == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"不支持的保存方法: {method}")


def load_model(file_path: str, method: str = 'joblib') -> Any:
    """
    加载模型

    Args:
        file_path: 模型文件路径
        method: 加载方法 ('joblib' 或 'pickle')

    Returns:
        模型对象
    """
    if method == 'joblib':
        return joblib.load(file_path)
    elif method == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"不支持的加载方法: {method}")


# ==================== JSON工具 ====================

def save_json(data: Dict, file_path: str):
    """保存字典到JSON文件"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)


def load_json(file_path: str) -> Dict:
    """从JSON文件加载字典"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==================== 数据切分 ====================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple:
    """
    将数据切分为训练集、验证集、测试集

    Args:
        X: 特征数据
        y: 目标变量
        train_size: 训练集比例
        val_size: 验证集比例
        test_size: 测试集比例
        random_state: 随机种子
        stratify: 是否分层抽样

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # 检查比例
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        f"切分比例之和必须为1，当前为{train_size + val_size + test_size}"

    # 第一次切分：分出测试集
    stratify_y = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )

    # 第二次切分：从剩余数据中分出验证集
    val_ratio = val_size / (train_size + val_size)
    stratify_y_temp = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==================== 时间工具 ====================

def get_timestamp(format: str = '%Y%m%d_%H%M%S') -> str:
    """
    获取当前时间戳字符串

    Args:
        format: 时间格式

    Returns:
        格式化的时间字符串
    """
    return datetime.now().strftime(format)


class Timer:
    """计时器上下文管理器"""

    def __init__(self, name: str = 'Timer'):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"{self.name} 耗时: {self.elapsed:.2f}秒")


# ==================== 文件操作 ====================

def ensure_dir(path: str):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def list_files(directory: str, pattern: str = '*', recursive: bool = False) -> List[str]:
    """
    列出目录下的文件

    Args:
        directory: 目录路径
        pattern: 文件模式（如'*.csv'）
        recursive: 是否递归搜索

    Returns:
        文件路径列表
    """
    path = Path(directory)

    if recursive:
        files = path.rglob(pattern)
    else:
        files = path.glob(pattern)

    return [str(f) for f in files if f.is_file()]


# ==================== 内存优化 ====================

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    优化DataFrame的内存使用
    通过降低数值类型的精度来减少内存占用

    Args:
        df: 数据DataFrame
        verbose: 是否打印优化信息

    Returns:
        优化后的DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(f'内存使用优化: {start_mem:.2f} MB -> {end_mem:.2f} MB '
              f'(减少 {100 * (start_mem - end_mem) / start_mem:.1f}%)')

    return df


# ==================== 数据验证 ====================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None):
    """
    验证DataFrame的基本属性

    Args:
        df: 数据DataFrame
        required_columns: 必需的列名列表
    """
    if df is None or df.empty:
        raise ValueError("数据为空")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")


# ==================== 进度显示 ====================

def print_progress(current: int, total: int, message: str = ''):
    """
    打印进度条

    Args:
        current: 当前进度
        total: 总数
        message: 附加信息
    """
    percent = current / total * 100
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '=' * filled + '-' * (bar_length - filled)

    print(f'\r[{bar}] {percent:.1f}% {message}', end='', flush=True)

    if current == total:
        print()  # 完成后换行


if __name__ == '__main__':
    # 测试示例
    print("=== 工具函数测试 ===\n")

    # 测试计时器
    with Timer("测试计时器"):
        import time
        time.sleep(1)

    # 测试随机种子
    set_seed(42)
    print(f"\n随机数: {np.random.rand(3)}")

    # 测试时间戳
    print(f"当前时间戳: {get_timestamp()}")

    print("\n✅ 所有测试通过！")
