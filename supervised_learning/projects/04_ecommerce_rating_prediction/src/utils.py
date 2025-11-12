"""
工具函数模块
包含日志记录、模型保存加载、性能监控等通用工具
"""

import os
import json
import joblib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


def setup_logger(name: str = "Ecommerce_Rating",
                 log_file: Optional[Path] = None,
                 level: str = "INFO") -> logging.Logger:
    """
    配置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（如果为None则只输出到控制台）
        level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）

    Returns:
        配置好的日志记录器

    TODO 1: 创建日志记录器
    # logger = logging.getLogger(name)
    # logger.setLevel(getattr(logging, level.upper()))

    TODO 2: 清除已有的处理器（避免重复日志）
    # logger.handlers.clear()

    TODO 3: 创建格式化器
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S'
    # )

    TODO 4: 添加控制台处理器
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    TODO 5: 如果指定了日志文件，添加文件处理器
    # if log_file is not None:
    #     log_file.parent.mkdir(parents=True, exist_ok=True)
    #     file_handler = logging.FileHandler(log_file, encoding='utf-8')
    #     file_handler.setFormatter(formatter)
    #     logger.addHandler(file_handler)

    TODO 6: 返回配置好的logger
    # return logger
    """
    # TODO: 实现日志记录器配置
    pass


def save_model(model: Any,
               filepath: Path,
               metadata: Optional[Dict] = None) -> None:
    """
    保存模型到文件

    Args:
        model: 要保存的模型对象
        filepath: 保存路径
        metadata: 模型元数据（可选）

    TODO 1: 确保保存目录存在
    # filepath.parent.mkdir(parents=True, exist_ok=True)

    TODO 2: 使用joblib保存模型
    # joblib.dump(model, filepath)

    TODO 3: 如果有元数据，保存到同目录下的metadata.json
    # if metadata is not None:
    #     metadata_path = filepath.parent / "metadata.json"
    #     save_json(metadata, metadata_path)

    TODO 4: 记录日志
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info(f"模型已保存到: {filepath}")
    """
    # TODO: 实现模型保存
    pass


def load_model(filepath: Path) -> Any:
    """
    从文件加载模型

    Args:
        filepath: 模型文件路径

    Returns:
        加载的模型对象

    TODO 1: 检查文件是否存在
    # if not filepath.exists():
    #     raise FileNotFoundError(f"模型文件不存在: {filepath}")

    TODO 2: 使用joblib加载模型
    # model = joblib.load(filepath)

    TODO 3: 记录日志并返回模型
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info(f"模型已加载: {filepath}")
    # return model
    """
    # TODO: 实现模型加载
    pass


def save_json(data: Dict, filepath: Path) -> None:
    """
    保存字典到JSON文件

    Args:
        data: 要保存的字典数据
        filepath: 保存路径

    TODO 1: 确保目录存在
    # filepath.parent.mkdir(parents=True, exist_ok=True)

    TODO 2: 定义类型转换函数（NumPy和Pandas类型转Python原生类型）
    # def convert_types(obj):
    #     if isinstance(obj, np.integer):
    #         return int(obj)
    #     elif isinstance(obj, np.floating):
    #         return float(obj)
    #     elif isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     elif isinstance(obj, pd.Series):
    #         return obj.to_dict()
    #     elif isinstance(obj, pd.DataFrame):
    #         return obj.to_dict('records')
    #     elif isinstance(obj, datetime):
    #         return obj.isoformat()
    #     return obj

    TODO 3: 递归转换所有数据
    # def deep_convert(data):
    #     if isinstance(data, dict):
    #         return {k: deep_convert(v) for k, v in data.items()}
    #     elif isinstance(data, list):
    #         return [deep_convert(item) for item in data]
    #     else:
    #         return convert_types(data)

    TODO 4: 保存到JSON文件
    # converted_data = deep_convert(data)
    # with open(filepath, 'w', encoding='utf-8') as f:
    #     json.dump(converted_data, f, indent=4, ensure_ascii=False)
    """
    # TODO: 实现JSON保存
    pass


def load_json(filepath: Path) -> Dict:
    """
    从JSON文件加载数据

    Args:
        filepath: JSON文件路径

    Returns:
        加载的字典数据

    TODO: 实现JSON加载（检查文件存在性，读取并返回数据）
    """
    # TODO: 实现JSON加载
    pass


class Timer:
    """
    计时器类，用于性能监控

    用法:
        with Timer("数据加载"):
            # 执行操作
            pass

    TODO 1: 在__init__中初始化name、start_time和logger
    # self.name = name
    # self.start_time = None
    # self.logger = logging.getLogger("Ecommerce_Rating")

    TODO 2: 在__enter__中记录开始时间并打印日志
    # self.start_time = time.time()
    # self.logger.info(f"开始 {self.name}...")
    # return self

    TODO 3: 在__exit__中计算耗时并打印日志
    # elapsed = time.time() - self.start_time
    # if elapsed < 60:
    #     time_str = f"{elapsed:.2f}秒"
    # else:
    #     time_str = f"{elapsed/60:.2f}分钟"
    # self.logger.info(f"{self.name} 完成，耗时: {time_str}")
    """

    def __init__(self, name: str = "操作"):
        # TODO: 实现初始化
        pass

    def __enter__(self):
        # TODO: 实现进入上下文管理器
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: 实现退出上下文管理器
        pass


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    优化DataFrame内存使用
    通过降低数值类型精度来减少内存占用

    Args:
        df: 输入DataFrame
        verbose: 是否打印详细信息

    Returns:
        优化后的DataFrame

    TODO 1: 计算初始内存使用
    # start_mem = df.memory_usage().sum() / 1024**2

    TODO 2: 遍历所有列，对数值列进行类型压缩
    # for col in df.columns:
    #     col_type = df[col].dtype
    #     if col_type != object:  # 只处理数值列
    #         c_min = df[col].min()
    #         c_max = df[col].max()

    TODO 3: 对整数类型选择最小的数据类型
    #         if str(col_type)[:3] == 'int':
    #             if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
    #                 df[col] = df[col].astype(np.int8)
    #             elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
    #                 df[col] = df[col].astype(np.int16)
    #             elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
    #                 df[col] = df[col].astype(np.int32)

    TODO 4: 对浮点类型选择最小的数据类型
    #         else:  # float类型
    #             if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
    #                 df[col] = df[col].astype(np.float32)

    TODO 5: 计算优化后的内存使用并打印
    # end_mem = df.memory_usage().sum() / 1024**2
    # if verbose:
    #     logger = logging.getLogger("Ecommerce_Rating")
    #     logger.info(f"内存使用从 {start_mem:.2f} MB 减少到 {end_mem:.2f} MB "
    #                f"(减少 {100 * (start_mem - end_mem) / start_mem:.1f}%)")
    """
    # TODO: 实现内存优化
    pass


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    打印DataFrame详细信息

    Args:
        df: 要查看的DataFrame
        name: DataFrame名称

    TODO 1: 打印基本信息（形状、内存使用、数据类型分布）
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info("=" * 60)
    # logger.info(f"{name} 基本信息")
    # logger.info("=" * 60)
    # logger.info(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    # logger.info(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    # logger.info(f"\n数据类型分布:\n{df.dtypes.value_counts()}")

    TODO 2: 打印缺失值统计
    # missing = df.isnull().sum()
    # if missing.sum() > 0:
    #     logger.info("\n缺失值统计:")
    #     missing_df = pd.DataFrame({
    #         '缺失数量': missing[missing > 0],
    #         '缺失比例': (missing[missing > 0] / len(df) * 100).round(2)
    #     })
    #     logger.info(f"\n{missing_df}")
    # else:
    #     logger.info("\n无缺失值")
    """
    # TODO: 实现DataFrame信息打印
    pass


def get_timestamp() -> str:
    """
    获取当前时间戳字符串

    Returns:
        格式化的时间戳字符串 (YYYYMMDD_HHMMSS)

    TODO: 使用datetime.now().strftime()生成时间戳
    # return datetime.now().strftime("%Y%m%d_%H%M%S")
    """
    # TODO: 实现时间戳获取
    pass


def ensure_dir(directory: Path) -> None:
    """
    确保目录存在，如果不存在则创建

    Args:
        directory: 目录路径

    TODO: 使用mkdir创建目录
    # directory.mkdir(parents=True, exist_ok=True)
    """
    # TODO: 实现目录创建
    pass


if __name__ == '__main__':
    # 测试工具函数
    print("=" * 60)
    print("工具函数测试")
    print("=" * 60)

    # TODO: 添加测试代码
    # 1. 测试日志记录器
    # 2. 测试计时器
    # 3. 测试时间戳生成
    # 4. 测试内存优化（创建测试DataFrame）

    print("\n提示：实现上述TODO后运行此文件进行测试")
