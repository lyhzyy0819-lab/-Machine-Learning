"""
工具函数模块
包含日志、计时、模型保存加载等通用功能
"""

import logging
import time
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str, log_file: Path, level: str = "INFO") -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别

    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 文件处理器
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, level))

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class Timer:
    """
    计时器上下文管理器
    """

    def __init__(self, description: str = "Operation"):
        """
        初始化计时器

        Args:
            description: 操作描述
        """
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """进入上下文"""
        self.start_time = time.time()
        print(f"\n[{self.description}] 开始...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"[{self.description}] 完成! 耗时: {elapsed:.2f}秒")


def save_model(model: Any, filepath: Path) -> None:
    """
    保存模型到文件

    Args:
        model: 要保存的模型对象
        filepath: 保存路径
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"模型已保存: {filepath}")


def load_model(filepath: Path) -> Any:
    """
    从文件加载模型

    Args:
        filepath: 模型文件路径

    Returns:
        加载的模型对象
    """
    if not filepath.exists():
        raise FileNotFoundError(f"模型文件不存在: {filepath}")

    model = joblib.load(filepath)
    print(f"模型已加载: {filepath}")
    return model


def save_json(data: Dict, filepath: Path) -> None:
    """
    保存字典到JSON文件

    Args:
        data: 要保存的字典
        filepath: 保存路径
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"JSON已保存: {filepath}")


def load_json(filepath: Path) -> Dict:
    """
    从JSON文件加载字典

    Args:
        filepath: JSON文件路径

    Returns:
        加载的字典
    """
    if not filepath.exists():
        raise FileNotFoundError(f"JSON文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"JSON已加载: {filepath}")
    return data


def print_section(title: str, width: int = 80) -> None:
    """
    打印格式化的章节标题

    Args:
        title: 标题文本
        width: 总宽度
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection(title: str, width: int = 80) -> None:
    """
    打印格式化的子章节标题

    Args:
        title: 标题文本
        width: 总宽度
    """
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def optimize_dataframe_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    优化DataFrame内存使用

    Args:
        df: 输入DataFrame
        verbose: 是否打印信息

    Returns:
        优化后的DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f"\n优化前内存使用: {start_mem:.2f} MB")

    # 优化数值型列
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # 优化object类型列
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f"优化后内存使用: {end_mem:.2f} MB")
        print(f"减少: {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df


def get_feature_names_from_column_transformer(column_transformer, feature_names):
    """
    从ColumnTransformer获取转换后的特征名称

    Args:
        column_transformer: sklearn的ColumnTransformer对象
        feature_names: 原始特征名称列表

    Returns:
        转换后的特征名称列表
    """
    output_features = []

    for name, pipe, features in column_transformer.transformers_:
        if name == 'remainder':
            continue

        if hasattr(pipe, 'get_feature_names_out'):
            # 对于支持get_feature_names_out的转换器
            if isinstance(features, list):
                pipe_features = pipe.get_feature_names_out(features)
            else:
                pipe_features = pipe.get_feature_names_out([features])
            output_features.extend(pipe_features)
        else:
            # 对于不支持的转换器,直接使用原始特征名
            if isinstance(features, list):
                output_features.extend(features)
            else:
                output_features.append(features)

    return output_features


def format_time(seconds: float) -> str:
    """
    格式化时间显示

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"


def ensure_dir(path: Path) -> None:
    """
    确保目录存在，如果不存在则创建

    Args:
        path: 目录路径
    """
    path.mkdir(parents=True, exist_ok=True)


def create_directory_structure(base_dir: Path) -> None:
    """
    创建项目目录结构

    Args:
        base_dir: 项目根目录
    """
    directories = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "models",
        base_dir / "figures",
        base_dir / "logs",
        base_dir / "src"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("目录结构创建完成")


if __name__ == '__main__':
    # 测试工具函数
    print("=" * 60)
    print("工具函数模块测试")
    print("=" * 60)

    # 测试计时器
    with Timer("测试计时器"):
        time.sleep(1)

    # 测试格式化函数
    print_section("测试章节标题")
    print_subsection("测试子章节标题")

    # 测试时间格式化
    print(f"\n时间格式化测试:")
    print(f"  30秒: {format_time(30)}")
    print(f"  90秒: {format_time(90)}")
    print(f"  3700秒: {format_time(3700)}")

    print("\n工具函数测试完成!")
