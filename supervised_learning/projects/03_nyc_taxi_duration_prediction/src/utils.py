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


def setup_logger(name: str = "NYC_Taxi",
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
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 清除已有的处理器
    logger.handlers.clear()

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_model(model: Any,
               filepath: Path,
               metadata: Optional[Dict] = None) -> None:
    """
    保存模型到文件

    Args:
        model: 要保存的模型对象
        filepath: 保存路径
        metadata: 模型元数据（可选）
    """
    try:
        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 保存模型
        joblib.dump(model, filepath)

        # 保存元数据
        if metadata is not None:
            metadata_path = filepath.parent / "metadata.json"
            save_json(metadata, metadata_path)

        logger = logging.getLogger("NYC_Taxi")
        logger.info(f"模型已保存到: {filepath}")

    except Exception as e:
        logger = logging.getLogger("NYC_Taxi")
        logger.error(f"保存模型失败: {str(e)}")
        raise


def load_model(filepath: Path) -> Any:
    """
    从文件加载模型

    Args:
        filepath: 模型文件路径

    Returns:
        加载的模型对象
    """
    try:
        if not filepath.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        model = joblib.load(filepath)

        logger = logging.getLogger("NYC_Taxi")
        logger.info(f"模型已加载: {filepath}")

        return model

    except Exception as e:
        logger = logging.getLogger("NYC_Taxi")
        logger.error(f"加载模型失败: {str(e)}")
        raise


def save_json(data: Dict, filepath: Path) -> None:
    """
    保存字典到JSON文件

    Args:
        data: 要保存的字典数据
        filepath: 保存路径
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 转换NumPy类型为Python原生类型
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        # 递归转换
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(item) for item in data]
            else:
                return convert_types(data)

        converted_data = deep_convert(data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=4, ensure_ascii=False)

        logger = logging.getLogger("NYC_Taxi")
        logger.info(f"JSON数据已保存到: {filepath}")

    except Exception as e:
        logger = logging.getLogger("NYC_Taxi")
        logger.error(f"保存JSON失败: {str(e)}")
        raise


def load_json(filepath: Path) -> Dict:
    """
    从JSON文件加载数据

    Args:
        filepath: JSON文件路径

    Returns:
        加载的字典数据
    """
    try:
        if not filepath.exists():
            raise FileNotFoundError(f"JSON文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger = logging.getLogger("NYC_Taxi")
        logger.info(f"JSON数据已加载: {filepath}")

        return data

    except Exception as e:
        logger = logging.getLogger("NYC_Taxi")
        logger.error(f"加载JSON失败: {str(e)}")
        raise


class Timer:
    """
    计时器类，用于性能监控

    用法:
        with Timer("数据加载"):
            # 执行操作
            pass
    """

    def __init__(self, name: str = "操作"):
        self.name = name
        self.start_time = None
        self.logger = logging.getLogger("NYC_Taxi")

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"开始 {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            time_str = f"{elapsed:.2f}秒"
        else:
            time_str = f"{elapsed/60:.2f}分钟"

        self.logger.info(f"{self.name} 完成，耗时: {time_str}")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    优化DataFrame内存使用
    通过降低数值类型精度来减少内存占用

    Args:
        df: 输入DataFrame
        verbose: 是否打印详细信息

    Returns:
        优化后的DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger = logging.getLogger("NYC_Taxi")

    for col in df.columns:
        col_type = df[col].dtype

        # 跳过 object、datetime 和 category 类型
        if col_type == object or str(col_type).startswith('datetime') or str(col_type) == 'category':
            continue

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
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        logger.info(f"内存使用从 {start_mem:.2f} MB 减少到 {end_mem:.2f} MB "
                   f"(减少 {100 * (start_mem - end_mem) / start_mem:.1f}%)")

    return df


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    打印DataFrame详细信息

    Args:
        df: 要查看的DataFrame
        name: DataFrame名称
    """
    logger = logging.getLogger("NYC_Taxi")

    logger.info("=" * 60)
    logger.info(f"{name} 基本信息")
    logger.info("=" * 60)
    logger.info(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    logger.info(f"内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info("\n数据类型分布:")
    logger.info(f"{df.dtypes.value_counts()}")

    # 缺失值统计
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info("\n缺失值统计:")
        missing_df = pd.DataFrame({
            '缺失数量': missing[missing > 0],
            '缺失比例': (missing[missing > 0] / len(df) * 100).round(2)
        })
        logger.info(f"\n{missing_df}")
    else:
        logger.info("\n无缺失值")

    logger.info("=" * 60)


def create_prediction_function(model, scaler, feature_names: List[str]):
    """
    创建预测函数

    Args:
        model: 训练好的模型
        scaler: 数据缩放器
        feature_names: 特征名称列表

    Returns:
        预测函数
    """
    def predict(input_data: Dict) -> Dict:
        """
        对单个样本进行预测

        Args:
            input_data: 输入特征字典

        Returns:
            预测结果字典
        """
        try:
            # 转换为DataFrame
            df = pd.DataFrame([input_data])

            # 确保所有特征存在
            for feat in feature_names:
                if feat not in df.columns:
                    raise ValueError(f"缺少特征: {feat}")

            # 选择特征并排序
            X = df[feature_names]

            # 缩放
            X_scaled = scaler.transform(X)

            # 预测
            pred = model.predict(X_scaled)[0]

            # 返回结果
            return {
                'predicted_duration_seconds': float(pred),
                'predicted_duration_minutes': float(pred / 60),
                'status': 'success'
            }

        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }

    return predict


def format_duration(seconds: float) -> str:
    """
    格式化时长显示

    Args:
        seconds: 秒数

    Returns:
        格式化的字符串
    """
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: Path) -> None:
    """确保目录存在"""
    directory.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # 测试工具函数
    print("=" * 60)
    print("工具函数测试")
    print("=" * 60)

    # 测试日志
    logger = setup_logger("TEST", config.LOG_FILE, "INFO")
    logger.info("日志记录器测试成功")

    # 测试计时器
    with Timer("测试操作"):
        time.sleep(1)

    # 测试时长格式化
    print(f"\n时长格式化测试:")
    print(f"45秒: {format_duration(45)}")
    print(f"150秒: {format_duration(150)}")
    print(f"3700秒: {format_duration(3700)}")

    # 测试时间戳
    print(f"\n当前时间戳: {get_timestamp()}")

    print("\n所有工具函数测试完成！")
