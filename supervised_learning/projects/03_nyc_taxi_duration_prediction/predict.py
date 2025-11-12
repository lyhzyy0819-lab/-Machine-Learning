"""
NYC出租车行程时长预测 - 测试集预测脚本
用于Kaggle竞赛提交或生产环境预测
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from setuptools import logging

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

import config
from src.utils import setup_logger, load_model, Timer
from src.data_loader import load_data_from_csv
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features, select_features


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NYC出租车测试集预测')

    parser.add_argument('--test-file', type=str,
                       default=str(config.TEST_DATA_PATH),
                       help='测试集文件路径')
    parser.add_argument('--model-file', type=str,
                       default=str(config.BEST_MODEL_PATH),
                       help='模型文件路径')
    parser.add_argument('--scaler-file', type=str,
                       default=str(config.SCALER_PATH),
                       help='缩放器文件路径')
    parser.add_argument('--output', type=str,
                       default='submission.csv',
                       help='输出文件路径')
    parser.add_argument('--sample', type=int, default=None,
                       help='样本数量（用于测试）')

    return parser.parse_args()


def load_test_data(test_file: Path, sample_size: int = None):
    """
    加载测试集数据

    Args:
        test_file: 测试集文件路径
        sample_size: 样本大小（可选）

    Returns:
        测试集DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("=" * 60)
    logger.info("加载测试集")
    logger.info("=" * 60)

    test_file = Path(test_file)

    if not test_file.exists():
        raise FileNotFoundError(f"测试集文件不存在: {test_file}")

    # 读取数据
    df = load_data_from_csv(
        test_file,
        nrows=sample_size,
        optimize_memory=True
    )

    logger.info(f"测试集加载完成: {df.shape}")

    return df


def preprocess_test_data(df: pd.DataFrame):
    """
    预处理测试集

    Args:
        df: 测试集DataFrame

    Returns:
        预处理后的DataFrame
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("=" * 60)
    logger.info("预处理测试集")
    logger.info("=" * 60)

    # 对于测试集，不移除异常值（因为需要预测所有样本）
    # 但需要进行基本的数据清洗
    df_clean = df.copy()

    # 转换数据类型
    if 'pickup_datetime' in df_clean.columns:
        df_clean['pickup_datetime'] = pd.to_datetime(df_clean['pickup_datetime'])

    # 检查必需列
    required_cols = ['pickup_longitude', 'pickup_latitude',
                    'dropoff_longitude', 'dropoff_latitude', 'pickup_datetime']

    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"测试集缺少必需列: {missing_cols}")

    logger.info(f"预处理完成: {df_clean.shape}")

    return df_clean


def predict_test_set(model, scaler, test_df: pd.DataFrame):
    """
    对测试集进行预测

    Args:
        model: 训练好的模型
        scaler: 数据缩放器
        test_df: 测试集DataFrame

    Returns:
        预测结果数组
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("=" * 60)
    logger.info("特征工程和预测")
    logger.info("=" * 60)

    with Timer("特征工程"):
        # 特征工程
        test_feat = engineer_features(test_df)

        # 选择建模特征
        X_test = select_features(test_feat)

        logger.info(f"特征矩阵: {X_test.shape}")

    with Timer("模型预测"):
        # 缩放
        X_test_scaled = scaler.transform(X_test)

        # 预测
        predictions = model.predict(X_test_scaled)

        logger.info(f"预测完成: {len(predictions)} 条记录")
        logger.info(f"预测统计:")
        logger.info(f"  均值: {predictions.mean():.2f} 秒 ({predictions.mean()/60:.2f} 分钟)")
        logger.info(f"  中位数: {np.median(predictions):.2f} 秒")
        logger.info(f"  最小值: {predictions.min():.2f} 秒")
        logger.info(f"  最大值: {predictions.max():.2f} 秒")

    return predictions


def generate_submission(test_df: pd.DataFrame, predictions: np.ndarray,
                       output_file: str):
    """
    生成提交文件

    Args:
        test_df: 测试集DataFrame
        predictions: 预测结果
        output_file: 输出文件路径
    """
    logger = logging.getLogger("NYC_Taxi")
    logger.info("=" * 60)
    logger.info("生成提交文件")
    logger.info("=" * 60)

    # 创建提交DataFrame
    if 'id' in test_df.columns:
        submission = pd.DataFrame({
            'id': test_df['id'],
            'trip_duration': predictions
        })
    else:
        # 如果没有id列，使用索引
        submission = pd.DataFrame({
            'id': range(len(predictions)),
            'trip_duration': predictions
        })

    # 确保预测值为正数
    submission['trip_duration'] = submission['trip_duration'].clip(lower=0)

    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    logger.info(f"提交文件已保存: {output_path}")
    logger.info(f"提交文件大小: {len(submission)} 行")
    logger.info(f"\n前5行预测结果:")
    print(submission.head())

    return submission


def main():
    """主函数"""

    # 解析参数
    args = parse_args()

    # 设置日志
    import logging
    logger = setup_logger("NYC_Taxi", config.LOG_FILE, "INFO")

    logger.info("=" * 80)
    logger.info(" " * 20 + "NYC出租车测试集预测")
    logger.info("=" * 80)

    try:
        with Timer("完整预测流程"):

            # 1. 加载模型
            logger.info("\n" + "=" * 60)
            logger.info("步骤 1/4: 加载模型")
            logger.info("=" * 60)

            model = load_model(Path(args.model_file))
            scaler = load_model(Path(args.scaler_file))

            logger.info(f"模型类型: {type(model).__name__}")

            # 2. 加载测试集
            logger.info("\n" + "=" * 60)
            logger.info("步骤 2/4: 加载测试集")
            logger.info("=" * 60)

            test_df = load_test_data(args.test_file, args.sample)

            # 3. 预处理
            logger.info("\n" + "=" * 60)
            logger.info("步骤 3/4: 数据预处理")
            logger.info("=" * 60)

            test_clean = preprocess_test_data(test_df)

            # 4. 预测
            logger.info("\n" + "=" * 60)
            logger.info("步骤 4/4: 预测")
            logger.info("=" * 60)

            predictions = predict_test_set(model, scaler, test_clean)

            # 5. 生成提交文件
            submission = generate_submission(test_clean, predictions, args.output)

        logger.info("\n" + "=" * 80)
        logger.info("预测完成！")
        logger.info("=" * 80)
        logger.info(f"\n输出文件: {args.output}")
        logger.info("可以将此文件提交到Kaggle竞赛平台")

        return 0

    except Exception as e:
        logger.error(f"\n预测过程中出现错误: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    """
    使用示例:

    # 基本用法（使用默认路径）
    python predict.py

    # 指定测试集文件
    python predict.py --test-file data/raw/test.csv

    # 指定输出文件名
    python predict.py --output my_submission.csv

    # 指定模型文件
    python predict.py --model-file models/best_model.pkl

    # 快速测试（只预测前1000条）
    python predict.py --sample 1000

    # 完整示例
    python predict.py --test-file data/raw/test.csv --output submission.csv
    """

    exit_code = main()
    sys.exit(exit_code)
