"""
客户流失预测 - 预测脚本
用于对新客户数据进行流失风险预测
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

import config
from src.utils import setup_logger, load_model, Timer
from src.data_loader import load_data_from_csv
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features, select_features


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='客户流失预测')

    parser.add_argument('--input', type=str,
                       default=None,
                       help='输入数据文件路径（CSV格式）')
    parser.add_argument('--model-file', type=str,
                       default=str(config.BEST_MODEL_PATH),
                       help='模型文件路径')
    parser.add_argument('--metadata-file', type=str,
                       default=str(config.METADATA_PATH),
                       help='模型元数据文件路径')
    parser.add_argument('--output', type=str,
                       default='predictions.csv',
                       help='输出文件路径')
    parser.add_argument('--sample', type=int, default=None,
                       help='样本数量（用于测试）')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='分类阈值（默认0.5）')
    parser.add_argument('--show-probability', action='store_true',
                       help='显示流失概率')
    parser.add_argument('--top-k', type=int, default=None,
                       help='显示流失风险最高的前K个客户')

    return parser.parse_args()


def load_model_and_metadata(model_path: Path, metadata_path: Path):
    """
    加载模型和元数据

    Args:
        model_path: 模型文件路径
        metadata_path: 元数据文件路径

    Returns:
        (模型, 元数据字典)
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info("=" * 60)
    logger.info("加载模型")
    logger.info("=" * 60)

    # 加载模型
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = load_model(model_path)
    logger.info(f"模型加载成功: {type(model).__name__}")

    # 加载元数据（如果存在）
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"元数据加载成功")

        # 打印模型信息
        if 'model_name' in metadata:
            logger.info(f"  模型名称: {metadata['model_name']}")
        if 'train_date' in metadata:
            logger.info(f"  训练日期: {metadata['train_date']}")
        if 'metrics' in metadata:
            logger.info(f"  性能指标:")
            for metric, value in metadata['metrics'].items():
                logger.info(f"    {metric}: {value:.4f}")
        if 'feature_count' in metadata:
            logger.info(f"  特征数量: {metadata['feature_count']}")
    else:
        logger.warning(f"元数据文件不存在: {metadata_path}")

    return model, metadata


def load_input_data(input_file: Path, sample_size: int = None):
    """
    加载输入数据

    Args:
        input_file: 输入文件路径
        sample_size: 样本大小（可选）

    Returns:
        输入数据DataFrame
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info("=" * 60)
    logger.info("加载输入数据")
    logger.info("=" * 60)

    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 读取数据
    df = load_data_from_csv(
        input_file,
        nrows=sample_size,
        optimize_memory=True
    )

    logger.info(f"输入数据加载完成: {df.shape}")

    return df


def preprocess_input_data(df: pd.DataFrame):
    """
    预处理输入数据

    Args:
        df: 输入DataFrame

    Returns:
        预处理后的DataFrame
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info("=" * 60)
    logger.info("预处理输入数据")
    logger.info("=" * 60)

    # 保存customerID（如果存在）
    customer_ids = None
    if config.ID_COL in df.columns:
        customer_ids = df[config.ID_COL].copy()
        logger.info(f"  保存客户ID列: {config.ID_COL}")

    # 预处理（不包含目标变量编码）
    df_clean = preprocess_data(
        df,
        handle_missing=True,
        remove_outliers=False,  # 预测时不移除异常值
        encode_target=False  # 预测时没有目标变量
    )

    logger.info(f"预处理完成: {df_clean.shape}")

    # 恢复customerID
    if customer_ids is not None:
        df_clean[config.ID_COL] = customer_ids

    return df_clean


def predict_churn(model, input_df: pd.DataFrame, metadata: dict = None,
                 threshold: float = 0.5):
    """
    预测客户流失

    Args:
        model: 训练好的模型
        input_df: 输入数据DataFrame
        metadata: 模型元数据
        threshold: 分类阈值

    Returns:
        (预测结果DataFrame, 预测概率)
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info("=" * 60)
    logger.info("特征工程和预测")
    logger.info("=" * 60)

    with Timer("特征工程"):
        # 特征工程
        input_feat = engineer_features(input_df)

        # 如果存在目标变量列，移除它
        if config.TARGET_COL in input_feat.columns:
            input_feat = input_feat.drop(config.TARGET_COL, axis=1)

        # 保存ID列
        customer_ids = None
        if config.ID_COL in input_feat.columns:
            customer_ids = input_feat[config.ID_COL].copy()
            input_feat = input_feat.drop(config.ID_COL, axis=1)

        # 选择建模特征
        # 注意：这里需要确保特征顺序与训练时一致
        if metadata and 'feature_names' in metadata:
            feature_names = metadata['feature_names']
            logger.info(f"使用元数据中的特征列表: {len(feature_names)} 个特征")

            # 检查缺失的特征
            missing_features = set(feature_names) - set(input_feat.columns)
            if missing_features:
                logger.warning(f"输入数据缺少特征: {missing_features}")
                # 添加缺失特征（填充0）
                for feat in missing_features:
                    input_feat[feat] = 0

            # 选择特征并保持顺序
            X_input = input_feat[feature_names]
        else:
            # 如果没有元数据，使用所有数值特征
            logger.warning("未找到特征名称列表，使用所有数值特征")
            X_input = input_feat.select_dtypes(include=[np.number])

        logger.info(f"特征矩阵: {X_input.shape}")

    with Timer("模型预测"):
        # 预测概率
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_input)[:, 1]  # 流失概率
        else:
            # 如果模型不支持predict_proba，使用decision_function
            logger.warning("模型不支持predict_proba，使用predict")
            probabilities = model.predict(X_input)

        # 根据阈值预测类别
        predictions = (probabilities >= threshold).astype(int)

        logger.info(f"预测完成: {len(predictions)} 条记录")
        logger.info(f"预测统计:")
        logger.info(f"  流失客户数: {predictions.sum():,} ({predictions.sum()/len(predictions)*100:.2f}%)")
        logger.info(f"  留存客户数: {(1-predictions).sum():,} ({(1-predictions).sum()/len(predictions)*100:.2f}%)")
        logger.info(f"  平均流失概率: {probabilities.mean():.4f}")
        logger.info(f"  流失概率中位数: {np.median(probabilities):.4f}")
        logger.info(f"  流失概率范围: [{probabilities.min():.4f}, {probabilities.max():.4f}]")

    # 创建结果DataFrame
    results = pd.DataFrame({
        'churn_probability': probabilities,
        'churn_prediction': predictions
    })

    # 添加风险等级
    results['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['低风险', '中风险', '高风险']
    )

    # 添加客户ID（如果有）
    if customer_ids is not None:
        results.insert(0, config.ID_COL, customer_ids.values)

    return results, probabilities


def analyze_high_risk_customers(results_df: pd.DataFrame, top_k: int = None):
    """
    分析高风险客户

    Args:
        results_df: 预测结果DataFrame
        top_k: 显示前K个高风险客户
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info("=" * 60)
    logger.info("高风险客户分析")
    logger.info("=" * 60)

    # 统计各风险等级的客户数
    risk_counts = results_df['risk_level'].value_counts().sort_index()
    logger.info(f"\n风险等级分布:")
    for level, count in risk_counts.items():
        logger.info(f"  {level}: {count:,} ({count/len(results_df)*100:.2f}%)")

    # 高风险客户
    high_risk = results_df[results_df['risk_level'] == '高风险']
    logger.info(f"\n高风险客户数: {len(high_risk):,}")

    if len(high_risk) > 0:
        logger.info(f"  平均流失概率: {high_risk['churn_probability'].mean():.4f}")

    # 显示前K个高风险客户
    if top_k and len(results_df) > 0:
        logger.info(f"\n流失风险最高的前 {top_k} 个客户:")
        top_customers = results_df.nlargest(top_k, 'churn_probability')
        print("\n" + top_customers.to_string(index=False))


def save_predictions(results_df: pd.DataFrame, output_file: str):
    """
    保存预测结果

    Args:
        results_df: 预测结果DataFrame
        output_file: 输出文件路径
    """
    logger = logging.getLogger("ChurnPrediction")
    logger.info("=" * 60)
    logger.info("保存预测结果")
    logger.info("=" * 60)

    # 保存到CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logger.info(f"预测结果已保存: {output_path}")
    logger.info(f"  文件大小: {len(results_df)} 行")
    logger.info(f"\n前5行预测结果:")
    print(results_df.head().to_string(index=False))

    # 保存摘要统计
    summary_file = output_path.parent / (output_path.stem + '_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("客户流失预测摘要\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"预测客户总数: {len(results_df):,}\n\n")

        # 预测分布
        f.write("预测分布:\n")
        churn_counts = results_df['churn_prediction'].value_counts()
        f.write(f"  预测流失: {churn_counts.get(1, 0):,} ({churn_counts.get(1, 0)/len(results_df)*100:.2f}%)\n")
        f.write(f"  预测留存: {churn_counts.get(0, 0):,} ({churn_counts.get(0, 0)/len(results_df)*100:.2f}%)\n\n")

        # 风险等级分布
        f.write("风险等级分布:\n")
        risk_counts = results_df['risk_level'].value_counts().sort_index()
        for level, count in risk_counts.items():
            f.write(f"  {level}: {count:,} ({count/len(results_df)*100:.2f}%)\n")

        # 概率统计
        f.write(f"\n流失概率统计:\n")
        f.write(f"  平均值: {results_df['churn_probability'].mean():.4f}\n")
        f.write(f"  中位数: {results_df['churn_probability'].median():.4f}\n")
        f.write(f"  标准差: {results_df['churn_probability'].std():.4f}\n")
        f.write(f"  最小值: {results_df['churn_probability'].min():.4f}\n")
        f.write(f"  最大值: {results_df['churn_probability'].max():.4f}\n")

    logger.info(f"预测摘要已保存: {summary_file}")


def main():
    """主函数"""

    # 解析参数
    args = parse_args()

    # 设置日志
    logger = setup_logger("ChurnPrediction", config.LOG_FILE, "INFO")

    logger.info("=" * 80)
    logger.info(" " * 25 + "客户流失预测")
    logger.info("=" * 80)

    try:
        with Timer("完整预测流程"):

            # 1. 加载模型
            logger.info("\n" + "=" * 60)
            logger.info("步骤 1/5: 加载模型")
            logger.info("=" * 60)

            model, metadata = load_model_and_metadata(
                Path(args.model_file),
                Path(args.metadata_file)
            )

            # 2. 加载输入数据
            logger.info("\n" + "=" * 60)
            logger.info("步骤 2/5: 加载输入数据")
            logger.info("=" * 60)

            if args.input is None:
                # 如果没有指定输入文件，使用原始训练数据作为示例
                logger.info("未指定输入文件，使用训练数据进行演示")
                input_file = config.DATA_PATH
            else:
                input_file = Path(args.input)

            input_df = load_input_data(input_file, args.sample)

            # 3. 预处理
            logger.info("\n" + "=" * 60)
            logger.info("步骤 3/5: 数据预处理")
            logger.info("=" * 60)

            input_clean = preprocess_input_data(input_df)

            # 4. 预测
            logger.info("\n" + "=" * 60)
            logger.info("步骤 4/5: 预测客户流失")
            logger.info("=" * 60)

            results_df, probabilities = predict_churn(
                model,
                input_clean,
                metadata,
                threshold=args.threshold
            )

            # 分析高风险客户
            if args.top_k:
                analyze_high_risk_customers(results_df, args.top_k)

            # 5. 保存结果
            logger.info("\n" + "=" * 60)
            logger.info("步骤 5/5: 保存预测结果")
            logger.info("=" * 60)

            save_predictions(results_df, args.output)

        logger.info("\n" + "=" * 80)
        logger.info("预测完成！")
        logger.info("=" * 80)
        logger.info(f"\n输出文件: {args.output}")
        logger.info("\n业务建议:")
        logger.info("  1. 重点关注高风险客户，提供个性化挽留方案")
        logger.info("  2. 对中风险客户进行预防性营销活动")
        logger.info("  3. 分析流失客户特征，优化产品和服务")

        return 0

    except Exception as e:
        logger.error(f"\n预测过程中出现错误: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    """
    使用示例:

    # 基本用法（使用默认路径和训练数据作为示例）
    python predict.py

    # 指定输入文件
    python predict.py --input data/new_customers.csv

    # 指定输出文件名
    python predict.py --input data/new_customers.csv --output my_predictions.csv

    # 指定模型文件
    python predict.py --input data/new_customers.csv --model-file models/my_model.pkl

    # 快速测试（只预测前100条）
    python predict.py --input data/new_customers.csv --sample 100

    # 自定义分类阈值（提高阈值可以减少误报）
    python predict.py --input data/new_customers.csv --threshold 0.6

    # 显示流失风险最高的前20个客户
    python predict.py --input data/new_customers.csv --top-k 20

    # 显示预测概率
    python predict.py --input data/new_customers.csv --show-probability

    # 完整示例（生产环境）
    python predict.py \
        --input data/new_customers.csv \
        --model-file models/churn_model_best.pkl \
        --output predictions/2024_01_predictions.csv \
        --threshold 0.55 \
        --top-k 50

    # 批量预测（大文件）
    python predict.py --input data/all_customers.csv --output predictions/all_predictions.csv

    # 开发测试
    python predict.py --sample 500 --top-k 10

    # 高风险客户筛选（提高阈值）
    python predict.py --input data/new_customers.csv --threshold 0.7 --output high_risk.csv
    """

    exit_code = main()
    sys.exit(exit_code)
