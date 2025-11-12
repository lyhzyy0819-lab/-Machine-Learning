"""
NYC出租车行程时长预测 - 一键生成Kaggle提交文件
完整流程：训练 → 全量重训练 → 预测 → 生成submission.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import config
from src.utils import setup_logger, Timer
from src.data_loader import load_train_data, load_data_from_csv
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features, select_features
from src.model_training import ModelTrainer
import pandas as pd
import numpy as np


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='一键生成Kaggle提交文件')

    parser.add_argument('--train-file', type=str,
                       default=str(config.TRAIN_DATA_PATH),
                       help='训练集文件路径')
    parser.add_argument('--test-file', type=str,
                       default=str(config.TEST_DATA_PATH),
                       help='测试集文件路径')
    parser.add_argument('--output', type=str,
                       default='submission.csv',
                       help='输出文件路径')
    parser.add_argument('--sample', type=int, default=None,
                       help='样本数量（用于快速测试）')
    parser.add_argument('--skip-retrain', action='store_true',
                       help='跳过全量重训练（快速模式）')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（仅训练基础模型）')

    return parser.parse_args()


def main():
    """主函数 - 完整的Kaggle提交流程"""

    args = parse_args()

    # 设置日志
    logger = setup_logger("NYC_Taxi", config.LOG_FILE, "INFO")

    logger.info("=" * 80)
    logger.info(" " * 15 + "NYC出租车 - 一键生成Kaggle提交文件")
    logger.info("=" * 80)

    try:
        with Timer("完整提交流程"):

            # ==================== 阶段1: 加载训练数据 ====================
            logger.info("\n" + "=" * 60)
            logger.info("阶段 1/5: 加载训练数据")
            logger.info("=" * 60)

            df_train = load_train_data(
                use_sample=args.sample is not None,
                sample_size=args.sample if args.sample else config.SAMPLE_SIZE
            )

            logger.info(f"训练数据: {df_train.shape}")

            # ==================== 阶段2: 数据预处理和特征工程 ====================
            logger.info("\n" + "=" * 60)
            logger.info("阶段 2/5: 数据预处理和特征工程")
            logger.info("=" * 60)

            # 预处理
            df_clean = preprocess_data(df_train, remove_outliers_flag=True)

            # 特征工程
            df_feat = engineer_features(df_clean)

            # 准备建模数据
            X_all = select_features(df_feat)
            y_all = df_feat['trip_duration']

            logger.info(f"特征矩阵: {X_all.shape}")

            # ==================== 阶段3: 模型训练和选择 ====================
            logger.info("\n" + "=" * 60)
            logger.info("阶段 3/5: 模型训练和选择")
            logger.info("=" * 60)

            # 初始化训练器
            trainer = ModelTrainer(random_state=config.RANDOM_STATE)

            # 准备数据（train_test_split用于验证）
            trainer.prepare_data(X_all, y_all, test_size=0.2, scale=True)

            # 训练模型
            if args.quick:
                logger.info("快速模式：仅训练基础模型")
                trainer.train_linear_regression()
                trainer.train_ridge()
                trainer.train_random_forest()
                trainer._select_best_model()
            else:
                logger.info("完整模式：训练所有模型")
                trainer.train_all_models(
                    include_xgboost=True,
                    include_stacking=True
                )

            logger.info(f"\n最佳模型: {trainer.best_model_name}")
            logger.info(f"验证集R²: {trainer.best_score:.4f}")

            # ==================== 阶段4: 全量重训练（可选） ====================
            if not args.skip_retrain:
                logger.info("\n" + "=" * 60)
                logger.info("阶段 4/5: 全量重训练")
                logger.info("=" * 60)
                logger.info("使用全部训练数据重新训练以获得最佳性能...")

                trainer.retrain_on_full_data(X_all, y_all)
            else:
                logger.info("\n跳过全量重训练")

            # 保存模型
            trainer.save_best_model()

            # ==================== 阶段5: 测试集预测 ====================
            logger.info("\n" + "=" * 60)
            logger.info("阶段 5/5: 测试集预测")
            logger.info("=" * 60)

            # 加载测试集
            test_file = Path(args.test_file)
            if not test_file.exists():
                logger.error(f"测试集文件不存在: {test_file}")
                logger.info("请将测试集文件放到: data/raw/nyc_taxi_test.csv")
                return 1

            df_test = load_data_from_csv(test_file, optimize_memory=True)
            logger.info(f"测试集: {df_test.shape}")

            # 预处理测试集
            df_test_clean = df_test.copy()
            if 'pickup_datetime' in df_test_clean.columns:
                df_test_clean['pickup_datetime'] = pd.to_datetime(df_test_clean['pickup_datetime'])

            # 特征工程
            df_test_feat = engineer_features(df_test_clean)
            X_test = select_features(df_test_feat)

            logger.info(f"测试集特征: {X_test.shape}")

            # 预测
            X_test_scaled = trainer.scaler.transform(X_test)
            predictions = trainer.best_model.predict(X_test_scaled)

            logger.info(f"\n预测统计:")
            logger.info(f"  均值: {predictions.mean():.2f} 秒 ({predictions.mean()/60:.2f} 分钟)")
            logger.info(f"  中位数: {np.median(predictions):.2f} 秒")
            logger.info(f"  最小值: {predictions.min():.2f} 秒")
            logger.info(f"  最大值: {predictions.max():.2f} 秒")

            # 生成提交文件
            if 'id' in df_test.columns:
                submission = pd.DataFrame({
                    'id': df_test['id'],
                    'trip_duration': predictions
                })
            else:
                submission = pd.DataFrame({
                    'id': range(len(predictions)),
                    'trip_duration': predictions
                })

            # 确保预测值为正数
            submission['trip_duration'] = submission['trip_duration'].clip(lower=0)

            # 保存
            output_path = Path(args.output)
            submission.to_csv(output_path, index=False)

            logger.info(f"\n提交文件已保存: {output_path}")
            logger.info(f"提交文件大小: {len(submission)} 行")

            # 显示前几行
            logger.info(f"\n前5行预测结果:")
            print(submission.head().to_string(index=False))

        logger.info("\n" + "=" * 80)
        logger.info("Kaggle提交文件生成完成！")
        logger.info("=" * 80)
        logger.info(f"\n下一步:")
        logger.info(f"  1. 检查输出文件: {args.output}")
        logger.info(f"  2. 提交到Kaggle竞赛平台")
        logger.info(f"  3. 查看排行榜得分")

        return 0

    except Exception as e:
        logger.error(f"\n生成过程中出现错误: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    """
    使用示例:

    # 基本用法（完整流程）
    python generate_submission.py

    # 快速测试（使用样本数据）
    python generate_submission.py --sample 5000 --quick

    # 跳过全量重训练（更快）
    python generate_submission.py --skip-retrain

    # 指定文件路径
    python generate_submission.py --train-file data/raw/train.csv --test-file data/raw/test.csv

    # 指定输出文件名
    python generate_submission.py --output my_submission.csv

    # 完整示例（推荐）
    python generate_submission.py --train-file data/raw/train.csv --test-file data/raw/test.csv --output submission.csv
    """

    exit_code = main()
    sys.exit(exit_code)
