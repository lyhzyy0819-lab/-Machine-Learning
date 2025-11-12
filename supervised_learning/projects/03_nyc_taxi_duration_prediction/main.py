"""
NYC出租车行程时长预测 - 主程序
完整的端到端机器学习项目流程
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

import config
from src.utils import setup_logger, Timer
from src.data_loader import load_train_data
from src.data_preprocessing import preprocess_data, split_features_target
from src.feature_engineering import engineer_features, select_features
from src.model_training import ModelTrainer, compare_models
from src.model_evaluation import ModelEvaluator
from src.visualization import create_visualization_report


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NYC出租车行程时长预测')

    parser.add_argument('--sample', action='store_true',
                       help='使用样本数据（用于快速测试）')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='样本数据大小')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--no-xgboost', action='store_true',
                       help='跳过XGBoost模型')
    parser.add_argument('--no-stacking', action='store_true',
                       help='跳过Stacking模型')
    parser.add_argument('--no-viz', action='store_true',
                       help='跳过可视化报告')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（仅训练基础模型）')
    parser.add_argument('--retrain-full', action='store_true',
                       help='用全部训练数据重新训练最佳模型（Kaggle模式）')

    return parser.parse_args()


def main():
    """主函数 - 完整的ML流程"""

    # 解析参数
    args = parse_args()

    # 设置日志
    logger = setup_logger("NYC_Taxi", config.LOG_FILE, "INFO")

    logger.info("=" * 80)
    logger.info(" " * 20 + "NYC出租车行程时长预测系统")
    logger.info("=" * 80)

    try:
        with Timer("完整流程"):

            # ==================== 阶段1: 数据加载 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 1/7: 数据加载")
            logger.info("=" * 80)

            use_sample = args.sample or config.USE_SAMPLE
            sample_size = args.sample_size if args.sample else config.SAMPLE_SIZE

            df = load_train_data(use_sample=use_sample, sample_size=sample_size)

            logger.info(f"\n数据加载完成:")
            logger.info(f"  样本数量: {len(df):,}")
            logger.info(f"  特征数量: {df.shape[1]}")

            # ==================== 阶段2: 数据预处理 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 2/7: 数据预处理")
            logger.info("=" * 80)

            df_clean = preprocess_data(
                df,
                remove_outliers_flag=True,
                outlier_method='iqr'
            )

            logger.info(f"\n数据预处理完成:")
            logger.info(f"  清洗后样本数: {len(df_clean):,}")
            logger.info(f"  数据保留率: {len(df_clean)/len(df)*100:.2f}%")

            # ==================== 阶段3: 探索性数据分析 (EDA) ====================
            if not args.no_viz:
                logger.info("\n" + "=" * 80)
                logger.info("阶段 3/7: 探索性数据分析 (EDA)")
                logger.info("=" * 80)

                create_visualization_report(
                    df_clean,
                    target_col='trip_duration',
                    save_dir=config.FIGURES_DIR / "eda"
                )

                logger.info("\nEDA可视化报告已生成")
            else:
                logger.info("\n跳过EDA阶段")

            # ==================== 阶段4: 特征工程 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 4/7: 特征工程")
            logger.info("=" * 80)

            df_feat = engineer_features(df_clean)

            logger.info(f"\n特征工程完成:")
            logger.info(f"  总特征数: {df_feat.shape[1]}")

            # 分离特征和目标
            X_all = select_features(df_feat)
            y_all = df_feat['trip_duration']

            logger.info(f"  建模特征数: {X_all.shape[1]}")
            logger.info(f"  目标变量: trip_duration")

            # ==================== 阶段5: 模型训练 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 5/7: 模型训练")
            logger.info("=" * 80)

            # 初始化训练器
            trainer = ModelTrainer(random_state=config.RANDOM_STATE)

            # 准备数据
            trainer.prepare_data(X_all, y_all, test_size=args.test_size, scale=True)

            # 训练模型
            if args.quick:
                # 快速模式：只训练基础模型
                logger.info("快速模式：仅训练基础模型")
                trainer.train_linear_regression()
                trainer.train_ridge()
                trainer.train_random_forest()
            else:
                # 完整模式：训练所有模型
                include_xgb = not args.no_xgboost
                include_stack = not args.no_stacking

                trainer.train_all_models(
                    include_xgboost=include_xgb,
                    include_stacking=include_stack
                )

            # 对比模型
            logger.info("\n" + "-" * 80)
            logger.info("模型性能对比")
            logger.info("-" * 80)
            comparison_df = compare_models(trainer.models)
            print("\n" + comparison_df.to_string(index=False))

            # ==================== 可选: 全量重训练 ====================
            if args.retrain_full:
                logger.info("\n" + "=" * 80)
                logger.info("全量重训练模式")
                logger.info("=" * 80)
                logger.info("使用全部训练数据重新训练最佳模型...")
                logger.info("这是Kaggle竞赛的最佳实践，可以充分利用所有数据")

                trainer.retrain_on_full_data(X_all, y_all)

            # 保存最佳模型
            trainer.save_best_model()

            # ==================== 阶段6: 模型评估 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 6/7: 模型评估")
            logger.info("=" * 80)

            evaluator = ModelEvaluator()

            # 评估所有模型
            eval_results = evaluator.evaluate_all_models(
                trainer.models,
                trainer.X_train_scaled,
                trainer.X_test_scaled,
                trainer.y_train,
                trainer.y_test,
                list(X_all.columns)
            )

            logger.info("\n模型评估完成")

            # ==================== 阶段7: 结果总结 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 7/7: 结果总结")
            logger.info("=" * 80)

            best_model_name = trainer.best_model_name
            best_metrics = trainer.models[best_model_name]['metrics']

            logger.info(f"\n最佳模型: {best_model_name}")
            logger.info(f"\n性能指标:")
            logger.info(f"  R² Score: {best_metrics['r2']:.4f}")
            logger.info(f"  RMSE: {best_metrics['rmse']:.2f} 秒 ({best_metrics['rmse']/60:.2f} 分钟)")
            logger.info(f"  MAE: {best_metrics['mae']:.2f} 秒 ({best_metrics['mae']/60:.2f} 分钟)")
            logger.info(f"  MAPE: {best_metrics['mape']:.2f}%")

            # 与目标对比
            logger.info(f"\n目标达成情况:")
            targets = config.TARGET_METRICS
            logger.info(f"  R² > {targets['r2']}: {'✓' if best_metrics['r2'] > targets['r2'] else '✗'}")
            logger.info(f"  RMSE < {targets['rmse']}秒: {'✓' if best_metrics['rmse'] < targets['rmse'] else '✗'}")
            logger.info(f"  MAE < {targets['mae']}秒: {'✓' if best_metrics['mae'] < targets['mae'] else '✗'}")

            # 输出文件位置
            logger.info(f"\n输出文件:")
            logger.info(f"  模型文件: {config.BEST_MODEL_PATH}")
            logger.info(f"  缩放器: {config.SCALER_PATH}")
            logger.info(f"  元数据: {config.METADATA_PATH}")
            logger.info(f"  图表目录: {config.FIGURES_DIR}")
            logger.info(f"  日志文件: {config.LOG_FILE}")

        logger.info("\n" + "=" * 80)
        logger.info("项目执行完成！")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n执行过程中出现错误: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    """
    运行示例:

    # 完整模式（使用全部数据）
    python main.py

    # 快速测试模式（使用样本数据）
    python main.py --sample --sample-size 5000

    # 快速模式（仅训练基础模型）
    python main.py --sample --quick

    # Kaggle竞赛模式（全量重训练）
    python main.py --retrain-full

    # Kaggle模式 + 样本测试
    python main.py --sample --retrain-full

    # 不训练XGBoost
    python main.py --no-xgboost

    # 跳过可视化
    python main.py --no-viz

    # 自定义测试集比例
    python main.py --test-size 0.3

    # 完整Kaggle提交流程
    # 1. 训练并全量重训练
    python main.py --retrain-full
    # 2. 生成提交文件
    python predict.py
    """

    exit_code = main()
    sys.exit(exit_code)
