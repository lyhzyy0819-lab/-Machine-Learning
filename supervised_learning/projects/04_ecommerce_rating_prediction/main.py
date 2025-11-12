"""
电商评分预测 - 主程序
完整的端到端机器学习项目流程
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

import config
from src.utils import setup_logger, Timer
from src.data_loader import load_raw_data
from src.data_preprocessing import preprocess_data, split_features_target
from src.feature_engineering import engineer_features, select_features
from src.model_training import ModelTrainer, compare_models
from src.model_evaluation import ModelEvaluator
from src.visualization import create_visualization_report


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='电商评分预测')

    parser.add_argument('--sample', action='store_true',
                       help='使用样本数据（用于快速测试）')
    parser.add_argument('--sample-size', type=int, default=500,
                       help='样本数据大小')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--no-xgboost', action='store_true',
                       help='跳过XGBoost模型')
    parser.add_argument('--no-viz', action='store_true',
                       help='跳过可视化报告')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（仅训练基础模型）')
    parser.add_argument('--tune', action='store_true',
                       help='启用超参数调优')
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'classification', 'both'],
                       help='任务类型: regression（评分预测）, classification（高/低评分分类）, both（两者都做）')

    return parser.parse_args()


def main():
    """主函数 - 完整的ML流程"""

    # 解析参数
    args = parse_args()

    # 设置日志
    # TODO 1: 使用setup_logger创建日志记录器
    # logger = setup_logger("Ecommerce_Rating", config.LOG_DIR / "main.log", "INFO")

    # TODO 2: 打印项目标题
    # logger.info("=" * 80)
    # logger.info(" " * 25 + "电商评分预测系统")
    # logger.info("=" * 80)

    try:
        # TODO 3: 使用Timer包装整个流程
        # with Timer("完整流程"):

            # ==================== 阶段1: 数据加载 ====================
            # TODO 4: 打印阶段标题
            # logger.info("\n" + "=" * 80)
            # logger.info("阶段 1/7: 数据加载")
            # logger.info("=" * 80)

            # TODO 5: 根据参数决定是否使用样本数据
            # use_sample = args.sample or config.USE_SAMPLE
            # sample_size = args.sample_size if args.sample else config.SAMPLE_SIZE

            # TODO 6: 调用load_raw_data加载数据
            # df = load_raw_data(use_sample=use_sample, sample_size=sample_size)

            # TODO 7: 打印数据加载完成信息
            # logger.info(f"\n数据加载完成:")
            # logger.info(f"  样本数量: {len(df):,}")
            # logger.info(f"  特征数量: {df.shape[1]}")

            # ==================== 阶段2: 数据预处理 ====================
            # TODO 8: 打印阶段标题并调用preprocess_data
            # logger.info("\n" + "=" * 80)
            # logger.info("阶段 2/7: 数据预处理")
            # logger.info("=" * 80)

            # df_clean = preprocess_data(
            #     df,
            #     remove_missing_target=True,
            #     handle_outliers=True
            # )

            # TODO 9: 打印预处理完成信息
            # logger.info(f"\n数据预处理完成:")
            # logger.info(f"  清洗后样本数: {len(df_clean):,}")
            # logger.info(f"  数据保留率: {len(df_clean)/len(df)*100:.2f}%")

            # ==================== 阶段3: 探索性数据分析 (EDA) ====================
            # TODO 10: 如果不跳过可视化，创建EDA报告
            # if not args.no_viz:
            #     logger.info("\n" + "=" * 80)
            #     logger.info("阶段 3/7: 探索性数据分析 (EDA)")
            #     logger.info("=" * 80)

            #     create_visualization_report(
            #         df_clean,
            #         target_col='rating',
            #         save_dir=config.FIGURE_DIR / "eda"
            #     )

            #     logger.info("\nEDA可视化报告已生成")
            # else:
            #     logger.info("\n跳过EDA阶段")

            # ==================== 阶段4: 特征工程 ====================
            # TODO 11: 打印阶段标题并执行特征工程
            # logger.info("\n" + "=" * 80)
            # logger.info("阶段 4/7: 特征工程")
            # logger.info("=" * 80)

            # df_feat = engineer_features(df_clean)

            # logger.info(f"\n特征工程完成:")
            # logger.info(f"  总特征数: {df_feat.shape[1]}")

            # TODO 12: 分离特征和目标变量
            # X_all = select_features(df_feat)
            # y_regression = df_feat['rating']  # 回归任务的目标
            # y_classification = (df_feat['rating'] >= config.HIGH_RATING_THRESHOLD).astype(int)  # 分类任务的目标

            # logger.info(f"  建模特征数: {X_all.shape[1]}")
            # logger.info(f"  回归目标: rating (1.0-5.0)")
            # logger.info(f"  分类目标: high_rating (>= {config.HIGH_RATING_THRESHOLD})")

            # ==================== 阶段5: 模型训练 ====================
            # TODO 13: 打印阶段标题
            # logger.info("\n" + "=" * 80)
            # logger.info("阶段 5/7: 模型训练")
            # logger.info("=" * 80)

            # TODO 14: 根据args.task决定训练哪些模型（回归/分类/两者）
            # tasks = []
            # if args.task in ['regression', 'both']:
            #     tasks.append('regression')
            # if args.task in ['classification', 'both']:
            #     tasks.append('classification')

            # results = {}

            # TODO 15: 对每个任务训练模型
            # for task in tasks:
            #     logger.info(f"\n{'=' * 60}")
            #     logger.info(f"训练{task}模型")
            #     logger.info(f"{'=' * 60}")

            #     # 选择目标变量
            #     y_target = y_regression if task == 'regression' else y_classification

            #     # 初始化训练器
            #     trainer = ModelTrainer(task=task, random_state=config.RANDOM_STATE)

            #     # 准备数据
            #     trainer.prepare_data(X_all, y_target, test_size=args.test_size, scale=True)

            #     # 训练模型
            #     if args.quick:
            #         # 快速模式：只训练基础模型
            #         logger.info("快速模式：仅训练基础模型")
            #         if task == 'regression':
            #             trainer.train_linear_regression()
            #             trainer.train_ridge()
            #         else:
            #             trainer.train_logistic_regression()
            #     else:
            #         # 完整模式：训练所有模型
            #         include_xgb = not args.no_xgboost and config.HAS_XGBOOST
            #         trainer.train_all_models(
            #             include_xgboost=include_xgb,
            #             tune_hyperparameters=args.tune
            #         )

            #     # 对比模型
            #     logger.info("\n" + "-" * 80)
            #     logger.info(f"{task}模型性能对比")
            #     logger.info("-" * 80)
            #     comparison_df = compare_models(trainer.models, task=task)
            #     print("\n" + comparison_df.to_string(index=False))

            #     # 保存最佳模型
            #     trainer.save_best_model()

            #     results[task] = {
            #         'trainer': trainer,
            #         'comparison': comparison_df
            #     }

            # ==================== 阶段6: 模型评估 ====================
            # TODO 16: 打印阶段标题并评估所有模型
            # logger.info("\n" + "=" * 80)
            # logger.info("阶段 6/7: 模型评估")
            # logger.info("=" * 80)

            # for task in tasks:
            #     trainer = results[task]['trainer']
            #     evaluator = ModelEvaluator(task=task)

            #     # 评估所有模型
            #     eval_results = evaluator.evaluate_all_models(
            #         trainer.models,
            #         trainer.X_train_scaled,
            #         trainer.X_test_scaled,
            #         trainer.y_train,
            #         trainer.y_test,
            #         list(X_all.columns)
            #     )

            #     logger.info(f"\n{task}模型评估完成")

            # ==================== 阶段7: 结果总结 ====================
            # TODO 17: 打印最终结果总结
            # logger.info("\n" + "=" * 80)
            # logger.info("阶段 7/7: 结果总结")
            # logger.info("=" * 80)

            # for task in tasks:
            #     trainer = results[task]['trainer']
            #     best_model_name = trainer.best_model_name
            #     best_metrics = trainer.models[best_model_name]['metrics']

            #     logger.info(f"\n{task}最佳模型: {best_model_name}")
            #     logger.info(f"\n性能指标:")

            #     if task == 'regression':
            #         logger.info(f"  R² Score: {best_metrics['r2']:.4f}")
            #         logger.info(f"  RMSE: {best_metrics['rmse']:.4f}")
            #         logger.info(f"  MAE: {best_metrics['mae']:.4f}")
            #     else:
            #         logger.info(f"  Accuracy: {best_metrics['accuracy']:.4f}")
            #         logger.info(f"  Precision: {best_metrics['precision']:.4f}")
            #         logger.info(f"  Recall: {best_metrics['recall']:.4f}")
            #         logger.info(f"  F1 Score: {best_metrics['f1']:.4f}")
            #         logger.info(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")

            # TODO 18: 输出文件位置
            # logger.info(f"\n输出文件:")
            # logger.info(f"  模型目录: {config.MODEL_DIR}")
            # logger.info(f"  图表目录: {config.FIGURE_DIR}")
            # logger.info(f"  日志文件: {config.LOG_DIR}")

        # TODO 19: 打印完成信息
        # logger.info("\n" + "=" * 80)
        # logger.info("项目执行完成！")
        # logger.info("=" * 80)

        # return 0

        # ===== 临时占位代码（删除上面的TODO实现后删除此行） =====
        print("=" * 80)
        print(" " * 25 + "电商评分预测系统")
        print("=" * 80)
        print("\n请实现上面的TODO注释，完成主程序逻辑")
        print("\n项目结构:")
        print("  1. 数据加载 (src.data_loader.load_raw_data)")
        print("  2. 数据预处理 (src.data_preprocessing.preprocess_data)")
        print("  3. 探索性数据分析 (src.visualization.create_visualization_report)")
        print("  4. 特征工程 (src.feature_engineering.engineer_features)")
        print("  5. 模型训练 (src.model_training.ModelTrainer)")
        print("  6. 模型评估 (src.model_evaluation.ModelEvaluator)")
        print("  7. 结果总结")
        print("\n提示：参考 NYC Taxi 项目的 main.py 实现")
        print("=" * 80)
        return 0

    except Exception as e:
        # TODO 20: 记录错误日志
        # logger.error(f"\n执行过程中出现错误: {str(e)}", exc_info=True)
        print(f"\n执行过程中出现错误: {str(e)}")
        return 1


if __name__ == '__main__':
    """
    运行示例:

    # 完整模式（使用全部数据，回归任务）
    python main.py

    # 快速测试模式（使用样本数据）
    python main.py --sample --sample-size 200

    # 快速模式（仅训练基础模型）
    python main.py --sample --quick

    # 分类任务
    python main.py --task classification

    # 同时训练回归和分类模型
    python main.py --task both

    # 启用超参数调优
    python main.py --tune

    # 不训练XGBoost
    python main.py --no-xgboost

    # 跳过可视化
    python main.py --no-viz

    # 自定义测试集比例
    python main.py --test-size 0.3

    # 完整流程（推荐）
    # 1. 先下载数据
    python download_data.py
    # 2. 训练模型
    python main.py --task both
    # 3. 生成预测
    python predict.py
    """

    exit_code = main()
    sys.exit(exit_code)
