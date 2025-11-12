"""
客户流失预测 - 主程序
完整的端到端机器学习项目流程
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

import config
from src.utils import setup_logger, Timer
from src.data_loader import load_train_data, split_features_target
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features, select_features
from src.model_training import ModelTrainer, compare_models
from src.model_evaluation import ModelEvaluator
from src.visualization import create_visualization_report


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='客户流失预测系统')

    parser.add_argument('--sample', action='store_true',
                       help='使用样本数据（用于快速测试）')
    parser.add_argument('--sample-size', type=int, default=5000,
                       help='样本数据大小')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--no-smote', action='store_true',
                       help='不使用SMOTE处理类别不平衡')
    parser.add_argument('--no-xgboost', action='store_true',
                       help='跳过XGBoost模型')
    parser.add_argument('--no-lightgbm', action='store_true',
                       help='跳过LightGBM模型')
    parser.add_argument('--no-viz', action='store_true',
                       help='跳过可视化报告')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（仅训练基础模型）')
    parser.add_argument('--no-tuning', action='store_true',
                       help='跳过超参数调优')
    parser.add_argument('--retrain-full', action='store_true',
                       help='用全部训练数据重新训练最佳模型')

    return parser.parse_args()


def main():
    """主函数 - 完整的ML流程"""

    # 解析参数
    args = parse_args()

    # 设置日志
    logger = setup_logger("ChurnPrediction", config.LOG_FILE, "INFO")

    logger.info("=" * 80)
    logger.info(" " * 25 + "客户流失预测系统")
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

            # 打印目标变量分布
            if config.TARGET_COL in df.columns:
                churn_counts = df[config.TARGET_COL].value_counts()
                logger.info(f"\n目标变量分布:")
                for label, count in churn_counts.items():
                    logger.info(f"  {label}: {count:,} ({count/len(df)*100:.2f}%)")

            # ==================== 阶段2: 数据预处理 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 2/7: 数据预处理")
            logger.info("=" * 80)

            df_clean = preprocess_data(
                df,
                remove_outliers_flag=True,
                outlier_method='iqr',
                handle_missing='auto'
            )

            logger.info(f"\n数据预处理完成:")
            logger.info(f"  清洗后样本数: {len(df_clean):,}")
            logger.info(f"  数据保留率: {len(df_clean)/len(df)*100:.2f}%")

            # 检查类别分布
            if config.TARGET_COL in df_clean.columns:
                churn_counts_clean = df_clean[config.TARGET_COL].value_counts()
                logger.info(f"\n清洗后目标变量分布:")
                for label, count in churn_counts_clean.items():
                    logger.info(f"  {label}: {count:,} ({count/len(df_clean)*100:.2f}%)")

            # ==================== 阶段3: 探索性数据分析 (EDA) ====================
            if not args.no_viz:
                logger.info("\n" + "=" * 80)
                logger.info("阶段 3/7: 探索性数据分析 (EDA)")
                logger.info("=" * 80)

                # 创建EDA可视化报告
                create_visualization_report(
                    df_clean,
                    target_col=config.TARGET_COL,
                    save_dir=config.FIGURES_DIR / "eda"
                )

                logger.info("\nEDA可视化报告已生成")
                logger.info(f"  图表保存位置: {config.FIGURES_DIR / 'eda'}")
            else:
                logger.info("\n跳过EDA阶段（使用 --no-viz 参数）")

            # ==================== 阶段4: 特征工程 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 4/7: 特征工程")
            logger.info("=" * 80)

            # 执行特征工程
            df_feat = engineer_features(df_clean)

            logger.info(f"\n特征工程完成:")
            logger.info(f"  原始特征数: {df_clean.shape[1]}")
            logger.info(f"  工程后特征数: {df_feat.shape[1]}")

            # 分离特征和目标
            X_all, y_all, ids = split_features_target(
                df_feat,
                target_col=config.TARGET_COL,
                id_col=config.ID_COL
            )

            # 选择建模特征
            X_selected = select_features(X_all, y_all)

            logger.info(f"  最终建模特征数: {X_selected.shape[1]}")
            logger.info(f"  目标变量: {config.TARGET_COL}")
            logger.info(f"  样本总数: {len(y_all):,}")

            # 打印前10个特征名
            logger.info(f"\n前10个特征:")
            for i, feature in enumerate(X_selected.columns[:10], 1):
                logger.info(f"  {i:2d}. {feature}")
            if len(X_selected.columns) > 10:
                logger.info(f"  ... 还有 {len(X_selected.columns) - 10} 个特征")

            # ==================== 阶段5: 模型训练 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 5/7: 模型训练")
            logger.info("=" * 80)

            # 初始化训练器
            trainer = ModelTrainer(random_state=config.RANDOM_STATE)

            # 准备数据（包括数据分割和SMOTE处理）
            use_smote = not args.no_smote
            trainer.prepare_data(
                X_selected,
                y_all,
                test_size=args.test_size,
                use_smote=use_smote
            )

            logger.info(f"\n训练集大小: {len(trainer.X_train):,}")
            logger.info(f"测试集大小: {len(trainer.X_test):,}")

            if use_smote and trainer.X_train_resampled is not None:
                logger.info(f"SMOTE处理后训练集大小: {len(trainer.X_train_resampled):,}")
                logger.info(f"  类别0: {(trainer.y_train_resampled == 0).sum():,}")
                logger.info(f"  类别1: {(trainer.y_train_resampled == 1).sum():,}")

            # 决定是否进行超参数调优
            tune_hyperparameters = not args.no_tuning and config.TUNE_HYPERPARAMETERS

            if args.quick:
                # 快速模式：只训练基础模型，不调优
                logger.info("\n快速模式：仅训练基础模型（不进行超参数调优）")
                trainer.train_logistic_regression(use_grid_search=False)
                trainer.train_decision_tree(use_grid_search=False)
                trainer.train_random_forest(use_randomized=False)
            else:
                # 完整模式：训练所有模型
                include_xgb = not args.no_xgboost
                include_lgb = not args.no_lightgbm

                logger.info("\n开始训练所有模型...")
                logger.info(f"  超参数调优: {'开启' if tune_hyperparameters else '关闭'}")
                logger.info(f"  包含XGBoost: {'是' if include_xgb else '否'}")
                logger.info(f"  包含LightGBM: {'是' if include_lgb else '否'}")

                trainer.train_all_models(
                    include_xgboost=include_xgb,
                    include_lightgbm=include_lgb,
                    tune_hyperparameters=tune_hyperparameters
                )

            # 对比模型性能
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
                logger.info("这可以充分利用所有数据以获得更好的泛化性能")

                trainer.retrain_on_full_data(X_selected, y_all, use_smote=use_smote)
                logger.info(f"最佳模型已使用全部数据重新训练")

            # 保存最佳模型
            logger.info("\n" + "-" * 80)
            logger.info("保存模型")
            logger.info("-" * 80)

            trainer.save_best_model(
                model_path=config.BEST_MODEL_PATH,
                metadata_path=config.METADATA_PATH,
                feature_names=list(X_selected.columns)
            )

            logger.info(f"最佳模型已保存: {config.BEST_MODEL_PATH}")
            logger.info(f"模型元数据已保存: {config.METADATA_PATH}")

            # ==================== 阶段6: 模型评估 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 6/7: 模型评估")
            logger.info("=" * 80)

            evaluator = ModelEvaluator()

            # 评估所有模型
            eval_results = evaluator.evaluate_all_models(
                trainer.models,
                trainer.X_train,
                trainer.X_test,
                trainer.y_train,
                trainer.y_test,
                feature_names=list(X_selected.columns)
            )

            logger.info("\n模型评估完成")

            # 生成评估报告
            if not args.no_viz:
                logger.info("\n生成评估可视化报告...")
                evaluator.plot_all_models_comparison(
                    trainer.models,
                    trainer.X_test,
                    trainer.y_test,
                    save_dir=config.FIGURES_DIR / "evaluation"
                )
                logger.info(f"  评估图表保存位置: {config.FIGURES_DIR / 'evaluation'}")

            # ==================== 阶段7: 结果总结 ====================
            logger.info("\n" + "=" * 80)
            logger.info("阶段 7/7: 结果总结")
            logger.info("=" * 80)

            best_model_name = trainer.best_model_name
            best_metrics = trainer.models[best_model_name]['metrics']

            logger.info(f"\n最佳模型: {best_model_name}")
            logger.info(f"\n性能指标:")
            logger.info(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {best_metrics['precision']:.4f}")
            logger.info(f"  Recall:    {best_metrics['recall']:.4f}")
            logger.info(f"  F1 Score:  {best_metrics['f1']:.4f}")
            logger.info(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f}")

            # 与目标对比
            logger.info(f"\n目标达成情况:")
            targets = config.TARGET_METRICS
            logger.info(f"  Accuracy  > {targets['accuracy']:.2f}: "
                       f"{'✓ 达成' if best_metrics['accuracy'] > targets['accuracy'] else '✗ 未达成'}")
            logger.info(f"  Precision > {targets['precision']:.2f}: "
                       f"{'✓ 达成' if best_metrics['precision'] > targets['precision'] else '✗ 未达成'}")
            logger.info(f"  Recall    > {targets['recall']:.2f}: "
                       f"{'✓ 达成' if best_metrics['recall'] > targets['recall'] else '✗ 未达成'}")
            logger.info(f"  F1 Score  > {targets['f1']:.2f}: "
                       f"{'✓ 达成' if best_metrics['f1'] > targets['f1'] else '✗ 未达成'}")
            logger.info(f"  ROC-AUC   > {targets['roc_auc']:.2f}: "
                       f"{'✓ 达成' if best_metrics['roc_auc'] > targets['roc_auc'] else '✗ 未达成'}")

            # 输出文件位置
            logger.info(f"\n输出文件:")
            logger.info(f"  最佳模型: {config.BEST_MODEL_PATH}")
            logger.info(f"  模型元数据: {config.METADATA_PATH}")
            logger.info(f"  图表目录: {config.FIGURES_DIR}")
            logger.info(f"  日志文件: {config.LOG_FILE}")

            # 业务洞察
            logger.info(f"\n业务洞察:")
            logger.info(f"  模型可以帮助识别高流失风险客户")
            logger.info(f"  预测准确率: {best_metrics['accuracy']*100:.2f}%")
            logger.info(f"  召回率: {best_metrics['recall']*100:.2f}% (能识别出的实际流失客户比例)")
            logger.info(f"  精确率: {best_metrics['precision']*100:.2f}% (预测为流失的客户中真正流失的比例)")

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
    python main.py --sample --sample-size 3000

    # 快速模式（仅训练基础模型）
    python main.py --sample --quick

    # 完整训练但不进行超参数调优（节省时间）
    python main.py --no-tuning

    # 全量重训练模式（使用所有数据训练最佳模型）
    python main.py --retrain-full

    # 样本测试 + 全量重训练
    python main.py --sample --retrain-full

    # 不使用SMOTE处理
    python main.py --no-smote

    # 跳过XGBoost和LightGBM
    python main.py --no-xgboost --no-lightgbm

    # 跳过可视化
    python main.py --no-viz

    # 自定义测试集比例
    python main.py --test-size 0.3

    # 完整生产环境训练流程（推荐）
    # 1. 先用样本快速验证
    python main.py --sample --sample-size 2000 --quick
    # 2. 完整训练
    python main.py --retrain-full
    # 3. 使用模型预测
    python predict.py --input new_customers.csv

    # 开发调试模式（快速迭代）
    python main.py --sample --quick --no-viz

    # 生产环境最佳实践
    python main.py --retrain-full --no-viz
    """

    exit_code = main()
    sys.exit(exit_code)
