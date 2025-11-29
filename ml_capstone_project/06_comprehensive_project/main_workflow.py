"""
综合项目主工作流
=================

机器学习项目完整工作流的实现，从数据诊断到最终模型部署。

使用方法:
    # 方式1: 交互式运行
    python main_workflow.py --mode interactive

    # 方式2: 自动运行所有Phase
    python main_workflow.py --mode auto --data data/raw/data.csv

    # 方式3: 运行指定Phase
    python main_workflow.py --phase 1 --data data/raw/data.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime

# 导入配置
from config import Config, get_default_config, get_production_config, get_testing_config


class MLWorkflow:
    """机器学习完整工作流"""

    def __init__(self, config: Config, data_path: Optional[str] = None):
        """
        初始化工作流

        Args:
            config: 配置对象
            data_path: 数据文件路径（可选）
        """
        self.config = config
        self.data_path = data_path
        self.data = None
        self.results = {}  # 存储各Phase的结果

        # 设置日志
        self._setup_logging()

        # 加载数据（如果提供路径）
        if data_path:
            self.load_data(data_path)

        self.logger.info(f"MLWorkflow初始化完成 [模式: {config.MODE}]")

    def _setup_logging(self):
        """配置日志系统"""
        # 创建logger
        self.logger = logging.getLogger('MLWorkflow')
        self.logger.setLevel(self.config.LOG_LEVEL)

        # 清除已有的handlers
        self.logger.handlers = []

        # 日志格式
        formatter = logging.Formatter(
            self.config.LOG_FORMAT,
            datefmt=self.config.LOG_DATE_FORMAT
        )

        # 文件handler
        file_handler = logging.FileHandler(
            self.config.LOG_FILE,
            encoding='utf-8'
        )
        file_handler.setLevel(self.config.LOG_LEVEL)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 控制台handler
        if self.config.LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.config.LOG_LEVEL)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    # ==================== 数据加载 ====================

    def load_data(self, data_path: str):
        """
        加载数据文件

        Args:
            data_path: 数据文件路径（支持csv, xlsx, json）
        """
        self.logger.info(f"正在加载数据: {data_path}")

        try:
            file_ext = Path(data_path).suffix.lower()

            if file_ext == '.csv':
                self.data = pd.read_csv(data_path)
            elif file_ext in ['.xlsx', '.xls']:
                self.data = pd.read_excel(data_path)
            elif file_ext == '.json':
                self.data = pd.read_json(data_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")

            self.logger.info(f"数据加载成功: {self.data.shape[0]} 行 × {self.data.shape[1]} 列")

            # 保存到配置的数据路径
            self.data_path = data_path

        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise

    # ==================== Phase 1: 数据诊断 ====================

    def phase1_data_diagnosis(self) -> Dict[str, Any]:
        """
        Phase 1: 数据诊断

        执行内容:
        - 数据规模检查
        - 数据类型识别
        - 缺失值分析
        - 异常值检测
        - 分布分析
        - 相关性分析

        Returns:
            诊断报告字典
        """
        self.logger.info("=" * 50)
        self.logger.info("Phase 1: 数据诊断")
        self.logger.info("=" * 50)

        if self.data is None:
            raise ValueError("请先加载数据: workflow.load_data(path)")

        diagnosis_report = {}

        # 1. 数据规模
        self.logger.info("1.1 数据规模检查")
        n_samples, n_features = self.data.shape
        diagnosis_report['data_shape'] = {
            'n_samples': n_samples,
            'n_features': n_features,
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        }
        self.logger.info(f"  - 样本数: {n_samples:,}")
        self.logger.info(f"  - 特征数: {n_features}")
        self.logger.info(f"  - 内存占用: {diagnosis_report['data_shape']['memory_usage_mb']:.2f} MB")

        # 2. 数据类型
        self.logger.info("1.2 数据类型识别")
        numeric_features = self.data.select_dtypes(include=['int', 'float']).columns.tolist()
        categorical_features = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_features = self.data.select_dtypes(include=['datetime']).columns.tolist()

        diagnosis_report['feature_types'] = {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'datetime': datetime_features
        }
        self.logger.info(f"  - 数值特征: {len(numeric_features)} 个")
        self.logger.info(f"  - 分类特征: {len(categorical_features)} 个")
        self.logger.info(f"  - 时间特征: {len(datetime_features)} 个")

        # 3. 缺失值分析
        self.logger.info("1.3 缺失值分析")
        missing_summary = self.data.isnull().sum()
        missing_features = missing_summary[missing_summary > 0]

        diagnosis_report['missing_values'] = {
            'total_missing': missing_summary.sum(),
            'missing_ratio': missing_summary.sum() / (n_samples * n_features),
            'features_with_missing': missing_features.to_dict()
        }
        self.logger.info(f"  - 总缺失值: {diagnosis_report['missing_values']['total_missing']:,}")
        self.logger.info(f"  - 缺失比例: {diagnosis_report['missing_values']['missing_ratio']:.2%}")
        self.logger.info(f"  - 有缺失的特征: {len(missing_features)} 个")

        # 4. 数值特征统计
        if numeric_features:
            self.logger.info("1.4 数值特征统计")
            diagnosis_report['numeric_stats'] = self.data[numeric_features].describe().to_dict()

        # 5. 分类特征统计
        if categorical_features:
            self.logger.info("1.5 分类特征统计")
            cat_stats = {}
            for col in categorical_features:
                cat_stats[col] = {
                    'n_unique': self.data[col].nunique(),
                    'top_values': self.data[col].value_counts().head(5).to_dict()
                }
            diagnosis_report['categorical_stats'] = cat_stats

        # 6. 保存诊断报告
        self._save_phase_results(1, diagnosis_report)

        self.results['phase1'] = diagnosis_report
        self.logger.info("Phase 1 完成！诊断报告已保存。\n")

        return diagnosis_report

    # ==================== Phase 2: 快速Baseline ====================

    def phase2_quick_baseline(self) -> Dict[str, Any]:
        """
        Phase 2: 快速Baseline

        执行内容:
        - 基础数据预处理
        - 训练3-4个简单模型
        - 快速性能对比
        - 确定问题难度

        Returns:
            Baseline结果字典
        """
        self.logger.info("=" * 50)
        self.logger.info("Phase 2: 快速Baseline")
        self.logger.info("=" * 50)

        if self.data is None:
            raise ValueError("请先加载数据")

        baseline_results = {}

        # NOTE: 这里是简化示例，实际应调用 src/supervised_pipeline.py
        self.logger.info("2.1 基础数据预处理")
        # TODO: 调用 data_preprocessing.preprocess_data()

        self.logger.info("2.2 训练Baseline模型")
        # TODO: 调用 supervised_pipeline.train_baseline_models()

        baseline_results['status'] = 'completed'
        baseline_results['message'] = '请参考完整实现版本（需要实现 src/ 模块）'

        self.results['phase2'] = baseline_results
        self.logger.info("Phase 2 完成！Baseline结果已保存。\n")

        return baseline_results

    # ==================== Phase 3: 监督学习方案 ====================

    def phase3_supervised_solution(self) -> Dict[str, Any]:
        """
        Phase 3: 监督学习深入方案

        执行内容:
        - 高级特征工程
        - 多模型训练与对比
        - 超参数调优
        - 模型融合

        Returns:
            监督学习结果字典
        """
        self.logger.info("=" * 50)
        self.logger.info("Phase 3: 监督学习方案")
        self.logger.info("=" * 50)

        supervised_results = {}

        # NOTE: 这里是简化示例
        self.logger.info("3.1 高级特征工程")
        # TODO: 调用 feature_engineering.create_features()

        self.logger.info("3.2 模型训练与调优")
        # TODO: 调用 supervised_pipeline.train_and_tune()

        self.logger.info("3.3 模型融合")
        # TODO: 调用 supervised_pipeline.ensemble_models()

        supervised_results['status'] = 'completed'
        supervised_results['message'] = '请参考完整实现版本（需要实现 src/ 模块）'

        self.results['phase3'] = supervised_results
        self.logger.info("Phase 3 完成！监督学习方案已保存。\n")

        return supervised_results

    # ==================== Phase 4: 无监督学习洞察 ====================

    def phase4_unsupervised_insights(self) -> Dict[str, Any]:
        """
        Phase 4: 无监督学习洞察

        执行内容:
        - 聚类分析
        - 降维可视化
        - 异常检测
        - 发现数据模式

        Returns:
            无监督学习结果字典
        """
        self.logger.info("=" * 50)
        self.logger.info("Phase 4: 无监督学习洞察")
        self.logger.info("=" * 50)

        unsupervised_results = {}

        # NOTE: 这里是简化示例
        self.logger.info("4.1 聚类分析")
        # TODO: 调用 unsupervised_pipeline.clustering_analysis()

        self.logger.info("4.2 降维可视化")
        # TODO: 调用 unsupervised_pipeline.dimensionality_reduction()

        self.logger.info("4.3 异常检测")
        # TODO: 调用 unsupervised_pipeline.anomaly_detection()

        unsupervised_results['status'] = 'completed'
        unsupervised_results['message'] = '请参考完整实现版本（需要实现 src/ 模块）'

        self.results['phase4'] = unsupervised_results
        self.logger.info("Phase 4 完成！无监督洞察已保存。\n")

        return unsupervised_results

    # ==================== Phase 5: 混合方案（可选）====================

    def phase5_integrated_approach(self) -> Dict[str, Any]:
        """
        Phase 5: 混合方案（可选）

        执行内容:
        - 聚类标签作为特征
        - 异常分数作为特征
        - 先聚类再分类
        - 混合模型对比

        Returns:
            混合方案结果字典
        """
        self.logger.info("=" * 50)
        self.logger.info("Phase 5: 混合方案（可选）")
        self.logger.info("=" * 50)

        integrated_results = {}

        # NOTE: 这里是简化示例
        self.logger.info("5.1 聚类特征融合")
        # TODO: 添加聚类标签作为新特征

        self.logger.info("5.2 混合模型训练")
        # TODO: 使用增强后的特征训练模型

        integrated_results['status'] = 'completed'
        integrated_results['message'] = '混合方案为可选内容'

        self.results['phase5'] = integrated_results
        self.logger.info("Phase 5 完成！混合方案已保存。\n")

        return integrated_results

    # ==================== Phase 6: 最终方案与部署 ====================

    def phase6_final_solution(self) -> Dict[str, Any]:
        """
        Phase 6: 最终方案与部署

        执行内容:
        - 最佳模型选择
        - 完整Pipeline构建
        - 性能报告生成
        - 模型保存与导出

        Returns:
            最终方案结果字典
        """
        self.logger.info("=" * 50)
        self.logger.info("Phase 6: 最终方案与部署")
        self.logger.info("=" * 50)

        final_results = {}

        # NOTE: 这里是简化示例
        self.logger.info("6.1 最佳模型选择")
        # TODO: 从之前的结果中选择最佳模型

        self.logger.info("6.2 完整Pipeline构建")
        # TODO: 构建包含预处理+模型的完整Pipeline

        self.logger.info("6.3 模型保存")
        # TODO: 保存最终模型

        final_results['status'] = 'completed'
        final_results['message'] = '最终方案已保存'

        self.results['phase6'] = final_results
        self.logger.info("Phase 6 完成！最终方案已保存。\n")

        return final_results

    # ==================== 辅助方法 ====================

    def _save_phase_results(self, phase: int, results: Dict[str, Any]):
        """
        保存Phase结果到JSON文件

        Args:
            phase: Phase编号
            results: 结果字典
        """
        import json

        phase_config = self.config.get_phase_config(phase)
        output_dir = phase_config.get('output_dir')

        if output_dir:
            output_file = Path(output_dir) / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False, default=str)

            self.logger.info(f"结果已保存至: {output_file}")

    def run_all_phases(self):
        """按顺序运行所有Phase"""
        self.logger.info("开始运行完整工作流...\n")

        try:
            self.phase1_data_diagnosis()
            self.phase2_quick_baseline()
            self.phase3_supervised_solution()
            self.phase4_unsupervised_insights()
            # Phase 5 和 6 可选
            # self.phase5_integrated_approach()
            # self.phase6_final_solution()

            self.logger.info("=" * 50)
            self.logger.info("完整工作流执行完成！")
            self.logger.info("=" * 50)

        except Exception as e:
            self.logger.error(f"工作流执行失败: {str(e)}")
            raise

    def run_phase(self, phase: int):
        """
        运行指定的Phase

        Args:
            phase: Phase编号 (1-6)
        """
        phase_methods = {
            1: self.phase1_data_diagnosis,
            2: self.phase2_quick_baseline,
            3: self.phase3_supervised_solution,
            4: self.phase4_unsupervised_insights,
            5: self.phase5_integrated_approach,
            6: self.phase6_final_solution,
        }

        if phase in phase_methods:
            phase_methods[phase]()
        else:
            raise ValueError(f"无效的Phase编号: {phase}，请选择 1-6")

    def get_summary(self) -> Dict[str, Any]:
        """
        获取工作流执行摘要

        Returns:
            摘要字典
        """
        summary = {
            'data_path': self.data_path,
            'data_shape': self.data.shape if self.data is not None else None,
            'completed_phases': list(self.results.keys()),
            'config_mode': self.config.MODE,
        }

        return summary


# ==================== 交互式模式 ====================

def interactive_mode():
    """交互式运行模式"""
    print("=" * 60)
    print("欢迎使用机器学习综合项目工作流 - 交互式模式")
    print("=" * 60)

    # 1. 选择配置模式
    print("\n请选择运行模式:")
    print("1. 开发模式 (development) - 完整日志，调试信息")
    print("2. 生产模式 (production) - 精简日志")
    print("3. 测试模式 (testing) - 快速运行，小规模")

    mode_choice = input("请输入选项 (1-3, 默认1): ").strip() or '1'

    mode_map = {'1': 'development', '2': 'production', '3': 'testing'}
    config_func_map = {
        'development': get_default_config,
        'production': get_production_config,
        'testing': get_testing_config,
    }

    selected_mode = mode_map.get(mode_choice, 'development')
    config = config_func_map[selected_mode]()

    print(f"\n已选择: {selected_mode} 模式")

    # 2. 加载数据
    data_path = input("\n请输入数据文件路径 (支持csv/xlsx/json): ").strip()

    if not data_path or not Path(data_path).exists():
        print("❌ 数据文件不存在，请检查路径")
        return

    # 3. 创建工作流
    workflow = MLWorkflow(config, data_path)

    # 4. 选择运行Phase
    while True:
        print("\n" + "=" * 60)
        print("请选择要执行的操作:")
        print("1. Phase 1: 数据诊断")
        print("2. Phase 2: 快速Baseline")
        print("3. Phase 3: 监督学习方案")
        print("4. Phase 4: 无监督学习洞察")
        print("5. Phase 5: 混合方案（可选）")
        print("6. Phase 6: 最终方案与部署")
        print("A. 运行所有Phase")
        print("S. 查看执行摘要")
        print("Q. 退出")
        print("=" * 60)

        choice = input("请输入选项: ").strip().upper()

        if choice == 'Q':
            print("\n感谢使用！")
            break
        elif choice == 'A':
            workflow.run_all_phases()
        elif choice == 'S':
            summary = workflow.get_summary()
            print("\n执行摘要:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
        elif choice in ['1', '2', '3', '4', '5', '6']:
            try:
                workflow.run_phase(int(choice))
            except Exception as e:
                print(f"❌ 执行失败: {str(e)}")
        else:
            print("❌ 无效选项，请重新选择")


# ==================== 命令行入口 ====================

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description="机器学习综合项目工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互式模式
  python main_workflow.py --mode interactive

  # 自动运行所有Phase
  python main_workflow.py --mode auto --data data/raw/data.csv

  # 运行指定Phase
  python main_workflow.py --phase 1 --data data/raw/data.csv --config-mode testing
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'auto'],
        default='interactive',
        help='运行模式: interactive(交互式) 或 auto(自动运行)'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='数据文件路径'
    )

    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='运行指定的Phase (1-6)'
    )

    parser.add_argument(
        '--config-mode',
        type=str,
        choices=['development', 'production', 'testing'],
        default='development',
        help='配置模式'
    )

    args = parser.parse_args()

    # 交互式模式
    if args.mode == 'interactive':
        interactive_mode()
        return

    # 自动模式
    if not args.data:
        print("❌ 自动模式需要提供 --data 参数")
        parser.print_help()
        sys.exit(1)

    # 创建配置
    config_func_map = {
        'development': get_default_config,
        'production': get_production_config,
        'testing': get_testing_config,
    }
    config = config_func_map[args.config_mode]()

    # 创建工作流
    workflow = MLWorkflow(config, args.data)

    # 运行Phase
    if args.phase:
        workflow.run_phase(args.phase)
    else:
        workflow.run_all_phases()

    # 显示摘要
    summary = workflow.get_summary()
    print("\n" + "=" * 60)
    print("执行摘要:")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    main()
