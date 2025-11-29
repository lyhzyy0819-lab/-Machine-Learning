"""
综合项目配置文件
=================

集中管理所有配置参数，包括数据路径、模型参数、训练配置等。

使用方法:
    from config import Config
    cfg = Config()
    print(cfg.DATA_RAW_PATH)
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import logging


class Config:
    """综合项目配置类"""

    def __init__(self, mode: str = 'development'):
        """
        初始化配置

        Args:
            mode: 运行模式 ('development', 'production', 'testing')
        """
        self.MODE = mode
        self._setup_paths()
        self._setup_data_config()
        self._setup_model_config()
        self._setup_training_config()
        self._setup_evaluation_config()
        self._setup_visualization_config()
        self._setup_logging_config()

    # ==================== 1. 路径配置 ====================

    def _setup_paths(self):
        """设置所有路径配置"""
        # 项目根目录
        self.PROJECT_ROOT = Path(__file__).parent

        # 数据目录
        self.DATA_DIR = self.PROJECT_ROOT / 'data'
        self.DATA_RAW_PATH = self.DATA_DIR / 'raw'
        self.DATA_PROCESSED_PATH = self.DATA_DIR / 'processed'

        # Phase 输出目录
        self.OUTPUT_DIR = self.PROJECT_ROOT / 'output'
        self.PHASE1_OUTPUT = self.OUTPUT_DIR / 'phase1_diagnosis'
        self.PHASE2_OUTPUT = self.OUTPUT_DIR / 'phase2_baseline'
        self.PHASE3_OUTPUT = self.OUTPUT_DIR / 'phase3_supervised'
        self.PHASE4_OUTPUT = self.OUTPUT_DIR / 'phase4_unsupervised'
        self.PHASE5_OUTPUT = self.OUTPUT_DIR / 'phase5_integrated'
        self.PHASE6_OUTPUT = self.OUTPUT_DIR / 'phase6_final'

        # 模型保存目录
        self.MODELS_DIR = self.PROJECT_ROOT / 'models'
        self.CHECKPOINT_DIR = self.MODELS_DIR / 'checkpoints'

        # 日志和可视化目录
        self.LOGS_DIR = self.PROJECT_ROOT / 'logs'
        self.FIGURES_DIR = self.PROJECT_ROOT / 'figures'

        # 创建所有必要的目录
        self._create_directories()

    def _create_directories(self):
        """创建所有必要的目录"""
        directories = [
            self.DATA_RAW_PATH,
            self.DATA_PROCESSED_PATH,
            self.PHASE1_OUTPUT,
            self.PHASE2_OUTPUT,
            self.PHASE3_OUTPUT,
            self.PHASE4_OUTPUT,
            self.PHASE5_OUTPUT,
            self.PHASE6_OUTPUT,
            self.MODELS_DIR,
            self.CHECKPOINT_DIR,
            self.LOGS_DIR,
            self.FIGURES_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    # ==================== 2. 数据配置 ====================

    def _setup_data_config(self):
        """设置数据处理配置"""

        # 数据文件名（默认配置，可被覆盖）
        self.DATA_FILENAME = 'data.csv'

        # 目标变量名（需要根据实际数据修改）
        self.TARGET_COLUMN = 'target'

        # 特征列（None表示自动识别，排除target）
        self.FEATURE_COLUMNS = None

        # ID列（不参与建模）
        self.ID_COLUMN = None

        # 时间列（用于时序数据）
        self.TIME_COLUMN = None

        # 数据切分比例
        self.TRAIN_SIZE = 0.7
        self.VAL_SIZE = 0.15
        self.TEST_SIZE = 0.15

        # 随机种子（保证可复现性）
        self.RANDOM_STATE = 42

        # 缺失值处理配置
        self.MISSING_VALUE_CONFIG = {
            'numeric_strategy': 'median',   # 'mean', 'median', 'mode', 'constant', 'knn'
            'categorical_strategy': 'mode', # 'mode', 'constant', 'unknown'
            'threshold': 0.5,               # 缺失值超过此比例的特征将被删除
        }

        # 异常值处理配置
        self.OUTLIER_CONFIG = {
            'method': 'iqr',                # 'iqr', 'zscore', 'isolation_forest'
            'iqr_multiplier': 1.5,          # IQR方法的倍数
            'zscore_threshold': 3,          # Z-score方法的阈值
            'contamination': 0.05,          # Isolation Forest的污染比例
        }

        # 特征编码配置
        self.ENCODING_CONFIG = {
            'low_cardinality_threshold': 10,    # 低基数阈值（使用One-Hot）
            'high_cardinality_method': 'target', # 高基数编码方法: 'target', 'frequency', 'label'
        }

        # 特征缩放配置
        self.SCALING_CONFIG = {
            'method': 'standard',            # 'standard', 'minmax', 'robust', 'none'
        }

    # ==================== 3. 模型配置 ====================

    def _setup_model_config(self):
        """设置模型参数配置"""

        # Phase 2: Baseline 模型
        self.BASELINE_MODELS = {
            'LogisticRegression': {
                'random_state': self.RANDOM_STATE,
                'max_iter': 1000,
                'class_weight': 'balanced',  # 自动处理类别不平衡
            },
            'DecisionTree': {
                'random_state': self.RANDOM_STATE,
                'max_depth': 5,
                'min_samples_split': 20,
            },
            'RandomForest': {
                'n_estimators': 100,
                'random_state': self.RANDOM_STATE,
                'max_depth': 10,
                'class_weight': 'balanced',
                'n_jobs': -1,
            },
        }

        # Phase 3: 监督学习模型（完整参数）
        self.SUPERVISED_MODELS = {
            'LogisticRegression': {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'random_state': self.RANDOM_STATE,
                'max_iter': 1000,
                'class_weight': 'balanced',
            },
            'RandomForest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'random_state': self.RANDOM_STATE,
                'class_weight': 'balanced',
                'n_jobs': -1,
            },
            'XGBoost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.RANDOM_STATE,
                'objective': 'binary:logistic',  # 根据任务类型修改
                'eval_metric': 'auc',
                'n_jobs': -1,
            },
            'LightGBM': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.RANDOM_STATE,
                'objective': 'binary',           # 根据任务类型修改
                'metric': 'auc',
                'n_jobs': -1,
                'verbose': -1,
            },
        }

        # Phase 4: 无监督学习模型
        self.UNSUPERVISED_MODELS = {
            'KMeans': {
                'n_clusters': 3,                 # 需要根据肘部法则确定
                'init': 'k-means++',
                'n_init': 10,
                'random_state': self.RANDOM_STATE,
            },
            'DBSCAN': {
                'eps': 0.5,                      # 需要根据k-距离图确定
                'min_samples': 5,
                'metric': 'euclidean',
                'n_jobs': -1,
            },
            'PCA': {
                'n_components': 0.95,            # 保留95%方差
                'random_state': self.RANDOM_STATE,
            },
        }

        # 超参数搜索配置
        self.HYPERPARAMETER_SEARCH_CONFIG = {
            'method': 'random',                  # 'grid', 'random', 'bayesian'
            'n_iter': 20,                        # RandomizedSearchCV的迭代次数
            'cv': 5,                             # 交叉验证折数
            'scoring': 'roc_auc',                # 优化指标
            'n_jobs': -1,
            'verbose': 1,
        }

    # ==================== 4. 训练配置 ====================

    def _setup_training_config(self):
        """设置训练流程配置"""

        # 交叉验证配置
        self.CV_CONFIG = {
            'n_splits': 5,                       # K折交叉验证的折数
            'shuffle': True,
            'random_state': self.RANDOM_STATE,
        }

        # 早停配置（针对支持的模型）
        self.EARLY_STOPPING_CONFIG = {
            'enabled': True,
            'metric': 'auc',
            'patience': 10,
            'min_delta': 0.001,
        }

        # 集成学习配置
        self.ENSEMBLE_CONFIG = {
            'method': 'voting',                  # 'voting', 'stacking', 'blending'
            'voting': 'soft',                    # 'hard', 'soft'
            'weights': None,                     # None表示等权重
        }

        # 模型保存配置
        self.MODEL_SAVE_CONFIG = {
            'save_best_only': True,              # 只保存最佳模型
            'save_interval': None,               # 保存间隔（轮数），None表示仅最终保存
            'format': 'joblib',                  # 'joblib', 'pickle'
        }

    # ==================== 5. 评估配置 ====================

    def _setup_evaluation_config(self):
        """设置模型评估配置"""

        # 分类任务评估指标
        self.CLASSIFICATION_METRICS = [
            'accuracy',
            'precision',
            'recall',
            'f1',
            'roc_auc',
            'confusion_matrix',
        ]

        # 回归任务评估指标
        self.REGRESSION_METRICS = [
            'mae',
            'mse',
            'rmse',
            'r2',
            'mape',
        ]

        # 聚类任务评估指标
        self.CLUSTERING_METRICS = [
            'silhouette_score',
            'calinski_harabasz_score',
            'davies_bouldin_score',
        ]

        # 评估报告配置
        self.EVALUATION_REPORT_CONFIG = {
            'include_train_metrics': True,       # 报告中包含训练集指标
            'include_val_metrics': True,         # 报告中包含验证集指标
            'include_test_metrics': True,        # 报告中包含测试集指标
            'include_confusion_matrix': True,    # 包含混淆矩阵
            'include_roc_curve': True,           # 包含ROC曲线
            'include_feature_importance': True,  # 包含特征重要性
        }

        # 性能对比配置
        self.COMPARISON_CONFIG = {
            'primary_metric': 'roc_auc',         # 主要对比指标
            'sort_by': 'test_score',             # 排序依据
            'ascending': False,                  # 降序排列
        }

    # ==================== 6. 可视化配置 ====================

    def _setup_visualization_config(self):
        """设置可视化配置"""

        # 图表样式
        self.PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # matplotlib样式

        # 图表尺寸
        self.FIGURE_SIZE = (10, 6)               # (width, height)
        self.FIGURE_DPI = 100

        # 颜色配置
        self.COLOR_PALETTE = 'viridis'           # matplotlib色板

        # 保存格式
        self.FIGURE_FORMAT = 'png'               # 'png', 'jpg', 'svg', 'pdf'

        # 可视化选项
        self.VISUALIZATION_OPTIONS = {
            # Phase 1: 数据诊断可视化
            'phase1': {
                'missing_values_heatmap': True,
                'distribution_plots': True,
                'correlation_heatmap': True,
                'outliers_boxplot': True,
            },
            # Phase 2: Baseline可视化
            'phase2': {
                'model_comparison_bar': True,
                'confusion_matrix': True,
            },
            # Phase 3: 监督学习可视化
            'phase3': {
                'learning_curves': True,
                'roc_curves': True,
                'feature_importance': True,
                'confusion_matrix': True,
            },
            # Phase 4: 无监督学习可视化
            'phase4': {
                'elbow_curve': True,              # K-Means肘部曲线
                'silhouette_plot': True,          # 轮廓系数图
                'cluster_visualization': True,    # 聚类结果可视化（2D/3D）
                'pca_variance_plot': True,        # PCA方差解释图
            },
        }

    # ==================== 7. 日志配置 ====================

    def _setup_logging_config(self):
        """设置日志配置"""

        # 日志级别
        if self.MODE == 'production':
            self.LOG_LEVEL = logging.INFO
        elif self.MODE == 'development':
            self.LOG_LEVEL = logging.DEBUG
        else:  # testing
            self.LOG_LEVEL = logging.WARNING

        # 日志格式
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

        # 日志文件
        self.LOG_FILE = self.LOGS_DIR / f'workflow_{self.MODE}.log'

        # 控制台输出
        self.LOG_TO_CONSOLE = True

        # 日志文件大小限制（MB）
        self.LOG_MAX_SIZE = 10

        # 日志文件备份数量
        self.LOG_BACKUP_COUNT = 5

    # ==================== 8. 运行模式配置 ====================

    def get_phase_config(self, phase: int) -> Dict[str, Any]:
        """
        获取指定Phase的完整配置

        Args:
            phase: Phase编号 (1-6)

        Returns:
            该Phase的配置字典
        """
        phase_configs = {
            1: {
                'name': 'Data Diagnosis',
                'output_dir': self.PHASE1_OUTPUT,
                'visualizations': self.VISUALIZATION_OPTIONS['phase1'],
            },
            2: {
                'name': 'Quick Baseline',
                'output_dir': self.PHASE2_OUTPUT,
                'models': self.BASELINE_MODELS,
                'visualizations': self.VISUALIZATION_OPTIONS['phase2'],
            },
            3: {
                'name': 'Supervised Learning',
                'output_dir': self.PHASE3_OUTPUT,
                'models': self.SUPERVISED_MODELS,
                'hyperparameter_search': self.HYPERPARAMETER_SEARCH_CONFIG,
                'visualizations': self.VISUALIZATION_OPTIONS['phase3'],
            },
            4: {
                'name': 'Unsupervised Insights',
                'output_dir': self.PHASE4_OUTPUT,
                'models': self.UNSUPERVISED_MODELS,
                'visualizations': self.VISUALIZATION_OPTIONS['phase4'],
            },
        }

        return phase_configs.get(phase, {})

    def update_config(self, **kwargs):
        """
        动态更新配置参数

        Args:
            **kwargs: 要更新的配置参数

        Example:
            cfg.update_config(RANDOM_STATE=123, TRAIN_SIZE=0.8)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"配置参数 '{key}' 不存在")

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典

        Returns:
            配置字典
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                # 将Path对象转换为字符串
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict

    def save_config(self, filepath: str = None):
        """
        保存配置到JSON文件

        Args:
            filepath: 保存路径，默认为 logs/config_{mode}.json
        """
        import json

        if filepath is None:
            filepath = self.LOGS_DIR / f'config_{self.MODE}.json'

        config_dict = self.to_dict()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False, default=str)

        print(f"配置已保存至: {filepath}")

    @classmethod
    def load_config(cls, filepath: str) -> 'Config':
        """
        从JSON文件加载配置

        Args:
            filepath: 配置文件路径

        Returns:
            Config对象
        """
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # 创建Config对象
        config = cls()

        # 更新配置
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def __repr__(self):
        """配置对象的字符串表示"""
        return f"Config(mode='{self.MODE}', random_state={self.RANDOM_STATE})"


# ==================== 预定义配置 ====================

def get_default_config() -> Config:
    """获取默认配置（开发模式）"""
    return Config(mode='development')


def get_production_config() -> Config:
    """获取生产配置"""
    config = Config(mode='production')
    # 生产环境特定配置
    config.update_config(
        LOG_LEVEL=logging.INFO,
        MODEL_SAVE_CONFIG={'save_best_only': True, 'format': 'joblib'}
    )
    return config


def get_testing_config() -> Config:
    """获取测试配置（快速运行，小规模）"""
    config = Config(mode='testing')
    # 测试环境特定配置（减少计算量）
    config.update_config(
        TRAIN_SIZE=0.6,
        CV_CONFIG={'n_splits': 3, 'shuffle': True, 'random_state': 42},
    )
    # 减少模型复杂度
    config.BASELINE_MODELS['RandomForest']['n_estimators'] = 50
    config.SUPERVISED_MODELS['RandomForest']['n_estimators'] = 100
    config.SUPERVISED_MODELS['XGBoost']['n_estimators'] = 100
    config.SUPERVISED_MODELS['LightGBM']['n_estimators'] = 100
    config.HYPERPARAMETER_SEARCH_CONFIG['n_iter'] = 5

    return config


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 示例1: 使用默认配置
    cfg = get_default_config()
    print(cfg)
    print(f"数据路径: {cfg.DATA_RAW_PATH}")
    print(f"随机种子: {cfg.RANDOM_STATE}")

    # 示例2: 更新配置
    cfg.update_config(RANDOM_STATE=123, TRAIN_SIZE=0.8)
    print(f"更新后随机种子: {cfg.RANDOM_STATE}")

    # 示例3: 保存配置
    cfg.save_config()

    # 示例4: 获取Phase配置
    phase2_config = cfg.get_phase_config(2)
    print(f"\nPhase 2 配置: {phase2_config['name']}")
    print(f"Baseline模型: {list(phase2_config['models'].keys())}")

    # 示例5: 使用生产配置
    prod_cfg = get_production_config()
    print(f"\n生产模式配置: {prod_cfg}")

    # 示例6: 使用测试配置
    test_cfg = get_testing_config()
    print(f"测试模式配置: {test_cfg}")
    print(f"测试模式CV折数: {test_cfg.CV_CONFIG['n_splits']}")
