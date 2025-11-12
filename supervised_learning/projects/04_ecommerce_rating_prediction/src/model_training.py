"""
模型训练模块
包含回归和分类模型的训练、调优和评估
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, save_model, save_json


class RegressionTrainer:
    """
    回归模型训练器类
    用于预测评分（连续值1.0-5.0）
    """

    def __init__(self, random_state: int = None):
        """
        初始化回归训练器

        Args:
            random_state: 随机种子

        TODO 1: 初始化基本属性
        # self.random_state = random_state or config.RANDOM_STATE
        # self.logger = logging.getLogger("Ecommerce_Rating")

        TODO 2: 初始化缩放器
        # self.scaler = StandardScaler()

        TODO 3: 初始化模型存储字典
        # self.models = {}  # 存储训练好的模型
        # self.best_model = None
        # self.best_model_name = None
        # self.best_score = -np.inf

        TODO 4: 初始化数据存储
        # self.X_train = None
        # self.X_test = None
        # self.y_train = None
        # self.y_test = None
        # self.X_train_scaled = None
        # self.X_test_scaled = None
        """
        # TODO: 实现初始化
        pass

    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                    test_size: float = None,
                    scale: bool = True) -> None:
        """
        准备训练和测试数据

        Args:
            X: 特征DataFrame
            y: 目标Series
            test_size: 测试集比例
            scale: 是否进行特征缩放

        TODO 1: 打印准备数据信息
        # self.logger.info("=" * 60)
        # self.logger.info("准备回归训练数据")
        # self.logger.info("=" * 60)

        TODO 2: 设置默认test_size
        # if test_size is None:
        #     test_size = config.TEST_SIZE

        TODO 3: 分割数据
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #     X, y, test_size=test_size, random_state=self.random_state
        # )
        # self.logger.info(f"训练集大小: {self.X_train.shape}")
        # self.logger.info(f"测试集大小: {self.X_test.shape}")

        TODO 4: 特征缩放（如果启用）
        # if scale:
        #     self.logger.info("执行特征缩放...")
        #     self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        #     self.X_test_scaled = self.scaler.transform(self.X_test)
        #     self.logger.info("特征缩放完成")
        # else:
        #     self.X_train_scaled = self.X_train.values
        #     self.X_test_scaled = self.X_test.values
        """
        # TODO: 实现数据准备
        pass

    def train_model(self, model_name: str,
                   tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        训练单个回归模型

        Args:
            model_name: 模型名称（在config中定义）
            tune_hyperparameters: 是否进行超参数调优

        Returns:
            模型结果字典

        TODO 1: 获取模型配置
        # if model_name not in config.REGRESSION_MODELS:
        #     raise ValueError(f"未知模型: {model_name}")
        #
        # model_config = config.REGRESSION_MODELS[model_name]
        # self.logger.info(f"\n训练回归模型: {model_config['description']}")

        TODO 2: 使用Timer计时训练
        # with Timer(f"{model_name}训练"):

        TODO 3: 如果启用超参数调优且有param_grid
        #     if tune_hyperparameters and 'param_grid' in model_config:
        #         self.logger.info("  执行网格搜索...")
        #         grid_search = GridSearchCV(
        #             model_config['model'],
        #             model_config['param_grid'],
        #             cv=config.CV_FOLDS,
        #             scoring='r2',
        #             n_jobs=-1
        #         )
        #         grid_search.fit(self.X_train_scaled, self.y_train)
        #         model = grid_search.best_estimator_
        #         self.logger.info(f"  最佳参数: {grid_search.best_params_}")
        #     else:
        #         # 使用默认参数训练
        #         model = model_config['model']
        #         model.fit(self.X_train_scaled, self.y_train)

        TODO 4: 预测
        #     y_pred = model.predict(self.X_test_scaled)

        TODO 5: 计算评估指标
        #     metrics = self._calculate_metrics(self.y_test, y_pred)

        TODO 6: 存储模型结果
        #     self.models[model_name] = {
        #         'model': model,
        #         'metrics': metrics,
        #         'predictions': y_pred
        #     }

        TODO 7: 记录指标
        #     self._log_metrics(model_name, metrics)

        TODO 8: 返回结果
        # return self.models[model_name]
        """
        # TODO: 实现模型训练
        pass

    def train_all_models(self, tune_hyperparameters: bool = False) -> Dict[str, Dict]:
        """
        训练所有回归模型

        Args:
            tune_hyperparameters: 是否进行超参数调优

        Returns:
            所有模型结果字典

        TODO 1: 打印开始信息
        # self.logger.info("=" * 60)
        # self.logger.info("开始训练所有回归模型")
        # self.logger.info("=" * 60)

        TODO 2: 遍历config中定义的所有回归模型
        # for model_name in config.REGRESSION_MODELS.keys():
        #     try:
        #         self.train_model(model_name, tune_hyperparameters)
        #     except Exception as e:
        #         self.logger.error(f"训练 {model_name} 失败: {str(e)}")

        TODO 3: 选择最佳模型
        # self._select_best_model()

        TODO 4: 打印完成信息
        # self.logger.info("\n" + "=" * 60)
        # self.logger.info("所有回归模型训练完成！")
        # self.logger.info(f"最佳模型: {self.best_model_name} (R² = {self.best_score:.4f})")
        # self.logger.info("=" * 60)

        TODO 5: 返回所有模型结果
        # return self.models
        """
        # TODO: 实现训练所有模型
        pass

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            指标字典

        TODO: 计算R²、RMSE、MAE、MAPE
        # r2 = r2_score(y_true, y_pred)
        # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # mae = mean_absolute_error(y_true, y_pred)
        # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        #
        # return {
        #     'r2': r2,
        #     'rmse': rmse,
        #     'mae': mae,
        #     'mape': mape
        # }
        """
        # TODO: 实现指标计算
        pass

    def _log_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """记录模型评估指标"""
        # TODO: 打印各项指标
        pass

    def _select_best_model(self) -> None:
        """选择R²最高的模型作为最佳模型"""
        # TODO: 遍历所有模型，找到R²最高的
        pass


class ClassificationTrainer:
    """
    分类模型训练器类
    用于预测高/低评分（二分类：rating >= 4.0）
    """

    def __init__(self, random_state: int = None):
        """
        初始化分类训练器

        TODO: 参考RegressionTrainer的__init__实现
        提示：需要初始化random_state、logger、scaler、models字典、数据存储等
        """
        # TODO: 实现初始化
        pass

    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                    test_size: float = None,
                    scale: bool = True) -> None:
        """
        准备训练和测试数据

        TODO: 参考RegressionTrainer的prepare_data实现
        注意：y应该是二分类标签（0或1）
        """
        # TODO: 实现数据准备
        pass

    def train_model(self, model_name: str,
                   tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        训练单个分类模型

        TODO: 参考RegressionTrainer的train_model实现
        注意：使用config.CLASSIFICATION_MODELS而不是REGRESSION_MODELS
        """
        # TODO: 实现模型训练
        pass

    def train_all_models(self, tune_hyperparameters: bool = False) -> Dict[str, Dict]:
        """
        训练所有分类模型

        TODO: 参考RegressionTrainer的train_all_models实现
        注意：使用config.CLASSIFICATION_MODELS
        """
        # TODO: 实现训练所有模型
        pass

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        计算分类评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率（用于计算ROC-AUC）

        Returns:
            指标字典

        TODO: 计算准确率、精确率、召回率、F1、ROC-AUC
        # accuracy = accuracy_score(y_true, y_pred)
        # precision = precision_score(y_true, y_pred, zero_division=0)
        # recall = recall_score(y_true, y_pred, zero_division=0)
        # f1 = f1_score(y_true, y_pred, zero_division=0)
        #
        # metrics = {
        #     'accuracy': accuracy,
        #     'precision': precision,
        #     'recall': recall,
        #     'f1': f1
        # }
        #
        # # 如果有预测概率，计算ROC-AUC
        # if y_pred_proba is not None:
        #     roc_auc = roc_auc_score(y_true, y_pred_proba)
        #     metrics['roc_auc'] = roc_auc
        #
        # return metrics
        """
        # TODO: 实现分类指标计算
        pass

    def _log_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """记录分类模型评估指标"""
        # TODO: 打印各项指标（accuracy, precision, recall, f1, roc_auc）
        pass

    def _select_best_model(self) -> None:
        """选择F1-score最高的模型作为最佳模型"""
        # TODO: 遍历所有模型，找到F1最高的
        pass


def create_classification_target(y: pd.Series,
                                 threshold: float = None) -> pd.Series:
    """
    创建分类目标变量（高评分 vs 低评分）

    Args:
        y: 原始评分Series
        threshold: 高评分阈值

    Returns:
        二分类标签Series（1=高评分，0=低评分）

    TODO 1: 设置默认阈值
    # if threshold is None:
    #     threshold = config.HIGH_RATING_THRESHOLD

    TODO 2: 创建二分类标签
    # y_binary = (y >= threshold).astype(int)

    TODO 3: 打印统计信息
    # logger = logging.getLogger("Ecommerce_Rating")
    # logger.info(f"创建分类目标（阈值={threshold}）:")
    # logger.info(f"  高评分样本: {y_binary.sum()} ({y_binary.sum()/len(y_binary)*100:.2f}%)")
    # logger.info(f"  低评分样本: {(1-y_binary).sum()} ({(1-y_binary).sum()/len(y_binary)*100:.2f}%)")

    TODO 4: 返回二分类标签
    # return y_binary
    """
    # TODO: 实现分类目标创建
    pass


def compare_models(models_dict: Dict[str, Dict], task: str = 'regression') -> pd.DataFrame:
    """
    对比多个模型的性能

    Args:
        models_dict: 模型结果字典
        task: 任务类型 ('regression' 或 'classification')

    Returns:
        对比结果DataFrame

    TODO 1: 创建对比列表
    # comparison = []

    TODO 2: 遍历模型字典，提取指标
    # for name, result in models_dict.items():
    #     metrics = result['metrics']
    #     row = {'模型': name}
    #     row.update(metrics)
    #     comparison.append(row)

    TODO 3: 转换为DataFrame并排序
    # df = pd.DataFrame(comparison)
    #
    # if task == 'regression':
    #     df = df.sort_values('r2', ascending=False)
    # else:  # classification
    #     df = df.sort_values('f1', ascending=False)
    #
    # df = df.reset_index(drop=True)

    TODO 4: 返回对比DataFrame
    # return df
    """
    # TODO: 实现模型对比
    pass


if __name__ == '__main__':
    # 测试模型训练
    from src.utils import setup_logger
    from src.data_loader import load_raw_data
    from src.data_preprocessing import preprocess_data, split_features_target
    from src.feature_engineering import engineer_features, select_features

    # TODO: 设置日志
    # logger = setup_logger("Ecommerce_Rating", config.LOG_DIR / "model_training_test.log", "INFO")

    print("=" * 60)
    print("模型训练模块测试")
    print("=" * 60)

    # TODO 1: 加载和准备数据
    # try:
    #     df = load_raw_data(use_sample=True, sample_size=300)
    #     df_clean = preprocess_data(df)
    #     df_feat, _ = engineer_features(df_clean, include_text_features=False)
    #
    #     X, y = split_features_target(df_feat, target_col='rating')
    #
    #     print(f"\n数据准备完成: X={X.shape}, y={y.shape}")

    # TODO 2: 测试回归训练器
    #     print("\n" + "="*60)
    #     print("测试回归模型训练")
    #     print("="*60)
    #
    #     reg_trainer = RegressionTrainer()
    #     reg_trainer.prepare_data(X, y)
    #     reg_results = reg_trainer.train_all_models(tune_hyperparameters=False)
    #
    #     # 对比回归模型
    #     comparison_df = compare_models(reg_results, task='regression')
    #     print("\n回归模型对比:")
    #     print(comparison_df)

    # TODO 3: 测试分类训练器
    #     print("\n" + "="*60)
    #     print("测试分类模型训练")
    #     print("="*60)
    #
    #     # 创建分类目标
    #     y_binary = create_classification_target(y)
    #
    #     clf_trainer = ClassificationTrainer()
    #     clf_trainer.prepare_data(X, y_binary)
    #     clf_results = clf_trainer.train_all_models(tune_hyperparameters=False)
    #
    #     # 对比分类模型
    #     comparison_df = compare_models(clf_results, task='classification')
    #     print("\n分类模型对比:")
    #     print(comparison_df)

    # except Exception as e:
    #     print(f"\n错误: {str(e)}")

    print("\n提示：实现上述TODO后运行此文件进行测试")
