"""
模型训练模块 - 客户流失预测
包含多种分类模型的训练、SMOTE处理和超参数调优
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings

# 禁用警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', module='xgboost')
warnings.filterwarnings('ignore', module='lightgbm')

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, classification_report)

# SMOTE for imbalanced data
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logging.warning("imbalanced-learn未安装，SMOTE功能不可用")

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost未安装，将跳过XGBoost模型")

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM未安装，将跳过LightGBM模型")

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, save_model, setup_logger


class ModelTrainer:
    """
    模型训练器类
    管理多个分类模型的训练、评估和优化
    """

    def __init__(self, random_state: int = None):
        """
        初始化模型训练器

        Args:
            random_state: 随机种子
        """
        self.random_state = random_state or config.RANDOM_STATE
        self.logger = logging.getLogger("ChurnPrediction")

        # 存储训练好的模型
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf

        # 存储训练和测试数据
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None

    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                    test_size: float = None,
                    use_smote: bool = None) -> None:
        """
        准备训练和测试数据

        Args:
            X: 特征DataFrame
            y: 目标Series
            test_size: 测试集比例
            use_smote: 是否使用SMOTE处理不平衡数据
        """
        self.logger.info("=" * 60)
        self.logger.info("准备训练数据")
        self.logger.info("=" * 60)

        if test_size is None:
            test_size = config.TEST_SIZE

        if use_smote is None:
            use_smote = config.USE_SMOTE

        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        self.logger.info(f"训练集大小: {self.X_train.shape}")
        self.logger.info(f"测试集大小: {self.X_test.shape}")

        # 检查类别分布
        self.logger.info(f"\n训练集类别分布:")
        train_dist = self.y_train.value_counts()
        for label, count in train_dist.items():
            pct = count / len(self.y_train) * 100
            self.logger.info(f"  类别 {label}: {count} ({pct:.2f}%)")

        # 使用SMOTE处理不平衡数据
        if use_smote and SMOTE_AVAILABLE:
            self.logger.info("\n应用SMOTE处理类别不平衡...")
            smote = SMOTE(
                sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
                k_neighbors=config.SMOTE_K_NEIGHBORS,
                random_state=self.random_state
            )
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
                self.X_train, self.y_train
            )

            self.logger.info(f"SMOTE后训练集大小: {self.X_train_resampled.shape}")
            self.logger.info(f"SMOTE后类别分布:")
            resampled_dist = pd.Series(self.y_train_resampled).value_counts()
            for label, count in resampled_dist.items():
                pct = count / len(self.y_train_resampled) * 100
                self.logger.info(f"  类别 {label}: {count} ({pct:.2f}%)")
        else:
            self.X_train_resampled = self.X_train.copy()
            self.y_train_resampled = self.y_train.copy()

    def train_logistic_regression(self, param_grid: Optional[Dict] = None,
                                  use_grid_search: bool = True) -> Dict[str, Any]:
        """
        训练逻辑回归模型

        Args:
            param_grid: 参数网格
            use_grid_search: 是否使用网格搜索

        Returns:
            模型和评估结果字典
        """
        self.logger.info("\n训练逻辑回归模型...")

        if param_grid is None:
            param_grid = config.LOGISTIC_PARAMS

        with Timer("逻辑回归训练"):
            if use_grid_search and config.TUNE_HYPERPARAMETERS and param_grid:
                lr = LogisticRegression(random_state=self.random_state, max_iter=1000)

                if config.SEARCH_METHOD == 'random':
                    search = RandomizedSearchCV(
                        lr, param_grid,
                        n_iter=config.RANDOM_SEARCH_ITERATIONS,
                        cv=config.RANDOM_SEARCH_CV,
                        scoring=config.PRIMARY_METRIC,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                else:
                    search = GridSearchCV(
                        lr, param_grid,
                        cv=config.CV_FOLDS,
                        scoring=config.PRIMARY_METRIC,
                        n_jobs=-1
                    )

                search.fit(self.X_train_resampled, self.y_train_resampled)
                model = search.best_estimator_
                self.logger.info(f"最佳参数: {search.best_params_}")
            else:
                model = LogisticRegression(
                    C=1.0,
                    random_state=self.random_state,
                    max_iter=1000
                )
                model.fit(self.X_train_resampled, self.y_train_resampled)

            # 预测和评估
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)

            self.models['LogisticRegression'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            self._log_metrics("LogisticRegression", metrics)

        return self.models['LogisticRegression']

    def train_decision_tree(self, param_grid: Optional[Dict] = None,
                           use_grid_search: bool = True) -> Dict[str, Any]:
        """
        训练决策树模型

        Args:
            param_grid: 参数网格
            use_grid_search: 是否使用网格搜索

        Returns:
            模型和评估结果字典
        """
        self.logger.info("\n训练决策树模型...")

        if param_grid is None:
            param_grid = config.DECISION_TREE_PARAMS

        with Timer("决策树训练"):
            if use_grid_search and config.TUNE_HYPERPARAMETERS and param_grid:
                dt = DecisionTreeClassifier(random_state=self.random_state)

                # 简化参数网格
                simplified_grid = {
                    'max_depth': param_grid['max_depth'][:3],
                    'min_samples_split': param_grid['min_samples_split'][:2],
                    'min_samples_leaf': param_grid['min_samples_leaf'][:2]
                }

                search = GridSearchCV(
                    dt, simplified_grid,
                    cv=config.CV_FOLDS,
                    scoring=config.PRIMARY_METRIC,
                    n_jobs=-1
                )
                search.fit(self.X_train_resampled, self.y_train_resampled)
                model = search.best_estimator_
                self.logger.info(f"最佳参数: {search.best_params_}")
            else:
                model = DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_split=5,
                    random_state=self.random_state
                )
                model.fit(self.X_train_resampled, self.y_train_resampled)

            # 预测和评估
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)

            # 特征重要性
            feature_importance = model.feature_importances_

            self.models['DecisionTree'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'feature_importance': feature_importance
            }

            self._log_metrics("DecisionTree", metrics)

        return self.models['DecisionTree']

    def train_random_forest(self, param_grid: Optional[Dict] = None,
                           use_randomized: bool = True) -> Dict[str, Any]:
        """
        训练随机森林模型

        Args:
            param_grid: 参数网格
            use_randomized: 是否使用随机搜索

        Returns:
            模型和评估结果字典
        """
        self.logger.info("\n训练随机森林模型...")

        if param_grid is None:
            param_grid = config.RANDOM_FOREST_PARAMS

        with Timer("随机森林训练"):
            if config.TUNE_HYPERPARAMETERS and param_grid:
                rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)

                if use_randomized:
                    search = RandomizedSearchCV(
                        rf, param_grid,
                        n_iter=config.RANDOM_SEARCH_ITERATIONS,
                        cv=config.RANDOM_SEARCH_CV,
                        scoring=config.PRIMARY_METRIC,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                else:
                    # 简化参数网格
                    simplified_grid = {
                        'n_estimators': param_grid['n_estimators'][:2],
                        'max_depth': param_grid['max_depth'][:2],
                        'min_samples_split': param_grid['min_samples_split'][:2],
                        'min_samples_leaf': param_grid['min_samples_leaf'][:2]
                    }
                    search = GridSearchCV(
                        rf, simplified_grid,
                        cv=config.CV_FOLDS,
                        scoring=config.PRIMARY_METRIC,
                        n_jobs=-1
                    )

                search.fit(self.X_train_resampled, self.y_train_resampled)
                model = search.best_estimator_
                self.logger.info(f"最佳参数: {search.best_params_}")
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(self.X_train_resampled, self.y_train_resampled)

            # 预测和评估
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)

            # 特征重要性
            feature_importance = model.feature_importances_

            self.models['RandomForest'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'feature_importance': feature_importance
            }

            self._log_metrics("RandomForest", metrics)

        return self.models['RandomForest']

    def train_gradient_boosting(self, param_grid: Optional[Dict] = None,
                               use_randomized: bool = True) -> Dict[str, Any]:
        """
        训练梯度提升模型

        Args:
            param_grid: 参数网格
            use_randomized: 是否使用随机搜索

        Returns:
            模型和评估结果字典
        """
        self.logger.info("\n训练梯度提升模型...")

        if param_grid is None:
            param_grid = config.GRADIENT_BOOSTING_PARAMS

        with Timer("梯度提升训练"):
            if config.TUNE_HYPERPARAMETERS and param_grid:
                gb = GradientBoostingClassifier(random_state=self.random_state)

                if use_randomized:
                    search = RandomizedSearchCV(
                        gb, param_grid,
                        n_iter=config.RANDOM_SEARCH_ITERATIONS,
                        cv=config.RANDOM_SEARCH_CV,
                        scoring=config.PRIMARY_METRIC,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                else:
                    # 简化参数网格
                    simplified_grid = {
                        'n_estimators': param_grid['n_estimators'][:2],
                        'learning_rate': param_grid['learning_rate'][:2],
                        'max_depth': param_grid['max_depth'][:2],
                        'subsample': param_grid['subsample'][:1]
                    }
                    search = GridSearchCV(
                        gb, simplified_grid,
                        cv=config.CV_FOLDS,
                        scoring=config.PRIMARY_METRIC,
                        n_jobs=-1
                    )

                search.fit(self.X_train_resampled, self.y_train_resampled)
                model = search.best_estimator_
                self.logger.info(f"最佳参数: {search.best_params_}")
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=self.random_state
                )
                model.fit(self.X_train_resampled, self.y_train_resampled)

            # 预测和评估
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)

            # 特征重要性
            feature_importance = model.feature_importances_

            self.models['GradientBoosting'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'feature_importance': feature_importance
            }

            self._log_metrics("GradientBoosting", metrics)

        return self.models['GradientBoosting']

    def train_xgboost(self, param_grid: Optional[Dict] = None,
                     use_randomized: bool = True) -> Optional[Dict[str, Any]]:
        """
        训练XGBoost模型

        Args:
            param_grid: 参数网格
            use_randomized: 是否使用随机搜索

        Returns:
            模型和评估结果字典（如果XGBoost可用）
        """
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost不可用，跳过训练")
            return None

        self.logger.info("\n训练XGBoost模型...")

        if param_grid is None:
            param_grid = config.XGBOOST_PARAMS

        with Timer("XGBoost训练"):
            if config.TUNE_HYPERPARAMETERS and param_grid:
                xgb_model = xgb.XGBClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )

                if use_randomized:
                    search = RandomizedSearchCV(
                        xgb_model, param_grid,
                        n_iter=config.RANDOM_SEARCH_ITERATIONS,
                        cv=config.RANDOM_SEARCH_CV,
                        scoring=config.PRIMARY_METRIC,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                else:
                    # 简化参数网格
                    simplified_grid = {
                        'n_estimators': param_grid['n_estimators'][:2],
                        'max_depth': param_grid['max_depth'][:2],
                        'learning_rate': param_grid['learning_rate'][:2],
                        'subsample': param_grid['subsample'][:2],
                        'colsample_bytree': param_grid['colsample_bytree'][:2]
                    }
                    search = GridSearchCV(
                        xgb_model, simplified_grid,
                        cv=config.CV_FOLDS,
                        scoring=config.PRIMARY_METRIC,
                        n_jobs=-1
                    )

                search.fit(self.X_train_resampled, self.y_train_resampled)
                model = search.best_estimator_
                self.logger.info(f"最佳参数: {search.best_params_}")
            else:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                model.fit(self.X_train_resampled, self.y_train_resampled)

            # 预测和评估
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)

            # 特征重要性
            feature_importance = model.feature_importances_

            self.models['XGBoost'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'feature_importance': feature_importance
            }

            self._log_metrics("XGBoost", metrics)

        return self.models['XGBoost']

    def train_lightgbm(self, param_grid: Optional[Dict] = None,
                      use_randomized: bool = True) -> Optional[Dict[str, Any]]:
        """
        训练LightGBM模型

        Args:
            param_grid: 参数网格
            use_randomized: 是否使用随机搜索

        Returns:
            模型和评估结果字典（如果LightGBM可用）
        """
        if not LIGHTGBM_AVAILABLE:
            self.logger.warning("LightGBM不可用，跳过训练")
            return None

        self.logger.info("\n训练LightGBM模型...")

        if param_grid is None:
            param_grid = config.LIGHTGBM_PARAMS

        with Timer("LightGBM训练"):
            if config.TUNE_HYPERPARAMETERS and param_grid:
                lgb_model = lgb.LGBMClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )

                if use_randomized:
                    search = RandomizedSearchCV(
                        lgb_model, param_grid,
                        n_iter=config.RANDOM_SEARCH_ITERATIONS,
                        cv=config.RANDOM_SEARCH_CV,
                        scoring=config.PRIMARY_METRIC,
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                else:
                    # 简化参数网格
                    simplified_grid = {
                        'n_estimators': param_grid['n_estimators'][:2],
                        'max_depth': param_grid['max_depth'][:2],
                        'learning_rate': param_grid['learning_rate'][:2],
                        'num_leaves': param_grid['num_leaves'][:2],
                        'subsample': param_grid['subsample'][:2]
                    }
                    search = GridSearchCV(
                        lgb_model, simplified_grid,
                        cv=config.CV_FOLDS,
                        scoring=config.PRIMARY_METRIC,
                        n_jobs=-1
                    )

                search.fit(self.X_train_resampled, self.y_train_resampled)
                model = search.best_estimator_
                self.logger.info(f"最佳参数: {search.best_params_}")
            else:
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
                model.fit(self.X_train_resampled, self.y_train_resampled)

            # 预测和评估
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)

            # 特征重要性
            feature_importance = model.feature_importances_

            self.models['LightGBM'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'feature_importance': feature_importance
            }

            self._log_metrics("LightGBM", metrics)

        return self.models['LightGBM']

    def train_all_models(self,
                        include_xgboost: bool = True,
                        include_lightgbm: bool = True,
                        tune_hyperparameters: bool = True) -> Dict[str, Dict]:
        """
        训练所有配置的模型

        Args:
            include_xgboost: 是否包含XGBoost模型
            include_lightgbm: 是否包含LightGBM模型
            tune_hyperparameters: 是否进行超参数调优

        Returns:
            所有模型结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info("开始训练所有模型")
        self.logger.info("=" * 60)

        # 训练配置中指定的模型
        if 'logistic_regression' in config.MODELS_TO_TRAIN:
            self.train_logistic_regression(use_grid_search=tune_hyperparameters)

        if 'decision_tree' in config.MODELS_TO_TRAIN:
            self.train_decision_tree(use_grid_search=tune_hyperparameters)

        if 'random_forest' in config.MODELS_TO_TRAIN:
            self.train_random_forest(use_randomized=tune_hyperparameters)

        if 'gradient_boosting' in config.MODELS_TO_TRAIN:
            self.train_gradient_boosting(use_randomized=tune_hyperparameters)

        if 'xgboost' in config.MODELS_TO_TRAIN and XGBOOST_AVAILABLE and include_xgboost:
            self.train_xgboost(use_randomized=tune_hyperparameters)

        if 'lightgbm' in config.MODELS_TO_TRAIN and LIGHTGBM_AVAILABLE and include_lightgbm:
            self.train_lightgbm(use_randomized=tune_hyperparameters)

        # 选择最佳模型
        self._select_best_model()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("所有模型训练完成！")
        self.logger.info(f"最佳模型: {self.best_model_name}")
        self.logger.info(f"最佳 {config.PRIMARY_METRIC.upper()}: {self.best_score:.4f}")
        self.logger.info("=" * 60)

        return self.models

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率

        Returns:
            指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }

        return metrics

    def _log_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """
        记录模型评估指标

        Args:
            model_name: 模型名称
            metrics: 指标字典
        """
        self.logger.info(f"\n{model_name} 评估结果:")
        self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        self.logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    def _select_best_model(self) -> None:
        """根据主要指标选择最佳模型"""
        best_score = -np.inf
        best_name = None

        for name, result in self.models.items():
            score = result['metrics'][config.PRIMARY_METRIC]
            if score > best_score:
                best_score = score
                best_name = name

        self.best_model = self.models[best_name]['model']
        self.best_model_name = best_name
        self.best_score = best_score

    def retrain_on_full_data(self, X: pd.DataFrame, y: pd.Series,
                            use_smote: bool = False) -> None:
        """
        使用全部数据重新训练最佳模型

        Args:
            X: 完整特征数据
            y: 完整目标数据
            use_smote: 是否使用SMOTE
        """
        if self.best_model is None or self.best_model_name is None:
            self.logger.error("没有最佳模型可供重训练，请先训练模型")
            return

        self.logger.info(f"\n使用全部数据重新训练最佳模型: {self.best_model_name}")
        self.logger.info(f"  训练样本数: {len(X):,}")

        X_train = X.copy()
        y_train = y.copy()

        # 如果需要SMOTE处理
        if use_smote and SMOTE_AVAILABLE:
            from imblearn.over_sampling import SMOTE
            self.logger.info("  应用SMOTE处理类别不平衡...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            self.logger.info(f"  SMOTE后样本数: {len(X_train):,}")

        # 重新训练最佳模型
        with Timer(f"重新训练{self.best_model_name}"):
            self.best_model.fit(X_train, y_train)

        self.logger.info(f"{self.best_model_name} 已使用全部数据重新训练完成")

    def save_best_model(self, model_path: Path = None,
                       metadata_path: Path = None,
                       feature_names: list = None) -> None:
        """
        保存最佳模型和元数据

        Args:
            model_path: 模型保存路径
            metadata_path: 元数据保存路径
        """
        if self.best_model is None:
            self.logger.warning("没有可保存的最佳模型")
            return

        if model_path is None:
            model_path = config.BEST_MODEL_PATH
        if metadata_path is None:
            metadata_path = config.METADATA_PATH

        # 保存模型
        save_model(self.best_model, model_path)

        # 保存元数据
        from src.utils import save_json
        metadata = {
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'primary_metric': config.PRIMARY_METRIC,
            'metrics': self.models[self.best_model_name]['metrics'],
            'train_size': len(self.X_train) if self.X_train is not None else 0,
            'test_size': len(self.X_test) if self.X_test is not None else 0,
            'use_smote': config.USE_SMOTE,
            'random_state': self.random_state
        }
        save_json(metadata, metadata_path)

        self.logger.info(f"\n最佳模型已保存:")
        self.logger.info(f"  模型: {model_path}")
        self.logger.info(f"  元数据: {metadata_path}")


def compare_models(models_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    对比多个模型的性能

    Args:
        models_dict: 模型结果字典

    Returns:
        对比结果DataFrame
    """
    comparison = []

    for name, result in models_dict.items():
        metrics = result['metrics']
        comparison.append({
            '模型': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'ROC_AUC': metrics['roc_auc']
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values('ROC_AUC', ascending=False).reset_index(drop=True)

    return df


if __name__ == '__main__':
    # 测试模型训练模块
    from src.utils import setup_logger
    from src.data_loader import load_data
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import FeatureEngineer

    logger = setup_logger("Churn_Prediction", config.LOG_FILE, "INFO")

    print("=" * 60)
    print("模型训练模块测试")
    print("=" * 60)

    # 1. 加载和准备数据
    print("\n1. 加载数据")
    df = load_data()

    print("\n2. 数据预处理")
    df_clean = preprocess_data(df)

    print("\n3. 特征工程")
    engineer = FeatureEngineer()
    df_feat = engineer.engineer_features(df_clean)

    # 4. 分离特征和目标
    print("\n4. 分离特征和目标")
    X = df_feat.drop(columns=[config.TARGET_COL, config.ID_COL], errors='ignore')
    y = df_feat[config.TARGET_COL]

    print(f"特征形状: {X.shape}")
    print(f"目标形状: {y.shape}")

    # 5. 训练模型
    print("\n5. 训练模型")
    trainer = ModelTrainer()
    trainer.prepare_data(X, y, use_smote=True)

    # 训练所有模型
    results = trainer.train_all_models()

    # 6. 模型对比
    print("\n6. 模型对比")
    comparison_df = compare_models(results)
    print("\n" + str(comparison_df))

    # 7. 保存最佳模型
    print("\n7. 保存最佳模型")
    trainer.save_best_model()

    print("\n模型训练模块测试完成！")
