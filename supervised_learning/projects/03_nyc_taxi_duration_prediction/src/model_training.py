"""
模型训练模块
包含各种回归模型的训练、调优和集成
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost未安装，将跳过XGBoost模型")

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, save_model, setup_logger


class ModelTrainer:
    """
    模型训练器类
    管理多个模型的训练、评估和优化
    """

    def __init__(self, random_state: int = None):
        """
        初始化模型训练器

        Args:
            random_state: 随机种子
        """
        self.random_state = random_state or config.RANDOM_STATE
        self.logger = logging.getLogger("NYC_Taxi")

        # 初始化缩放器
        self.scaler = StandardScaler()

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
        self.X_train_scaled = None
        self.X_test_scaled = None

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
        """
        self.logger.info("=" * 60)
        self.logger.info("准备训练数据")
        self.logger.info("=" * 60)

        if test_size is None:
            test_size = config.TEST_SIZE

        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        self.logger.info(f"训练集大小: {self.X_train.shape}")
        self.logger.info(f"测试集大小: {self.X_test.shape}")

        # 特征缩放
        if scale:
            self.logger.info("执行特征缩放...")
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            self.logger.info("特征缩放完成")
        else:
            self.X_train_scaled = self.X_train.values
            self.X_test_scaled = self.X_test.values

    def train_linear_regression(self) -> Dict[str, Any]:
        """
        训练线性回归模型（Baseline）

        Returns:
            模型和评估结果字典
        """
        self.logger.info("\n训练线性回归模型...")

        with Timer("线性回归训练"):
            model = LinearRegression()
            model.fit(self.X_train_scaled, self.y_train)

            # 预测
            y_pred = model.predict(self.X_test_scaled)

            # 评估
            metrics = self._calculate_metrics(self.y_test, y_pred)

            # 存储模型
            self.models['LinearRegression'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }

            self._log_metrics("LinearRegression", metrics)

        return self.models['LinearRegression']

    def train_ridge(self, alpha: Optional[List[float]] = None,
                   use_cv: bool = True) -> Dict[str, Any]:
        """
        训练Ridge回归模型（L2正则化）

        Args:
            alpha: 正则化强度参数列表
            use_cv: 是否使用交叉验证选择最佳alpha

        Returns:
            模型和评估结果字典
        """
        self.logger.info("\n训练Ridge回归模型...")

        if alpha is None:
            alpha = config.RIDGE_PARAMS['alpha']

        with Timer("Ridge回归训练"):
            if use_cv and len(alpha) > 1:
                # 使用GridSearch寻找最佳alpha
                ridge = Ridge(random_state=self.random_state)
                grid_search = GridSearchCV(
                    ridge,
                    {'alpha': alpha},
                    cv=config.CV_FOLDS,
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(self.X_train_scaled, self.y_train)
                model = grid_search.best_estimator_
                self.logger.info(f"最佳alpha: {grid_search.best_params_['alpha']}")
            else:
                model = Ridge(alpha=alpha[0] if isinstance(alpha, list) else alpha,
                            random_state=self.random_state)
                model.fit(self.X_train_scaled, self.y_train)

            # 预测和评估
            y_pred = model.predict(self.X_test_scaled)
            metrics = self._calculate_metrics(self.y_test, y_pred)

            self.models['Ridge'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }

            self._log_metrics("Ridge", metrics)

        return self.models['Ridge']

    def train_lasso(self, alpha: Optional[List[float]] = None,
                   use_cv: bool = True) -> Dict[str, Any]:
        """
        训练Lasso回归模型（L1正则化）

        Args:
            alpha: 正则化强度参数列表
            use_cv: 是否使用交叉验证选择最佳alpha

        Returns:
            模型和评估结果字典
        """
        self.logger.info("\n训练Lasso回归模型...")

        if alpha is None:
            alpha = config.LASSO_PARAMS['alpha']

        with Timer("Lasso回归训练"):
            if use_cv and len(alpha) > 1:
                lasso = Lasso(random_state=self.random_state, max_iter=10000)
                grid_search = GridSearchCV(
                    lasso,
                    {'alpha': alpha},
                    cv=config.CV_FOLDS,
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(self.X_train_scaled, self.y_train)
                model = grid_search.best_estimator_
                self.logger.info(f"最佳alpha: {grid_search.best_params_['alpha']}")
            else:
                model = Lasso(alpha=alpha[0] if isinstance(alpha, list) else alpha,
                            random_state=self.random_state, max_iter=10000)
                model.fit(self.X_train_scaled, self.y_train)

            # 预测和评估
            y_pred = model.predict(self.X_test_scaled)
            metrics = self._calculate_metrics(self.y_test, y_pred)

            self.models['Lasso'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }

            self._log_metrics("Lasso", metrics)

        return self.models['Lasso']

    def train_elasticnet(self, param_grid: Optional[Dict] = None,
                        use_cv: bool = True) -> Dict[str, Any]:
        """
        训练ElasticNet模型（L1+L2正则化）

        Args:
            param_grid: 参数网格
            use_cv: 是否使用交叉验证

        Returns:
            模型和评估结果字典
        """
        self.logger.info("\n训练ElasticNet模型...")

        if param_grid is None:
            param_grid = config.ELASTICNET_PARAMS

        with Timer("ElasticNet训练"):
            if use_cv and param_grid:
                elasticnet = ElasticNet(random_state=self.random_state, max_iter=10000)
                grid_search = GridSearchCV(
                    elasticnet,
                    param_grid,
                    cv=config.CV_FOLDS,
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(self.X_train_scaled, self.y_train)
                model = grid_search.best_estimator_
                self.logger.info(f"最佳参数: {grid_search.best_params_}")
            else:
                model = ElasticNet(random_state=self.random_state, max_iter=10000)
                model.fit(self.X_train_scaled, self.y_train)

            # 预测和评估
            y_pred = model.predict(self.X_test_scaled)
            metrics = self._calculate_metrics(self.y_test, y_pred)

            self.models['ElasticNet'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }

            self._log_metrics("ElasticNet", metrics)

        return self.models['ElasticNet']

    def train_random_forest(self, param_grid: Optional[Dict] = None,
                          use_randomized: bool = False,
                          n_iter: int = 20) -> Dict[str, Any]:
        """
        训练随机森林模型

        Args:
            param_grid: 参数网格
            use_randomized: 是否使用随机搜索
            n_iter: 随机搜索迭代次数

        Returns:
            模型和评估结果字典
        """
        self.logger.info("\n训练随机森林模型...")

        if param_grid is None:
            param_grid = config.RANDOM_FOREST_PARAMS

        with Timer("随机森林训练"):
            if param_grid:
                rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)

                if use_randomized:
                    search = RandomizedSearchCV(
                        rf, param_grid, n_iter=n_iter,
                        cv=config.CV_FOLDS, scoring='r2',
                        random_state=self.random_state, n_jobs=-1
                    )
                else:
                    # 简化参数网格避免过长时间
                    simplified_grid = {
                        'n_estimators': param_grid['n_estimators'][:2],
                        'max_depth': param_grid['max_depth'][:2],
                        'min_samples_split': param_grid['min_samples_split'][:1],
                        'min_samples_leaf': param_grid['min_samples_leaf'][:1]
                    }
                    search = GridSearchCV(
                        rf, simplified_grid,
                        cv=config.CV_FOLDS, scoring='r2', n_jobs=-1
                    )

                search.fit(self.X_train_scaled, self.y_train)
                model = search.best_estimator_
                self.logger.info(f"最佳参数: {search.best_params_}")
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(self.X_train_scaled, self.y_train)

            # 预测和评估
            y_pred = model.predict(self.X_test_scaled)
            metrics = self._calculate_metrics(self.y_test, y_pred)

            # 获取特征重要性
            feature_importance = model.feature_importances_

            self.models['RandomForest'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'feature_importance': feature_importance
            }

            self._log_metrics("RandomForest", metrics)

        return self.models['RandomForest']

    def train_xgboost(self, param_grid: Optional[Dict] = None,
                     use_randomized: bool = True,
                     n_iter: int = 10) -> Optional[Dict[str, Any]]:
        """
        训练XGBoost模型

        Args:
            param_grid: 参数网格
            use_randomized: 是否使用随机搜索
            n_iter: 随机搜索迭代次数

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
            if param_grid and use_randomized:
                xgb_model = xgb.XGBRegressor(
                    random_state=self.random_state,
                    n_jobs=-1,
                    tree_method='auto'
                )

                # 使用随机搜索
                search = RandomizedSearchCV(
                    xgb_model, param_grid, n_iter=n_iter,
                    cv=config.CV_FOLDS, scoring='r2',
                    random_state=self.random_state, n_jobs=-1
                )
                search.fit(self.X_train_scaled, self.y_train)
                model = search.best_estimator_
                self.logger.info(f"最佳参数: {search.best_params_}")
            else:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(self.X_train_scaled, self.y_train)

            # 预测和评估
            y_pred = model.predict(self.X_test_scaled)
            metrics = self._calculate_metrics(self.y_test, y_pred)

            # 获取特征重要性
            feature_importance = model.feature_importances_

            self.models['XGBoost'] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'feature_importance': feature_importance
            }

            self._log_metrics("XGBoost", metrics)

        return self.models['XGBoost']

    def train_stacking(self, base_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        训练Stacking集成模型

        Args:
            base_models: 基学习器名称列表

        Returns:
            模型和评估结果字典
        """
        self.logger.info("\n训练Stacking集成模型...")

        # 默认使用已训练的所有模型作为基学习器
        if base_models is None:
            base_models = ['Ridge', 'RandomForest']
            if XGBOOST_AVAILABLE and 'XGBoost' in self.models:
                base_models.append('XGBoost')

        with Timer("Stacking模型训练"):
            # 构建基学习器列表
            estimators = []
            for name in base_models:
                if name in self.models:
                    estimators.append((name.lower(), self.models[name]['model']))

            if len(estimators) < 2:
                self.logger.warning("基学习器数量不足，无法训练Stacking模型")
                return None

            # 创建Stacking模型，使用Ridge作为最终估计器
            stacking_model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=config.CV_FOLDS,
                n_jobs=-1
            )

            # 训练
            stacking_model.fit(self.X_train_scaled, self.y_train)

            # 预测和评估
            y_pred = stacking_model.predict(self.X_test_scaled)
            metrics = self._calculate_metrics(self.y_test, y_pred)

            self.models['Stacking'] = {
                'model': stacking_model,
                'metrics': metrics,
                'predictions': y_pred,
                'base_models': base_models
            }

            self._log_metrics("Stacking", metrics)

        return self.models['Stacking']

    def train_all_models(self, include_xgboost: bool = True,
                        include_stacking: bool = True) -> Dict[str, Dict]:
        """
        训练所有模型

        Args:
            include_xgboost: 是否包含XGBoost
            include_stacking: 是否包含Stacking

        Returns:
            所有模型结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info("开始训练所有模型")
        self.logger.info("=" * 60)

        # 线性模型
        self.train_linear_regression()
        self.train_ridge()
        self.train_lasso()
        self.train_elasticnet()

        # 树模型
        self.train_random_forest()

        # XGBoost
        if include_xgboost and XGBOOST_AVAILABLE:
            self.train_xgboost()

        # Stacking集成
        if include_stacking and len(self.models) >= 2:
            self.train_stacking()

        # 选择最佳模型
        self._select_best_model()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("所有模型训练完成！")
        self.logger.info(f"最佳模型: {self.best_model_name} (R² = {self.best_score:.4f})")
        self.logger.info("=" * 60)

        return self.models

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            指标字典
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

    def _log_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """
        记录模型评估指标

        Args:
            model_name: 模型名称
            metrics: 指标字典
        """
        self.logger.info(f"\n{model_name} 评估结果:")
        self.logger.info(f"  R² Score: {metrics['r2']:.4f}")
        self.logger.info(f"  RMSE: {metrics['rmse']:.2f} 秒")
        self.logger.info(f"  MAE: {metrics['mae']:.2f} 秒")
        self.logger.info(f"  MAPE: {metrics['mape']:.2f}%")

    def _select_best_model(self) -> None:
        """选择R²最高的模型作为最佳模型"""
        best_score = -np.inf
        best_name = None

        for name, result in self.models.items():
            score = result['metrics']['r2']
            if score > best_score:
                best_score = score
                best_name = name

        self.best_model = self.models[best_name]['model']
        self.best_model_name = best_name
        self.best_score = best_score

    def retrain_on_full_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        使用全部数据重新训练最佳模型

        这是Kaggle竞赛的最佳实践：
        1. 先用train_test_split验证和选择模型
        2. 确定最佳模型后，用全部训练数据重新训练
        3. 这样可以充分利用所有数据，提升最终性能

        Args:
            X: 全部特征数据
            y: 全部目标数据
        """
        if self.best_model_name is None:
            self.logger.error("请先训练模型并选择最佳模型")
            return

        self.logger.info("\n" + "=" * 60)
        self.logger.info("使用全部数据重新训练最佳模型")
        self.logger.info("=" * 60)
        self.logger.info(f"最佳模型: {self.best_model_name}")
        self.logger.info(f"数据量: {len(X)} 条")

        with Timer(f"{self.best_model_name}全量重训练"):
            # 缩放全部数据
            X_scaled = self.scaler.fit_transform(X)

            # 重新训练最佳模型
            if self.best_model_name == 'LinearRegression':
                model = LinearRegression()
            elif self.best_model_name == 'Ridge':
                model = self.best_model  # 保持调优后的参数
            elif self.best_model_name == 'Lasso':
                model = self.best_model
            elif self.best_model_name == 'ElasticNet':
                model = self.best_model
            elif self.best_model_name == 'RandomForest':
                model = self.best_model
            elif self.best_model_name == 'XGBoost':
                model = self.best_model
            elif self.best_model_name == 'Stacking':
                model = self.best_model
            else:
                model = self.best_model

            # 训练
            model.fit(X_scaled, y)

            # 更新最佳模型
            self.best_model = model

            self.logger.info(f"\n全量重训练完成！")
            self.logger.info(f"模型已使用全部 {len(X)} 条数据训练")
            self.logger.info(f"建议立即保存模型: trainer.save_best_model()")

    def save_best_model(self, model_path: Path = None,
                       scaler_path: Path = None,
                       metadata_path: Path = None) -> None:
        """
        保存最佳模型和相关文件

        Args:
            model_path: 模型保存路径
            scaler_path: 缩放器保存路径
            metadata_path: 元数据保存路径
        """
        if self.best_model is None:
            self.logger.warning("没有可保存的最佳模型")
            return

        if model_path is None:
            model_path = config.BEST_MODEL_PATH
        if scaler_path is None:
            scaler_path = config.SCALER_PATH
        if metadata_path is None:
            metadata_path = config.METADATA_PATH

        # 保存模型
        save_model(self.best_model, model_path)

        # 保存缩放器
        save_model(self.scaler, scaler_path)

        # 保存元数据
        from src.utils import save_json
        metadata = {
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'metrics': self.models[self.best_model_name]['metrics'],
            'feature_names': list(self.X_train.columns) if self.X_train is not None else [],
            'train_size': len(self.X_train) if self.X_train is not None else 0,
            'test_size': len(self.X_test) if self.X_test is not None else 0,
            'random_state': self.random_state
        }
        save_json(metadata, metadata_path)

        self.logger.info(f"\n最佳模型已保存:")
        self.logger.info(f"  模型: {model_path}")
        self.logger.info(f"  缩放器: {scaler_path}")
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
            'R²': metrics['r2'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'MAPE': metrics['mape']
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values('R²', ascending=False).reset_index(drop=True)

    return df


if __name__ == '__main__':
    # 测试模型训练
    from src.utils import setup_logger
    from src.data_loader import load_train_data
    from src.data_preprocessing import preprocess_data, split_features_target
    from src.feature_engineering import engineer_features, select_features

    logger = setup_logger("NYC_Taxi", config.LOG_FILE, "INFO")

    print("=" * 60)
    print("模型训练模块测试")
    print("=" * 60)

    # 加载和准备数据
    print("\n1. 加载数据")
    df = load_train_data(use_sample=True, sample_size=1000)

    print("\n2. 数据预处理")
    df_clean = preprocess_data(df)

    print("\n3. 特征工程")
    df_feat = engineer_features(df_clean)

    print("\n4. 分离特征和目标")
    X, y = split_features_target(df_feat)
    X_selected = select_features(df_feat)
    y = df_feat['trip_duration']

    # 训练模型
    print("\n5. 训练模型")
    trainer = ModelTrainer()
    trainer.prepare_data(X_selected, y)

    # 训练所有模型
    results = trainer.train_all_models(include_xgboost=True, include_stacking=True)

    # 对比模型
    print("\n6. 模型对比")
    comparison_df = compare_models(results)
    print("\n" + str(comparison_df))

    # 保存最佳模型
    print("\n7. 保存最佳模型")
    trainer.save_best_model()

    print("\n模型训练模块测试完成！")
