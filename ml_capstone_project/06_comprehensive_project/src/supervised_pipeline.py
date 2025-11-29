"""
监督学习Pipeline模块
===================

提供完整的监督学习工作流，从数据准备到模型部署。

主要功能:
- 自动化数据预处理
- 特征工程Pipeline
- 多模型训练与评估
- 超参数优化
- 模型集成
- Pipeline持久化

支持的算法:
- 分类: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM
- 回归: Linear Regression, Ridge, Lasso, Random Forest, XGBoost

使用方式:
    pipeline = SupervisedPipeline(problem_type='classification')
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# ==================== 监督学习Pipeline类 ====================

class SupervisedPipeline:
    """
    监督学习完整Pipeline

    自动处理数据预处理、特征工程、模型训练、评估全流程
    """

    def __init__(self, problem_type: str = 'classification',
                metric: Optional[str] = None,
                random_state: int = 42):
        """
        Args:
            problem_type: 问题类型 ('classification', 'regression')
            metric: 评估指标（None则使用默认）
            random_state: 随机种子
        """
        self.problem_type = problem_type
        self.random_state = random_state
        self.metric = metric or ('accuracy' if problem_type == 'classification' else 'neg_mean_squared_error')

        self.preprocessor = None
        self.feature_names = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None

    def _get_default_models(self) -> Dict[str, Any]:
        """
        获取默认模型集合

        Returns:
            模型字典
        """
        if self.problem_type == 'classification':
            return {
                'LogisticRegression': LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state
                ),
                'RandomForest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'XGBoost': XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    eval_metric='logloss'
                ),
                'LightGBM': LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
            }
        else:  # regression
            return {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(
                    alpha=1.0,
                    random_state=self.random_state
                ),
                'RandomForest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'XGBoost': XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'LightGBM': LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
            }

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        构建数据预处理器

        Args:
            X: 特征数据

        Returns:
            预处理Pipeline
        """
        # 识别数值型和类别型特征
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"数值型特征: {len(numeric_features)} 个")
        print(f"类别型特征: {len(categorical_features)} 个")

        # 数值型特征Pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # 类别型特征Pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # 组合
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return preprocessor

    def fit(self, X: pd.DataFrame, y: pd.Series,
           models: Optional[Dict[str, Any]] = None,
           cv: int = 5) -> 'SupervisedPipeline':
        """
        训练多个模型

        Args:
            X: 训练特征
            y: 训练标签
            models: 自定义模型字典（None则使用默认模型）
            cv: 交叉验证折数

        Returns:
            self
        """
        print("\n" + "=" * 60)
        print("开始训练监督学习模型")
        print("=" * 60)

        # 构建预处理器
        self.preprocessor = self._build_preprocessor(X)

        # 获取模型列表
        if models is None:
            models = self._get_default_models()

        results = {}

        # 训练每个模型
        for model_name, model in models.items():
            print(f"\n训练 {model_name}...")

            # 创建完整Pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])

            # 交叉验证
            try:
                scores = cross_val_score(pipeline, X, y, cv=cv,
                                       scoring=self.metric, n_jobs=-1)

                results[model_name] = {
                    'model': pipeline,
                    'scores': scores,
                    'mean_score': scores.mean(),
                    'std_score': scores.std()
                }

                print(f"  交叉验证得分: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

            except Exception as e:
                print(f"  ⚠️  训练失败: {e}")
                continue

        # 保存结果
        self.models = results

        # 选择最佳模型
        best_model_name = max(results.keys(),
                             key=lambda k: results[k]['mean_score'])
        self.best_model_name = best_model_name
        self.best_model = results[best_model_name]['model']

        # 在全部数据上重新训练最佳模型
        print(f"\n✅ 最佳模型: {best_model_name}")
        print(f"   得分: {results[best_model_name]['mean_score']:.4f}")
        print(f"\n在全部训练数据上重新训练最佳模型...")
        self.best_model.fit(X, y)

        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60 + "\n")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用最佳模型进行预测

        Args:
            X: 测试特征

        Returns:
            预测结果
        """
        if self.best_model is None:
            raise ValueError("模型尚未训练，请先调用fit()方法")

        return self.best_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率（仅分类问题）

        Args:
            X: 测试特征

        Returns:
            预测概率
        """
        if self.problem_type != 'classification':
            raise ValueError("只有分类问题支持predict_proba")

        if self.best_model is None:
            raise ValueError("模型尚未训练，请先调用fit()方法")

        return self.best_model.predict_proba(X)

    def get_model_comparison(self) -> pd.DataFrame:
        """
        获取所有模型的对比结果

        Returns:
            对比结果DataFrame
        """
        if not self.models:
            raise ValueError("尚未训练任何模型")

        comparison = []
        for model_name, result in self.models.items():
            comparison.append({
                '模型': model_name,
                '平均得分': result['mean_score'],
                '标准差': result['std_score'],
                '最佳': '✓' if model_name == self.best_model_name else ''
            })

        df = pd.DataFrame(comparison).sort_values('平均得分', ascending=False)
        return df

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                           param_grid: Dict[str, Any],
                           cv: int = 5,
                           n_iter: Optional[int] = None) -> Dict[str, Any]:
        """
        超参数优化

        Args:
            X: 训练特征
            y: 训练标签
            param_grid: 参数网格
            cv: 交叉验证折数
            n_iter: 随机搜索迭代次数（None则使用网格搜索）

        Returns:
            最佳参数和得分
        """
        if self.best_model is None:
            raise ValueError("请先训练基础模型")

        print(f"\n开始超参数优化（{'随机搜索' if n_iter else '网格搜索'}）...")

        if n_iter:
            # 随机搜索
            search = RandomizedSearchCV(
                self.best_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=self.metric,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        else:
            # 网格搜索
            search = GridSearchCV(
                self.best_model,
                param_grid,
                cv=cv,
                scoring=self.metric,
                n_jobs=-1,
                verbose=1
            )

        search.fit(X, y)

        # 更新最佳模型
        self.best_model = search.best_estimator_

        print(f"\n✅ 优化完成！")
        print(f"最佳参数: {search.best_params_}")
        print(f"最佳得分: {search.best_score_:.4f}")

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }


# ==================== 快速训练函数 ====================

def quick_train_classification(X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: Optional[pd.DataFrame] = None,
                              y_test: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    快速训练分类模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征（可选）
        y_test: 测试标签（可选）

    Returns:
        训练结果字典
    """
    from . import model_evaluation

    # 创建Pipeline
    pipeline = SupervisedPipeline(problem_type='classification')

    # 训练
    pipeline.fit(X_train, y_train)

    # 打印对比结果
    print("\n模型对比:")
    print(pipeline.get_model_comparison())

    # 测试集评估（如果提供）
    results = {'pipeline': pipeline}

    if X_test is not None and y_test is not None:
        print("\n" + "=" * 60)
        print("测试集评估")
        print("=" * 60)

        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None

        metrics = model_evaluation.evaluate_classification(
            y_test, y_pred, y_pred_proba, verbose=True
        )

        results['test_metrics'] = metrics

        # 绘制混淆矩阵
        model_evaluation.plot_confusion_matrix(y_test, y_pred)

        # 绘制ROC曲线（二分类）
        if y_pred_proba is not None:
            model_evaluation.plot_roc_curve(y_test, y_pred_proba)

    return results


def quick_train_regression(X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: Optional[pd.DataFrame] = None,
                          y_test: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    快速训练回归模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征（可选）
        y_test: 测试标签（可选）

    Returns:
        训练结果字典
    """
    from . import model_evaluation

    # 创建Pipeline
    pipeline = SupervisedPipeline(problem_type='regression')

    # 训练
    pipeline.fit(X_train, y_train)

    # 打印对比结果
    print("\n模型对比:")
    print(pipeline.get_model_comparison())

    # 测试集评估（如果提供）
    results = {'pipeline': pipeline}

    if X_test is not None and y_test is not None:
        print("\n" + "=" * 60)
        print("测试集评估")
        print("=" * 60)

        y_pred = pipeline.predict(X_test)

        metrics = model_evaluation.evaluate_regression(
            y_test, y_pred, verbose=True
        )

        results['test_metrics'] = metrics

        # 绘制回归结果
        model_evaluation.plot_regression_results(y_test, y_pred)

    return results


# ==================== 超参数搜索空间 ====================

PARAM_GRIDS = {
    'RandomForest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, 15, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.3],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0]
    },
    'LightGBM': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7, -1],
        'classifier__learning_rate': [0.01, 0.1, 0.3],
        'classifier__num_leaves': [15, 31, 63]
    },
    'LogisticRegression': {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga']
    }
}


if __name__ == '__main__':
    # 测试示例
    print("=== 监督学习Pipeline测试 ===\n")

    # 创建测试数据
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, n_redundant=5,
                              random_state=42)

    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y, name='target')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 快速训练
    results = quick_train_classification(X_train, y_train, X_test, y_test)

    print("\n✅ 测试完成！")
