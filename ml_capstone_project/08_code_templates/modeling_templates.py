"""
模型训练代码模板库
=================

快速使用:
    from code_templates.modeling_templates import (
        quick_train,
        quick_baseline_comparison,
        quick_tune,
        get_default_param_space
    )

    # 5行代码完成建模
    model, metrics = quick_train(X, y, algorithm='xgboost')

    # 对比多个算法
    results = quick_baseline_comparison(X, y, algorithms=['rf', 'xgb', 'lgb'])

    # 超参数调优
    best_model = quick_tune(X, y, algorithm='xgboost', method='grid')

对应决策模板: 07_decision_templates/algorithm_selection_template.md
               07_decision_templates/hyperparameter_tuning_template.md
参考实现: 06_comprehensive_project/src/supervised_pipeline.py (475行)

项目定位: ML实战操作手册（非教学项目）
核心价值: 5-15分钟快速代码落地
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 尝试导入XGBoost和LightGBM（如果已安装）
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ==================== 1. 快速训练 ====================

def quick_train(X: pd.DataFrame,
               y: pd.Series,
               algorithm: str = 'auto',
               problem_type: str = 'auto',
               test_size: float = 0.2,
               random_state: int = 42,
               verbose: bool = True) -> Tuple[Any, Dict[str, float]]:
    """
    快速训练单个模型（5分钟）

    对应决策: algorithm_selection_template.md - 推荐算法

    Parameters
    ----------
    X : DataFrame
        特征数据
    y : Series
        目标变量
    algorithm : str, default='auto'
        算法选择
        'auto' - 自动选择（推荐）
        'rf' - 随机森林
        'xgboost' - XGBoost
        'lightgbm' - LightGBM
        'logistic' - 逻辑回归
        'svm' - 支持向量机
        'knn' - K近邻
    problem_type : {'auto', 'classification', 'regression'}
        问题类型，auto自动识别
    test_size : float, default=0.2
        测试集比例
    random_state : int, default=42
        随机种子
    verbose : bool, default=True
        是否打印信息

    Returns
    -------
    model, metrics : tuple
        训练好的模型和评估指标

    Examples
    --------
    >>> # 快速模式：自动选择算法
    >>> model, metrics = quick_train(X, y, algorithm='auto')
    >>> # ✓ 模型训练完成: XGBoost, AUC=0.85, F1=0.78

    >>> # 指定算法
    >>> model, metrics = quick_train(X, y, algorithm='xgboost')

    Decision Logic
    --------------
    auto模式选择策略:
    - 分类问题 → XGBoost/RandomForest
    - 回归问题 → XGBoost/RandomForest
    - 样本<1000 → KNN/RandomForest
    - 样本>100K → LightGBM

    Notes
    -----
    - 适合快速Baseline建立
    - 参考06章src/supervised_pipeline.py:85-156
    """
    # 自动识别问题类型
    if problem_type == 'auto':
        if y.nunique() <= 10:
            problem_type = 'classification'
        else:
            problem_type = 'regression'

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if verbose:
        print("🚀 快速训练模型...")
        print(f"   问题类型: {problem_type}")
        print(f"   训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 自动选择算法
    if algorithm == 'auto':
        if len(X_train) < 1000:
            algorithm = 'rf'
        elif XGBOOST_AVAILABLE:
            algorithm = 'xgboost'
        else:
            algorithm = 'rf'

    # 创建模型
    if problem_type == 'classification':
        if algorithm == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
            model = XGBClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, verbosity=0)
        elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model = LGBMClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, verbosity=-1)
        elif algorithm == 'logistic':
            model = LogisticRegression(random_state=random_state, max_iter=1000)
        elif algorithm == 'knn':
            model = KNeighborsClassifier()
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

    else:  # regression
        if algorithm == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
            model = XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, verbosity=0)
        elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model = LGBMRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, verbosity=-1)
        elif algorithm == 'ridge':
            model = Ridge(random_state=random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)

    # 训练模型
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)

    if problem_type == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                metrics['auc'] = roc_auc_score(y_test, y_proba[:, 1])
    else:
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

    if verbose:
        print(f"✓ 模型训练完成: {algorithm.upper()}")
        print("   测试集指标:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name.upper()}: {value:.4f}")
        print()

    return model, metrics


def quick_baseline_comparison(X: pd.DataFrame,
                              y: pd.Series,
                              algorithms: List[str] = ['rf', 'xgboost', 'lightgbm'],
                              problem_type: str = 'auto',
                              test_size: float = 0.2,
                              random_state: int = 42,
                              verbose: bool = True) -> Dict[str, Dict]:
    """
    快速对比多个算法（10-15分钟）

    对应决策: algorithm_selection_template.md - 推荐算法Top 3

    Parameters
    ----------
    algorithms : list, default=['rf', 'xgboost', 'lightgbm']
        算法列表
    problem_type : {'auto', 'classification', 'regression'}
        问题类型
    test_size : float, default=0.2
        测试集比例
    random_state : int, default=42
        随机种子
    verbose : bool, default=True
        是否打印对比结果

    Returns
    -------
    Dict[str, Dict]
        每个算法的模型和指标
        {
            'rf': {'model': model, 'metrics': {...}},
            'xgboost': {'model': model, 'metrics': {...}},
            ...
        }

    Examples
    --------
    >>> # 对比3个算法
    >>> results = quick_baseline_comparison(
    ...     X, y,
    ...     algorithms=['rf', 'xgboost', 'logistic']
    ... )
    >>> # ✓ Baseline对比完成
    >>> #   RandomForest: AUC=0.85, F1=0.78
    >>> #   XGBoost:      AUC=0.87, F1=0.80  ← 最佳
    >>> #   Logistic:     AUC=0.75, F1=0.72

    >>> # 获取最佳模型
    >>> best_algo = max(results.items(), key=lambda x: x[1]['metrics'].get('auc', 0))
    >>> best_model = best_algo[1]['model']

    Notes
    -----
    - 适合快速筛选最佳算法
    - 参考06章src/supervised_pipeline.py:160-245
    """
    if verbose:
        print("="*60)
        print("   Baseline算法对比")
        print("="*60 + "\n")

    results = {}

    for algo in algorithms:
        try:
            model, metrics = quick_train(
                X, y,
                algorithm=algo,
                problem_type=problem_type,
                test_size=test_size,
                random_state=random_state,
                verbose=False
            )
            results[algo] = {
                'model': model,
                'metrics': metrics
            }
        except Exception as e:
            if verbose:
                print(f"⚠️  {algo}训练失败: {str(e)}")

    if verbose:
        print("="*60)
        print("   对比结果")
        print("="*60)

        # 自动识别问题类型
        if problem_type == 'auto':
            if y.nunique() <= 10:
                problem_type = 'classification'
            else:
                problem_type = 'regression'

        # 打印对比表
        if problem_type == 'classification':
            print(f"{'算法':<15} {'Accuracy':<12} {'F1':<12} {'AUC':<12}")
            print("-"*60)
            for algo, result in results.items():
                metrics = result['metrics']
                print(f"{algo:<15} {metrics.get('accuracy', 0):<12.4f} "
                      f"{metrics.get('f1', 0):<12.4f} {metrics.get('auc', 0):<12.4f}")
        else:
            print(f"{'算法':<15} {'RMSE':<12} {'R2':<12}")
            print("-"*60)
            for algo, result in results.items():
                metrics = result['metrics']
                print(f"{algo:<15} {metrics.get('rmse', 0):<12.4f} "
                      f"{metrics.get('r2', 0):<12.4f}")

        print("="*60 + "\n")

    return results


# ==================== 2. 超参数调优 ====================

def get_default_param_space(algorithm: str) -> Dict[str, List]:
    """
    获取默认参数空间（基于经验和03章算法对比表）

    对应决策: hyperparameter_tuning_template.md - 参数空间速查表

    Parameters
    ----------
    algorithm : str
        算法名称

    Returns
    -------
    Dict[str, List]
        参数空间

    Examples
    --------
    >>> param_space = get_default_param_space('xgboost')
    >>> # {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3], ...}
    """
    param_spaces = {
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        },
        'lightgbm': {
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'num_leaves': [15, 31, 63],
            'subsample': [0.7, 0.8, 0.9, 1.0]
        },
        'logistic': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    }

    return param_spaces.get(algorithm, {})


def quick_tune(X: pd.DataFrame,
              y: pd.Series,
              algorithm: str = 'xgboost',
              method: str = 'grid',
              param_space: Dict = None,
              cv: int = 5,
              n_iter: int = 20,
              random_state: int = 42,
              verbose: bool = True) -> Any:
    """
    快速超参数调优（15-30分钟）

    对应决策: hyperparameter_tuning_template.md - 调优策略选择

    Parameters
    ----------
    algorithm : str
        算法名称
    method : {'grid', 'random'}
        调优方法
        grid   - 网格搜索（参数空间小时）
        random - 随机搜索（参数空间大时）
    param_space : dict, optional
        参数空间，None则使用默认
    cv : int, default=5
        交叉验证折数
    n_iter : int, default=20
        随机搜索迭代次数
    random_state : int, default=42
        随机种子
    verbose : bool, default=True
        是否打印信息

    Returns
    -------
    model
        调优后的最佳模型

    Examples
    --------
    >>> # 使用默认参数空间
    >>> best_model = quick_tune(X, y, algorithm='xgboost', method='grid')
    >>> # ✓ 调优完成: 最佳参数 {'max_depth': 5, 'learning_rate': 0.1, ...}

    >>> # 自定义参数空间
    >>> param_space = {'max_depth': [3, 5, 7], 'n_estimators': [100, 200]}
    >>> best_model = quick_tune(X, y, param_space=param_space)

    Decision Logic
    --------------
    参数数量 < 20  → Grid Search
    参数数量 >= 20 → Random Search（n_iter=20-50）

    Notes
    -----
    - Grid Search适合小参数空间（<20组合）
    - Random Search适合大参数空间
    - 参考06章src/supervised_pipeline.py:249-315
    """
    if verbose:
        print("🔧 超参数调优...")
        print(f"   算法: {algorithm.upper()}, 方法: {method}")

    # 获取参数空间
    if param_space is None:
        param_space = get_default_param_space(algorithm)

    if not param_space:
        if verbose:
            print(f"⚠️  未找到{algorithm}的默认参数空间")
        return None

    # 自动识别问题类型
    if y.nunique() <= 10:
        problem_type = 'classification'
        scoring = 'roc_auc'
    else:
        problem_type = 'regression'
        scoring = 'r2'

    # 创建基础模型
    if problem_type == 'classification':
        if algorithm == 'rf':
            base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
            base_model = XGBClassifier(random_state=random_state, n_jobs=-1, verbosity=0)
        elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
            base_model = LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1)
        elif algorithm == 'logistic':
            base_model = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            if verbose:
                print(f"⚠️  不支持的算法: {algorithm}")
            return None
    else:
        if algorithm == 'rf':
            base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
            base_model = XGBRegressor(random_state=random_state, n_jobs=-1, verbosity=0)
        elif algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
            base_model = LGBMRegressor(random_state=random_state, n_jobs=-1, verbosity=-1)
        else:
            if verbose:
                print(f"⚠️  不支持的算法: {algorithm}")
            return None

    # 执行调优
    if method == 'grid':
        search = GridSearchCV(
            base_model,
            param_space,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
    else:  # random
        search = RandomizedSearchCV(
            base_model,
            param_space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )

    search.fit(X, y)

    if verbose:
        print(f"✓ 调优完成")
        print(f"   最佳得分: {search.best_score_:.4f}")
        print(f"   最佳参数: {search.best_params_}")
        print()

    return search.best_estimator_


# ==================== 3. 无监督学习模板 ====================

def quick_kmeans(X: pd.DataFrame,
                n_clusters: int = 3,
                random_state: int = 42,
                verbose: bool = True) -> Tuple[KMeans, np.ndarray]:
    """
    快速K-Means聚类

    Parameters
    ----------
    X : DataFrame
        特征数据
    n_clusters : int, default=3
        聚类数量
    random_state : int, default=42
        随机种子

    Returns
    -------
    model, labels : tuple
        聚类模型和标签

    Examples
    --------
    >>> model, labels = quick_kmeans(X, n_clusters=3)
    >>> # ✓ K-Means聚类完成: 3类
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)

    if verbose:
        print(f"✓ K-Means聚类完成: {n_clusters}类")
        for i in range(n_clusters):
            count = (labels == i).sum()
            print(f"   类{i}: {count}个样本")
        print()

    return model, labels


def quick_pca(X: pd.DataFrame,
             n_components: int = 2,
             verbose: bool = True) -> Tuple[PCA, np.ndarray]:
    """
    快速PCA降维

    Parameters
    ----------
    X : DataFrame
        特征数据
    n_components : int, default=2
        降维后的维度

    Returns
    -------
    model, X_reduced : tuple
        PCA模型和降维后的数据

    Examples
    --------
    >>> model, X_reduced = quick_pca(X, n_components=2)
    >>> # ✓ PCA降维完成: 100维 → 2维, 解释方差: 85.3%
    """
    model = PCA(n_components=n_components)
    X_reduced = model.fit_transform(X)

    if verbose:
        variance_ratio = model.explained_variance_ratio_.sum() * 100
        print(f"✓ PCA降维完成: {X.shape[1]}维 → {n_components}维")
        print(f"   解释方差: {variance_ratio:.1f}%")
        print()

    return model, X_reduced
