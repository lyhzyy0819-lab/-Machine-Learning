"""
无监督学习Pipeline模块
=====================

提供完整的无监督学习工作流，包括聚类、降维、异常检测。

主要功能:
- 聚类分析（K-Means、DBSCAN、层次聚类、GMM）
- 降维分析（PCA、t-SNE、UMAP）
- 异常检测（Isolation Forest、One-Class SVM、LOF）
- 自动确定最佳参数（如K值）
- 聚类结果评估与可视化

使用场景:
- 客户分群
- 数据探索
- 异常检测
- 特征提取

使用方式:
    # 聚类
    pipeline = ClusteringPipeline(method='kmeans')
    labels = pipeline.fit_predict(X)

    # 降维
    pipeline = DimensionReductionPipeline(method='pca', n_components=2)
    X_reduced = pipeline.fit_transform(X)

    # 异常检测
    pipeline = AnomalyDetectionPipeline(method='isolation_forest')
    anomalies = pipeline.fit_predict(X)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ==================== 聚类Pipeline ====================

class ClusteringPipeline:
    """
    聚类分析Pipeline

    支持多种聚类算法，自动寻找最佳参数
    """

    def __init__(self, method: str = 'kmeans',
                n_clusters: Optional[int] = None,
                random_state: int = 42):
        """
        Args:
            method: 聚类方法 ('kmeans', 'dbscan', 'hierarchical', 'gmm')
            n_clusters: 聚类数（None则自动确定）
            random_state: 随机种子
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.labels_ = None
        self.cluster_centers_ = None

    def _find_optimal_k(self, X: np.ndarray,
                       k_range: range = range(2, 11)) -> int:
        """
        使用肘部法则和轮廓系数确定最佳K值

        Args:
            X: 特征数据
            k_range: K值范围

        Returns:
            最佳K值
        """
        print("\n寻找最佳聚类数...")

        inertias = []
        silhouettes = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))

        # 绘制肘部图
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 左图：惯性（肘部法则）
        axes[0].plot(k_range, inertias, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('聚类数 K')
        axes[0].set_ylabel('Inertia (WCSS)')
        axes[0].set_title('肘部法则')
        axes[0].grid(alpha=0.3)

        # 右图：轮廓系数
        axes[1].plot(k_range, silhouettes, 'o-', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('聚类数 K')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('轮廓系数')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 选择轮廓系数最大的K
        best_k = k_range[np.argmax(silhouettes)]
        print(f"✅ 推荐聚类数: K = {best_k}")
        print(f"   轮廓系数: {max(silhouettes):.3f}")

        return best_k

    def fit(self, X: pd.DataFrame) -> 'ClusteringPipeline':
        """
        训练聚类模型

        Args:
            X: 特征数据

        Returns:
            self
        """
        print(f"\n开始聚类分析（方法: {self.method.upper()}）...")

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 自动确定K值（仅K-Means和GMM）
        if self.n_clusters is None and self.method in ['kmeans', 'gmm']:
            self.n_clusters = self._find_optimal_k(X_scaled)

        # 创建模型
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )

        elif self.method == 'dbscan':
            # DBSCAN不需要指定聚类数
            self.model = DBSCAN(
                eps=0.5,
                min_samples=5
            )

        elif self.method == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters or 3
            )

        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state
            )

        else:
            raise ValueError(f"不支持的聚类方法: {self.method}")

        # 训练
        if self.method == 'gmm':
            self.model.fit(X_scaled)
            self.labels_ = self.model.predict(X_scaled)
        else:
            self.labels_ = self.model.fit_predict(X_scaled)

        # 保存聚类中心
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers_ = self.model.cluster_centers_

        # 评估
        self._evaluate(X_scaled)

        return self

    def _evaluate(self, X: np.ndarray):
        """评估聚类质量"""
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels[unique_labels != -1])  # 排除噪声点

        print(f"\n聚类结果:")
        print(f"  聚类数: {n_clusters}")
        print(f"  样本数: {len(self.labels_)}")

        # 打印每个簇的样本数
        for label in unique_labels:
            count = (self.labels_ == label).sum()
            if label == -1:
                print(f"  噪声点: {count}")
            else:
                print(f"  簇 {label}: {count} 个样本")

        # 评估指标（需要至少2个簇）
        if n_clusters > 1:
            # 移除噪声点
            mask = self.labels_ != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X[mask], self.labels_[mask])
                ch_score = calinski_harabasz_score(X[mask], self.labels_[mask])
                db_score = davies_bouldin_score(X[mask], self.labels_[mask])

                print(f"\n评估指标:")
                print(f"  Silhouette Score: {silhouette:.3f} (越接近1越好)")
                print(f"  Calinski-Harabasz: {ch_score:.2f} (越大越好)")
                print(f"  Davies-Bouldin: {db_score:.3f} (越小越好)")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测新数据的聚类标签

        Args:
            X: 特征数据

        Returns:
            聚类标签
        """
        X_scaled = self.scaler.transform(X)

        if self.method == 'gmm':
            return self.model.predict(X_scaled)
        elif self.method in ['kmeans']:
            return self.model.predict(X_scaled)
        else:
            raise ValueError(f"{self.method} 不支持predict方法，请使用fit_predict")

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        训练并预测

        Args:
            X: 特征数据

        Returns:
            聚类标签
        """
        self.fit(X)
        return self.labels_

    def visualize(self, X: pd.DataFrame, method: str = 'pca'):
        """
        可视化聚类结果

        Args:
            X: 原始特征数据
            method: 降维方法 ('pca', 'tsne')
        """
        from . import visualization

        X_scaled = self.scaler.transform(X)

        visualization.plot_clustering_results(
            X_scaled,
            self.labels_,
            self.cluster_centers_,
            method=method
        )


# ==================== 降维Pipeline ====================

class DimensionReductionPipeline:
    """
    降维Pipeline

    支持PCA、t-SNE等降维方法
    """

    def __init__(self, method: str = 'pca',
                n_components: int = 2,
                random_state: int = 42):
        """
        Args:
            method: 降维方法 ('pca', 'tsne')
            n_components: 降维后的维度
            random_state: 随机种子
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame) -> 'DimensionReductionPipeline':
        """
        训练降维模型

        Args:
            X: 特征数据

        Returns:
            self
        """
        print(f"\n开始降维分析（方法: {self.method.upper()}）...")

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 创建模型
        if self.method == 'pca':
            self.model = PCA(n_components=self.n_components)

        elif self.method == 'tsne':
            self.model = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                perplexity=min(30, len(X) - 1)
            )

        else:
            raise ValueError(f"不支持的降维方法: {self.method}")

        # 训练
        self.model.fit(X_scaled)

        # PCA解释方差
        if self.method == 'pca':
            print(f"\n主成分解释方差:")
            for i, var in enumerate(self.model.explained_variance_ratio_):
                print(f"  PC{i+1}: {var:.2%}")
            print(f"  累计: {self.model.explained_variance_ratio_.sum():.2%}")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        降维转换

        Args:
            X: 特征数据

        Returns:
            降维后的数据
        """
        X_scaled = self.scaler.transform(X)

        if self.method == 'tsne':
            # t-SNE不支持transform，需要重新fit
            print("⚠️  t-SNE不支持transform，建议使用fit_transform")
            return self.model.fit_transform(X_scaled)

        return self.model.transform(X_scaled)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        训练并转换

        Args:
            X: 特征数据

        Returns:
            降维后的数据
        """
        self.fit(X)
        X_scaled = self.scaler.transform(X)
        return self.model.fit_transform(X_scaled)

    def visualize(self, X: pd.DataFrame, y: Optional[np.ndarray] = None,
                 labels: Optional[List[str]] = None):
        """
        可视化降维结果

        Args:
            X: 原始特征数据
            y: 标签（可选）
            labels: 标签名称（可选）
        """
        if self.n_components != 2:
            print("⚠️  只支持2D可视化")
            return

        from . import visualization

        X_scaled = self.scaler.transform(X)

        if self.method == 'pca':
            visualization.plot_pca_2d(X_scaled, y, labels)
        elif self.method == 'tsne':
            visualization.plot_tsne_2d(X_scaled, y, labels)


# ==================== 异常检测Pipeline ====================

class AnomalyDetectionPipeline:
    """
    异常检测Pipeline

    支持多种异常检测算法
    """

    def __init__(self, method: str = 'isolation_forest',
                contamination: float = 0.1,
                random_state: int = 42):
        """
        Args:
            method: 异常检测方法 ('isolation_forest', 'one_class_svm', 'lof')
            contamination: 预期异常比例
            random_state: 随机种子
        """
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame) -> 'AnomalyDetectionPipeline':
        """
        训练异常检测模型

        Args:
            X: 特征数据

        Returns:
            self
        """
        print(f"\n开始异常检测（方法: {self.method.upper()}）...")

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 创建模型
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )

        elif self.method == 'one_class_svm':
            self.model = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto'
            )

        elif self.method == 'lof':
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=20,
                n_jobs=-1
            )

        else:
            raise ValueError(f"不支持的异常检测方法: {self.method}")

        # 训练
        if self.method == 'lof':
            # LOF的fit_predict在fit时就会计算
            self.predictions_ = self.model.fit_predict(X_scaled)
        else:
            self.model.fit(X_scaled)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测异常

        Args:
            X: 特征数据

        Returns:
            预测结果 (1: 正常, -1: 异常)
        """
        X_scaled = self.scaler.transform(X)

        if self.method == 'lof':
            raise ValueError("LOF不支持predict方法，请使用fit_predict")

        return self.model.predict(X_scaled)

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        训练并预测

        Args:
            X: 特征数据

        Returns:
            预测结果 (1: 正常, -1: 异常)
        """
        self.fit(X)

        if self.method == 'lof':
            predictions = self.predictions_
        else:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)

        # 统计
        n_anomalies = (predictions == -1).sum()
        anomaly_ratio = n_anomalies / len(predictions) * 100

        print(f"\n异常检测结果:")
        print(f"  总样本数: {len(predictions)}")
        print(f"  异常样本数: {n_anomalies}")
        print(f"  异常比例: {anomaly_ratio:.2f}%")

        return predictions

    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        获取异常分数

        Args:
            X: 特征数据

        Returns:
            异常分数（越低越异常）
        """
        X_scaled = self.scaler.transform(X)

        if self.method == 'isolation_forest':
            # 分数越低越异常
            return self.model.score_samples(X_scaled)
        elif self.method == 'lof':
            # 负离群因子
            return -self.model.negative_outlier_factor_
        else:
            raise ValueError(f"{self.method} 不支持异常分数")


# ==================== 快速使用函数 ====================

def quick_clustering(X: pd.DataFrame, method: str = 'kmeans',
                    n_clusters: Optional[int] = None,
                    visualize: bool = True) -> np.ndarray:
    """
    快速聚类分析

    Args:
        X: 特征数据
        method: 聚类方法
        n_clusters: 聚类数
        visualize: 是否可视化

    Returns:
        聚类标签
    """
    pipeline = ClusteringPipeline(method=method, n_clusters=n_clusters)
    labels = pipeline.fit_predict(X)

    if visualize:
        pipeline.visualize(X, method='pca')

    return labels


def quick_dimension_reduction(X: pd.DataFrame, method: str = 'pca',
                              n_components: int = 2,
                              y: Optional[np.ndarray] = None,
                              visualize: bool = True) -> np.ndarray:
    """
    快速降维分析

    Args:
        X: 特征数据
        method: 降维方法
        n_components: 降维维度
        y: 标签（可选，用于可视化）
        visualize: 是否可视化

    Returns:
        降维后的数据
    """
    pipeline = DimensionReductionPipeline(method=method, n_components=n_components)
    X_reduced = pipeline.fit_transform(X)

    if visualize and n_components == 2:
        pipeline.visualize(X, y)

    return X_reduced


def quick_anomaly_detection(X: pd.DataFrame, method: str = 'isolation_forest',
                           contamination: float = 0.1) -> np.ndarray:
    """
    快速异常检测

    Args:
        X: 特征数据
        method: 异常检测方法
        contamination: 预期异常比例

    Returns:
        异常标签 (1: 正常, -1: 异常)
    """
    pipeline = AnomalyDetectionPipeline(method=method, contamination=contamination)
    predictions = pipeline.fit_predict(X)

    return predictions


if __name__ == '__main__':
    # 测试示例
    print("=== 无监督学习Pipeline测试 ===\n")

    # 创建测试数据
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=500, n_features=10, centers=3, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

    # 测试聚类
    print("1. 聚类分析")
    labels = quick_clustering(X, method='kmeans', n_clusters=3)

    # 测试降维
    print("\n2. 降维分析")
    X_reduced = quick_dimension_reduction(X, method='pca', n_components=2, y=y)

    # 测试异常检测
    print("\n3. 异常检测")
    anomalies = quick_anomaly_detection(X, contamination=0.05)

    print("\n✅ 所有测试完成！")
