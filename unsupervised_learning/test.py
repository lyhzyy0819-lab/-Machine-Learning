# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = [
        'Arial Unicode MS',  # macOS通用
        'PingFang SC',       # macOS系统字体
        'STHeiti',           # 华文黑体
        'Heiti TC',          # 黑体-繁
        'SimHei',            # 黑体
    ]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
np.random.seed(42)

print("✅ 环境准备完成！")


X, y_true = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=0.6,
    random_state=42
)

print(X.shape)
print(f"特征范围：X1 [{X[:, 0].min()}, {X[:, 0].max():.2f}], [{X[:, 1].min()}, {X[:, 1].max():.2f}]")


class KMeansFromScratch:
    """
    从零实现的K-Means聚类算法

    Parameters:
    -----------
    n_clusters : int, default=3
        簇的数量
    max_iters : int, default=100
        最大迭代次数
    tol : float, default=1e-4
        收敛阈值
    """

    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None  # 簇内距离平方和

    def fit(self, X):
        """
        训练K-Means模型

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据
        """
        # ==================== 步骤1: 初始化簇中心 ====================
        # 从数据集中随机选择k个样本作为初始簇中心
        # 例如: 如果n_clusters=3, X有300个样本, 则随机选3个不重复的索引
        random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        # 用选中的样本点作为初始簇中心
        # 形状: (n_clusters, n_features), 例如 (3, 2)
        self.centroids = X[random_idx]

        # ==================== 迭代优化 ====================
        for i in range(self.max_iters):
            # ---------- 步骤2: 分配阶段 (Expectation) ----------
            # 计算每个样本点到所有簇中心的距离
            # distances形状: (n_samples, n_clusters), 例如 (300, 3)
            # distances[i, k] 表示第i个样本到第k个簇中心的距离
            distances = self._compute_distances(X, self.centroids)

            # 为每个样本分配最近的簇标签
            # np.argmin(distances, axis=1) 沿着列方向找最小值的索引
            # 即找到每个样本距离最近的簇中心编号(0, 1, 2)
            # 形状: (n_samples,), 例如 (300,)
            self.labels_ = np.argmin(distances, axis=1)

            # ---------- 步骤3: 更新阶段 (Maximization) ----------
            # 重新计算每个簇的中心点（均值）
            new_centroids = np.array([
                # 对于每个簇k, 找到属于该簇的所有样本点
                # X[self.labels_ == k] 筛选出标签为k的所有样本
                # .mean(axis=0) 计算这些样本点的坐标均值
                # 例如: 簇0有100个点, 计算这100个点的x均值和y均值
                X[self.labels_ == k].mean(axis=0)
                for k in range(self.n_clusters)
            ])
            # new_centroids形状: (n_clusters, n_features), 例如 (3, 2)

            # ---------- 步骤4: 检查收敛 ----------
            # 判断新旧簇中心是否足够接近（变化小于阈值tol）
            # np.allclose() 检查两个数组是否在给定容差内相等
            # atol=self.tol 表示绝对容差, 例如 1e-4
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                print(f"✅ 算法在第 {i + 1} 次迭代后收敛")
                break

            # 更新簇中心为新计算的值
            self.centroids = new_centroids

        # ==================== 计算最终评估指标 ====================
        # 计算惯性（inertia）: 所有样本到其簇中心的距离平方和
        # 惯性越小说明簇内聚合度越高
        self.inertia_ = self._compute_inertia(X)

        return self

    def predict(self, X):
        """
        预测新数据点的簇标签
        """
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X, centroids):
        """
        计算每个数据点到各簇中心的欧氏距离

        Returns:
        --------
        distances : array, shape (n_samples, n_clusters)
        """
        # 初始化距离矩阵, 形状: (n_samples, n_clusters)
        # 例如: (300, 3) 表示300个样本到3个簇中心的距离
        distances = np.zeros((len(X), self.n_clusters))

        # 对每个簇中心k计算所有样本点到它的距离
        for k in range(self.n_clusters):
            # X - centroids[k]: 广播机制, 每个样本点减去第k个簇中心坐标
            # 例如: X是(300,2), centroids[k]是(2,), 广播后得到(300,2)
            # 每一行是样本点与簇中心的坐标差值

            # np.linalg.norm(..., axis=1): 计算每行的L2范数(欧氏距离)
            # axis=1表示沿着列方向计算, 即对每个样本点的[x差, y差]计算距离
            # 公式: sqrt((x1-cx)^2 + (x2-cy)^2)
            # 结果形状: (n_samples,), 例如 (300,)
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)

        return distances

    def _compute_inertia(self, X):
        """
        计算惯性（簇内距离平方和）
        Inertia = Σ ||xi - μk||^2  (对所有簇k和簇内所有点xi求和)
        """
        # 初始化惯性为0
        inertia = 0

        # 对每个簇k计算簇内距离平方和
        for k in range(self.n_clusters):
            # 筛选出属于簇k的所有样本点
            # X[self.labels_ == k] 布尔索引, 选出标签为k的样本
            # 例如: 簇0有100个点, cluster_points形状为(100, 2)
            cluster_points = X[self.labels_ == k]

            # 计算该簇内所有点到簇中心的距离平方和
            # cluster_points - self.centroids[k]: 广播, 每个点减去簇中心
            # ** 2: 平方
            # np.sum(): 求和所有元素
            # 例如: 对于簇0的100个点, 计算每个点到簇中心距离的平方, 然后求和
            inertia += np.sum((cluster_points - self.centroids[k]) ** 2)

        # 返回总惯性值
        # 惯性越小 → 簇内紧密度越高 → 聚类效果越好
        return inertia

kmeans_scratch = KMeansFromScratch(n_clusters=3, max_iters=100)
kmeans_scratch.fit(X)

print(f"\n簇中心:\n{kmeans_scratch.centroids}")
print(f"\n惯性（Inertia）: {kmeans_scratch.inertia_:.2f}")
print(f"\n每个簇的样本数: {np.bincount(kmeans_scratch.labels_)}")
