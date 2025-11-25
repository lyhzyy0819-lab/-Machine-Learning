"""
鸢尾花(Iris)聚类练习
===================

练习目标：
1. 使用多种聚类算法对Iris数据集进行聚类
2. 对比不同算法的聚类效果
3. 理解聚类评估指标的含义

运行方式：
python iris_clustering_exercise.py
"""

# ==================== 导入必要的库 ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 聚类算法
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# 评估指标
from sklearn.metrics import (
    silhouette_score,           # 轮廓系数：衡量簇内紧密度和簇间分离度
    davies_bouldin_score,       # DB指数：簇内距离与簇间距离的比值，越小越好
    adjusted_rand_score,        # 调整兰德系数：与真实标签的一致性
    calinski_harabasz_score     # CH指数：簇间方差与簇内方差的比值，越大越好
)

import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS',  # macOS通用
    'PingFang SC',       # macOS系统字体
    'STHeiti',           # 华文黑体
    'Heiti TC',          # 黑体-繁
    'SimHei',            # 黑体
]
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)

print("=" * 80)
print("鸢尾花(Iris)聚类分析练习".center(70))
print("=" * 80)


# ==================== 1. 数据加载与探索 ====================
print("\n【步骤1】数据加载与探索")
print("-" * 80)

# 加载Iris数据集
# Iris数据集包含150个样本，3个类别(Setosa、Versicolor、Virginica)
# 每个样本有4个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
iris = load_iris()
X = iris.data           # 特征数据 (150, 4)
y_true = iris.target    # 真实标签 (150,) - 用于后续评估对比

# 创建DataFrame方便查看
feature_names = iris.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['species'] = iris.target_names[y_true]

print(f"数据集形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"类别名称: {iris.target_names}")
print(f"\n各类别样本数量:")
print(df['species'].value_counts())

print("\n数据统计信息:")
print(df.describe())


# ==================== 2. 数据预处理 ====================
print("\n【步骤2】数据预处理")
print("-" * 80)

# 数据标准化
# 原因：不同特征的量纲可能不同，标准化可以消除量纲影响
# 方法：Z-score标准化，使每个特征均值为0，标准差为1
# 公式：x_scaled = (x - mean) / std
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"标准化前特征范围:")
for i, name in enumerate(feature_names):
    print(f"  {name}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")

print(f"\n标准化后特征范围:")
for i, name in enumerate(feature_names):
    print(f"  {name}: [{X_scaled[:, i].min():.2f}, {X_scaled[:, i].max():.2f}]")

# PCA降维用于可视化
# 原因：Iris有4个特征，无法直接在2D平面上可视化
# 方法：使用PCA提取前2个主成分，保留最多的方差信息
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA降维后形状: {X_pca.shape}")
print(f"前2个主成分解释的方差比例: {pca.explained_variance_ratio_}")
print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.2%}")


# ==================== 3. K-Means聚类 ====================
print("\n【步骤3】K-Means聚类")
print("-" * 80)

# K-Means算法原理：
# 1. 随机初始化k个簇中心
# 2. 将每个样本分配到最近的簇中心
# 3. 重新计算每个簇的中心点（均值）
# 4. 重复步骤2-3，直到收敛

# 参数说明：
# - n_clusters=3: 簇的数量（我们知道Iris有3个类别）
# - n_init=10: 用不同的初始中心运行10次，选择最好的结果
# - max_iter=300: 最大迭代次数
# - random_state=42: 随机种子，保证结果可复现
kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# 评估K-Means聚类效果
silhouette_kmeans = silhouette_score(X_scaled, y_kmeans)
db_kmeans = davies_bouldin_score(X_scaled, y_kmeans)
ari_kmeans = adjusted_rand_score(y_true, y_kmeans)
ch_kmeans = calinski_harabasz_score(X_scaled, y_kmeans)

print(f"K-Means聚类结果:")
print(f"  Silhouette Score (轮廓系数): {silhouette_kmeans:.4f}  # 范围[-1,1]，越大越好")
print(f"  Davies-Bouldin Index (DB指数): {db_kmeans:.4f}  # 越小越好")
print(f"  Adjusted Rand Index (ARI): {ari_kmeans:.4f}  # 范围[-1,1]，1表示完全一致")
print(f"  Calinski-Harabasz Index (CH指数): {ch_kmeans:.4f}  # 越大越好")
print(f"  Inertia (惯性): {kmeans.inertia_:.4f}  # 簇内距离平方和，越小越好")

print(f"\n每个簇的样本数量:")
for i in range(3):
    count = np.sum(y_kmeans == i)
    print(f"  簇 {i}: {count} 个样本")


# ==================== 4. DBSCAN聚类 ====================
print("\n【步骤4】DBSCAN聚类")
print("-" * 80)

# DBSCAN算法原理：
# - 基于密度的聚类，不需要预先指定簇的数量
# - 核心思想：高密度区域形成簇，低密度区域为噪声点
# - 参数：
#   1. eps: 邻域半径（两点之间的最大距离）
#   2. min_samples: 核心点的最小邻居数量

# 参数选择：
# - eps=0.5: 经过尝试，0.5对Iris数据效果较好
# - min_samples=5: 一般设置为特征数+1，这里4+1=5
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

# DBSCAN会将噪声点标记为-1
n_clusters_dbscan = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise = list(y_dbscan).count(-1)

print(f"DBSCAN聚类结果:")
print(f"  识别的簇数量: {n_clusters_dbscan}")
print(f"  噪声点数量: {n_noise}")

# 只对非噪声点计算评估指标
if n_clusters_dbscan > 1 and n_noise < len(y_dbscan):
    # 筛选非噪声点
    mask = y_dbscan != -1
    X_no_noise = X_scaled[mask]
    y_dbscan_no_noise = y_dbscan[mask]
    y_true_no_noise = y_true[mask]

    silhouette_dbscan = silhouette_score(X_no_noise, y_dbscan_no_noise)
    db_dbscan = davies_bouldin_score(X_no_noise, y_dbscan_no_noise)
    ari_dbscan = adjusted_rand_score(y_true_no_noise, y_dbscan_no_noise)
    ch_dbscan = calinski_harabasz_score(X_no_noise, y_dbscan_no_noise)

    print(f"  Silhouette Score: {silhouette_dbscan:.4f}")
    print(f"  Davies-Bouldin Index: {db_dbscan:.4f}")
    print(f"  Adjusted Rand Index: {ari_dbscan:.4f}")
    print(f"  Calinski-Harabasz Index: {ch_dbscan:.4f}")
else:
    print("  ⚠️ DBSCAN未能有效聚类（簇数量<=1或噪声点过多）")
    silhouette_dbscan = db_dbscan = ari_dbscan = ch_dbscan = 0

print(f"\n每个簇的样本数量:")
for i in sorted(set(y_dbscan)):
    count = np.sum(y_dbscan == i)
    label = "噪声点" if i == -1 else f"簇 {i}"
    print(f"  {label}: {count} 个样本")


# ==================== 5. 层次聚类 ====================
print("\n【步骤5】层次聚类 (Hierarchical Clustering)")
print("-" * 80)

# 层次聚类算法原理：
# - 凝聚型(Agglomerative)：自底向上，每个点初始为一个簇，逐步合并
# - 参数：
#   1. n_clusters: 最终簇的数量
#   2. linkage: 簇间距离计算方法
#      - 'ward': 最小化簇内方差（最常用）
#      - 'complete': 最大距离
#      - 'average': 平均距离
#      - 'single': 最小距离

# 使用ward连接方法
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_hierarchical = hierarchical.fit_predict(X_scaled)

# 评估层次聚类效果
silhouette_hier = silhouette_score(X_scaled, y_hierarchical)
db_hier = davies_bouldin_score(X_scaled, y_hierarchical)
ari_hier = adjusted_rand_score(y_true, y_hierarchical)
ch_hier = calinski_harabasz_score(X_scaled, y_hierarchical)

print(f"层次聚类结果 (linkage='ward'):")
print(f"  Silhouette Score: {silhouette_hier:.4f}")
print(f"  Davies-Bouldin Index: {db_hier:.4f}")
print(f"  Adjusted Rand Index: {ari_hier:.4f}")
print(f"  Calinski-Harabasz Index: {ch_hier:.4f}")

print(f"\n每个簇的样本数量:")
for i in range(3):
    count = np.sum(y_hierarchical == i)
    print(f"  簇 {i}: {count} 个样本")


# ==================== 6. 结果对比 ====================
print("\n【步骤6】三种算法性能对比")
print("-" * 80)

# 创建对比表格
results = pd.DataFrame({
    'Algorithm': ['K-Means', 'DBSCAN', 'Hierarchical'],
    'Silhouette Score ↑': [silhouette_kmeans, silhouette_dbscan, silhouette_hier],
    'Davies-Bouldin ↓': [db_kmeans, db_dbscan, db_hier],
    'Adjusted Rand Index ↑': [ari_kmeans, ari_dbscan, ari_hier],
    'Calinski-Harabasz ↑': [ch_kmeans, ch_dbscan, ch_hier]
})

print("\n性能对比 (↑越大越好，↓越小越好):")
print(results.to_string(index=False))

# 找出最佳算法
print("\n各指标最佳算法:")
print(f"  Silhouette Score: {results.loc[results['Silhouette Score ↑'].idxmax(), 'Algorithm']}")
print(f"  Davies-Bouldin: {results.loc[results['Davies-Bouldin ↓'].idxmin(), 'Algorithm']}")
print(f"  Adjusted Rand Index: {results.loc[results['Adjusted Rand Index ↑'].idxmax(), 'Algorithm']}")
print(f"  Calinski-Harabasz: {results.loc[results['Calinski-Harabasz ↑'].idxmax(), 'Algorithm']}")


# ==================== 7. 可视化 ====================
print("\n【步骤7】可视化聚类结果")
print("-" * 80)

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('鸢尾花(Iris)聚类结果对比 (基于PCA降维)', fontsize=16, fontweight='bold')

# 定义颜色映射
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# 子图1: 真实标签
ax1 = axes[0, 0]
for i, species in enumerate(iris.target_names):
    mask = y_true == i
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[i], label=species, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax1.set_title('真实标签 (Ground Truth)', fontsize=14, fontweight='bold')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 子图2: K-Means聚类
ax2 = axes[0, 1]
for i in range(3):
    mask = y_kmeans == i
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[i], label=f'Cluster {i}', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
# 绘制簇中心 (需要将簇中心投影到PCA空间)
centers_pca = pca.transform(kmeans.cluster_centers_)
ax2.scatter(centers_pca[:, 0], centers_pca[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidth=2,
           label='Centroids', zorder=10)
ax2.set_title(f'K-Means (ARI={ari_kmeans:.3f})', fontsize=14, fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# 子图3: DBSCAN聚类
ax3 = axes[1, 0]
unique_labels = set(y_dbscan)
for i, label in enumerate(sorted(unique_labels)):
    mask = y_dbscan == label
    if label == -1:
        # 噪声点用灰色X标记
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c='gray', marker='x', s=50, alpha=0.5, label='Noise')
    else:
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[label], label=f'Cluster {label}', s=50, alpha=0.7,
                   edgecolors='black', linewidth=0.5)
ax3.set_title(f'DBSCAN (ARI={ari_dbscan:.3f}, eps=0.5)', fontsize=14, fontweight='bold')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# 子图4: 层次聚类
ax4 = axes[1, 1]
for i in range(3):
    mask = y_hierarchical == i
    ax4.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[i], label=f'Cluster {i}', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax4.set_title(f'Hierarchical (ARI={ari_hier:.3f}, ward)', fontsize=14, fontweight='bold')
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/lyh/Desktop/ Machine Learning/unsupervised_learning/iris_clustering_results.png',
            dpi=300, bbox_inches='tight')
print("✅ 可视化图表已保存: iris_clustering_results.png")
plt.show()


# ==================== 8. 评估指标雷达图 ====================
print("\n【步骤8】绘制评估指标雷达图")
print("-" * 80)

# 归一化指标 (使其在0-1范围内，便于对比)
def normalize_score(score, metric_type='maximize'):
    """
    归一化评分到0-1范围
    metric_type: 'maximize' 表示越大越好, 'minimize' 表示越小越好
    """
    if metric_type == 'minimize':
        # 对于越小越好的指标，使用倒数
        return 1 / (1 + score)
    else:
        # 对于越大越好的指标，直接使用
        return max(0, min(1, (score + 1) / 2))  # ARI范围[-1,1]，归一化到[0,1]

# 准备雷达图数据
categories = ['Silhouette\n(内聚性)', 'Davies-Bouldin\n(分离度)',
              'Adjusted Rand\n(准确性)', 'Calinski-Harabasz\n(对比度)']

# 归一化各指标
kmeans_scores = [
    normalize_score(silhouette_kmeans),
    normalize_score(db_kmeans, 'minimize'),
    normalize_score(ari_kmeans),
    normalize_score(ch_kmeans / 1000)  # CH指数较大，缩放一下
]

dbscan_scores = [
    normalize_score(silhouette_dbscan),
    normalize_score(db_dbscan, 'minimize'),
    normalize_score(ari_dbscan),
    normalize_score(ch_dbscan / 1000)
]

hierarchical_scores = [
    normalize_score(silhouette_hier),
    normalize_score(db_hier, 'minimize'),
    normalize_score(ari_hier),
    normalize_score(ch_hier / 1000)
]

# 雷达图需要闭合，复制第一个值到最后
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
kmeans_scores += kmeans_scores[:1]
dbscan_scores += dbscan_scores[:1]
hierarchical_scores += hierarchical_scores[:1]
angles += angles[:1]

# 绘制雷达图
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

ax.plot(angles, kmeans_scores, 'o-', linewidth=2, label='K-Means', color='#FF6B6B')
ax.fill(angles, kmeans_scores, alpha=0.25, color='#FF6B6B')

ax.plot(angles, dbscan_scores, 'o-', linewidth=2, label='DBSCAN', color='#4ECDC4')
ax.fill(angles, dbscan_scores, alpha=0.25, color='#4ECDC4')

ax.plot(angles, hierarchical_scores, 'o-', linewidth=2, label='Hierarchical', color='#45B7D1')
ax.fill(angles, hierarchical_scores, alpha=0.25, color='#45B7D1')

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.grid(True, linestyle='--', alpha=0.7)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
plt.title('聚类算法评估指标对比 (归一化)', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/Users/lyh/Desktop/ Machine Learning/unsupervised_learning/iris_clustering_radar.png',
            dpi=300, bbox_inches='tight')
print("✅ 雷达图已保存: iris_clustering_radar.png")
plt.show()


# ==================== 9. 总结 ====================
print("\n" + "=" * 80)
print("【总结】聚类分析结果".center(70))
print("=" * 80)

print("""
📊 Iris数据集聚类分析总结:

1️⃣  K-Means聚类:
   ✅ 优点: 速度快，结果稳定，适合球形簇
   ⚠️  缺点: 需要预先指定k值，对初始值敏感
   📈 性能: Silhouette={:.3f}, ARI={:.3f}
   💡 适用场景: 簇大小相近、形状规则的数据

2️⃣  DBSCAN聚类:
   ✅ 优点: 不需要指定k值，可以识别噪声点和任意形状的簇
   ⚠️  缺点: 对eps和min_samples参数敏感，密度差异大时效果差
   📈 性能: Silhouette={:.3f}, ARI={:.3f}
   💡 适用场景: 簇密度相近、存在噪声的数据

3️⃣  层次聚类:
   ✅ 优点: 可以生成层次结构树，不需要预先指定k值
   ⚠️  缺点: 计算复杂度高O(n²)，不适合大数据集
   📈 性能: Silhouette={:.3f}, ARI={:.3f}
   💡 适用场景: 需要多层次结构分析的小数据集

🏆 对于Iris数据集，{}表现最佳！

💡 关键启示:
   - Iris数据集的3个类别中，Setosa与其他两类分离明显
   - Versicolor和Virginica存在一定重叠，聚类难度较大
   - 选择聚类算法需要根据数据特点和业务需求
   - 评估指标应综合考虑，不能只看单一指标
""".format(
    silhouette_kmeans, ari_kmeans,
    silhouette_dbscan, ari_dbscan,
    silhouette_hier, ari_hier,
    results.loc[results['Adjusted Rand Index ↑'].idxmax(), 'Algorithm']
))

print("=" * 80)
print("✅ 练习完成！".center(70))
print("=" * 80)
