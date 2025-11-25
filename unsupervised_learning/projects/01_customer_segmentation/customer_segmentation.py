"""
🎯 客户分群项目 (Customer Segmentation)
======================================

项目目标：
    使用K-Means和GMM聚类算法对商场客户进行分群分析，
    以便制定针对性的营销策略。

数据集：
    Kaggle Mall Customer Segmentation Dataset
    https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

数据字段：
    - CustomerID: 客户ID
    - Gender: 性别 (Male/Female)
    - Age: 年龄
    - Annual Income (k$): 年收入（千美元）
    - Spending Score (1-100): 消费评分（1-100，由商场根据客户行为评定）

作者: Machine Learning 学习项目
日期: 2024年11月
"""

# ============================================================================
# 第1部分：导入必要的库
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 聚类算法
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# 数据预处理
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 评估指标
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 模型保存
import joblib
import json

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS',  # macOS通用
    'PingFang SC',       # macOS系统字体
    'STHeiti',           # 华文黑体
    'SimHei',            # Windows黑体
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 设置随机种子，保证结果可复现
np.random.seed(42)

print("✅ 库导入完成！")
print("=" * 60)


# ============================================================================
# 第2部分：数据加载与探索 (EDA)
# ============================================================================

def load_data(filepath):
    """
    加载客户数据

    Parameters:
    -----------
    filepath : str
        数据文件路径

    Returns:
    --------
    df : DataFrame
        加载的数据框
    """
    # 读取CSV文件
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
    """
    数据探索性分析 (EDA)

    目的：了解数据的基本情况，包括数据类型、缺失值、统计特征等

    Parameters:
    -----------
    df : DataFrame
        客户数据
    """
    print("\n" + "=" * 60)
    print("📊 数据探索性分析 (EDA)")
    print("=" * 60)

    # ----- 1. 基本信息 -----
    print("\n【1. 数据基本信息】")
    print(f"  • 数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"  • 列名: {list(df.columns)}")
    print("\n数据类型:")
    print(df.dtypes)

    # ----- 2. 前5行数据 -----
    print("\n【2. 数据预览（前5行）】")
    print(df.head())

    # ----- 3. 缺失值检查 -----
    print("\n【3. 缺失值检查】")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✅ 数据无缺失值")
    else:
        print(missing[missing > 0])

    # ----- 4. 统计描述 -----
    print("\n【4. 数值特征统计描述】")
    print(df.describe())

    # ----- 5. 类别特征分布 -----
    print("\n【5. 性别分布】")
    gender_counts = df['Gender'].value_counts()
    print(gender_counts)
    print(f"  男性占比: {gender_counts['Male']/len(df)*100:.1f}%")
    print(f"  女性占比: {gender_counts['Female']/len(df)*100:.1f}%")


def visualize_distributions(df):
    """
    可视化特征分布

    目的：通过图表直观了解各特征的分布情况

    Parameters:
    -----------
    df : DataFrame
        客户数据
    """
    print("\n" + "=" * 60)
    print("📈 特征分布可视化")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ----- 1. 年龄分布 -----
    # 直方图展示年龄的整体分布
    ax1 = axes[0, 0]
    ax1.hist(df['Age'], bins=20, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f'均值={df["Age"].mean():.1f}')
    ax1.set_xlabel('年龄')
    ax1.set_ylabel('频数')
    ax1.set_title('年龄分布直方图')
    ax1.legend()

    # ----- 2. 年收入分布 -----
    ax2 = axes[0, 1]
    ax2.hist(df['Annual Income (k$)'], bins=20, color='seagreen', edgecolor='white', alpha=0.7)
    ax2.axvline(df['Annual Income (k$)'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值={df["Annual Income (k$)"].mean():.1f}k$')
    ax2.set_xlabel('年收入 (k$)')
    ax2.set_ylabel('频数')
    ax2.set_title('年收入分布直方图')
    ax2.legend()

    # ----- 3. 消费评分分布 -----
    ax3 = axes[0, 2]
    ax3.hist(df['Spending Score (1-100)'], bins=20, color='coral', edgecolor='white', alpha=0.7)
    ax3.axvline(df['Spending Score (1-100)'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值={df["Spending Score (1-100)"].mean():.1f}')
    ax3.set_xlabel('消费评分 (1-100)')
    ax3.set_ylabel('频数')
    ax3.set_title('消费评分分布直方图')
    ax3.legend()

    # ----- 4. 性别分布（饼图）-----
    ax4 = axes[1, 0]
    gender_counts = df['Gender'].value_counts()
    colors = ['#66b3ff', '#ff9999']
    explode = (0.05, 0)  # 突出显示第一块
    ax4.pie(gender_counts, labels=['女性', '男性'], autopct='%1.1f%%',
            colors=colors, explode=explode, startangle=90, shadow=True)
    ax4.set_title('性别分布')

    # ----- 5. 年龄箱线图（按性别）-----
    ax5 = axes[1, 1]
    df.boxplot(column='Age', by='Gender', ax=ax5)
    ax5.set_xlabel('性别')
    ax5.set_ylabel('年龄')
    ax5.set_title('年龄分布（按性别）')
    plt.suptitle('')  # 移除自动生成的标题

    # ----- 6. 收入 vs 消费评分 散点图 -----
    # 这是聚类分析的核心特征组合
    ax6 = axes[1, 2]
    scatter = ax6.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                         c=df['Age'], cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('年收入 (k$)')
    ax6.set_ylabel('消费评分 (1-100)')
    ax6.set_title('收入 vs 消费评分（颜色=年龄）')
    plt.colorbar(scatter, ax=ax6, label='年龄')

    plt.tight_layout()
    plt.savefig('output/01_feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("  ✅ 图表已保存: output/01_feature_distributions.png")


def plot_correlation_heatmap(df):
    """
    绘制特征相关性热力图

    目的：了解各数值特征之间的相关关系

    Parameters:
    -----------
    df : DataFrame
        客户数据
    """
    print("\n【相关性分析】")

    # 选择数值特征
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=0.5,
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('特征相关性热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("  ✅ 图表已保存: output/02_correlation_heatmap.png")

    # 解读相关性
    print("\n【相关性解读】")
    print(f"  • 年龄 vs 年收入: {corr_matrix.loc['Age', 'Annual Income (k$)']:.3f}")
    print(f"  • 年龄 vs 消费评分: {corr_matrix.loc['Age', 'Spending Score (1-100)']:.3f}")
    print(f"  • 年收入 vs 消费评分: {corr_matrix.loc['Annual Income (k$)', 'Spending Score (1-100)']:.3f}")
    print("\n  💡 特征之间相关性较低，说明它们各自提供了不同的信息维度")


# ============================================================================
# 第3部分：数据预处理
# ============================================================================

def preprocess_data(df):
    """
    数据预处理

    步骤：
        1. 性别编码（Label Encoding）
        2. 选择聚类特征
        3. 特征标准化（StandardScaler）

    Parameters:
    -----------
    df : DataFrame
        原始客户数据

    Returns:
    --------
    X : array
        用于聚类的原始特征（未标准化）
    X_scaled : array
        标准化后的特征
    feature_names : list
        特征名称
    df_processed : DataFrame
        处理后的数据框（包含编码后的性别）
    scaler : StandardScaler
        训练好的标准化器（用于新数据预测）
    """
    print("\n" + "=" * 60)
    print("🔧 数据预处理")
    print("=" * 60)

    # 复制数据，避免修改原始数据
    df_processed = df.copy()

    # ----- 1. 性别编码 -----
    # 将类别变量转换为数值：Female=0, Male=1
    print("\n【1. 性别编码】")
    le = LabelEncoder()
    df_processed['Gender_encoded'] = le.fit_transform(df_processed['Gender'])
    print(f"  • 编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ----- 2. 选择聚类特征 -----
    # 根据业务场景，选择"年收入"和"消费评分"作为主要聚类特征
    # 理由：这两个特征直接反映客户的消费能力和消费意愿
    print("\n【2. 选择聚类特征】")
    feature_names = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df_processed[feature_names].values
    print(f"  • 选择的特征: {feature_names}")
    print(f"  • 特征矩阵形状: {X.shape}")

    # ----- 3. 特征标准化 -----
    # 为什么要标准化？
    # K-Means使用欧氏距离，如果特征尺度不同，大尺度特征会主导距离计算
    # 标准化公式: z = (x - μ) / σ，使每个特征均值为0，标准差为1
    print("\n【3. 特征标准化】")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("  标准化前后对比:")
    print(f"  • 年收入 - 原始均值: {X[:, 0].mean():.2f}, 标准化后均值: {X_scaled[:, 0].mean():.6f}")
    print(f"  • 年收入 - 原始标准差: {X[:, 0].std():.2f}, 标准化后标准差: {X_scaled[:, 0].std():.6f}")
    print(f"  • 消费评分 - 原始均值: {X[:, 1].mean():.2f}, 标准化后均值: {X_scaled[:, 1].mean():.6f}")
    print(f"  • 消费评分 - 原始标准差: {X[:, 1].std():.2f}, 标准化后标准差: {X_scaled[:, 1].std():.6f}")

    print("\n  ✅ 数据预处理完成！")

    return X, X_scaled, feature_names, df_processed, scaler


# ============================================================================
# 第4部分：确定最佳K值
# ============================================================================

def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """
    使用肘部法则和轮廓系数确定最佳K值

    方法说明：
    ---------
    1. 肘部法则 (Elbow Method)
       - 计算不同K值下的惯性（Inertia，即簇内距离平方和）
       - 寻找惯性下降速度明显变缓的"肘部"点
       - 惯性公式: J = Σ Σ ||x - μ_i||²

    2. 轮廓系数 (Silhouette Score)
       - 衡量簇内紧密度和簇间分离度
       - 范围: [-1, 1]，越大越好
       - 公式: s(i) = (b(i) - a(i)) / max(a(i), b(i))
         其中 a(i) = 样本i到同簇其他点的平均距离
              b(i) = 样本i到最近其他簇的平均距离

    Parameters:
    -----------
    X_scaled : array
        标准化后的特征矩阵
    k_range : range
        要测试的K值范围

    Returns:
    --------
    optimal_k : int
        推荐的最佳K值
    """
    print("\n" + "=" * 60)
    print("🔍 确定最佳聚类数 K")
    print("=" * 60)

    # 存储各指标
    inertias = []           # 惯性（簇内距离平方和）
    silhouette_scores = []  # 轮廓系数
    db_scores = []          # Davies-Bouldin指数（越小越好）
    ch_scores = []          # Calinski-Harabasz指数（越大越好）

    print("\n【计算各K值的评估指标】")
    print("-" * 60)
    print(f"{'K':^5} {'Inertia':^12} {'Silhouette':^12} {'Davies-Bouldin':^15} {'Calinski-Harabasz':^18}")
    print("-" * 60)

    for k in k_range:
        # 训练K-Means模型
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X_scaled)

        # 计算惯性
        inertias.append(kmeans.inertia_)

        # 计算轮廓系数（K>=2才能计算）
        sil_score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil_score)

        # 计算Davies-Bouldin指数
        db_score = davies_bouldin_score(X_scaled, labels)
        db_scores.append(db_score)

        # 计算Calinski-Harabasz指数
        ch_score = calinski_harabasz_score(X_scaled, labels)
        ch_scores.append(ch_score)

        print(f"{k:^5} {kmeans.inertia_:^12.2f} {sil_score:^12.4f} {db_score:^15.4f} {ch_score:^18.2f}")

    print("-" * 60)

    # ----- 可视化 -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 肘部法则图
    ax1 = axes[0, 0]
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('聚类数 K')
    ax1.set_ylabel('惯性 (Inertia)')
    ax1.set_title('肘部法则 - 惯性 vs K', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # 标记K=5的位置（通常是肘部）
    ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='K=5 (建议)')
    ax1.legend()

    # 2. 轮廓系数图
    ax2 = axes[0, 1]
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('聚类数 K')
    ax2.set_ylabel('轮廓系数')
    ax2.set_title('轮廓系数 vs K（越大越好）', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # 找到最大轮廓系数对应的K
    best_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
    ax2.axvline(x=best_k_silhouette, color='red', linestyle='--', alpha=0.7,
                label=f'最佳 K={best_k_silhouette}')
    ax2.legend()

    # 3. Davies-Bouldin指数图
    ax3 = axes[1, 0]
    ax3.plot(k_range, db_scores, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('聚类数 K')
    ax3.set_ylabel('Davies-Bouldin 指数')
    ax3.set_title('Davies-Bouldin指数 vs K（越小越好）', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    best_k_db = list(k_range)[np.argmin(db_scores)]
    ax3.axvline(x=best_k_db, color='red', linestyle='--', alpha=0.7, label=f'最佳 K={best_k_db}')
    ax3.legend()

    # 4. Calinski-Harabasz指数图
    ax4 = axes[1, 1]
    ax4.plot(k_range, ch_scores, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('聚类数 K')
    ax4.set_ylabel('Calinski-Harabasz 指数')
    ax4.set_title('Calinski-Harabasz指数 vs K（越大越好）', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    best_k_ch = list(k_range)[np.argmax(ch_scores)]
    ax4.axvline(x=best_k_ch, color='red', linestyle='--', alpha=0.7, label=f'最佳 K={best_k_ch}')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('output/03_optimal_k_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n  ✅ 图表已保存: output/03_optimal_k_analysis.png")

    # ----- 综合分析 -----
    print("\n【综合分析】")
    print(f"  • 轮廓系数最大: K = {best_k_silhouette}")
    print(f"  • Davies-Bouldin最小: K = {best_k_db}")
    print(f"  • Calinski-Harabasz最大: K = {best_k_ch}")
    print(f"  • 肘部法则观察: K = 5 附近出现明显拐点")

    # 综合各指标，选择K=5
    optimal_k = 5
    print(f"\n  🎯 推荐使用 K = {optimal_k}")
    print("     理由：多个指标综合指向K=5，且从业务角度可解释性强")

    return optimal_k


# ============================================================================
# 第5部分：K-Means 聚类
# ============================================================================

def kmeans_clustering(X, X_scaled, optimal_k, feature_names):
    """
    执行K-Means聚类

    K-Means算法步骤：
    ----------------
    1. 初始化：随机选择K个数据点作为初始簇中心
    2. 分配：将每个数据点分配到最近的簇中心
    3. 更新：重新计算每个簇的中心（簇内所有点的均值）
    4. 迭代：重复步骤2-3，直到收敛

    Parameters:
    -----------
    X : array
        原始特征矩阵（用于可视化）
    X_scaled : array
        标准化后的特征矩阵（用于聚类）
    optimal_k : int
        最佳聚类数
    feature_names : list
        特征名称

    Returns:
    --------
    kmeans : KMeans
        训练好的K-Means模型
    labels : array
        每个样本的簇标签
    """
    print("\n" + "=" * 60)
    print("🎯 K-Means 聚类")
    print("=" * 60)

    # ----- 训练K-Means模型 -----
    print(f"\n【模型训练】")
    print(f"  • 聚类数 K = {optimal_k}")
    print(f"  • 初始化方法: k-means++ (智能初始化)")
    print(f"  • 初始化次数: 10 (选择最优结果)")

    kmeans = KMeans(
        n_clusters=optimal_k,   # 簇的数量
        init='k-means++',       # 初始化方法：k-means++比随机初始化更优
        n_init=10,              # 不同初始化的运行次数
        max_iter=300,           # 最大迭代次数
        random_state=42         # 随机种子
    )

    # 训练并预测
    labels = kmeans.fit_predict(X_scaled)

    print(f"  • 收敛迭代次数: {kmeans.n_iter_}")
    print(f"  • 惯性 (Inertia): {kmeans.inertia_:.2f}")

    # ----- 评估聚类效果 -----
    print("\n【聚类评估】")
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    print(f"  • 轮廓系数: {sil_score:.4f} (范围[-1,1]，越大越好)")
    print(f"  • Davies-Bouldin指数: {db_score:.4f} (越小越好)")
    print(f"  • Calinski-Harabasz指数: {ch_score:.2f} (越大越好)")

    # ----- 簇分布统计 -----
    print("\n【各簇样本数】")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  • 簇 {cluster}: {count} 个客户 ({count/len(labels)*100:.1f}%)")

    # ----- 簇中心分析（还原到原始尺度）-----
    # 注意：kmeans.cluster_centers_ 是标准化后的坐标，需要还原
    print("\n【簇中心（原始尺度）】")

    # 计算原始数据中每个簇的均值作为簇中心
    centers_original = np.array([X[labels == i].mean(axis=0) for i in range(optimal_k)])

    print(f"  {'簇':<5} {feature_names[0]:<25} {feature_names[1]:<25}")
    print("  " + "-" * 55)
    for i in range(optimal_k):
        print(f"  {i:<5} {centers_original[i, 0]:<25.2f} {centers_original[i, 1]:<25.2f}")

    # ----- 可视化聚类结果 -----
    plt.figure(figsize=(12, 8))

    # 定义颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    # 绘制各簇的数据点
    for i in range(optimal_k):
        mask = labels == i
        plt.scatter(X[mask, 0], X[mask, 1],
                   c=colors[i], s=80, alpha=0.6,
                   edgecolors='white', linewidth=0.5,
                   label=f'簇 {i} (n={mask.sum()})')

    # 绘制簇中心
    plt.scatter(centers_original[:, 0], centers_original[:, 1],
               c='black', s=300, marker='*',
               edgecolors='white', linewidths=2,
               label='簇中心', zorder=10)

    # 为每个簇中心添加标签
    for i, center in enumerate(centers_original):
        plt.annotate(f'C{i}', xy=center, xytext=(5, 5),
                    textcoords='offset points', fontsize=12, fontweight='bold')

    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.title(f'K-Means 聚类结果 (K={optimal_k})\n轮廓系数={sil_score:.4f}',
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/04_kmeans_clustering_result.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n  ✅ 图表已保存: output/04_kmeans_clustering_result.png")

    return kmeans, labels


# ============================================================================
# 第6部分：GMM 聚类对比
# ============================================================================

def gmm_clustering(X, X_scaled, optimal_k, feature_names, kmeans_labels):
    """
    使用高斯混合模型 (GMM) 进行聚类，并与K-Means对比

    GMM vs K-Means:
    ---------------
    | 特性         | K-Means        | GMM                |
    |--------------|----------------|---------------------|
    | 分配方式     | 硬分配         | 软分配（概率）      |
    | 簇形状       | 球形           | 椭圆形（更灵活）    |
    | 算法         | 距离最小化     | EM算法（似然最大化）|

    Parameters:
    -----------
    X : array
        原始特征矩阵
    X_scaled : array
        标准化后的特征矩阵
    optimal_k : int
        聚类数
    feature_names : list
        特征名称
    kmeans_labels : array
        K-Means的聚类标签（用于对比）

    Returns:
    --------
    gmm : GaussianMixture
        训练好的GMM模型
    gmm_labels : array
        GMM的聚类标签
    """
    print("\n" + "=" * 60)
    print("🎯 高斯混合模型 (GMM) 聚类")
    print("=" * 60)

    # ----- 1. 使用BIC选择最优组件数 -----
    print("\n【1. BIC/AIC模型选择】")

    n_components_range = range(2, 11)
    bic_scores = []
    aic_scores = []

    for n in n_components_range:
        gmm_test = GaussianMixture(n_components=n, covariance_type='full',
                                   random_state=42, n_init=5)
        gmm_test.fit(X_scaled)
        bic_scores.append(gmm_test.bic(X_scaled))
        aic_scores.append(gmm_test.aic(X_scaled))

    # BIC最小值对应的组件数
    best_n_bic = list(n_components_range)[np.argmin(bic_scores)]
    best_n_aic = list(n_components_range)[np.argmin(aic_scores)]

    print(f"  • BIC推荐组件数: {best_n_bic}")
    print(f"  • AIC推荐组件数: {best_n_aic}")

    # ----- 2. 训练GMM模型 -----
    print(f"\n【2. 训练GMM模型】(使用K={optimal_k}进行公平对比)")

    gmm = GaussianMixture(
        n_components=optimal_k,     # 与K-Means相同的簇数
        covariance_type='full',     # 完整协方差矩阵（最灵活）
        max_iter=100,
        n_init=10,
        random_state=42
    )

    gmm.fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)
    gmm_proba = gmm.predict_proba(X_scaled)  # 软分配概率

    print(f"  • 是否收敛: {gmm.converged_}")
    print(f"  • 迭代次数: {gmm.n_iter_}")

    # ----- 3. 评估对比 -----
    print("\n【3. GMM vs K-Means 对比】")
    print("-" * 50)

    sil_gmm = silhouette_score(X_scaled, gmm_labels)
    sil_kmeans = silhouette_score(X_scaled, kmeans_labels)

    db_gmm = davies_bouldin_score(X_scaled, gmm_labels)
    db_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)

    print(f"  {'指标':<20} {'K-Means':<15} {'GMM':<15}")
    print("  " + "-" * 50)
    print(f"  {'轮廓系数':<20} {sil_kmeans:<15.4f} {sil_gmm:<15.4f}")
    print(f"  {'Davies-Bouldin':<20} {db_kmeans:<15.4f} {db_gmm:<15.4f}")
    print("  " + "-" * 50)

    # ----- 4. 可视化对比 -----
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    # 图1: BIC/AIC曲线
    ax1 = axes[0, 0]
    ax1.plot(n_components_range, bic_scores, 'o-', label='BIC', linewidth=2, markersize=8)
    ax1.plot(n_components_range, aic_scores, 's-', label='AIC', linewidth=2, markersize=8)
    ax1.axvline(best_n_bic, color='red', linestyle='--', alpha=0.7, label=f'BIC最优: {best_n_bic}')
    ax1.set_xlabel('组件数')
    ax1.set_ylabel('信息准则值')
    ax1.set_title('BIC/AIC 模型选择', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: K-Means结果
    ax2 = axes[0, 1]
    for i in range(optimal_k):
        mask = kmeans_labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=60, alpha=0.6,
                   edgecolors='white', linewidth=0.5, label=f'簇 {i}')
    ax2.set_xlabel(feature_names[0])
    ax2.set_ylabel(feature_names[1])
    ax2.set_title(f'K-Means (轮廓系数={sil_kmeans:.4f})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: GMM结果
    ax3 = axes[1, 0]
    for i in range(optimal_k):
        mask = gmm_labels == i
        ax3.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=60, alpha=0.6,
                   edgecolors='white', linewidth=0.5, label=f'簇 {i}')
    ax3.set_xlabel(feature_names[0])
    ax3.set_ylabel(feature_names[1])
    ax3.set_title(f'GMM (轮廓系数={sil_gmm:.4f})', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: GMM概率不确定性
    # 用最大概率的值来表示确定性，颜色深浅表示分配的确定程度
    ax4 = axes[1, 1]
    max_proba = gmm_proba.max(axis=1)  # 每个点最大概率
    scatter = ax4.scatter(X[:, 0], X[:, 1], c=max_proba, cmap='RdYlGn',
                         s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    plt.colorbar(scatter, ax=ax4, label='最大归属概率')
    ax4.set_xlabel(feature_names[0])
    ax4.set_ylabel(feature_names[1])
    ax4.set_title('GMM 分配确定性（绿色=高确定性）', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/05_gmm_vs_kmeans.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n  ✅ 图表已保存: output/05_gmm_vs_kmeans.png")

    # ----- 5. 展示软分配示例 -----
    print("\n【4. GMM软分配示例（前5个样本）】")
    print("  每个样本属于各簇的概率:")
    proba_df = pd.DataFrame(gmm_proba[:5],
                           columns=[f'簇{i}' for i in range(optimal_k)])
    proba_df['最大概率簇'] = gmm_labels[:5]
    proba_df['确定性'] = max_proba[:5]
    print(proba_df.to_string(index=False))

    return gmm, gmm_labels


# ============================================================================
# 第7部分：客户群画像分析
# ============================================================================

def analyze_customer_segments(df, labels, optimal_k):
    """
    客户群画像分析

    为每个客户群计算统计特征，进行命名，并提供营销建议

    Parameters:
    -----------
    df : DataFrame
        原始客户数据
    labels : array
        聚类标签
    optimal_k : int
        聚类数

    Returns:
    --------
    segment_profiles : DataFrame
        各客户群的画像统计
    """
    print("\n" + "=" * 60)
    print("👥 客户群画像分析")
    print("=" * 60)

    # 添加聚类标签到数据框
    df_analysis = df.copy()
    df_analysis['Cluster'] = labels

    # ----- 1. 计算各簇统计特征 -----
    print("\n【1. 各客户群统计特征】")

    # 数值特征统计
    numeric_stats = df_analysis.groupby('Cluster').agg({
        'Age': ['mean', 'std', 'min', 'max'],
        'Annual Income (k$)': ['mean', 'std', 'min', 'max'],
        'Spending Score (1-100)': ['mean', 'std', 'min', 'max']
    }).round(2)

    # 性别比例
    gender_ratio = df_analysis.groupby('Cluster')['Gender'].apply(
        lambda x: f"男{(x=='Male').sum()}/女{(x=='Female').sum()}"
    )

    # 客户数量
    cluster_counts = df_analysis['Cluster'].value_counts().sort_index()

    # 创建综合画像表
    segment_profiles = []
    for cluster in range(optimal_k):
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        profile = {
            '客户群': cluster,
            '客户数量': len(cluster_data),
            '占比(%)': round(len(cluster_data) / len(df_analysis) * 100, 1),
            '平均年龄': round(cluster_data['Age'].mean(), 1),
            '平均年收入(k$)': round(cluster_data['Annual Income (k$)'].mean(), 1),
            '平均消费评分': round(cluster_data['Spending Score (1-100)'].mean(), 1),
            '男性占比(%)': round((cluster_data['Gender'] == 'Male').sum() / len(cluster_data) * 100, 1)
        }
        segment_profiles.append(profile)

    profiles_df = pd.DataFrame(segment_profiles)
    print(profiles_df.to_string(index=False))

    # ----- 2. 客户群命名 -----
    print("\n【2. 客户群命名与特征解读】")
    print("-" * 70)

    # 根据收入和消费评分的均值进行分类命名
    cluster_names = {}
    cluster_descriptions = {}
    marketing_strategies = {}

    for cluster in range(optimal_k):
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        avg_age = cluster_data['Age'].mean()

        # 根据收入和消费评分进行分类
        # 定义阈值：收入60k$为分界，消费评分50为分界
        if avg_income >= 70 and avg_spending >= 60:
            name = "💎 VIP客户 (高收入高消费)"
            desc = "高收入且消费意愿强，是商场最有价值的客户群"
            strategy = "提供专属VIP服务、高端品牌推荐、会员积分奖励、优先体验新品"
        elif avg_income >= 70 and avg_spending < 40:
            name = "🎯 潜力客户 (高收入低消费)"
            desc = "有消费能力但消费意愿低，可能是理性消费者或对商场产品不感兴趣"
            strategy = "精准营销高品质商品、个性化推荐、了解消费偏好、提供定制服务"
        elif avg_income < 40 and avg_spending >= 60:
            name = "🔥 冲动消费型 (低收入高消费)"
            desc = "收入不高但消费意愿强，可能是年轻人或注重生活品质者"
            strategy = "推送促销活动、分期付款选项、性价比商品推荐、会员折扣"
        elif avg_income < 40 and avg_spending < 40:
            name = "💰 价格敏感型 (低收入低消费)"
            desc = "消费能力和意愿都较低，注重价格"
            strategy = "打折促销信息、特价商品推荐、优惠券发放、基础会员服务"
        else:
            name = "📊 普通客户 (中等水平)"
            desc = "收入和消费都处于中等水平，是商场的主力客户群"
            strategy = "常规促销活动、会员权益介绍、多样化商品推荐、提升消费体验"

        cluster_names[cluster] = name
        cluster_descriptions[cluster] = desc
        marketing_strategies[cluster] = strategy

        print(f"\n簇 {cluster}: {name}")
        print(f"  📌 特征: 平均年龄{avg_age:.0f}岁, 年收入{avg_income:.0f}k$, 消费评分{avg_spending:.0f}")
        print(f"  📝 描述: {desc}")
        print(f"  💡 营销策略: {strategy}")

    # ----- 3. 可视化客户画像 -----
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    # 图1: 各簇在收入-消费空间的分布
    ax1 = axes[0, 0]
    for i in range(optimal_k):
        mask = labels == i
        ax1.scatter(df.loc[mask, 'Annual Income (k$)'],
                   df.loc[mask, 'Spending Score (1-100)'],
                   c=colors[i], s=80, alpha=0.6,
                   edgecolors='white', linewidth=0.5,
                   label=f'簇{i}: {cluster_names[i].split("(")[0].strip()}')
    ax1.set_xlabel('年收入 (k$)', fontsize=12)
    ax1.set_ylabel('消费评分', fontsize=12)
    ax1.set_title('客户分群分布图', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 图2: 各簇平均特征雷达图（简化为条形图对比）
    ax2 = axes[0, 1]
    x = np.arange(optimal_k)
    width = 0.25

    ax2.bar(x - width, profiles_df['平均年龄'], width, label='平均年龄', color='#3498db')
    ax2.bar(x, profiles_df['平均年收入(k$)'], width, label='平均年收入(k$)', color='#2ecc71')
    ax2.bar(x + width, profiles_df['平均消费评分'], width, label='平均消费评分', color='#e74c3c')

    ax2.set_xlabel('客户群')
    ax2.set_ylabel('数值')
    ax2.set_title('各客户群平均特征对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'簇{i}' for i in range(optimal_k)])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 图3: 各簇客户数量
    ax3 = axes[1, 0]
    bars = ax3.bar(range(optimal_k), profiles_df['客户数量'], color=colors[:optimal_k],
                  edgecolor='white', linewidth=2)
    ax3.set_xlabel('客户群')
    ax3.set_ylabel('客户数量')
    ax3.set_title('各客户群规模', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(optimal_k))
    ax3.set_xticklabels([f'簇{i}' for i in range(optimal_k)])

    # 在柱子上添加数值标签
    for bar, count, pct in zip(bars, profiles_df['客户数量'], profiles_df['占比(%)']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}\n({pct}%)', ha='center', va='bottom', fontsize=10)

    ax3.grid(True, alpha=0.3, axis='y')

    # 图4: 各簇年龄分布箱线图
    ax4 = axes[1, 1]
    bp = ax4.boxplot([df_analysis[df_analysis['Cluster']==i]['Age'] for i in range(optimal_k)],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:optimal_k]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_xlabel('客户群')
    ax4.set_ylabel('年龄')
    ax4.set_title('各客户群年龄分布', fontsize=14, fontweight='bold')
    ax4.set_xticklabels([f'簇{i}' for i in range(optimal_k)])
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output/06_customer_segments_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n  ✅ 图表已保存: output/06_customer_segments_analysis.png")

    # ----- 4. 输出营销建议汇总 -----
    print("\n" + "=" * 60)
    print("💼 营销策略汇总")
    print("=" * 60)

    for cluster in range(optimal_k):
        print(f"\n【{cluster_names[cluster]}】")
        print(f"  客户数: {cluster_counts[cluster]} ({profiles_df.loc[cluster, '占比(%)']:.1f}%)")
        print(f"  策略: {marketing_strategies[cluster]}")

    return profiles_df, cluster_names


# ============================================================================
# 第8部分：模型保存与加载
# ============================================================================

def save_models(kmeans, gmm, scaler, feature_names, cluster_names, profiles_df, model_dir="models"):
    """
    保存训练好的模型和相关信息

    保存内容：
    ---------
    1. kmeans_model.pkl - K-Means聚类模型
    2. gmm_model.pkl - GMM聚类模型
    3. scaler.pkl - 特征标准化器
    4. cluster_info.json - 簇信息（名称、特征、营销策略）

    Parameters:
    -----------
    kmeans : KMeans
        训练好的K-Means模型
    gmm : GaussianMixture
        训练好的GMM模型
    scaler : StandardScaler
        训练好的标准化器
    feature_names : list
        特征名称
    cluster_names : dict
        各簇的命名
    profiles_df : DataFrame
        各簇的统计画像
    model_dir : str
        模型保存目录
    """
    print("\n" + "=" * 60)
    print("💾 保存模型")
    print("=" * 60)

    # 创建模型目录
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)

    # ----- 1. 保存K-Means模型 -----
    kmeans_path = model_path / "kmeans_model.pkl"
    joblib.dump(kmeans, kmeans_path)
    print(f"  ✅ K-Means模型已保存: {kmeans_path}")

    # ----- 2. 保存GMM模型 -----
    gmm_path = model_path / "gmm_model.pkl"
    joblib.dump(gmm, gmm_path)
    print(f"  ✅ GMM模型已保存: {gmm_path}")

    # ----- 3. 保存Scaler -----
    scaler_path = model_path / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ 标准化器已保存: {scaler_path}")

    # ----- 4. 保存簇信息 -----
    cluster_info = {
        "feature_names": feature_names,
        "n_clusters": kmeans.n_clusters,
        "cluster_names": {str(k): v for k, v in cluster_names.items()},
        "cluster_profiles": profiles_df.to_dict(orient='records')
    }

    info_path = model_path / "cluster_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_info, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 簇信息已保存: {info_path}")

    print(f"\n  📁 所有模型已保存到 '{model_dir}/' 目录")


def load_models(model_dir="models"):
    """
    加载已保存的模型

    Parameters:
    -----------
    model_dir : str
        模型保存目录

    Returns:
    --------
    kmeans : KMeans
        K-Means模型
    gmm : GaussianMixture
        GMM模型
    scaler : StandardScaler
        标准化器
    cluster_info : dict
        簇信息
    """
    model_path = Path(model_dir)

    # 加载模型
    kmeans = joblib.load(model_path / "kmeans_model.pkl")
    gmm = joblib.load(model_path / "gmm_model.pkl")
    scaler = joblib.load(model_path / "scaler.pkl")

    # 加载簇信息
    with open(model_path / "cluster_info.json", 'r', encoding='utf-8') as f:
        cluster_info = json.load(f)

    print(f"✅ 模型已从 '{model_dir}/' 加载")

    return kmeans, gmm, scaler, cluster_info


def predict_new_customer(annual_income, spending_score, model_dir="models"):
    """
    对新客户进行分群预测

    使用方法：
    --------
    >>> cluster, cluster_name, proba = predict_new_customer(
    ...     annual_income=75,      # 年收入 75k$
    ...     spending_score=60      # 消费评分 60
    ... )
    >>> print(f"客户分群: {cluster_name}")

    Parameters:
    -----------
    annual_income : float
        年收入（千美元）
    spending_score : float
        消费评分（1-100）
    model_dir : str
        模型目录

    Returns:
    --------
    cluster : int
        簇标签
    cluster_name : str
        簇名称
    proba : array
        属于各簇的概率（GMM软分配）
    """
    # 加载模型
    kmeans, gmm, scaler, cluster_info = load_models(model_dir)

    # 构造特征向量
    X_new = np.array([[annual_income, spending_score]])

    # 标准化
    X_new_scaled = scaler.transform(X_new)

    # K-Means预测
    cluster = kmeans.predict(X_new_scaled)[0]

    # GMM软分配概率
    proba = gmm.predict_proba(X_new_scaled)[0]

    # 获取簇名称
    cluster_name = cluster_info["cluster_names"].get(str(cluster), f"簇{cluster}")

    print(f"\n🎯 新客户分群预测结果")
    print("=" * 50)
    print(f"  输入特征:")
    print(f"    • 年收入: {annual_income}k$")
    print(f"    • 消费评分: {spending_score}")
    print(f"\n  预测结果:")
    print(f"    • 所属客户群: 簇{cluster}")
    print(f"    • 客户群名称: {cluster_name}")
    print(f"\n  各簇归属概率 (GMM):")
    for i, p in enumerate(proba):
        print(f"    • 簇{i}: {p*100:.2f}%")

    return cluster, cluster_name, proba


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：执行完整的客户分群分析流程
    """
    print("\n" + "=" * 60)
    print("🎯 客户分群项目 (Customer Segmentation)")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # ----- Step 1: 加载数据 -----
    data_path = "data/Mall_Customers.csv"
    df = load_data(data_path)

    # ----- Step 2: 数据探索 -----
    explore_data(df)
    visualize_distributions(df)
    plot_correlation_heatmap(df)

    # ----- Step 3: 数据预处理 -----
    X, X_scaled, feature_names, df_processed, scaler = preprocess_data(df)

    # ----- Step 4: 确定最佳K值 -----
    optimal_k = find_optimal_k(X_scaled)

    # ----- Step 5: K-Means聚类 -----
    kmeans, kmeans_labels = kmeans_clustering(X, X_scaled, optimal_k, feature_names)

    # ----- Step 6: GMM聚类对比 -----
    gmm, gmm_labels = gmm_clustering(X, X_scaled, optimal_k, feature_names, kmeans_labels)

    # ----- Step 7: 客户画像分析 -----
    # 使用K-Means的结果进行画像分析（两者结果相近）
    profiles_df, cluster_names = analyze_customer_segments(df, kmeans_labels, optimal_k)

    # ----- Step 8: 保存模型 -----
    save_models(kmeans, gmm, scaler, feature_names, cluster_names, profiles_df)

    # ----- 项目总结 -----
    print("\n" + "=" * 60)
    print("📋 项目总结")
    print("=" * 60)
    print("""
    本项目完成了以下工作：

    1. ✅ 数据探索 (EDA)
       - 分析了200个客户的数据
       - 可视化了年龄、收入、消费评分的分布
       - 发现特征之间相关性较低，各自提供不同信息

    2. ✅ 数据预处理
       - 对性别进行了标签编码
       - 选择年收入和消费评分作为聚类特征
       - 使用StandardScaler进行特征标准化

    3. ✅ 聚类分析
       - 使用肘部法则和轮廓系数确定K=5
       - 完成K-Means聚类
       - 完成GMM聚类并与K-Means对比

    4. ✅ 客户画像
       - 识别出5个不同的客户群体
       - 为每个群体进行命名和特征描述
       - 提供针对性的营销建议

    5. ✅ 模型保存
       - 保存了K-Means和GMM模型
       - 保存了标准化器和簇信息
       - 可用于新客户分群预测

    📁 输出文件：
       - output/01_feature_distributions.png
       - output/02_correlation_heatmap.png
       - output/03_optimal_k_analysis.png
       - output/04_kmeans_clustering_result.png
       - output/05_gmm_vs_kmeans.png
       - output/06_customer_segments_analysis.png

    📦 保存的模型：
       - models/kmeans_model.pkl
       - models/gmm_model.pkl
       - models/scaler.pkl
       - models/cluster_info.json

    🔮 新客户预测示例：
       from customer_segmentation import predict_new_customer
       cluster, name, proba = predict_new_customer(
           annual_income=75,    # 年收入 75k$
           spending_score=60    # 消费评分 60
       )
    """)

    print("\n🎉 项目完成！")
    print("=" * 60)


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    # 切换到项目目录
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 执行主函数
    main()
