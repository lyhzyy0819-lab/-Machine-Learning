"""
🎯 信用卡欺诈检测项目 (Credit Card Fraud Detection)
===================================================

项目目标：
    使用异常检测算法识别信用卡欺诈交易，处理高度不平衡的数据，
    对比Isolation Forest、One-Class SVM、LOF三种算法的性能。

业务场景：
    信用卡欺诈每年造成数十亿美元损失。通过机器学习实时检测异常交易，
    可以帮助银行及时发现欺诈行为，保护客户资金安全。

数据特点：
    - 高度不平衡：欺诈交易仅占 ~0.2%
    - 特征已脱敏：使用PCA降维保护隐私
    - 真实场景：需要权衡误报和漏报

核心算法：
    - Isolation Forest (隔离森林)：基于决策树的快速异常检测
    - One-Class SVM (单类SVM)：学习正常样本的边界
    - LOF (局部离群因子)：基于密度的异常检测

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
import time

# 数据生成和预处理
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 异常检测算法
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# 降维可视化
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 评估指标
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

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
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# 设置随机种子，保证结果可复现
np.random.seed(42)

print("✅ 库导入完成！")
print("=" * 80)


# ============================================================================
# 第2部分：数据生成（模拟真实欺诈检测场景）
# ============================================================================

def generate_fraud_data(n_samples=50000, fraud_ratio=0.002, n_features=30):
    """
    生成模拟的信用卡交易数据

    说明：
        由于Kaggle数据集需要手动下载，这里使用sklearn生成模拟数据。
        模拟真实场景特点：
        - 高度不平衡（欺诈率 ~0.2%）
        - 特征已PCA降维（模拟隐私保护）
        - 包含交易金额特征

    Parameters:
    -----------
    n_samples : int
        总样本数量
    fraud_ratio : float
        欺诈交易比例（默认0.2%）
    n_features : int
        特征数量（模拟V1-V28 + Time + Amount）

    Returns:
    --------
    df : DataFrame
        包含特征和标签的数据框
    """
    print("\n" + "=" * 80)
    print("🔧 生成模拟信用卡交易数据")
    print("=" * 80)

    # 计算欺诈样本数量
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    print(f"   - 总交易数: {n_samples:,}")
    print(f"   - 正常交易: {n_normal:,} ({(1-fraud_ratio)*100:.2f}%)")
    print(f"   - 欺诈交易: {n_fraud:,} ({fraud_ratio*100:.3f}%)")

    # 使用make_classification生成不平衡数据
    # weights: 控制类别比例
    # n_informative: 有信息的特征数量
    # n_redundant: 冗余特征数量
    # n_clusters_per_class: 每个类的簇数量
    # class_sep: 类别分离度（较大表示更容易区分）
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features - 2,  # 预留Time和Amount特征
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.01,  # 添加少量噪声
        random_state=42,
        class_sep=0.8  # 类别分离度（较难区分，模拟真实场景）
    )

    # 创建DataFrame
    # 特征V1-V28模拟PCA降维后的特征（隐私保护）
    feature_names = [f'V{i}' for i in range(1, n_features - 1)]
    df = pd.DataFrame(X, columns=feature_names)

    # 添加Time特征（距离第一笔交易的秒数）
    # 模拟2天的交易数据
    df['Time'] = np.random.randint(0, 172800, size=n_samples)  # 172800 = 48小时

    # 添加Amount特征（交易金额）
    # 正常交易：平均88美元，标准差250
    # 欺诈交易：金额分布略有不同（通常较小或较大）
    normal_amounts = np.random.gamma(shape=2, scale=44, size=n_normal)
    fraud_amounts = np.concatenate([
        np.random.gamma(shape=1, scale=30, size=n_fraud // 2),  # 小额欺诈
        np.random.gamma(shape=3, scale=100, size=n_fraud - n_fraud // 2)  # 大额欺诈
    ])

    amounts = np.zeros(n_samples)
    amounts[y == 0] = normal_amounts
    amounts[y == 1] = np.random.permutation(fraud_amounts)
    df['Amount'] = amounts

    # 添加标签
    df['Class'] = y

    print(f"\n   ✅ 数据生成完成！")
    print(f"      - 特征数量: {n_features}")
    print(f"      - 数据形状: {df.shape}")
    print(f"      - 欺诈比例: {y.mean():.4f}")

    return df


# ============================================================================
# 第3部分：数据探索分析 (EDA)
# ============================================================================

def explore_data(df):
    """
    探索性数据分析

    目的：
        了解数据的基本情况、类别分布、特征差异等

    Parameters:
    -----------
    df : DataFrame
        交易数据
    """
    print("\n" + "=" * 80)
    print("🔍 数据探索分析 (EDA)")
    print("=" * 80)

    # 1. 基本信息
    print("\n【数据基本信息】")
    print(f"   - 数据形状: {df.shape}")
    print(f"   - 特征数量: {df.shape[1] - 1}")
    print(f"   - 样本数量: {df.shape[0]:,}")
    print(f"   - 缺失值: {df.isnull().sum().sum()}")
    print(f"\n   前5行数据:")
    print(df.head())

    # 2. 类别分布
    print("\n【类别分布】")
    class_counts = df['Class'].value_counts()
    print(f"   - 正常交易 (Class=0): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.3f}%)")
    print(f"   - 欺诈交易 (Class=1): {class_counts[1]:,} ({class_counts[1]/len(df)*100:.3f}%)")
    print(f"   - 不平衡比例: 1:{class_counts[0]//class_counts[1]}")

    # 创建输出目录
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # 3. 可视化类别分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3.1 柱状图
    class_counts.plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('交易类别', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('交易数量', fontsize=12, fontweight='bold')
    axes[0].set_title('交易类别分布（柱状图）', fontsize=13, fontweight='bold')
    axes[0].set_xticklabels(['正常 (0)', '欺诈 (1)'], rotation=0)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, v in enumerate(class_counts):
        axes[0].text(i, v + max(class_counts)*0.02, f'{v:,}\n({v/len(df)*100:.3f}%)',
                    ha='center', fontsize=10, fontweight='bold')

    # 3.2 饼图
    colors = ['steelblue', 'coral']
    explode = (0, 0.1)  # 突出显示欺诈部分
    axes[1].pie(class_counts, labels=['正常交易', '欺诈交易'], autopct='%1.3f%%',
               colors=colors, explode=explode, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title('交易类别分布（饼图）', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '01_class_distribution.png', dpi=150, bbox_inches='tight')
    print("\n   ✅ 类别分布图已保存到: outputs/01_class_distribution.png")
    plt.show()

    # 4. 交易金额分析
    print("\n【交易金额分析】")
    normal_amounts = df[df['Class'] == 0]['Amount']
    fraud_amounts = df[df['Class'] == 1]['Amount']

    print(f"   正常交易金额:")
    print(f"      - 平均值: ${normal_amounts.mean():.2f}")
    print(f"      - 中位数: ${normal_amounts.median():.2f}")
    print(f"      - 标准差: ${normal_amounts.std():.2f}")

    print(f"\n   欺诈交易金额:")
    print(f"      - 平均值: ${fraud_amounts.mean():.2f}")
    print(f"      - 中位数: ${fraud_amounts.median():.2f}")
    print(f"      - 标准差: ${fraud_amounts.std():.2f}")

    # 可视化金额分布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('交易金额分布分析', fontsize=16, fontweight='bold', y=0.995)

    # 4.1 正常交易金额直方图
    axes[0, 0].hist(normal_amounts, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('交易金额 ($)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('频数', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('正常交易金额分布', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(normal_amounts.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: ${normal_amounts.mean():.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 4.2 欺诈交易金额直方图
    axes[0, 1].hist(fraud_amounts, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('交易金额 ($)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('频数', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('欺诈交易金额分布', fontsize=12, fontweight='bold')
    axes[0, 1].axvline(fraud_amounts.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: ${fraud_amounts.mean():.2f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 4.3 箱线图对比
    data_to_plot = [normal_amounts, fraud_amounts]
    bp = axes[1, 0].boxplot(data_to_plot, labels=['正常交易', '欺诈交易'],
                           patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel('交易金额 ($)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('交易金额箱线图对比', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4.4 对数尺度对比
    axes[1, 1].hist([normal_amounts, fraud_amounts], bins=50, label=['正常交易', '欺诈交易'],
                   color=['steelblue', 'coral'], alpha=0.6, edgecolor='black')
    axes[1, 1].set_xlabel('交易金额 ($)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('频数 (对数尺度)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('交易金额分布对比（对数尺度）', fontsize=12, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '02_amount_distribution.png', dpi=150, bbox_inches='tight')
    print("   ✅ 金额分布图已保存到: outputs/02_amount_distribution.png")
    plt.show()

    # 5. 时间分布分析
    print("\n【时间分布分析】")
    normal_times = df[df['Class'] == 0]['Time']
    fraud_times = df[df['Class'] == 1]['Time']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 5.1 正常交易时间分布
    axes[0].hist(normal_times / 3600, bins=48, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('时间（小时）', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('交易数量', fontsize=11, fontweight='bold')
    axes[0].set_title('正常交易时间分布', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 5.2 欺诈交易时间分布
    axes[1].hist(fraud_times / 3600, bins=48, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('时间（小时）', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('交易数量', fontsize=11, fontweight='bold')
    axes[1].set_title('欺诈交易时间分布', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '03_time_distribution.png', dpi=150, bbox_inches='tight')
    print("   ✅ 时间分布图已保存到: outputs/03_time_distribution.png")
    plt.show()


# ============================================================================
# 第4部分：数据预处理
# ============================================================================

def preprocess_data(df):
    """
    数据预处理

    步骤：
        1. 分离特征和标签
        2. 标准化处理（Amount特征通常需要标准化）
        3. 划分训练集和测试集
        4. 提取正常交易（用于One-Class训练）

    Parameters:
    -----------
    df : DataFrame
        原始数据

    Returns:
    --------
    X_train : ndarray
        训练集特征
    X_test : ndarray
        测试集特征
    y_train : ndarray
        训练集标签
    y_test : ndarray
        测试集标签
    X_train_normal : ndarray
        训练集中的正常交易（用于One-Class方法）
    scaler : StandardScaler
        标准化器（用于后续新数据预处理）
    """
    print("\n" + "=" * 80)
    print("🔧 数据预处理")
    print("=" * 80)

    # 1. 分离特征和标签
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    print(f"\n   特征矩阵形状: {X.shape}")
    print(f"   标签向量形状: {y.shape}")

    # 2. 标准化处理
    # 为什么需要标准化？
    # - Amount特征的数值范围与V1-V28不同
    # - SVM等算法对特征尺度敏感
    # - 标准化可以提高模型性能
    print("\n   正在进行标准化处理...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"   ✅ 标准化完成！")
    print(f"      - 特征均值: {X_scaled.mean(axis=0)[:5]}...")  # 显示前5个特征
    print(f"      - 特征标准差: {X_scaled.std(axis=0)[:5]}...")

    # 3. 划分训练集和测试集
    # stratify=y: 保持训练集和测试集的类别比例一致
    # test_size=0.3: 30%作为测试集
    print("\n   正在划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"   ✅ 数据集划分完成！")
    print(f"      - 训练集大小: {X_train.shape[0]:,}")
    print(f"      - 测试集大小: {X_test.shape[0]:,}")
    print(f"      - 训练集欺诈比例: {y_train.mean():.4f}")
    print(f"      - 测试集欺诈比例: {y_test.mean():.4f}")

    # 4. 提取正常交易（用于One-Class方法）
    # One-Class方法的核心思想：
    # - 只使用正常样本学习"正常"的模式
    # - 测试时，不符合正常模式的视为异常
    # - 适合极度不平衡的数据
    X_train_normal = X_train[y_train == 0]

    print(f"\n   提取正常交易用于One-Class训练:")
    print(f"      - 正常交易数量: {X_train_normal.shape[0]:,}")
    print(f"      - 占训练集比例: {X_train_normal.shape[0]/X_train.shape[0]*100:.2f}%")

    return X_train, X_test, y_train, y_test, X_train_normal, scaler


# ============================================================================
# 第5部分：异常检测模型训练
# ============================================================================

def train_isolation_forest(X_train_normal, contamination=0.002):
    """
    训练Isolation Forest模型

    算法原理：
        Isolation Forest（隔离森林）基于以下直觉：
        - 异常点更容易被"隔离"（与其他点分离）
        - 使用随机决策树，异常点通常在树的较浅层就被隔离
        - 正常点需要更深的路径才能被隔离

        算法步骤：
        1. 随机选择特征和分割点构建决策树
        2. 计算每个样本的平均路径长度（从根到叶）
        3. 路径长度短的样本为异常（容易被隔离）

    参数说明：
        - contamination: 训练数据中异常的比例（预期）
          设置为0.002表示预期0.2%的数据为异常
        - n_estimators: 决策树数量，默认100
        - max_samples: 构建每棵树使用的样本数，'auto'表示min(256, n_samples)
        - random_state: 随机种子

    Parameters:
    -----------
    X_train_normal : ndarray
        正常交易数据
    contamination : float
        预期的异常比例

    Returns:
    --------
    model : IsolationForest
        训练好的模型
    train_time : float
        训练时间（秒）
    """
    print("\n" + "=" * 80)
    print("🌲 训练 Isolation Forest 模型")
    print("=" * 80)

    print(f"\n   模型参数:")
    print(f"      - contamination: {contamination} (预期异常比例)")
    print(f"      - n_estimators: 100 (决策树数量)")
    print(f"      - max_samples: 'auto'")

    # 创建模型
    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        max_samples='auto',
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心
    )

    # 训练模型
    print("\n   正在训练模型...")
    start_time = time.time()
    model.fit(X_train_normal)
    train_time = time.time() - start_time

    print(f"   ✅ 模型训练完成！")
    print(f"      - 训练样本数: {X_train_normal.shape[0]:,}")
    print(f"      - 训练时间: {train_time:.3f} 秒")

    return model, train_time


def train_one_class_svm(X_train_normal, nu=0.002):
    """
    训练One-Class SVM模型

    算法原理：
        One-Class SVM（单类支持向量机）通过以下方式检测异常：
        - 在特征空间中找到一个最小的超球面（或超平面）包围正常数据
        - 使用核技巧将数据映射到高维空间
        - 在高维空间中，正常数据更容易被分离

        关键概念：
        - nu参数：控制异常的上界和支持向量的下界
          设置为0.002表示允许最多0.2%的训练数据为异常
        - RBF核：径向基函数核，适合非线性数据

    参数说明：
        - nu: 异常值的上界，取值范围(0, 1]
        - kernel: 核函数类型，'rbf'是常用选择
        - gamma: RBF核的参数，'scale'表示1/(n_features * X.var())

    注意：
        One-Class SVM训练较慢，特别是大数据集
        可能需要几分钟时间

    Parameters:
    -----------
    X_train_normal : ndarray
        正常交易数据
    nu : float
        异常值上界

    Returns:
    --------
    model : OneClassSVM
        训练好的模型
    train_time : float
        训练时间（秒）
    """
    print("\n" + "=" * 80)
    print("🔵 训练 One-Class SVM 模型")
    print("=" * 80)

    print(f"\n   模型参数:")
    print(f"      - nu: {nu} (异常值上界)")
    print(f"      - kernel: 'rbf' (径向基函数核)")
    print(f"      - gamma: 'scale'")

    # 由于One-Class SVM训练较慢，对大数据集进行采样
    max_samples = 10000
    if X_train_normal.shape[0] > max_samples:
        print(f"\n   ⚠️  数据量较大，采样 {max_samples:,} 个样本以加速训练")
        indices = np.random.choice(X_train_normal.shape[0], max_samples, replace=False)
        X_train_sample = X_train_normal[indices]
    else:
        X_train_sample = X_train_normal

    # 创建模型
    model = OneClassSVM(
        nu=nu,
        kernel='rbf',
        gamma='scale'
    )

    # 训练模型
    print("\n   正在训练模型（这可能需要一些时间）...")
    start_time = time.time()
    model.fit(X_train_sample)
    train_time = time.time() - start_time

    print(f"   ✅ 模型训练完成！")
    print(f"      - 训练样本数: {X_train_sample.shape[0]:,}")
    print(f"      - 训练时间: {train_time:.3f} 秒")

    return model, train_time


def train_local_outlier_factor(X_train_normal, contamination=0.002):
    """
    训练Local Outlier Factor模型

    算法原理：
        LOF（局部离群因子）通过比较样本的局部密度来检测异常：
        - 计算每个点与其k近邻的密度
        - 比较该点的密度与其邻居的密度
        - 密度明显低于邻居的点被认为是异常

        核心概念：
        - 局部密度：点到其k近邻的平均距离的倒数
        - LOF值：点的局部密度与其邻居局部密度的比值
        - LOF >> 1: 异常点（密度远低于邻居）
        - LOF ≈ 1: 正常点（密度与邻居相似）

    参数说明：
        - n_neighbors: 考虑的邻居数量，默认20
        - contamination: 数据集中异常的比例
        - novelty: True表示用于新数据检测，False表示用于已有数据

    注意：
        LOF有两种模式：
        - novelty=False: 只能对训练数据打分，不能预测新数据
        - novelty=True: 可以预测新数据（我们使用这种模式）

    Parameters:
    -----------
    X_train_normal : ndarray
        正常交易数据
    contamination : float
        预期的异常比例

    Returns:
    --------
    model : LocalOutlierFactor
        训练好的模型
    train_time : float
        训练时间（秒）
    """
    print("\n" + "=" * 80)
    print("📍 训练 Local Outlier Factor (LOF) 模型")
    print("=" * 80)

    print(f"\n   模型参数:")
    print(f"      - n_neighbors: 20 (邻居数量)")
    print(f"      - contamination: {contamination} (预期异常比例)")
    print(f"      - novelty: True (用于新数据检测)")

    # 创建模型
    # novelty=True 允许模型预测新数据
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=True,  # 重要：允许预测新数据
        n_jobs=-1
    )

    # 训练模型
    print("\n   正在训练模型...")
    start_time = time.time()
    model.fit(X_train_normal)
    train_time = time.time() - start_time

    print(f"   ✅ 模型训练完成！")
    print(f"      - 训练样本数: {X_train_normal.shape[0]:,}")
    print(f"      - 训练时间: {train_time:.3f} 秒")

    return model, train_time


# ============================================================================
# 第6部分：模型评估
# ============================================================================

def evaluate_model(model, model_name, X_test, y_test):
    """
    评估异常检测模型

    评估指标说明：
        - Precision (精确率): 预测为欺诈的交易中，真正欺诈的比例
          公式: TP / (TP + FP)
          重要性: 高精确率减少误报，避免错误冻结正常交易

        - Recall (召回率): 真实欺诈交易中，被检测出的比例
          公式: TP / (TP + FN)
          重要性: 高召回率减少漏报，降低经济损失

        - F1-Score: Precision和Recall的调和平均
          公式: 2 * (Precision * Recall) / (Precision + Recall)
          用途: 综合评估模型性能

        - ROC-AUC: ROC曲线下面积
          用途: 衡量模型区分正负样本的能力
          注意: 对不平衡数据可能不够敏感

        - PR-AUC: Precision-Recall曲线下面积
          用途: 更适合不平衡数据的评估指标

    Parameters:
    -----------
    model : 异常检测模型
        已训练的模型
    model_name : str
        模型名称
    X_test : ndarray
        测试集特征
    y_test : ndarray
        测试集标签

    Returns:
    --------
    metrics : dict
        包含各项评估指标的字典
    y_pred : ndarray
        预测结果 (0或1)
    y_scores : ndarray
        异常分数
    """
    print(f"\n" + "=" * 80)
    print(f"📊 评估 {model_name} 模型")
    print("=" * 80)

    # 1. 预测
    print("\n   正在进行预测...")
    start_time = time.time()

    # predict返回：1表示正常，-1表示异常
    y_pred_raw = model.predict(X_test)

    # 转换为0/1标签：0=正常，1=异常
    y_pred = np.where(y_pred_raw == -1, 1, 0)

    # 获取异常分数
    # decision_function返回：值越小越可能是异常
    y_scores = -model.decision_function(X_test)  # 取负数，使得分数越大越异常

    predict_time = time.time() - start_time
    print(f"   ✅ 预测完成！用时: {predict_time:.3f} 秒")

    # 2. 计算基本指标
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n   【分类指标】")
    print(f"      - Precision (精确率): {precision:.4f}")
    print(f"      - Recall (召回率):    {recall:.4f}")
    print(f"      - F1-Score:          {f1:.4f}")

    # 3. 计算ROC-AUC
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"\n   【ROC指标】")
    print(f"      - ROC-AUC: {roc_auc:.4f}")

    # 4. 计算PR-AUC
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    print(f"\n   【PR指标】")
    print(f"      - PR-AUC (Average Precision): {pr_auc:.4f}")

    # 5. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n   【混淆矩阵】")
    print(f"      真负例 (TN): {tn:,}  |  假正例 (FP): {fp:,}")
    print(f"      假负例 (FN): {fn:,}  |  真正例 (TP): {tp:,}")

    # 6. 检测统计
    n_detected = (y_pred == 1).sum()
    n_actual = (y_test == 1).sum()

    print(f"\n   【检测统计】")
    print(f"      - 实际欺诈数: {n_actual:,}")
    print(f"      - 检测欺诈数: {n_detected:,}")
    print(f"      - 检测率: {tp/n_actual*100:.2f}% (召回率)")
    print(f"      - 误报数: {fp:,}")
    print(f"      - 漏报数: {fn:,}")

    # 7. 组织返回结果
    metrics = {
        'model_name': model_name,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'predict_time': float(predict_time),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'pr_precision': pr_precision.tolist(),
        'pr_recall': pr_recall.tolist()
    }

    return metrics, y_pred, y_scores


# ============================================================================
# 第7部分：可视化
# ============================================================================

def plot_confusion_matrices(all_metrics):
    """
    绘制三个模型的混淆矩阵对比

    Parameters:
    -----------
    all_metrics : list of dict
        包含所有模型评估结果的列表
    """
    print("\n" + "=" * 80)
    print("📊 绘制混淆矩阵对比")
    print("=" * 80)

    output_dir = Path('outputs')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('三种异常检测算法的混淆矩阵对比', fontsize=16, fontweight='bold', y=1.02)

    for i, metrics in enumerate(all_metrics):
        cm = np.array(metrics['confusion_matrix'])
        model_name = metrics['model_name']

        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   cbar=True, square=True, linewidths=2, linecolor='black',
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})

        axes[i].set_xlabel('预测标签', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('真实标签', fontsize=11, fontweight='bold')
        axes[i].set_title(f'{model_name}\n(F1={metrics["f1_score"]:.4f})',
                         fontsize=12, fontweight='bold')
        axes[i].set_xticklabels(['正常', '欺诈'])
        axes[i].set_yticklabels(['正常', '欺诈'])

    plt.tight_layout()
    plt.savefig(output_dir / '04_confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("   ✅ 混淆矩阵已保存到: outputs/04_confusion_matrices.png")
    plt.show()


def plot_roc_curves(all_metrics):
    """
    绘制ROC曲线对比

    ROC曲线说明：
        - 横轴：假正例率 (FPR) = FP / (FP + TN)
        - 纵轴：真正例率 (TPR) = TP / (TP + FN) = Recall
        - 对角线：随机猜测的性能
        - 曲线越靠近左上角，模型性能越好
        - AUC（曲线下面积）：0.5=随机，1.0=完美

    Parameters:
    -----------
    all_metrics : list of dict
        包含所有模型评估结果的列表
    """
    print("\n【绘制ROC曲线】")

    output_dir = Path('outputs')

    plt.figure(figsize=(10, 8))

    colors = ['steelblue', 'coral', 'green']

    for i, metrics in enumerate(all_metrics):
        fpr = np.array(metrics['fpr'])
        tpr = np.array(metrics['tpr'])
        roc_auc = metrics['roc_auc']
        model_name = metrics['model_name']

        plt.plot(fpr, tpr, color=colors[i], linewidth=2.5,
                label=f'{model_name} (AUC = {roc_auc:.4f})')

    # 绘制对角线（随机猜测）
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='随机猜测 (AUC = 0.5000)')

    plt.xlabel('假正例率 (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('真正例率 (TPR / Recall)', fontsize=12, fontweight='bold')
    plt.title('ROC曲线对比', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / '05_roc_curves.png', dpi=150, bbox_inches='tight')
    print("   ✅ ROC曲线已保存到: outputs/05_roc_curves.png")
    plt.show()


def plot_pr_curves(all_metrics):
    """
    绘制Precision-Recall曲线对比

    PR曲线说明：
        - 横轴：Recall (召回率) = TP / (TP + FN)
        - 纵轴：Precision (精确率) = TP / (TP + FP)
        - 对于不平衡数据，PR曲线比ROC曲线更有意义
        - 曲线越靠近右上角，模型性能越好
        - PR-AUC（平均精确率）：越接近1越好

    为什么PR曲线更适合不平衡数据？
        - ROC曲线中的FPR在负样本极多时变化不敏感
        - PR曲线直接关注正样本的检测质量
        - 更符合欺诈检测的业务需求

    Parameters:
    -----------
    all_metrics : list of dict
        包含所有模型评估结果的列表
    """
    print("\n【绘制PR曲线】")

    output_dir = Path('outputs')

    plt.figure(figsize=(10, 8))

    colors = ['steelblue', 'coral', 'green']

    for i, metrics in enumerate(all_metrics):
        pr_recall = np.array(metrics['pr_recall'])
        pr_precision = np.array(metrics['pr_precision'])
        pr_auc = metrics['pr_auc']
        model_name = metrics['model_name']

        plt.plot(pr_recall, pr_precision, color=colors[i], linewidth=2.5,
                label=f'{model_name} (AP = {pr_auc:.4f})')

    # 绘制基准线（随机猜测）
    # 对于不平衡数据，随机猜测的PR = 正样本比例
    baseline = all_metrics[0]['true_positives'] + all_metrics[0]['false_negatives']
    total = baseline + all_metrics[0]['true_negatives'] + all_metrics[0]['false_positives']
    baseline_precision = baseline / total

    plt.axhline(y=baseline_precision, color='k', linestyle='--', linewidth=2,
               label=f'随机猜测 (AP = {baseline_precision:.4f})')

    plt.xlabel('召回率 (Recall)', fontsize=12, fontweight='bold')
    plt.ylabel('精确率 (Precision)', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall曲线对比（更适合不平衡数据）', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / '06_pr_curves.png', dpi=150, bbox_inches='tight')
    print("   ✅ PR曲线已保存到: outputs/06_pr_curves.png")
    plt.show()


def plot_metrics_comparison(all_metrics):
    """
    绘制模型性能指标对比柱状图

    Parameters:
    -----------
    all_metrics : list of dict
        包含所有模型评估结果的列表
    """
    print("\n【绘制性能指标对比】")

    output_dir = Path('outputs')

    # 提取指标
    model_names = [m['model_name'] for m in all_metrics]
    precisions = [m['precision'] for m in all_metrics]
    recalls = [m['recall'] for m in all_metrics]
    f1_scores = [m['f1_score'] for m in all_metrics]
    pr_aucs = [m['pr_auc'] for m in all_metrics]

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('异常检测模型性能指标对比', fontsize=16, fontweight='bold', y=0.995)

    x = np.arange(len(model_names))
    width = 0.6

    # 1. Precision对比
    bars1 = axes[0, 0].bar(x, precisions, width, color='steelblue', alpha=0.8, edgecolor='black')
    axes[0, 0].set_ylabel('Precision (精确率)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('精确率对比（减少误报）', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(precisions):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

    # 2. Recall对比
    bars2 = axes[0, 1].bar(x, recalls, width, color='coral', alpha=0.8, edgecolor='black')
    axes[0, 1].set_ylabel('Recall (召回率)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('召回率对比（减少漏报）', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(recalls):
        axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

    # 3. F1-Score对比
    bars3 = axes[1, 0].bar(x, f1_scores, width, color='green', alpha=0.8, edgecolor='black')
    axes[1, 0].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('F1-Score对比（综合指标）', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(f1_scores):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

    # 4. PR-AUC对比
    bars4 = axes[1, 1].bar(x, pr_aucs, width, color='purple', alpha=0.8, edgecolor='black')
    axes[1, 1].set_ylabel('PR-AUC', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('PR-AUC对比（不平衡数据推荐指标）', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1])
    for i, v in enumerate(pr_aucs):
        axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '07_metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✅ 指标对比图已保存到: outputs/07_metrics_comparison.png")
    plt.show()


def visualize_anomalies_tsne(X_test, y_test, y_pred_if, y_pred_svm, y_pred_lof):
    """
    使用t-SNE降维可视化异常检测结果

    t-SNE说明：
        - t-SNE (t-Distributed Stochastic Neighbor Embedding)
        - 将高维数据降维到2D/3D用于可视化
        - 保持数据点之间的相似性关系
        - 相似的点在低维空间中也会靠近

    Parameters:
    -----------
    X_test : ndarray
        测试集特征
    y_test : ndarray
        真实标签
    y_pred_if : ndarray
        Isolation Forest预测结果
    y_pred_svm : ndarray
        One-Class SVM预测结果
    y_pred_lof : ndarray
        LOF预测结果
    """
    print("\n" + "=" * 80)
    print("📊 使用t-SNE可视化异常检测结果")
    print("=" * 80)

    output_dir = Path('outputs')

    # 使用较小的数据集进行t-SNE（t-SNE计算较慢）
    max_samples = 5000
    if X_test.shape[0] > max_samples:
        print(f"   采样 {max_samples:,} 个样本进行可视化...")
        indices = np.random.choice(X_test.shape[0], max_samples, replace=False)
        X_sample = X_test[indices]
        y_sample = y_test[indices]
        y_if_sample = y_pred_if[indices]
        y_svm_sample = y_pred_svm[indices]
        y_lof_sample = y_pred_lof[indices]
    else:
        X_sample = X_test
        y_sample = y_test
        y_if_sample = y_pred_if
        y_svm_sample = y_pred_svm
        y_lof_sample = y_pred_lof

    # 应用t-SNE降维
    print("\n   正在进行t-SNE降维（这可能需要一些时间）...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)

    print("   ✅ t-SNE降维完成！")

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('异常检测结果可视化 (t-SNE降维)', fontsize=16, fontweight='bold', y=0.995)

    # 1. 真实标签
    scatter1 = axes[0, 0].scatter(X_tsne[y_sample == 0, 0], X_tsne[y_sample == 0, 1],
                                  c='steelblue', s=20, alpha=0.5, label='正常交易', edgecolors='none')
    scatter2 = axes[0, 0].scatter(X_tsne[y_sample == 1, 0], X_tsne[y_sample == 1, 1],
                                  c='red', s=50, alpha=0.8, label='欺诈交易', marker='X', edgecolors='black')
    axes[0, 0].set_title('真实标签', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('t-SNE 维度 1', fontsize=11)
    axes[0, 0].set_ylabel('t-SNE 维度 2', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Isolation Forest预测
    axes[0, 1].scatter(X_tsne[y_if_sample == 0, 0], X_tsne[y_if_sample == 0, 1],
                      c='lightblue', s=20, alpha=0.5, label='预测正常', edgecolors='none')
    axes[0, 1].scatter(X_tsne[y_if_sample == 1, 0], X_tsne[y_if_sample == 1, 1],
                      c='orange', s=50, alpha=0.8, label='预测欺诈', marker='X', edgecolors='black')
    # 标记真实欺诈
    axes[0, 1].scatter(X_tsne[y_sample == 1, 0], X_tsne[y_sample == 1, 1],
                      c='none', s=80, marker='o', edgecolors='red', linewidths=2, label='真实欺诈')
    axes[0, 1].set_title('Isolation Forest 预测', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('t-SNE 维度 1', fontsize=11)
    axes[0, 1].set_ylabel('t-SNE 维度 2', fontsize=11)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. One-Class SVM预测
    axes[1, 0].scatter(X_tsne[y_svm_sample == 0, 0], X_tsne[y_svm_sample == 0, 1],
                      c='lightblue', s=20, alpha=0.5, label='预测正常', edgecolors='none')
    axes[1, 0].scatter(X_tsne[y_svm_sample == 1, 0], X_tsne[y_svm_sample == 1, 1],
                      c='orange', s=50, alpha=0.8, label='预测欺诈', marker='X', edgecolors='black')
    axes[1, 0].scatter(X_tsne[y_sample == 1, 0], X_tsne[y_sample == 1, 1],
                      c='none', s=80, marker='o', edgecolors='red', linewidths=2, label='真实欺诈')
    axes[1, 0].set_title('One-Class SVM 预测', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('t-SNE 维度 1', fontsize=11)
    axes[1, 0].set_ylabel('t-SNE 维度 2', fontsize=11)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. LOF预测
    axes[1, 1].scatter(X_tsne[y_lof_sample == 0, 0], X_tsne[y_lof_sample == 0, 1],
                      c='lightblue', s=20, alpha=0.5, label='预测正常', edgecolors='none')
    axes[1, 1].scatter(X_tsne[y_lof_sample == 1, 0], X_tsne[y_lof_sample == 1, 1],
                      c='orange', s=50, alpha=0.8, label='预测欺诈', marker='X', edgecolors='black')
    axes[1, 1].scatter(X_tsne[y_sample == 1, 0], X_tsne[y_sample == 1, 1],
                      c='none', s=80, marker='o', edgecolors='red', linewidths=2, label='真实欺诈')
    axes[1, 1].set_title('Local Outlier Factor 预测', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('t-SNE 维度 1', fontsize=11)
    axes[1, 1].set_ylabel('t-SNE 维度 2', fontsize=11)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '08_tsne_visualization.png', dpi=150, bbox_inches='tight')
    print("   ✅ t-SNE可视化已保存到: outputs/08_tsne_visualization.png")
    plt.show()


# ============================================================================
# 第8部分：模型保存
# ============================================================================

def save_models_and_metrics(models, all_metrics, scaler):
    """
    保存训练好的模型和评估指标

    Parameters:
    -----------
    models : dict
        包含所有模型的字典
    all_metrics : list of dict
        所有模型的评估指标
    scaler : StandardScaler
        数据标准化器
    """
    print("\n" + "=" * 80)
    print("💾 保存模型和评估指标")
    print("=" * 80)

    # 创建模型目录
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    # 保存模型
    for name, model in models.items():
        model_path = models_dir / f'{name.lower().replace(" ", "_")}_model.pkl'
        joblib.dump(model, model_path)
        print(f"   ✅ 已保存模型: {model_path}")

    # 保存标准化器
    scaler_path = models_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ 已保存标准化器: {scaler_path}")

    # 保存评估指标（简化版，去除大数组）
    metrics_simplified = []
    for m in all_metrics:
        metrics_simple = {
            'model_name': m['model_name'],
            'precision': m['precision'],
            'recall': m['recall'],
            'f1_score': m['f1_score'],
            'roc_auc': m['roc_auc'],
            'pr_auc': m['pr_auc'],
            'confusion_matrix': m['confusion_matrix'],
            'true_positives': m['true_positives'],
            'false_positives': m['false_positives'],
            'true_negatives': m['true_negatives'],
            'false_negatives': m['false_negatives'],
            'predict_time': m['predict_time']
        }
        metrics_simplified.append(metrics_simple)

    metrics_path = models_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_simplified, f, indent=4, ensure_ascii=False)

    print(f"   ✅ 已保存评估指标: {metrics_path}")
    print("\n" + "=" * 80)


# ============================================================================
# 第9部分：主函数
# ============================================================================

def main():
    """
    主函数：执行完整的欺诈检测流程
    """
    print("\n" + "=" * 80)
    print("🚀 信用卡欺诈检测项目开始")
    print("=" * 80)

    # Step 1: 生成数据
    df = generate_fraud_data(n_samples=50000, fraud_ratio=0.002, n_features=30)

    # Step 2: 数据探索
    explore_data(df)

    # Step 3: 数据预处理
    X_train, X_test, y_train, y_test, X_train_normal, scaler = preprocess_data(df)

    # Step 4: 训练三种异常检测模型
    contamination = 0.002  # 预期异常比例

    # 4.1 Isolation Forest
    model_if, train_time_if = train_isolation_forest(X_train_normal, contamination)

    # 4.2 One-Class SVM
    model_svm, train_time_svm = train_one_class_svm(X_train_normal, nu=contamination)

    # 4.3 Local Outlier Factor
    model_lof, train_time_lof = train_local_outlier_factor(X_train_normal, contamination)

    # Step 5: 评估模型
    metrics_if, y_pred_if, y_scores_if = evaluate_model(model_if, "Isolation Forest", X_test, y_test)
    metrics_svm, y_pred_svm, y_scores_svm = evaluate_model(model_svm, "One-Class SVM", X_test, y_test)
    metrics_lof, y_pred_lof, y_scores_lof = evaluate_model(model_lof, "Local Outlier Factor", X_test, y_test)

    all_metrics = [metrics_if, metrics_svm, metrics_lof]

    # Step 6: 可视化对比
    print("\n" + "=" * 80)
    print("📊 生成可视化对比图")
    print("=" * 80)

    plot_confusion_matrices(all_metrics)
    plot_roc_curves(all_metrics)
    plot_pr_curves(all_metrics)
    plot_metrics_comparison(all_metrics)
    visualize_anomalies_tsne(X_test, y_test, y_pred_if, y_pred_svm, y_pred_lof)

    # Step 7: 保存模型
    models = {
        'Isolation Forest': model_if,
        'One-Class SVM': model_svm,
        'Local Outlier Factor': model_lof
    }
    save_models_and_metrics(models, all_metrics, scaler)

    # Step 8: 生成总结报告
    print("\n" + "=" * 80)
    print("📊 项目总结报告")
    print("=" * 80)

    print("\n【模型性能对比】")
    print(f"{'模型':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'PR-AUC':<12}")
    print("-" * 68)
    for m in all_metrics:
        print(f"{m['model_name']:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1_score']:<12.4f} {m['pr_auc']:<12.4f}")

    # 找出最佳模型
    best_f1_model = max(all_metrics, key=lambda x: x['f1_score'])
    best_recall_model = max(all_metrics, key=lambda x: x['recall'])
    best_precision_model = max(all_metrics, key=lambda x: x['precision'])

    print("\n【最佳模型】")
    print(f"   - 最高F1-Score: {best_f1_model['model_name']} ({best_f1_model['f1_score']:.4f})")
    print(f"   - 最高Recall: {best_recall_model['model_name']} ({best_recall_model['recall']:.4f})")
    print(f"   - 最高Precision: {best_precision_model['model_name']} ({best_precision_model['precision']:.4f})")

    print("\n【业务建议】")
    print("   1. 模型选择:")
    print(f"      - 如果关注减少漏报（抓住更多欺诈）:")
    print(f"        推荐使用 {best_recall_model['model_name']}")
    print(f"      - 如果关注减少误报（避免误伤正常用户）:")
    print(f"        推荐使用 {best_precision_model['model_name']}")
    print(f"      - 综合考虑:")
    print(f"        推荐使用 {best_f1_model['model_name']}")

    print("\n   2. 阈值调整:")
    print("      - 可以通过调整decision_function的阈值来权衡Precision和Recall")
    print("      - 降低阈值：提高Recall，降低Precision（抓更多欺诈，但误报增加）")
    print("      - 提高阈值：提高Precision，降低Recall（减少误报，但漏报增加）")

    print("\n   3. 实际部署:")
    print("      - Isolation Forest: 训练快，推理快，适合大规模实时检测")
    print("      - One-Class SVM: 准确度高，但计算较慢，适合批量检测")
    print("      - LOF: 适合检测局部异常，但不适合大规模数据")

    print("\n   4. 改进方向:")
    print("      - 特征工程：从Amount、Time提取更多特征")
    print("      - 集成方法：结合多个模型的预测结果")
    print("      - 半监督学习：利用少量标注的欺诈样本")
    print("      - 在线学习：随着新数据不断更新模型")

    print("\n" + "=" * 80)
    print("✅ 信用卡欺诈检测项目完成！")
    print("=" * 80)

    print("\n📁 输出文件:")
    print("   - outputs/01_class_distribution.png       # 类别分布")
    print("   - outputs/02_amount_distribution.png      # 金额分布")
    print("   - outputs/03_time_distribution.png        # 时间分布")
    print("   - outputs/04_confusion_matrices.png       # 混淆矩阵")
    print("   - outputs/05_roc_curves.png              # ROC曲线")
    print("   - outputs/06_pr_curves.png               # PR曲线")
    print("   - outputs/07_metrics_comparison.png       # 指标对比")
    print("   - outputs/08_tsne_visualization.png       # t-SNE可视化")
    print("   - models/isolation_forest_model.pkl       # IF模型")
    print("   - models/one-class_svm_model.pkl          # SVM模型")
    print("   - models/local_outlier_factor_model.pkl   # LOF模型")
    print("   - models/scaler.pkl                       # 标准化器")
    print("   - models/evaluation_metrics.json          # 评估指标")
    print("\n" + "=" * 80)


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    main()
