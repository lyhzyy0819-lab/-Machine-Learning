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

# 数据预处理
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
# 第2部分：加载真实信用卡欺诈检测数据集
# ============================================================================

def load_credit_card_data(file_path=None):
    """
    加载 Kaggle 信用卡欺诈检测数据集

    数据集来源：
        https://www.kaggle.com/mlg-ulb/creditcardfraud

    数据集说明：
        该数据集包含2013年9月欧洲持卡人的信用卡交易记录。
        数据集经过PCA降维处理以保护用户隐私。

    数据特点：
        - 284,807 笔交易记录
        - 492 笔欺诈交易（占比约 0.172%）
        - 高度不平衡数据集（正常:欺诈 ≈ 578:1）

    特征说明：
        - Time: 该交易距离数据集第一笔交易的秒数
                可用于分析交易时间模式
        - V1-V28: PCA降维后的特征（原始特征因隐私保护不可知）
                  这些特征已经过标准化处理
        - Amount: 交易金额（未经标准化）
        - Class: 标签，0=正常交易，1=欺诈交易

    Parameters:
    -----------
    file_path : str, optional
        CSV文件路径。如果为None，则使用默认路径 'creditcard.csv'

    Returns:
    --------
    df : DataFrame
        包含所有特征和标签的数据框

    Raises:
    -------
    FileNotFoundError
        如果数据集文件不存在
    """
    print("\n" + "=" * 80)
    print("📂 加载 Kaggle 信用卡欺诈检测数据集")
    print("=" * 80)

    # 确定文件路径
    if file_path is None:
        # 默认在当前目录下查找
        file_path = Path(__file__).parent / 'creditcard.csv'

    file_path = Path(file_path)

    # 检查文件是否存在
    if not file_path.exists():
        raise FileNotFoundError(
            f"\n❌ 数据集文件未找到: {file_path}\n"
            f"   请从 Kaggle 下载数据集:\n"
            f"   https://www.kaggle.com/mlg-ulb/creditcardfraud\n"
            f"   并将 creditcard.csv 放置在当前目录下。"
        )

    print(f"\n   数据集路径: {file_path}")
    print(f"   正在加载数据...")

    # 读取CSV文件
    # 注意：Class列在CSV中可能是字符串类型，需要转换
    df = pd.read_csv(file_path)

    # 确保Class列是整数类型
    # Kaggle数据集中Class列可能是"0"/"1"字符串
    df['Class'] = df['Class'].astype(int)

    # 统计基本信息
    n_samples = len(df)
    n_features = df.shape[1] - 1  # 减去Class列
    n_fraud = df['Class'].sum()
    n_normal = n_samples - n_fraud
    fraud_ratio = n_fraud / n_samples

    print(f"\n   ✅ 数据加载完成！")
    print(f"\n   【数据集统计】")
    print(f"      - 总交易数: {n_samples:,}")
    print(f"      - 正常交易: {n_normal:,} ({(1-fraud_ratio)*100:.3f}%)")
    print(f"      - 欺诈交易: {n_fraud:,} ({fraud_ratio*100:.3f}%)")
    print(f"      - 不平衡比例: 1:{n_normal//n_fraud}")
    print(f"      - 特征数量: {n_features}")

    print(f"\n   【特征列表】")
    print(f"      - Time: 距离第一笔交易的秒数")
    print(f"      - V1-V28: PCA降维后的匿名特征")
    print(f"      - Amount: 交易金额")
    print(f"      - Class: 标签 (0=正常, 1=欺诈)")

    # 显示数据集的基本统计
    print(f"\n   【数值统计摘要】")
    print(f"      Time范围: {df['Time'].min():.0f} - {df['Time'].max():.0f} 秒")
    print(f"      Time跨度: {df['Time'].max()/3600:.1f} 小时")
    print(f"      Amount范围: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
    print(f"      Amount均值: ${df['Amount'].mean():.2f}")
    print(f"      Amount中位数: ${df['Amount'].median():.2f}")

    return df


# ============================================================================
# 第3部分：数据探索分析 (EDA)
# ============================================================================

def explore_data(df):
    """
    探索性数据分析

    目的：
        了解数据的基本情况、类别分布、特征差异等
        针对真实 Kaggle 数据集进行全面的 EDA

    分析内容：
        1. 数据基本信息（形状、缺失值、数据类型）
        2. 类别分布分析（正常 vs 欺诈）
        3. 交易金额分析（正常 vs 欺诈的金额差异）
        4. 时间分布分析（交易时间模式）
        5. V1-V28 特征分析（PCA特征的分布差异）

    Parameters:
    -----------
    df : DataFrame
        交易数据（包含 Time, V1-V28, Amount, Class 列）
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
    print(f"   - 数据类型:")
    for col in ['Time', 'Amount', 'Class']:
        print(f"      {col}: {df[col].dtype}")
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

    # 6. V1-V28 PCA特征分析
    # 这些特征是原始交易特征经过PCA降维后的结果
    # 分析正常交易和欺诈交易在这些特征上的差异
    print("\n【V1-V28 PCA特征分析】")
    print("   说明: V1-V28是原始特征经PCA降维后的匿名特征")
    print("   分析正常交易与欺诈交易在这些特征上的分布差异...")

    # 获取V1-V28特征列
    v_features = [f'V{i}' for i in range(1, 29)]

    # 计算正常和欺诈交易的特征均值差异
    normal_means = df[df['Class'] == 0][v_features].mean()
    fraud_means = df[df['Class'] == 1][v_features].mean()
    mean_diff = fraud_means - normal_means

    # 找出差异最大的特征（对欺诈检测最有价值的特征）
    abs_diff = mean_diff.abs().sort_values(ascending=False)
    top_features = abs_diff.head(10).index.tolist()

    print(f"\n   欺诈交易与正常交易均值差异最大的10个特征:")
    for i, feat in enumerate(top_features, 1):
        diff = mean_diff[feat]
        direction = "↑ 欺诈更高" if diff > 0 else "↓ 欺诈更低"
        print(f"      {i}. {feat}: 差异 {diff:+.4f} ({direction})")

    # 可视化差异最大的4个特征分布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('欺诈交易与正常交易的PCA特征分布对比（差异最大的4个特征）',
                 fontsize=14, fontweight='bold', y=0.995)

    for idx, feat in enumerate(top_features[:4]):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        # 绘制正常交易的分布
        normal_data = df[df['Class'] == 0][feat]
        fraud_data = df[df['Class'] == 1][feat]

        ax.hist(normal_data, bins=50, alpha=0.6, color='steelblue',
                label=f'正常 (n={len(normal_data):,})', density=True, edgecolor='none')
        ax.hist(fraud_data, bins=50, alpha=0.7, color='coral',
                label=f'欺诈 (n={len(fraud_data):,})', density=True, edgecolor='none')

        # 添加均值线
        ax.axvline(normal_data.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'正常均值: {normal_data.mean():.2f}')
        ax.axvline(fraud_data.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'欺诈均值: {fraud_data.mean():.2f}')

        ax.set_xlabel(f'{feat} 值', fontsize=11, fontweight='bold')
        ax.set_ylabel('密度', fontsize=11, fontweight='bold')
        ax.set_title(f'{feat} 特征分布对比', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '03b_feature_distribution.png', dpi=150, bbox_inches='tight')
    print("\n   ✅ 特征分布图已保存到: outputs/03b_feature_distribution.png")
    plt.show()

    # 7. 特征相关性热力图（仅展示与Class相关性较高的特征）
    print("\n【特征与欺诈标签的相关性分析】")

    # 计算所有特征与Class的相关性
    correlations = df.corr()['Class'].drop('Class').sort_values(key=abs, ascending=False)

    print(f"\n   与欺诈标签相关性最高的10个特征:")
    for i, (feat, corr) in enumerate(correlations.head(10).items(), 1):
        direction = "正相关" if corr > 0 else "负相关"
        print(f"      {i}. {feat}: {corr:+.4f} ({direction})")

    # 绘制相关性条形图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 选择相关性绝对值最高的15个特征
    top_corr = correlations.head(15)
    colors = ['coral' if x > 0 else 'steelblue' for x in top_corr.values]

    bars = ax.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr.index)
    ax.set_xlabel('与欺诈标签的相关系数', fontsize=12, fontweight='bold')
    ax.set_title('特征与欺诈标签(Class)的相关性排名', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for bar, val in zip(bars, top_corr.values):
        ax.text(val + 0.01 if val > 0 else val - 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                va='center', ha='left' if val > 0 else 'right',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '03c_feature_correlation.png', dpi=150, bbox_inches='tight')
    print("   ✅ 相关性分析图已保存到: outputs/03c_feature_correlation.png")
    plt.show()


# ============================================================================
# 第4部分：数据预处理
# ============================================================================

def preprocess_data(df):
    """
    数据预处理

    针对 Kaggle 信用卡欺诈检测数据集的预处理步骤：

    步骤：
        1. 分离特征和标签
           - 特征: Time, V1-V28, Amount（共30维）
           - 标签: Class（0=正常, 1=欺诈）

        2. 标准化处理
           - V1-V28 已经过 PCA 处理，但仍需标准化以统一尺度
           - Time 和 Amount 特征尤其需要标准化
           - 使用 StandardScaler: (x - mean) / std

        3. 划分训练集和测试集
           - 使用分层采样保持类别比例一致
           - 70% 训练，30% 测试

        4. 提取正常交易样本
           - 用于 One-Class 方法（只用正常样本训练）

    Parameters:
    -----------
    df : DataFrame
        原始数据（包含 Time, V1-V28, Amount, Class 列）

    Returns:
    --------
    X_train : ndarray
        训练集特征，形状 (n_train_samples, 30)
    X_test : ndarray
        测试集特征，形状 (n_test_samples, 30)
    y_train : ndarray
        训练集标签，形状 (n_train_samples,)
    y_test : ndarray
        测试集标签，形状 (n_test_samples,)
    X_train_normal : ndarray
        训练集中的正常交易（用于 One-Class 方法）
    scaler : StandardScaler
        标准化器（用于后续新数据预处理）
    """
    print("\n" + "=" * 80)
    print("🔧 数据预处理")
    print("=" * 80)

    # 1. 分离特征和标签
    # 特征列: Time, V1-V28, Amount
    # 标签列: Class
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    print(f"\n   【数据分离】")
    print(f"      - 特征矩阵形状: {X.shape}")
    print(f"      - 标签向量形状: {y.shape}")
    print(f"      - 特征列: Time, V1-V28, Amount")

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
# 第8.5部分：训练模式对比实验（半监督 vs 无监督）
# ============================================================================

def compare_training_modes(X_train, X_train_normal, X_test, y_train, y_test, contamination):
    """
    对比实验：半监督模式 vs 无监督模式

    核心概念说明：
    ==============

    异常检测算法有两种主要的训练模式，理解它们的区别对于选择正确的方法至关重要：

    【半监督模式 (Semi-supervised)】
    --------------------------------
    - 训练数据：只使用正常样本 (X_train_normal)
    - 核心思想：学习"什么是正常"，然后将不符合正常模式的判定为异常
    - 适用场景：
        * 有标签数据，知道哪些是正常样本
        * 正常样本数量充足
        * 异常样本太少不足以学习异常模式
    - contamination 参数含义：预测时期望的异常比例（影响决策阈值）

    【无监督模式 (Unsupervised)】
    ----------------------------
    - 训练数据：使用全部数据 (X_train)，包含少量异常
    - 核心思想：假设数据中有 contamination 比例的异常，找出最"不正常"的那部分
    - 适用场景：
        * 无标签数据
        * 不确定哪些是正常样本
        * 真实生产环境中常见
    - contamination 参数含义：训练数据中实际的异常比例

    【为什么要对比？】
    -----------------
    在这个信用卡欺诈检测数据集中：
    - 我们有标签，可以使用半监督模式
    - 但真实业务中往往没有完整标签，需要用无监督模式
    - 对比两种模式可以帮助理解各自的优缺点

    Parameters:
    -----------
    X_train : ndarray
        完整训练集（包含正常+欺诈样本）
    X_train_normal : ndarray
        仅包含正常样本的训练集
    X_test : ndarray
        测试集
    y_train : ndarray
        训练集标签
    y_test : ndarray
        测试集标签
    contamination : float
        异常比例

    Returns:
    --------
    comparison_results : dict
        包含两种模式的对比结果
    """
    print("\n" + "=" * 80)
    print("🔬 训练模式对比实验：半监督 vs 无监督")
    print("=" * 80)

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  本实验对比 Isolation Forest 在两种训练模式下的性能差异              │
    ├─────────────────────────────────────────────────────────────────────┤
    │  半监督模式：只用正常样本训练 → 学习"正常是什么样"                   │
    │  无监督模式：用全部数据训练   → 找出"最不正常的那部分"               │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    output_dir = Path('outputs')
    results = {}

    # =========================================================================
    # 实验1：半监督模式（只用正常样本训练）
    # =========================================================================
    print("\n【实验1】半监督模式 (Semi-supervised)")
    print("-" * 60)
    print(f"   训练数据：仅正常样本")
    print(f"   训练样本数：{X_train_normal.shape[0]:,}")
    print(f"   contamination：{contamination:.5f}（作为预测阈值）")

    # 训练半监督模型
    model_semi = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    model_semi.fit(X_train_normal)  # 只用正常样本
    train_time_semi = time.time() - start_time

    # 预测
    y_pred_semi_raw = model_semi.predict(X_test)
    y_pred_semi = np.where(y_pred_semi_raw == -1, 1, 0)
    y_scores_semi = -model_semi.decision_function(X_test)

    # 评估
    precision_semi = precision_score(y_test, y_pred_semi, zero_division=0)
    recall_semi = recall_score(y_test, y_pred_semi, zero_division=0)
    f1_semi = f1_score(y_test, y_pred_semi, zero_division=0)
    pr_auc_semi = average_precision_score(y_test, y_scores_semi)

    print(f"\n   训练时间：{train_time_semi:.3f} 秒")
    print(f"   Precision：{precision_semi:.4f}")
    print(f"   Recall：{recall_semi:.4f}")
    print(f"   F1-Score：{f1_semi:.4f}")
    print(f"   PR-AUC：{pr_auc_semi:.4f}")

    results['semi_supervised'] = {
        'precision': precision_semi,
        'recall': recall_semi,
        'f1_score': f1_semi,
        'pr_auc': pr_auc_semi,
        'train_time': train_time_semi
    }

    # =========================================================================
    # 实验2：无监督模式（用全部数据训练）
    # =========================================================================
    print("\n【实验2】无监督模式 (Unsupervised)")
    print("-" * 60)
    print(f"   训练数据：全部训练样本（包含少量欺诈）")
    print(f"   训练样本数：{X_train.shape[0]:,}")
    print(f"   其中欺诈样本：{y_train.sum():,} ({y_train.mean()*100:.3f}%)")
    print(f"   contamination：{contamination:.5f}（训练数据中的实际异常比例）")

    # 训练无监督模型
    model_unsup = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    model_unsup.fit(X_train)  # 用全部数据
    train_time_unsup = time.time() - start_time

    # 预测
    y_pred_unsup_raw = model_unsup.predict(X_test)
    y_pred_unsup = np.where(y_pred_unsup_raw == -1, 1, 0)
    y_scores_unsup = -model_unsup.decision_function(X_test)

    # 评估
    precision_unsup = precision_score(y_test, y_pred_unsup, zero_division=0)
    recall_unsup = recall_score(y_test, y_pred_unsup, zero_division=0)
    f1_unsup = f1_score(y_test, y_pred_unsup, zero_division=0)
    pr_auc_unsup = average_precision_score(y_test, y_scores_unsup)

    print(f"\n   训练时间：{train_time_unsup:.3f} 秒")
    print(f"   Precision：{precision_unsup:.4f}")
    print(f"   Recall：{recall_unsup:.4f}")
    print(f"   F1-Score：{f1_unsup:.4f}")
    print(f"   PR-AUC：{pr_auc_unsup:.4f}")

    results['unsupervised'] = {
        'precision': precision_unsup,
        'recall': recall_unsup,
        'f1_score': f1_unsup,
        'pr_auc': pr_auc_unsup,
        'train_time': train_time_unsup
    }

    # =========================================================================
    # 对比分析
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 对比分析结果")
    print("=" * 60)

    print(f"\n{'指标':<15} {'半监督模式':<15} {'无监督模式':<15} {'差异':<15}")
    print("-" * 60)

    metrics_names = ['Precision', 'Recall', 'F1-Score', 'PR-AUC']
    semi_values = [precision_semi, recall_semi, f1_semi, pr_auc_semi]
    unsup_values = [precision_unsup, recall_unsup, f1_unsup, pr_auc_unsup]

    for name, semi_val, unsup_val in zip(metrics_names, semi_values, unsup_values):
        diff = unsup_val - semi_val
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        winner = "← 胜" if semi_val > unsup_val else "胜 →" if unsup_val > semi_val else "平"
        print(f"{name:<15} {semi_val:<15.4f} {unsup_val:<15.4f} {diff_str:<10} {winner}")

    # 结论
    print("\n【结论与建议】")
    if f1_semi > f1_unsup:
        print("   ✅ 半监督模式表现更好")
        print("   原因分析：")
        print("      - 训练数据中只有纯正常样本，模型学到了清晰的'正常'边界")
        print("      - 无监督模式中，少量欺诈样本可能干扰了模型对'正常'的学习")
    elif f1_unsup > f1_semi:
        print("   ✅ 无监督模式表现更好")
        print("   原因分析：")
        print("      - 训练数据中的欺诈样本帮助模型学习了异常模式")
        print("      - Isolation Forest 在无监督场景下的设计使其能自动识别异常")
    else:
        print("   ⚖️ 两种模式表现相当")

    print("\n   【何时使用哪种模式？】")
    print("   - 半监督模式：有标签、确信训练数据是干净的")
    print("   - 无监督模式：无标签、数据可能已被污染、生产环境")

    # =========================================================================
    # 可视化对比
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Isolation Forest 训练模式对比：半监督 vs 无监督',
                 fontsize=14, fontweight='bold', y=1.02)

    # 图1：性能指标对比
    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, semi_values, width, label='半监督模式',
                        color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = axes[0].bar(x + width/2, unsup_values, width, label='无监督模式',
                        color='coral', alpha=0.8, edgecolor='black')

    axes[0].set_xlabel('评估指标', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('分数', fontsize=11, fontweight='bold')
    axes[0].set_title('性能指标对比', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, max(max(semi_values), max(unsup_values)) * 1.2)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 图2：PR曲线对比
    pr_precision_semi, pr_recall_semi, _ = precision_recall_curve(y_test, y_scores_semi)
    pr_precision_unsup, pr_recall_unsup, _ = precision_recall_curve(y_test, y_scores_unsup)

    axes[1].plot(pr_recall_semi, pr_precision_semi, color='steelblue', linewidth=2.5,
                 label=f'半监督 (AP={pr_auc_semi:.4f})')
    axes[1].plot(pr_recall_unsup, pr_precision_unsup, color='coral', linewidth=2.5,
                 label=f'无监督 (AP={pr_auc_unsup:.4f})')

    axes[1].set_xlabel('召回率 (Recall)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('精确率 (Precision)', fontsize=11, fontweight='bold')
    axes[1].set_title('Precision-Recall 曲线对比', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / '09_training_mode_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n   ✅ 对比图已保存到: outputs/09_training_mode_comparison.png")
    plt.show()

    return results


# ============================================================================
# 第9部分：主函数
# ============================================================================

def main():
    """
    主函数：执行完整的欺诈检测流程

    使用 Kaggle 信用卡欺诈检测数据集，完成以下步骤：
        1. 加载数据集
        2. 数据探索分析 (EDA)
        3. 数据预处理（标准化、划分训练/测试集）
        4. 训练三种异常检测模型
           - Isolation Forest
           - One-Class SVM
           - Local Outlier Factor (LOF)
        5. 模型评估与对比
        6. 可视化结果
        7. 保存模型
        8. 生成总结报告
    """
    print("\n" + "=" * 80)
    print("🚀 信用卡欺诈检测项目开始")
    print("   使用 Kaggle 真实数据集 (creditcard.csv)")
    print("=" * 80)

    # Step 1: 加载真实数据集
    # 数据集来源: https://www.kaggle.com/mlg-ulb/creditcardfraud
    df = load_credit_card_data()

    # Step 2: 数据探索分析
    explore_data(df)

    # Step 3: 数据预处理
    X_train, X_test, y_train, y_test, X_train_normal, scaler = preprocess_data(df)

    # Step 4: 训练三种异常检测模型
    # 根据真实数据集的欺诈比例设置 contamination 参数
    # 实际欺诈比例约为 492/284807 ≈ 0.00173
    fraud_ratio = df['Class'].mean()
    contamination = fraud_ratio
    print(f"\n   使用欺诈比例作为 contamination 参数: {contamination:.5f}")

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

    # Step 6.5: 训练模式对比实验（半监督 vs 无监督）
    # 这个实验帮助学习者理解异常检测的两种训练策略
    comparison_results = compare_training_modes(
        X_train, X_train_normal, X_test, y_train, y_test, contamination
    )

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
