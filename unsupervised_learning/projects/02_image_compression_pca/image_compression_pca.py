"""
🎯 PCA图像压缩项目 (Image Compression with PCA)
================================================

项目目标：
    使用主成分分析(PCA)实现图像压缩，分析不同压缩率对图像质量的影响，
    理解PCA在降维和数据压缩中的实际应用。

数据集：
    Olivetti Faces Dataset (sklearn自带)
    - 400张64x64像素的人脸灰度图像
    - 40个不同的人，每人10张不同表情/角度的照片
    - 数据集网址: https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset

核心概念：
    - PCA降维原理：将高维数据投影到低维空间
    - 方差解释率：衡量主成分保留的信息量
    - 图像压缩：减少存储空间的同时保持视觉质量
    - 压缩比 vs 重构质量的权衡

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

# PCA算法
from sklearn.decomposition import PCA

# 数据集（使用本地文件，无需从远程下载）
# from sklearn.datasets import fetch_olivetti_faces  # 已改为本地加载

# 评估指标
from sklearn.metrics import mean_squared_error

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
# 第2部分：数据加载与探索
# ============================================================================

def load_olivetti_faces():
    """
    加载Olivetti Faces人脸数据集

    数据集说明：
        - 400张64x64像素的人脸灰度图像
        - 40个不同的人，每人10张照片
        - 像素值范围：[0, 1]，已归一化

    Returns:
    --------
    faces : ndarray, shape (400, 4096)
        人脸图像数据，每行是一张64x64图像展平后的向量
    targets : ndarray, shape (400,)
        人脸标签，表示是哪个人（0-39）
    """
    print("\n" + "=" * 80)
    print("📂 正在加载Olivetti Faces数据集...")
    print("=" * 80)

    # ========================================================================
    # 从本地 archive/ 目录加载数据（Kaggle 下载的数据集）
    # 解决了从远程服务器下载时的 403 错误问题
    # ========================================================================
    data_dir = Path(__file__).parent / 'archive'

    # 加载原始数据
    # olivetti_faces.npy: shape (400, 64, 64) - 400张64x64的灰度人脸图像
    # olivetti_faces_target.npy: shape (400,) - 每张图像对应的人物ID (0-39)
    faces_raw = np.load(data_dir / 'olivetti_faces.npy')
    targets = np.load(data_dir / 'olivetti_faces_target.npy')

    # 转换格式以匹配原始 sklearn API 的输出格式
    images = faces_raw                    # shape: (400, 64, 64) 原始图像格式
    faces = faces_raw.reshape(400, -1)    # shape: (400, 4096) 展平为向量

    # 打乱数据顺序（保持与原代码一致的随机种子，确保结果可复现）
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(faces))
    faces = faces[shuffle_idx]
    targets = targets[shuffle_idx]
    images = images[shuffle_idx]

    print(f"✅ 数据加载成功！")
    print(f"   - 图像数量: {faces.shape[0]}")
    print(f"   - 图像尺寸: 64 x 64 像素")
    print(f"   - 特征维度: {faces.shape[1]} (64*64像素展平)")
    print(f"   - 人物数量: {len(np.unique(targets))}")
    print(f"   - 像素值范围: [{faces.min():.2f}, {faces.max():.2f}]")
    print(f"   - 数据类型: {faces.dtype}")

    return faces, targets, images


def explore_data(faces, targets, images):
    """
    探索性数据分析 (EDA)

    目的：
        通过可视化了解数据集的基本情况

    Parameters:
    -----------
    faces : ndarray, shape (400, 4096)
        展平的图像数据
    targets : ndarray, shape (400,)
        人脸标签
    images : ndarray, shape (400, 64, 64)
        原始图像格式
    """
    print("\n" + "=" * 80)
    print("🔍 数据探索分析")
    print("=" * 80)

    # 1. 数据基本统计
    print("\n【数据统计】")
    print(f"   - 总样本数: {len(faces)}")
    print(f"   - 原始维度: {faces.shape[1]} (未压缩)")
    print(f"   - 数据大小: {faces.nbytes / 1024:.2f} KB")
    print(f"   - 平均像素值: {faces.mean():.4f}")
    print(f"   - 像素标准差: {faces.std():.4f}")

    # 2. 可视化部分样本
    print("\n【可视化样本】")
    print("   正在生成样本图像...")

    fig, axes = plt.subplots(4, 10, figsize=(15, 6))
    fig.suptitle('Olivetti Faces 数据集样本 (前40张图像)',
                 fontsize=14, fontweight='bold', y=1.00)

    for i in range(40):
        ax = axes[i // 10, i % 10]
        # 显示图像（64x64灰度图）
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'ID:{targets[i]}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()

    # 保存图像
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '01_data_samples.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ 样本图像已保存到: outputs/01_data_samples.png")
    plt.show()

    # 3. 像素分布分析
    print("\n【像素值分布】")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 像素值直方图
    axes[0].hist(faces.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('像素值', fontsize=11)
    axes[0].set_ylabel('频数', fontsize=11)
    axes[0].set_title('所有图像的像素值分布', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 单张图像的像素值分布
    sample_idx = 0
    axes[1].hist(faces[sample_idx], bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('像素值', fontsize=11)
    axes[1].set_ylabel('频数', fontsize=11)
    axes[1].set_title(f'单张图像的像素值分布 (ID: {targets[sample_idx]})',
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '02_pixel_distribution.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ 像素分布图已保存到: outputs/02_pixel_distribution.png")
    plt.show()


# ============================================================================
# 第3部分：PCA图像压缩实现
# ============================================================================

def apply_pca_compression(faces, n_components_list):
    """
    应用PCA对图像进行压缩

    原理：
        PCA通过找到数据方差最大的方向（主成分），将高维数据投影到低维空间。
        对于图像压缩：
        1. 原始图像: 64×64 = 4096 维
        2. PCA压缩: 投影到 n_components 维
        3. 重构图像: 从低维空间还原到原始空间

        压缩比 = 1 - (n_components / 4096)
        例如：n_components=100 时，压缩比 = 1 - 100/4096 ≈ 97.6%

    Parameters:
    -----------
    faces : ndarray, shape (400, 4096)
        原始图像数据
    n_components_list : list
        要测试的主成分数量列表，如 [10, 50, 100, ...]

    Returns:
    --------
    results : dict
        包含每个n_components对应的PCA模型、重构图像、评估指标等
    """
    print("\n" + "=" * 80)
    print("🔧 应用PCA压缩")
    print("=" * 80)

    results = {}
    n_features = faces.shape[1]  # 4096

    for n_comp in n_components_list:
        print(f"\n【压缩到 {n_comp} 个主成分】")

        # 创建PCA模型
        # n_components: 保留的主成分数量
        # svd_solver='randomized': 使用随机化算法，适合大数据集
        # whiten=False: 不进行白化处理
        start_time = time.time()
        pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=42)

        # 步骤1：拟合PCA模型并转换数据（降维）
        # fit_transform 会：
        #   1. 计算数据的协方差矩阵
        #   2. 求解特征值和特征向量
        #   3. 选择前n_comp个最大特征值对应的特征向量
        #   4. 将原始数据投影到这些主成分上
        # 结果形状: (400, n_comp)
        faces_compressed = pca.fit_transform(faces)

        # 步骤2：从压缩数据重构原始图像（升维）
        # inverse_transform 会：
        #   将低维数据投影回原始高维空间
        # 公式: X_reconstructed = X_compressed @ components + mean
        # 结果形状: (400, 4096)
        faces_reconstructed = pca.inverse_transform(faces_compressed)

        compression_time = time.time() - start_time

        # 计算压缩比
        # 原始存储: n_samples × n_features
        # 压缩存储: n_samples × n_components + n_components × n_features (PCA参数)
        # 简化计算: 1 - (n_components / n_features)
        compression_ratio = 1 - (n_comp / n_features)

        # 计算方差解释率
        # explained_variance_ratio_: 每个主成分解释的方差占比
        # cumsum: 累积方差解释率
        variance_explained = pca.explained_variance_ratio_
        cumsum_variance = np.cumsum(variance_explained)
        total_variance_explained = cumsum_variance[-1]

        # 计算重构误差 (MSE)
        # MSE = mean((原始 - 重构)^2)
        # 越小表示重构质量越好
        mse = mean_squared_error(faces, faces_reconstructed)

        # 计算峰值信噪比 (PSNR)
        # PSNR = 10 * log10(MAX^2 / MSE)
        # MAX 是像素最大值，这里是1.0（归一化后）
        # PSNR越大表示图像质量越好，通常 >30dB 认为质量较好
        psnr = 10 * np.log10(1.0**2 / mse) if mse > 0 else float('inf')

        # 保存结果
        results[n_comp] = {
            'pca_model': pca,
            'compressed_data': faces_compressed,
            'reconstructed_data': faces_reconstructed,
            'compression_ratio': compression_ratio,
            'variance_explained': variance_explained,
            'total_variance_explained': total_variance_explained,
            'mse': mse,
            'psnr': psnr,
            'compression_time': compression_time
        }

        # 打印统计信息
        print(f"   ✅ 压缩完成！")
        print(f"      - 压缩后维度: {faces_compressed.shape}")
        print(f"      - 压缩比: {compression_ratio:.2%}")
        print(f"      - 方差解释率: {total_variance_explained:.2%}")
        print(f"      - 重构误差 (MSE): {mse:.6f}")
        print(f"      - 峰值信噪比 (PSNR): {psnr:.2f} dB")
        print(f"      - 压缩用时: {compression_time:.3f} 秒")

    return results


def calculate_variance_curve(faces, max_components=400):
    """
    计算方差解释率曲线

    目的：
        了解需要多少个主成分才能保留足够的信息（如90%、95%、99%方差）

    Parameters:
    -----------
    faces : ndarray, shape (400, 4096)
        原始图像数据
    max_components : int
        最大主成分数量（不超过样本数）

    Returns:
    --------
    pca_full : PCA对象
        完整的PCA模型
    cumsum_variance : ndarray
        累积方差解释率
    """
    print("\n" + "=" * 80)
    print("📊 计算完整方差解释率曲线")
    print("=" * 80)

    # 使用所有可能的主成分（最多min(n_samples, n_features)）
    pca_full = PCA(n_components=max_components, random_state=42)
    pca_full.fit(faces)

    # 累积方差解释率
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # 找到保留不同方差比例所需的主成分数
    variance_thresholds = [0.80, 0.90, 0.95, 0.99]
    print("\n【方差解释率分析】")
    for threshold in variance_thresholds:
        # 找到第一个超过阈值的索引
        n_comp_needed = np.argmax(cumsum_variance >= threshold) + 1
        compression_ratio = 1 - (n_comp_needed / faces.shape[1])
        print(f"   保留 {threshold:.0%} 方差需要: {n_comp_needed:3d} 个主成分 "
              f"(压缩比: {compression_ratio:.2%})")

    return pca_full, cumsum_variance


# ============================================================================
# 第4部分：可视化
# ============================================================================

def visualize_compression_comparison(faces, images, results, n_components_list):
    """
    可视化原始图像与不同压缩率的重构图像对比

    Parameters:
    -----------
    faces : ndarray, shape (400, 4096)
        原始图像数据（展平）
    images : ndarray, shape (400, 64, 64)
        原始图像格式
    results : dict
        PCA压缩结果
    n_components_list : list
        主成分数量列表
    """
    print("\n" + "=" * 80)
    print("📊 可视化压缩效果对比")
    print("=" * 80)

    # 选择几张代表性图像
    sample_indices = [0, 10, 50, 100, 200]
    n_samples = len(sample_indices)
    n_compressions = len(n_components_list)

    # 创建子图：第一行是原始图像，后续行是不同压缩率
    fig, axes = plt.subplots(n_compressions + 1, n_samples,
                             figsize=(15, 2.5 * (n_compressions + 1)))

    fig.suptitle('PCA图像压缩效果对比', fontsize=16, fontweight='bold', y=0.995)

    # 第一行：原始图像
    for j, idx in enumerate(sample_indices):
        axes[0, j].imshow(images[idx], cmap='gray', vmin=0, vmax=1)
        axes[0, j].set_title(f'原始图像\n(4096维)', fontsize=10, fontweight='bold')
        axes[0, j].axis('off')

    # 后续行：不同压缩率的重构图像
    for i, n_comp in enumerate(n_components_list):
        reconstructed = results[n_comp]['reconstructed_data']
        compression_ratio = results[n_comp]['compression_ratio']
        variance = results[n_comp]['total_variance_explained']
        psnr = results[n_comp]['psnr']

        for j, idx in enumerate(sample_indices):
            # 重构图像reshape回64x64
            img_reconstructed = reconstructed[idx].reshape(64, 64)

            axes[i + 1, j].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=1)

            # 第一列显示详细信息
            if j == 0:
                title = (f'{n_comp}个主成分\n'
                        f'压缩比:{compression_ratio:.1%}\n'
                        f'方差:{variance:.1%}|PSNR:{psnr:.1f}dB')
            else:
                title = f'{n_comp}维'

            axes[i + 1, j].set_title(title, fontsize=9)
            axes[i + 1, j].axis('off')

    plt.tight_layout()

    # 保存图像
    output_dir = Path('outputs')
    plt.savefig(output_dir / '03_compression_comparison.png', dpi=200, bbox_inches='tight')
    print("   ✅ 对比图已保存到: outputs/03_compression_comparison.png")
    plt.show()


def visualize_variance_curve(cumsum_variance):
    """
    可视化累积方差解释率曲线

    目的：
        了解主成分数量与信息保留量的关系

    Parameters:
    -----------
    cumsum_variance : ndarray
        累积方差解释率
    """
    print("\n【绘制方差解释率曲线】")

    fig, ax = plt.subplots(figsize=(12, 6))

    n_components = len(cumsum_variance)

    # 绘制累积方差曲线
    ax.plot(range(1, n_components + 1), cumsum_variance * 100,
            linewidth=2.5, color='steelblue', label='累积方差解释率')

    # 添加参考线
    variance_thresholds = [80, 90, 95, 99]
    colors = ['green', 'orange', 'red', 'purple']

    for threshold, color in zip(variance_thresholds, colors):
        ax.axhline(y=threshold, color=color, linestyle='--', alpha=0.6, linewidth=1.5,
                  label=f'{threshold}% 方差阈值')

        # 找到对应的主成分数
        n_comp_needed = np.argmax(cumsum_variance >= threshold/100) + 1
        ax.axvline(x=n_comp_needed, color=color, linestyle=':', alpha=0.4, linewidth=1.5)

        # 标注点
        ax.plot(n_comp_needed, threshold, 'o', color=color, markersize=8)
        ax.text(n_comp_needed + 5, threshold - 3,
               f'n={n_comp_needed}', fontsize=9, color=color, fontweight='bold')

    ax.set_xlabel('主成分数量', fontsize=12, fontweight='bold')
    ax.set_ylabel('累积方差解释率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('PCA累积方差解释率曲线', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_components)
    ax.set_ylim(0, 105)

    plt.tight_layout()

    # 保存图像
    output_dir = Path('outputs')
    plt.savefig(output_dir / '04_variance_curve.png', dpi=150, bbox_inches='tight')
    print("   ✅ 方差曲线已保存到: outputs/04_variance_curve.png")
    plt.show()


def visualize_metrics_comparison(results, n_components_list):
    """
    可视化压缩比、方差解释率、MSE、PSNR的对比

    Parameters:
    -----------
    results : dict
        PCA压缩结果
    n_components_list : list
        主成分数量列表
    """
    print("\n【绘制评估指标对比图】")

    # 提取指标
    compression_ratios = [results[n]['compression_ratio'] * 100 for n in n_components_list]
    variance_explained = [results[n]['total_variance_explained'] * 100 for n in n_components_list]
    mse_values = [results[n]['mse'] for n in n_components_list]
    psnr_values = [results[n]['psnr'] for n in n_components_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PCA图像压缩评估指标对比', fontsize=16, fontweight='bold', y=0.995)

    # 1. 压缩比 vs 主成分数
    axes[0, 0].plot(n_components_list, compression_ratios,
                   marker='o', linewidth=2.5, markersize=8, color='steelblue')
    axes[0, 0].set_xlabel('主成分数量', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('压缩比 (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('压缩比 vs 主成分数量', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 105)

    # 2. 方差解释率 vs 主成分数
    axes[0, 1].plot(n_components_list, variance_explained,
                   marker='s', linewidth=2.5, markersize=8, color='coral')
    axes[0, 1].axhline(y=95, color='red', linestyle='--', alpha=0.6, label='95%阈值')
    axes[0, 1].set_xlabel('主成分数量', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('方差解释率 (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('方差解释率 vs 主成分数量', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 105)

    # 3. MSE vs 主成分数
    axes[1, 0].plot(n_components_list, mse_values,
                   marker='^', linewidth=2.5, markersize=8, color='green')
    axes[1, 0].set_xlabel('主成分数量', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('均方误差 (MSE)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('重构误差 vs 主成分数量', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')  # 使用对数坐标

    # 4. PSNR vs 主成分数
    axes[1, 1].plot(n_components_list, psnr_values,
                   marker='D', linewidth=2.5, markersize=8, color='purple')
    axes[1, 1].axhline(y=30, color='orange', linestyle='--', alpha=0.6, label='30dB阈值')
    axes[1, 1].set_xlabel('主成分数量', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('峰值信噪比 (PSNR, dB)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('图像质量 vs 主成分数量', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    output_dir = Path('outputs')
    plt.savefig(output_dir / '05_metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✅ 指标对比图已保存到: outputs/05_metrics_comparison.png")
    plt.show()


def visualize_eigenfaces(pca_model, n_eigenfaces=16):
    """
    可视化特征脸（主成分）

    特征脸 (Eigenfaces)：
        - PCA的主成分对应于人脸的"基本特征"
        - 前几个主成分捕获最显著的人脸变化（如光照、表情、角度）
        - 任何人脸都可以表示为这些特征脸的线性组合

    Parameters:
    -----------
    pca_model : PCA对象
        训练好的PCA模型
    n_eigenfaces : int
        要显示的特征脸数量
    """
    print("\n【可视化特征脸 (Eigenfaces)】")

    # 获取主成分（特征向量）
    # components_ 的形状: (n_components, n_features)
    # 每一行是一个主成分，代表一个"特征脸"
    components = pca_model.components_[:n_eigenfaces]

    # 绘制特征脸
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('前16个特征脸 (Eigenfaces) - PCA主成分可视化',
                 fontsize=14, fontweight='bold', y=0.995)

    for i, ax in enumerate(axes.flat):
        if i < n_eigenfaces:
            # 将特征向量reshape成64x64图像
            eigenface = components[i].reshape(64, 64)

            # 显示特征脸
            # 使用'RdBu_r'色图更好地显示正负特征
            im = ax.imshow(eigenface, cmap='RdBu_r')
            ax.set_title(f'特征脸 #{i+1}', fontsize=10, fontweight='bold')
            ax.axis('off')

            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')

    plt.tight_layout()

    # 保存图像
    output_dir = Path('outputs')
    plt.savefig(output_dir / '06_eigenfaces.png', dpi=150, bbox_inches='tight')
    print("   ✅ 特征脸图已保存到: outputs/06_eigenfaces.png")
    plt.show()


# ============================================================================
# 第5部分：模型保存
# ============================================================================

def save_models(results, n_components_list):
    """
    保存PCA模型和评估指标

    Parameters:
    -----------
    results : dict
        PCA压缩结果
    n_components_list : list
        主成分数量列表
    """
    print("\n" + "=" * 80)
    print("💾 保存模型和评估指标")
    print("=" * 80)

    # 创建模型保存目录
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    # 保存每个PCA模型
    for n_comp in n_components_list:
        pca_model = results[n_comp]['pca_model']
        model_path = models_dir / f'pca_model_{n_comp}_components.pkl'
        joblib.dump(pca_model, model_path)
        print(f"   ✅ 已保存模型: {model_path}")

    # 保存评估指标到JSON
    metrics = {}
    for n_comp in n_components_list:
        metrics[f'{n_comp}_components'] = {
            'n_components': n_comp,
            'compression_ratio': float(results[n_comp]['compression_ratio']),
            'total_variance_explained': float(results[n_comp]['total_variance_explained']),
            'mse': float(results[n_comp]['mse']),
            'psnr': float(results[n_comp]['psnr']),
            'compression_time_seconds': float(results[n_comp]['compression_time'])
        }

    metrics_path = models_dir / 'pca_compression_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"   ✅ 已保存评估指标: {metrics_path}")
    print("\n" + "=" * 80)


# ============================================================================
# 第6部分：主函数
# ============================================================================

def main():
    """
    主函数：执行完整的PCA图像压缩流程
    """
    print("\n" + "=" * 80)
    print("🚀 PCA图像压缩项目开始")
    print("=" * 80)

    # Step 1: 加载数据
    faces, targets, images = load_olivetti_faces()

    # Step 2: 数据探索
    explore_data(faces, targets, images)

    # Step 3: 定义要测试的主成分数量
    # 从极度压缩到轻度压缩
    n_components_list = [10, 25, 50, 75, 100, 150, 200, 300]

    # Step 4: 应用PCA压缩
    results = apply_pca_compression(faces, n_components_list)

    # Step 5: 计算完整的方差解释率曲线
    pca_full, cumsum_variance = calculate_variance_curve(faces, max_components=400)

    # Step 6: 可视化
    visualize_compression_comparison(faces, images, results, n_components_list)
    visualize_variance_curve(cumsum_variance)
    visualize_metrics_comparison(results, n_components_list)

    # 使用最大主成分数的模型可视化特征脸
    best_pca = results[300]['pca_model']
    visualize_eigenfaces(best_pca, n_eigenfaces=16)

    # Step 7: 保存模型
    save_models(results, n_components_list)

    # Step 8: 总结报告
    print("\n" + "=" * 80)
    print("📊 项目总结报告")
    print("=" * 80)

    print("\n【关键发现】")

    # 找到95%方差的最小主成分数
    idx_95 = np.argmax(cumsum_variance >= 0.95)
    n_comp_95 = idx_95 + 1
    compression_95 = 1 - (n_comp_95 / faces.shape[1])

    print(f"   1. 保留95%方差所需主成分: {n_comp_95} (压缩比: {compression_95:.2%})")

    # 找到PSNR > 30dB的最小主成分数
    psnr_30_candidates = [n for n in n_components_list if results[n]['psnr'] >= 30]
    if psnr_30_candidates:
        n_comp_30db = min(psnr_30_candidates)
        print(f"   2. PSNR>30dB的最小主成分: {n_comp_30db} "
              f"(PSNR: {results[n_comp_30db]['psnr']:.2f} dB)")

    # 对比极端情况
    n_min = min(n_components_list)
    n_max = max(n_components_list)
    print(f"\n   3. 极度压缩 (n={n_min}):")
    print(f"      - 压缩比: {results[n_min]['compression_ratio']:.2%}")
    print(f"      - 方差解释率: {results[n_min]['total_variance_explained']:.2%}")
    print(f"      - PSNR: {results[n_min]['psnr']:.2f} dB")

    print(f"\n   4. 轻度压缩 (n={n_max}):")
    print(f"      - 压缩比: {results[n_max]['compression_ratio']:.2%}")
    print(f"      - 方差解释率: {results[n_max]['total_variance_explained']:.2%}")
    print(f"      - PSNR: {results[n_max]['psnr']:.2f} dB")

    print("\n【实践建议】")
    print("   - 人脸识别应用: 建议使用100-150个主成分（保留>95%方差）")
    print("   - 缩略图生成: 可使用25-50个主成分（压缩比>98%）")
    print("   - 数据存储优化: 根据质量要求在50-200个主成分间权衡")

    print("\n" + "=" * 80)
    print("✅ PCA图像压缩项目完成！")
    print("=" * 80)
    print("\n📁 输出文件:")
    print("   - outputs/01_data_samples.png           # 数据样本")
    print("   - outputs/02_pixel_distribution.png      # 像素分布")
    print("   - outputs/03_compression_comparison.png  # 压缩效果对比")
    print("   - outputs/04_variance_curve.png          # 方差解释率曲线")
    print("   - outputs/05_metrics_comparison.png      # 评估指标对比")
    print("   - outputs/06_eigenfaces.png              # 特征脸可视化")
    print("   - models/pca_model_*_components.pkl      # PCA模型文件")
    print("   - models/pca_compression_metrics.json    # 评估指标JSON")
    print("\n" + "=" * 80)


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    main()
