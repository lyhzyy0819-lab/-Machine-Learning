# 📊 无监督学习 (Unsupervised Learning)

> 无标签数据中发现模式、结构和隐藏信息

---

## 📚 学习目标

完成本模块后，你将能够：

- ✅ 理解无监督学习与监督学习的区别
- ✅ 掌握聚类算法（K-Means、DBSCAN、层次聚类、GMM）
- ✅ 理解并应用降维技术（PCA、t-SNE、UMAP）
- ✅ 学会异常检测方法（Isolation Forest、One-Class SVM、LOF）
- ✅ 能够选择合适的无监督算法解决实际问题
- ✅ 掌握无监督学习的评估方法

---

## 📖 课程大纲

### 第一部分：聚类算法 (Clustering)
**目标**：将相似的数据点分组

#### 1. K-Means 聚类
- **核心概念**：中心点、欧氏距离、迭代优化
- **算法流程**：初始化 → 分配 → 更新 → 收敛
- **肘部法则**：确定最佳K值
- **优缺点**：
  - ✅ 简单高效、易于理解
  - ❌ 需要预先指定K、对异常值敏感、只能发现球形簇
- **应用场景**：客户分群、图像分割、文档聚类

#### 2. DBSCAN (Density-Based Spatial Clustering)
- **核心概念**：密度可达、核心点、边界点、噪声点
- **参数**：eps(邻域半径)、min_samples(最小样本数)
- **优点**：
  - ✅ 不需要指定簇数量
  - ✅ 可以发现任意形状的簇
  - ✅ 能识别噪声点
- **应用场景**：异常检测、地理空间聚类

#### 3. 层次聚类 (Hierarchical Clustering)
- **凝聚式** (Agglomerative)：自底向上
- **分裂式** (Divisive)：自顶向下
- **链接方式**：单链接、全链接、平均链接、Ward方法
- **树状图** (Dendrogram)：可视化聚类过程
- **应用场景**：生物分类、社交网络分析

#### 4. 高斯混合模型 (GMM - Gaussian Mixture Model)
- **概率聚类**：软聚类 vs 硬聚类
- **EM算法**：期望最大化
- **协方差类型**：球形、对角、完全
- **优势**：提供概率分布、更灵活的簇形状
- **应用场景**：图像分割、语音识别

---

### 第二部分：降维技术 (Dimensionality Reduction) ⭐

**目标**：减少特征数量，同时保留重要信息

#### 5. PCA (主成分分析)
- **核心思想**：找到方差最大的方向
- **数学原理**：特征值分解、协方差矩阵
- **步骤**：
  1. 数据标准化
  2. 计算协方差矩阵
  3. 特征值分解
  4. 选择主成分
  5. 数据投影
- **方差解释率**：确定保留的主成分数
- **应用**：
  - 数据可视化
  - 特征压缩
  - 去噪
  - 图像压缩

#### 6. t-SNE (t-分布随机邻域嵌入)
- **目标**：高维数据可视化
- **核心**：保持局部结构
- **超参数**：perplexity、learning_rate、n_iter
- **注意事项**：
  - 主要用于可视化，不用于降维后建模
  - 计算成本高
  - 不同运行可能得到不同结果
- **应用场景**：MNIST可视化、词向量可视化

#### 7. UMAP (Uniform Manifold Approximation and Projection)
- **优势**：比t-SNE更快、更好的全局结构保持
- **应用**：大规模数据可视化

#### 8. LDA (线性判别分析)
- **监督降维**：利用标签信息
- **目标**：最大化类间距离、最小化类内距离
- **应用**：分类前的降维

---

### 第三部分：异常检测 (Anomaly Detection)

**目标**：识别与大多数数据显著不同的样本

#### 9. Isolation Forest (孤立森林)
- **核心思想**：异常点更容易被孤立
- **算法**：随机选择特征和分割点
- **contamination参数**：预期异常比例
- **优点**：
  - ✅ 高效、可扩展
  - ✅ 不需要距离计算
- **应用场景**：欺诈检测、网络入侵检测

#### 10. One-Class SVM
- **思想**：学习正常数据的边界
- **核函数**：RBF常用
- **nu参数**：异常值上界
- **应用场景**：新颖性检测

#### 11. LOF (Local Outlier Factor)
- **局部密度**：相对邻居的密度
- **n_neighbors参数**：邻居数量
- **应用场景**：局部异常检测

---

## 📂 文件结构

```
unsupervised_learning/
├── README.md                                # 本文件
├── START_HERE.md                            # 学习指南
├── 01_kmeans_clustering.ipynb               # K-Means聚类
├── 02_dbscan_hierarchical.ipynb             # DBSCAN & 层次聚类
├── 03_gmm.ipynb                             # 高斯混合模型
├── 04_pca_dimensionality_reduction.ipynb    # PCA降维 ⭐
├── 05_tsne_umap_visualization.ipynb         # t-SNE & UMAP可视化
├── 06_anomaly_detection.ipynb               # 异常检测
└── projects/
    ├── 01_customer_segmentation/            # 项目1：客户分群
    ├── 02_image_compression_pca/            # 项目2：PCA图像压缩
    └── 03_fraud_detection/                  # 项目3：欺诈检测
```

---

## 🎯 学习路径

### 推荐学习顺序（2-3周）

#### 第1周：聚类算法
1. **K-Means** (2天)
   - 理解算法原理
   - 实现肘部法则
   - 客户分群小案例

2. **DBSCAN & 层次聚类** (2天)
   - 理解密度聚类
   - 绘制树状图
   - 对比不同聚类算法

3. **GMM** (2天)
   - 理解EM算法
   - 软聚类 vs 硬聚类
   - 实战应用

4. **复习与小结** (1天)

#### 第2周：降维技术 ⭐ 重点
1. **PCA** (3天)
   - 数学原理深入理解
   - 从零实现PCA
   - 图像压缩项目
   - 方差解释率分析

2. **t-SNE & UMAP** (2天)
   - 高维数据可视化
   - 参数调优
   - MNIST/CIFAR-10可视化

3. **综合应用** (2天)
   - 特征工程中的降维
   - PCA + 分类器组合

#### 第3周：异常检测 + 项目
1. **异常检测算法** (2天)
   - Isolation Forest
   - One-Class SVM
   - LOF
   - 对比不同方法

2. **实战项目** (5天)
   - 项目1：客户分群 (K-Means/GMM)
   - 项目2：图像压缩 (PCA)
   - 项目3：欺诈检测 (Isolation Forest)

---

## 📊 评估指标

### 聚类评估
- **轮廓系数** (Silhouette Score): [-1, 1]，越接近1越好
- **Davies-Bouldin指数**: 越小越好
- **Calinski-Harabasz指数**: 越大越好
- **肘部法则** (Elbow Method): 确定K值

### 降维评估
- **方差解释率**: PCA主成分保留的信息量
- **重构误差**: 降维后重构的损失

### 异常检测评估
- **Precision / Recall / F1**: 如果有标签
- **ROC-AUC**: 评估检测效果
- **可视化分析**: 检查异常点的合理性

---

## 🛠️ 必备工具

### Python库
```python
# 核心库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 聚类
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# 降维
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap  # pip install umap-learn

# 异常检测
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# 评估
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
```

---

## 📚 学习资源

### 视频课程
- **StatQuest with Josh Starmer** - 聚类与PCA可视化讲解 ⭐ 强烈推荐
- **Andrew Ng - Machine Learning (Coursera)** - K-Means与PCA理论
- **Kaggle Learn** - Clustering实战

### 书籍
- 《Pattern Recognition and Machine Learning》- Bishop
- 《Python机器学习》- Sebastian Raschka
- 《动手学机器学习》- Aurélien Géron

### 在线资源
- [Scikit-learn官方文档](https://scikit-learn.org/stable/modules/clustering.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets) - 寻找聚类/降维数据集
- [Papers With Code - Clustering](https://paperswithcode.com/task/clustering)

### 推荐数据集
- **Mall Customer Segmentation** - 客户分群
- **Credit Card Fraud Detection** - 异常检测
- **MNIST / Fashion-MNIST** - 降维可视化
- **Iris / Wine** - 聚类练习

---

## 💡 学习建议

### 1. 重点掌握 PCA ⭐⭐⭐
- PCA是深度学习中常用的降维工具
- 理解特征值分解对后续神经网络学习很有帮助
- 建议从零实现一次PCA加深理解

### 2. 聚类算法的选择
```
数据特点                   推荐算法
─────────────────────────────────────
已知簇数量                 K-Means
不知道簇数量               DBSCAN / 层次聚类
球形簇                     K-Means
任意形状簇                 DBSCAN / GMM
需要概率输出               GMM
需要层次结构               层次聚类
有噪声点                   DBSCAN
```

### 3. 降维技术的选择
```
目的                       推荐方法
─────────────────────────────────────
特征压缩(用于建模)         PCA
高维数据可视化             t-SNE / UMAP
保持全局+局部结构          UMAP
有监督降维                 LDA
去噪                       PCA
```

### 4. 理论与实践结合
- 理解算法数学原理（特别是PCA）
- 动手实现核心算法
- 在真实数据集上实验
- 对比不同算法的效果

### 5. 可视化很重要
- 聚类结果可视化（散点图、颜色编码）
- PCA降维可视化（2D/3D）
- t-SNE/UMAP可视化（高维数据）
- 异常点可视化（标记异常样本）

---

## ✅ 完成标准

完成本模块后，你应该能够：

- [ ] **聚类算法**
  - [ ] 理解K-Means算法原理，能从零实现
  - [ ] 使用肘部法则和轮廓系数确定最佳K值
  - [ ] 理解DBSCAN的密度聚类思想
  - [ ] 能绘制层次聚类树状图
  - [ ] 理解GMM的EM算法

- [ ] **降维技术** ⭐ 重点
  - [ ] 深入理解PCA的数学原理
  - [ ] 能从零实现PCA
  - [ ] 知道如何选择主成分数量
  - [ ] 会使用t-SNE/UMAP进行可视化
  - [ ] 理解降维在特征工程中的作用

- [ ] **异常检测**
  - [ ] 理解Isolation Forest的孤立思想
  - [ ] 会使用One-Class SVM
  - [ ] 理解LOF局部密度概念
  - [ ] 能选择合适的异常检测算法

- [ ] **实战能力**
  - [ ] 独立完成3个项目
  - [ ] 能根据业务场景选择合适算法
  - [ ] 会评估和调优无监督模型
  - [ ] 能清晰可视化无监督学习结果

---

## 🚀 下一步

完成无监督学习后，进入：

**👉 阶段3：神经网络基础** (深度学习入门)
- 感知器与多层感知器
- 反向传播算法 ⭐ 核心
- 激活函数、优化器、正则化
- 从零实现神经网络

---

## 📞 常见问题 (FAQ)

### Q1: 无监督学习和监督学习有什么区别？
**A:**
- **监督学习**：有标签(y)，目标是学习 X → y 的映射
- **无监督学习**：无标签，目标是发现数据的内在结构和模式

### Q2: K-Means如何确定最佳K值？
**A:** 常用方法：
1. **肘部法则**：绘制K与惯性(inertia)的关系，寻找拐点
2. **轮廓系数**：计算不同K的轮廓系数，选最大的
3. **业务知识**：结合实际业务需求

### Q3: PCA降维会丢失信息吗？
**A:** 会，但：
- 丢失的是方差较小的方向（通常是噪声）
- 通过方差解释率可以控制信息损失
- 通常保留95-99%的方差就足够了

### Q4: t-SNE和PCA的主要区别？
**A:**
| 特性 | PCA | t-SNE |
|------|-----|-------|
| 目标 | 保持全局方差 | 保持局部结构 |
| 线性/非线性 | 线性 | 非线性 |
| 速度 | 快 | 慢 |
| 用途 | 降维+可视化 | 主要用于可视化 |
| 确定性 | 确定 | 随机（每次结果不同） |

### Q5: 异常检测的阈值如何确定？
**A:**
- Isolation Forest: 调整 `contamination` 参数（预期异常比例）
- One-Class SVM: 调整 `nu` 参数
- LOF: 设置 `outlier_fraction`
- 实践中：通过可视化 + 业务经验确定

---

**最后更新**: 2025-11-14
**作者**: 您的AI学习助手
**版本**: v1.0

---

**💪 加油！无监督学习是连接传统机器学习和深度学习的重要桥梁！**
