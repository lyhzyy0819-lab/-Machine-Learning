# 🚀 无监督学习 - 从这里开始

> 欢迎来到无监督学习模块！这里你将学习如何从无标签数据中发现模式和结构。

---

## 📋 学习前检查清单

在开始之前，请确保：

- [x] 已完成监督学习模块（线性回归、逻辑回归、SVM、决策树、集成学习）
- [ ] 熟悉Python、NumPy、Pandas
- [ ] 了解基本的线性代数（向量、矩阵、特征值）
- [ ] 安装好Jupyter Notebook环境
- [ ] 安装必要的库：
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn umap-learn
  ```

---

## 🎯 学习目标与时间规划

### 总时间：2-3周

| 周次 | 内容 | 时间分配 | 完成标志 |
|------|------|----------|----------|
| **第1周** | 聚类算法 | 7天 | ✅ 理解4种聚类算法，能选择合适算法 |
| **第2周** | 降维技术 | 7天 | ✅ 深入掌握PCA，会使用t-SNE可视化 |
| **第3周** | 异常检测+项目 | 7天 | ✅ 完成3个实战项目 |

---

## 📚 学习路线图

```
📊 无监督学习
│
├── 🎯 聚类算法 (Clustering)
│   ├── 01. K-Means 聚类          [2天] ⭐⭐⭐
│   │   ├── 算法原理
│   │   ├── 肘部法则
│   │   ├── 轮廓系数
│   │   └── 客户分群案例
│   │
│   ├── 02. DBSCAN & 层次聚类     [2天] ⭐⭐
│   │   ├── 密度聚类
│   │   ├── 噪声处理
│   │   ├── 树状图绘制
│   │   └── 算法对比
│   │
│   ├── 03. 高斯混合模型 (GMM)    [2天] ⭐⭐
│   │   ├── 软聚类 vs 硬聚类
│   │   ├── EM算法
│   │   └── 概率分布
│   │
│   └── 📝 第1周复习            [1天]
│
├── 🔍 降维技术 (Dimensionality Reduction)
│   ├── 04. PCA 主成分分析       [3天] ⭐⭐⭐ 重点！
│   │   ├── 数学原理（特征值分解）
│   │   ├── 从零实现PCA
│   │   ├── 方差解释率
│   │   ├── 图像压缩应用
│   │   └── 特征工程中的降维
│   │
│   ├── 05. t-SNE & UMAP        [2天] ⭐⭐
│   │   ├── 高维数据可视化
│   │   ├── 参数调优
│   │   ├── MNIST可视化
│   │   └── t-SNE vs UMAP对比
│   │
│   └── 📝 第2周复习            [2天]
│
└── 🚨 异常检测 (Anomaly Detection)
    ├── 06. 异常检测算法          [2天] ⭐⭐
    │   ├── Isolation Forest
    │   ├── One-Class SVM
    │   ├── Local Outlier Factor
    │   └── 算法选择策略
    │
    └── 🎓 实战项目              [5天] ⭐⭐⭐
        ├── 项目1: 客户分群 (K-Means/GMM)
        ├── 项目2: 图像压缩 (PCA)
        └── 项目3: 欺诈检测 (Isolation Forest)
```

---

## 📖 详细学习计划

### 🗓️ 第1周：聚类算法

#### Day 1-2: K-Means聚类 ⭐⭐⭐
📘 **学习内容**：
- [ ] 观看 StatQuest - K-Means视频
- [ ] 阅读 `01_kmeans_clustering.ipynb`
- [ ] 理解算法流程：初始化 → 分配 → 更新 → 收敛
- [ ] 实现肘部法则确定K值
- [ ] 计算轮廓系数评估聚类质量

💻 **实践任务**：
- [ ] 从零实现K-Means（NumPy版）
- [ ] 使用sklearn.cluster.KMeans
- [ ] Iris数据集聚类实验
- [ ] 可视化聚类结果

🎯 **完成标志**：
- 能解释K-Means算法每一步
- 知道如何确定最佳K值
- 理解K-Means的优缺点和适用场景

---

#### Day 3-4: DBSCAN & 层次聚类 ⭐⭐
📘 **学习内容**：
- [ ] 阅读 `02_dbscan_hierarchical.ipynb`
- [ ] 理解DBSCAN的核心概念：核心点、边界点、噪声点
- [ ] 学习层次聚类的链接方式
- [ ] 绘制并解读树状图 (Dendrogram)

💻 **实践任务**：
- [ ] DBSCAN参数调优（eps、min_samples）
- [ ] 层次聚类不同链接方式对比
- [ ] 对比K-Means、DBSCAN、层次聚类的结果

🎯 **完成标志**：
- 理解密度聚类的优势
- 会调整DBSCAN参数
- 能解读树状图

---

#### Day 5-6: 高斯混合模型 (GMM) ⭐⭐
📘 **学习内容**：
- [ ] 阅读 `03_gmm.ipynb`
- [ ] 理解软聚类 vs 硬聚类
- [ ] 学习EM算法的基本思想
- [ ] 理解协方差类型的选择

💻 **实践任务**：
- [ ] 使用sklearn.mixture.GaussianMixture
- [ ] 对比GMM与K-Means的结果
- [ ] 可视化GMM的概率分布

🎯 **完成标志**：
- 理解概率聚类的概念
- 知道何时选择GMM而非K-Means

---

#### Day 7: 第1周复习与小结 📝
- [ ] 总结4种聚类算法的特点
- [ ] 制作算法选择决策树
- [ ] 完成小测验
- [ ] 准备进入降维部分

---

### 🗓️ 第2周：降维技术 ⭐⭐⭐

#### Day 8-10: PCA主成分分析 ⭐⭐⭐ 重点！
📘 **学习内容**：
- [ ] 观看 StatQuest - PCA视频（强烈推荐）
- [ ] 阅读 `04_pca_dimensionality_reduction.ipynb`
- [ ] 深入理解数学原理：
  - 协方差矩阵
  - 特征值与特征向量
  - 方差最大化
- [ ] 学习方差解释率的含义

💻 **实践任务**：
- [ ] **Day 8**: 从零实现PCA（NumPy版）⭐ 必做
- [ ] **Day 9**: 使用sklearn.decomposition.PCA
- [ ] **Day 10**: PCA应用实战
  - 特征降维后分类
  - 图像压缩
  - 数据去噪
  - 可视化高维数据

🎯 **完成标志**：
- 能用自己的话解释PCA原理
- 能从零实现PCA
- 理解如何选择主成分数量
- 知道PCA的局限性

**⚡ 重要提示**：
PCA是深度学习的重要基础！
- 理解特征值分解 → 有助于理解神经网络权重初始化
- 理解降维思想 → 有助于理解卷积、池化操作
- 建议多花时间深入理解！

---

#### Day 11-12: t-SNE & UMAP可视化 ⭐⭐
📘 **学习内容**：
- [ ] 阅读 `05_tsne_umap_visualization.ipynb`
- [ ] 理解t-SNE的核心思想
- [ ] 学习UMAP的优势
- [ ] 了解超参数的作用（perplexity、n_neighbors等）

💻 **实践任务**：
- [ ] MNIST数字可视化
- [ ] Fashion-MNIST可视化
- [ ] 调整参数观察变化
- [ ] 对比t-SNE和UMAP的效果

🎯 **完成标志**：
- 会使用t-SNE/UMAP进行可视化
- 理解何时用PCA、何时用t-SNE
- 知道t-SNE的局限性

---

#### Day 13-14: 第2周复习与综合应用 📝
- [ ] 总结PCA、t-SNE、UMAP的区别
- [ ] 降维 + 分类器的pipeline实验
- [ ] 特征工程中的降维应用
- [ ] 准备进入异常检测

---

### 🗓️ 第3周：异常检测 + 实战项目

#### Day 15-16: 异常检测算法 ⭐⭐
📘 **学习内容**：
- [ ] 阅读 `06_anomaly_detection.ipynb`
- [ ] 理解Isolation Forest的孤立思想
- [ ] 学习One-Class SVM
- [ ] 理解LOF局部密度概念

💻 **实践任务**：
- [ ] 信用卡欺诈检测数据集
- [ ] 对比3种异常检测算法
- [ ] 参数调优
- [ ] 可视化异常点

🎯 **完成标志**：
- 能选择合适的异常检测算法
- 会调整异常阈值

---

#### Day 17-21: 实战项目 ⭐⭐⭐ 最重要！

##### 🎓 项目1: 客户分群 (2天)
**目标**：使用K-Means/GMM对客户进行分群

**数据集**：Mall Customer Segmentation / 自定义数据

**步骤**：
1. [ ] 数据探索与预处理
2. [ ] 特征工程
3. [ ] 确定最佳簇数（肘部法则、轮廓系数）
4. [ ] K-Means聚类
5. [ ] GMM聚类
6. [ ] 对比两种方法
7. [ ] 分析每个簇的特征
8. [ ] 业务建议

**交付物**：
- Jupyter Notebook
- 聚类结果可视化
- 每个簇的客户画像

---

##### 🎓 项目2: PCA图像压缩 (1.5天)
**目标**：使用PCA压缩图像

**数据集**：Olivetti Faces / 自定义图像

**步骤**：
1. [ ] 加载图像数据
2. [ ] 应用PCA降维
3. [ ] 不同主成分数量的压缩率与质量对比
4. [ ] 可视化重构图像
5. [ ] 分析方差解释率
6. [ ] 计算压缩比和MSE

**交付物**：
- 不同压缩率的图像对比
- 压缩率 vs 质量曲线

---

##### 🎓 项目3: 欺诈检测 (1.5天)
**目标**：使用Isolation Forest检测欺诈交易

**数据集**：Credit Card Fraud Detection (Kaggle)

**步骤**：
1. [ ] 数据预处理（高度不平衡数据）
2. [ ] 特征工程
3. [ ] Isolation Forest建模
4. [ ] 调整contamination参数
5. [ ] 评估（Precision、Recall、F1）
6. [ ] 可视化异常点
7. [ ] 业务解读

**交付物**：
- 模型评估报告
- 异常点分析
- ROC曲线

---

## 🛠️ 环境设置

### 安装必要的库
```bash
# 基础库
pip install numpy pandas matplotlib seaborn

# 机器学习
pip install scikit-learn

# 可视化降维
pip install umap-learn

# Jupyter
pip install jupyter notebook

# 可选：更好的绘图
pip install plotly
```

### 验证安装
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.ensemble import IsolationForest

print("✅ 所有库安装成功！")
```

---

## 📊 评估标准

### 理论掌握 (30%)
- [ ] 能解释每种算法的核心思想
- [ ] 理解算法的优缺点和适用场景
- [ ] 掌握PCA的数学原理

### 编程能力 (40%)
- [ ] 能使用sklearn实现各种算法
- [ ] 能从零实现K-Means和PCA
- [ ] 能调整超参数优化结果

### 项目实战 (30%)
- [ ] 完成3个项目
- [ ] 代码规范、注释清晰
- [ ] 有完整的数据分析流程
- [ ] 有清晰的可视化和业务解读

---

## 💡 学习建议

### 1. 重点突出 ⭐
**必须深入掌握**：
- K-Means聚类
- PCA降维（从零实现）
- 聚类评估方法

**理解即可**：
- GMM（除非做概率模型）
- t-SNE（主要用于可视化）

### 2. 动手实践
- 每个算法都要自己写代码实验
- 在多个数据集上对比算法效果
- 尝试调整参数观察变化

### 3. 可视化驱动
- 无监督学习的结果需要大量可视化
- 学会用颜色、形状、大小表达信息
- 多用散点图、热力图、树状图

### 4. 对比学习
- 制作算法对比表格
- 在同一数据集上对比不同算法
- 总结各自的优劣和适用场景

### 5. 连接实际
- 思考业务中哪些问题可以用聚类解决
- 思考降维在特征工程中的作用
- 思考异常检测的实际应用

---

## 🎯 自我检测清单

### 第1周结束
- [ ] 我能解释K-Means算法的每一步
- [ ] 我知道如何确定最佳K值
- [ ] 我理解DBSCAN的密度聚类思想
- [ ] 我能对比不同聚类算法的优缺点

### 第2周结束
- [ ] 我能用自己的话解释PCA原理
- [ ] 我能从零实现PCA
- [ ] 我知道如何选择主成分数量
- [ ] 我会使用t-SNE可视化高维数据

### 第3周结束
- [ ] 我理解Isolation Forest的原理
- [ ] 我能选择合适的异常检测算法
- [ ] 我完成了3个实战项目
- [ ] 我能将无监督学习应用到实际问题

---

## 📚 推荐资源

### 必看视频 ⭐⭐⭐
1. **StatQuest - K-Means Clustering**
   - 链接: https://www.youtube.com/watch?v=4b5d3muPQmA

2. **StatQuest - PCA Main Ideas**
   - 链接: https://www.youtube.com/watch?v=HMOI_lkzW08
   - 最好的PCA讲解视频！

3. **StatQuest - PCA Step-by-Step**
   - 链接: https://www.youtube.com/watch?v=FgakZw6K1QQ

### 推荐阅读
- [Scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [PCA Explained Visually](https://setosa.io/ev/principal-component-analysis/)
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)

### Kaggle竞赛/数据集
- [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)

---

## 🚀 准备好了吗？

### 开始学习步骤：

1. **确认环境**
   ```bash
   jupyter notebook
   ```

2. **从第一个Notebook开始**
   ```
   📂 打开: 01_kmeans_clustering.ipynb
   ```

3. **按顺序学习**
   - 先看视频/阅读理论
   - 再运行Notebook代码
   - 自己修改参数实验
   - 完成练习题

4. **记录笔记**
   - 建立自己的知识库
   - 记录遇到的问题和解决方案

5. **参与社区**
   - Kaggle讨论区
   - Stack Overflow
   - GitHub开源项目

---

## 🎉 激励语

> "数据中蕴藏着我们未曾察觉的模式和结构，无监督学习帮助我们发现它们！"

**无监督学习的魅力**：
- 不需要标签，从数据本身发现规律
- 是数据探索的强大工具
- 是深度学习中许多技术的基础（如自编码器、对比学习）

**学习要点**：
- 重视PCA，深入理解数学原理
- 多做可视化，培养数据直觉
- 3个项目一定要认真完成

**💪 加油！2-3周后，你将掌握无监督学习的核心技能！**

---

## 📞 需要帮助？

遇到问题时：
1. 先查看Notebook中的FAQ部分
2. 查阅Scikit-learn官方文档
3. Google/Stack Overflow搜索
4. Kaggle讨论区提问

---

**🚀 现在就开始吧 → 打开 `01_kmeans_clustering.ipynb`**

---

**最后更新**: 2025-11-14
**版本**: v1.0
