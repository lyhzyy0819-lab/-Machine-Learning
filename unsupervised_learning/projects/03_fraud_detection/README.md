# 🎯 项目3: 欺诈检测 (Fraud Detection)

> 使用异常检测算法识别欺诈交易

---

## 📋 项目概述

**业务场景**：
信用卡欺诈每年造成数十亿美元损失。通过机器学习识别异常交易可以帮助银行及时发现欺诈行为。

**项目目标**：
1. 使用Isolation Forest等算法检测欺诈交易
2. 处理高度不平衡数据（欺诈交易<0.2%）
3. 评估异常检测模型性能
4. 可视化异常点分布

**技术栈**：
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
- 不平衡数据处理
- 评估指标：Precision, Recall, F1, ROC-AUC

---

## 📊 数据集

### 数据来源
Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 数据特征
- **Time**: 距离第一笔交易的秒数
- **V1-V28**: PCA降维后的特征（保护隐私）
- **Amount**: 交易金额
- **Class**: 标签（0=正常, 1=欺诈）

### 数据特点
- **高度不平衡**: 欺诈交易仅占 ~0.17%
- **样本数量**: 284,807笔交易
- **欺诈数量**: 492笔
- **特征已脱敏**: V1-V28是PCA处理后的结果

---

## 🎯 项目步骤

### Step 1: 数据探索 (EDA)
- [ ] 加载数据，查看基本信息
- [ ] 分析类别分布（不平衡程度）
- [ ] 可视化特征分布（正常 vs 欺诈）
- [ ] 分析交易金额分布
- [ ] 检查缺失值和异常值

### Step 2: 数据预处理
- [ ] 处理时间特征（可选）
- [ ] 标准化金额特征
- [ ] 划分训练集和测试集
- [ ] 处理不平衡数据
  - 方案1: 只用正常交易训练（One-Class）
  - 方案2: 下采样
  - 方案3: SMOTE（可选）

### Step 3: Isolation Forest
- [ ] 只使用正常交易训练
- [ ] 调整contamination参数
- [ ] 预测异常点
- [ ] 评估模型性能

### Step 4: One-Class SVM
- [ ] 训练One-Class SVM
- [ ] 调整nu参数
- [ ] 对比与Isolation Forest的结果

### Step 5: Local Outlier Factor
- [ ] 训练LOF模型
- [ ] 调整n_neighbors参数
- [ ] 分析局部异常密度

### Step 6: 模型评估与对比
- [ ] 计算Precision, Recall, F1
- [ ] 绘制ROC曲线和计算AUC
- [ ] 绘制PR曲线（Precision-Recall）
- [ ] 对比三种算法的性能
- [ ] 分析混淆矩阵

### Step 7: 可视化分析
- [ ] 使用t-SNE/PCA降维可视化
- [ ] 标记异常点
- [ ] 分析误检和漏检的样本

---

## 📈 期望输出

### 1. 数据分析报告
- 数据不平衡分析
- 特征分布对比（正常 vs 欺诈）
- 交易金额分析

### 2. 模型训练与评估
三种算法的性能对比：

| 模型 | Precision | Recall | F1 | ROC-AUC |
|------|-----------|--------|----|----|
| Isolation Forest | ? | ? | ? | ? |
| One-Class SVM | ? | ? | ? | ? |
| LOF | ? | ? | ? | ? |

### 3. 可视化结果
- ROC曲线对比
- PR曲线对比
- 混淆矩阵
- t-SNE异常点可视化

### 4. 业务建议
- 最佳模型推荐
- 阈值设置建议
- 误报与漏报的权衡

---

## 💡 项目提示

### 重要概念

**为什么用异常检测而非分类？**
- 欺诈交易极少（~0.17%）
- 欺诈模式不断变化
- One-Class方法只需学习"正常"模式

**评估指标选择**
- **Precision**: 预测为欺诈的交易中，真正欺诈的比例
  - 重要性：高Precision减少误报，降低用户体验损害
- **Recall**: 真实欺诈交易中，被检测出的比例
  - 重要性：高Recall减少漏报，降低经济损失
- **F1**: Precision和Recall的调和平均
- **ROC-AUC**: 不平衡数据可能不太准确
- **PR-AUC**: 更适合不平衡数据

### 代码示例

```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# 只使用正常交易训练
X_train_normal = X_train[y_train == 0]

# 训练Isolation Forest
iso_forest = IsolationForest(
    contamination=0.002,  # 预期异常比例
    random_state=42,
    n_estimators=100
)
iso_forest.fit(X_train_normal)

# 预测（-1表示异常，1表示正常）
y_pred = iso_forest.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)  # 转换为0/1

# 评估
print(classification_report(y_test, y_pred))
```

### 参数调优技巧

**Isolation Forest**:
- `contamination`: 预期异常比例（0.001-0.01）
- `n_estimators`: 树的数量（100-200）
- `max_samples`: 子样本大小

**One-Class SVM**:
- `nu`: 异常值上界（0.001-0.01）
- `kernel`: 'rbf'常用
- `gamma`: RBF核参数

**LOF**:
- `n_neighbors`: 邻居数量（20-50）
- `contamination`: 异常比例

---

## 📊 分析问题

### 问题1: 不平衡数据处理
- 为什么不用传统分类算法？
- One-Class方法的优势是什么？
- 如何设置contamination参数？

### 问题2: 评估指标选择
- 为什么PR曲线比ROC曲线更适合？
- 如何权衡Precision和Recall？
- 业务上更关注哪个指标？

### 问题3: 模型对比
- 三种算法各有什么特点？
- 哪个算法在本数据集上表现最好？
- 实际部署时如何选择？

### 问题4: 误检分析
- 误报的样本有什么特点？
- 漏报的样本有什么特点？
- 如何改进模型？

---

## ✅ 检查清单

完成项目后，确保：
- [ ] 完成完整的EDA
- [ ] 理解数据不平衡的影响
- [ ] 正确处理训练数据（只用正常样本）
- [ ] 训练并评估3种异常检测算法
- [ ] 使用正确的评估指标（PR-AUC）
- [ ] 绘制ROC和PR曲线
- [ ] 可视化异常点分布
- [ ] 分析误检和漏检
- [ ] 提供模型选择建议
- [ ] 代码注释清晰
- [ ] 有完整的结论和业务建议

---

## 🔬 进阶挑战

1. **集成方法**
   - 结合多个异常检测算法（投票法）
   - 对比单一模型和集成模型

2. **特征工程**
   - 从Amount提取新特征（如交易金额分组）
   - 从Time提取时间特征（如一天中的小时）

3. **半监督学习**
   - 使用少量标注的欺诈样本改进模型
   - 对比纯无监督和半监督方法

4. **实时检测**
   - 分析模型推理时间
   - 设计在线检测pipeline

---

## 📚 参考资源

- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Anomaly Detection with Sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Kaggle Solutions](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/code)

---

## 🎯 项目挑战性

这个项目具有挑战性因为：
1. **极度不平衡**: 0.17%的正样本
2. **评估复杂**: 需要理解多种评估指标
3. **业务权衡**: Precision vs Recall的取舍
4. **模型选择**: 需要对比多种算法

**但完成后你将获得**：
- 处理不平衡数据的能力
- 异常检测的实战经验
- 模型评估的深入理解
- 业务场景分析能力

---

**开始时间**: __________
**完成时间**: __________
**项目耗时**: __________

**💪 这是最具挑战性的项目，完成它后你将成为异常检测专家！**
