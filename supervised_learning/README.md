# 监督学习完整教程

本目录包含监督学习的完整Jupyter Notebook教程,从基础到实战项目。

## 📚 学习路径

### 第一部分：回归 (Regression)

#### 1. 线性回归 [`01_linear_regression.ipynb`]
**学习内容：**
- 线性回归的数学原理
- 从零实现（正规方程 + 梯度下降）
- 使用Scikit-learn
- 模型评估（MSE、RMSE、MAE、R²）
- 残差分析

**时间：** 2-3小时

---

#### 2. 多项式回归与正则化 [`02_polynomial_regression_regularization.ipynb`]
**学习内容：**
- 多项式回归处理非线性数据
- 过拟合 vs 欠拟合
- Ridge回归（L2正则化）
- Lasso回归（L1正则化）
- ElasticNet
- 超参数调优
- Pipeline使用

**时间：** 2-3小时

---

---

### 第二部分：分类 (Classification)

#### 3. K-近邻算法 (KNN) [`03_knn.ipynb`]
**学习内容：**
- KNN算法原理（近朱者赤）
- 距离度量方法（欧氏、曼哈顿）
- K值选择和参数调优
- 从零实现KNN
- KNN分类 vs KNN回归
- 维度灾难问题
- 特征缩放的重要性

**时间：** 2-3小时

---

#### 4. 朴素贝叶斯 [`04_naive_bayes.ipynb`]
**学习内容：**
- 贝叶斯定理基础
- 朴素贝叶斯假设
- 高斯朴素贝叶斯（连续特征）
- 多项式朴素贝叶斯（词频）
- 伯努利朴素贝叶斯（二值）
- 文本分类应用
- 平滑技术

**时间：** 2-3小时

---

#### 5. 逻辑回归 [`05_logistic_regression.ipynb`]
**学习内容：**
- 逻辑回归原理（Sigmoid函数）
- 二分类 vs 多分类
- 决策边界可视化
- 概率预测
- 分类评估指标

**时间：** 2-3小时

---

#### 6. 支持向量机(SVM) [`06_svm.ipynb`]
**学习内容：**
- SVM原理（最大间隔）
- 线性SVM vs 非线性SVM
- 核技巧（RBF、多项式）
- 参数调优（C和gamma）

**时间：** 2-3小时

---

#### 7. 决策树与随机森林 [`07_tree_ensemble.ipynb`]
**学习内容：**
- 决策树原理
- 信息增益 vs 基尼系数
- 随机森林
- XGBoost/LightGBM
- 特征重要性

**时间：** 3-4小时

---

#### 8. 模型评估与选择 [`08_model_evaluation.ipynb`]
**学习内容：**
- 混淆矩阵
- Precision、Recall、F1-Score
- ROC曲线和AUC
- 交叉验证策略
- 模型对比

**时间：** 2小时

---

---

### 第三部分：实战项目 (Projects)

学完以上6个基础教程后，进入 **[projects](./projects/)** 文件夹，完成以下企业级完整项目：

#### 7. [房价预测项目](./projects/01_house_price_prediction/)
**项目目标**:
- 回归问题完整流程
- 地理特征工程
- 模型融合(Stacking)
- 完整的端到端实战

**时间**: 4-6小时
**难度**: ⭐⭐⭐

---

#### 8. [客户流失预测项目](./projects/02_customer_churn_prediction/)
**项目目标**:
- 分类问题完整流程
- 类别不平衡处理(SMOTE)
- ROC-AUC、混淆矩阵分析
- 业务洞察与决策建议

**时间**: 5-7小时
**难度**: ⭐⭐⭐⭐

---

每个项目都包含：
- ✅ 真实业务场景
- ✅ 完整的数据处理流程
- ✅ 多模型对比与优化
- ✅ 模型部署准备
- ✅ 业务洞察与建议

💡 **更多项目详情**: 查看 [projects目录README](./projects/README.md)

---

## 🚀 快速开始

### 环境准备
```bash
# 激活conda环境
conda activate ml_env

# 启动Jupyter Lab
jupyter lab
```

### 学习建议
1. **按顺序学习**：每个notebook都基于前面的知识
2. **动手实践**：运行所有代码单元，观察结果
3. **完成练习**：每个notebook都有练习题
4. **做笔记**：记录重点和疑问
5. **调试学习**：修改参数，观察变化

---

## 📊 每个Notebook的结构

```
1. 理论简介
   - 算法原理
   - 数学公式（简洁）
   - 应用场景

2. 代码实现
   - 从零实现
   - Scikit-learn实现
   - 可视化

3. 实战练习
   - 真实数据集
   - 参数调优
   - 模型评估

4. 练习题
   - 动手编码
   - 巩固理解

5. 总结
   - 知识点回顾
   - 使用场景
```

---

## 🎯 学习目标检查

完成本系列教程后，你应该能够：

- [ ] 理解回归和分类的区别
- [ ] 从零实现基本的ML算法
- [ ] 熟练使用Scikit-learn
- [ ] 独立完成数据预处理
- [ ] 进行特征工程
- [ ] 选择合适的模型和参数
- [ ] 评估模型性能
- [ ] 完成Kaggle竞赛（Top 25%）

---

## 📦 数据集

### 内置数据集
- Scikit-learn自带数据集（make_regression、make_classification等）
- 无需下载，代码中直接生成

### Kaggle数据集
- **Titanic**: https://www.kaggle.com/c/titanic
- **House Prices**: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- **Credit Card Fraud**: https://www.kaggle.com/mlg-ulb/creditcardfraud

### 下载方式
```bash
# 配置Kaggle API
pip install kaggle
# 下载数据
kaggle competitions download -c titanic
```

---

## 🔧 常见问题

### Q1: 代码运行出错怎么办？
- 检查Python版本（需要3.8）
- 确认所有包已安装
- 查看错误信息，善用Google和Stack Overflow

### Q2: 某个概念不理解？
- 先跑通代码，观察结果
- 查看可视化图表
- 参考理论部分
- 记下疑问，稍后深入学习

### Q3: 练习题太难？
- 参考前面的代码示例
- 查看Scikit-learn文档
- 先实现基本版本，再优化

### Q4: 如何提高Kaggle排名？
- 仔细的EDA（探索性数据分析）
- 创造更多有用的特征
- 尝试多个模型
- 模型融合（Ensemble）
- 学习Top解决方案

---

## 📚 推荐资源

### 文档
- [Scikit-learn官方文档](https://scikit-learn.org/stable/)
- [Pandas文档](https://pandas.pydata.org/docs/)
- [NumPy文档](https://numpy.org/doc/)

### 教程
- [Scikit-learn用户指南](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle Learn](https://www.kaggle.com/learn)

### 书籍
- 《Python机器学习基础教程》
- 《机器学习实战》
- 《Hands-On Machine Learning》

---

## ✅ 学习检查点

### Week 1: 回归基础
- [ ] 完成01-线性回归
- [ ] 完成02-多项式回归与正则化
- [ ] 理解过拟合和正则化

### Week 2: 回归实战
- [ ] 完成03-房价预测项目
- [ ] Kaggle提交至少一次
- [ ] R² > 0.85

### Week 3: 分类基础
- [ ] 完成03-KNN
- [ ] 完成04-朴素贝叶斯
- [ ] 完成05-逻辑回归
- [ ] 理解分类评估指标

### Week 4: 高级分类
- [ ] 完成06-SVM
- [ ] 完成07-决策树与随机森林
- [ ] 理解核技巧和集成学习

### Week 5: 模型评估
- [ ] 完成08-模型评估与选择
- [ ] 理解模型融合
- [ ] 掌握交叉验证

### Week 6-7: 综合实战
- [ ] 完成项目1-房价预测
- [ ] 完成项目2-客户流失预测
- [ ] Kaggle排名前25%

---

## 🎓 下一步学习

完成本教程后，你可以：

1. **深度学习方向**
   - PyTorch基础
   - 卷积神经网络(CNN)
   - 循环神经网络(RNN)

2. **进阶机器学习**
   - 时间序列预测
   - 推荐系统
   - 异常检测

3. **参加竞赛**
   - Kaggle竞赛
   - 天池竞赛
   - 积累实战经验

4. **实际项目**
   - 找真实业务问题
   - 端到端项目
   - 模型部署

---

## 💡 学习建议

### 学习节奏
- **每天2-3小时**
- **完成一个notebook再开始下一个**
- **周末做综合项目**
- **总计4-6周完成全部内容**

### 学习方法
1. **先快速过一遍**：了解大致内容
2. **详细学习**：逐行理解代码
3. **修改实验**：改参数，观察结果
4. **独立实现**：不看答案完成练习
5. **总结记录**：写学习笔记

### 避免的陷阱
- ❌ 只看不做
- ❌ 追求完美理解才前进
- ❌ 跳过练习题
- ❌ 不做笔记和总结

### 推荐做法
- ✅ 边学边练
- ✅ 80%理解就继续
- ✅ 完成所有练习
- ✅ 写技术博客总结

---

## 📞 反馈与改进

如果你发现：
- 代码错误
- 解释不清楚
- 需要更多示例
- 其他建议

欢迎反馈！

---

**祝学习顺利！记住：实践是最好的老师！** 🚀
