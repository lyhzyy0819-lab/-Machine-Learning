# IMDB电影评论情感分析 - 朴素贝叶斯挑战练习

## 📋 项目概述

本项目使用**朴素贝叶斯分类器**对IMDB电影评论进行情感分析（正面/负面），目标是通过特征工程和模型优化将准确率提升至**85%以上**。

## 🎯 挑战目标

- **数据集**: IMDB 50,000条电影评论（25,000训练 + 25,000测试）
- **任务**: 二分类情感分析（正面/负面）
- **目标准确率**: > 85%
- **算法**: 朴素贝叶斯（MultinomialNB, BernoulliNB）

## 🚀 快速开始

### 环境要求

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### 运行实验

```bash
python imdb_sentiment_analysis.py
```

程序会自动：
1. 下载并缓存IMDB数据集（首次运行）
2. 运行5种不同的优化策略
3. 生成性能对比报告和可视化
4. 保存最佳模型

## 📊 优化策略

### 1. 基础模型
- **向量化**: TF-IDF
- **模型**: MultinomialNB
- **特征数**: 10,000
- **预期准确率**: ~83-85%

### 2. N-gram特征
- 测试不同n-gram组合：
  - Unigram (1,1)
  - Unigram + Bigram (1,2)
  - Unigram + Bigram + Trigram (1,3)
  - Bigram only (2,2)
- **预期提升**: +1-2%

### 3. 网格搜索参数调优
- **调优参数**:
  - `alpha`: [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
  - `fit_prior`: [True, False]
- **交叉验证**: 5折
- **预期准确率**: ~86-87%

### 4. 特征选择
- **方法**: 卡方检验（Chi-square）
- **测试特征数**: [5000, 10000, 15000, 20000]
- **目的**: 去除噪声特征，提高泛化能力

### 5. 投票集成
- **组合3个模型**:
  1. TF-IDF + MultinomialNB
  2. CountVectorizer + MultinomialNB
  3. Binary Features + BernoulliNB
- **投票方式**: 软投票（概率平均）
- **预期准确率**: ~87-88%

## 📈 预期结果

| 优化策略 | 预期准确率 | 关键技术 |
|---------|----------|---------|
| 基础模型 | 83-85% | TF-IDF + MNB |
| N-gram优化 | 85-86% | (1,2)-gram |
| 网格搜索 | 86-87% | alpha=0.1, fit_prior=True |
| 特征选择 | 85-87% | 卡方检验 top-k |
| 投票集成 | 87-88% | 3模型软投票 |

## 📁 输出文件

运行完成后会生成：

```
03_imdb_sentiment_analysis/
├── imdb_sentiment_analysis.py  # 主程序
├── README.md                   # 本文档
├── best_model.pkl             # 最佳模型
├── vectorizer.pkl             # 向量化器
├── confusion_matrix.png       # 混淆矩阵
├── top_features.png           # 最具区分性的词汇
└── model_comparison.png       # 模型性能对比
```

## 🔍 关键代码亮点

### 数据加载与预处理
```python
# 使用Keras加载IMDB数据（自动缓存）
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()

# 获取词索引
word_index = keras.datasets.imdb.get_word_index()

# 将整数序列转换回文本
def decode_review(encoded_review, word_index):
    reverse_word_index = {v: k for k, v in word_index.items()}
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
```

### N-gram特征工程
```python
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),  # unigram + bigram
    stop_words='english',
    max_df=0.7,          # 忽略出现在>70%文档的词
    min_df=5             # 忽略出现<5次的词
)
```

### 网格搜索调优
```python
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}

grid_search = GridSearchCV(
    MultinomialNB(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```

### 投票集成
```python
# 组合不同向量化方法的预测概率
y_proba_avg = (y_proba_tfidf + y_proba_count + y_proba_binary) / 3
y_pred = (y_proba_avg[:, 1] > 0.5).astype(int)
```

## 💡 进阶优化建议

如果想进一步提升准确率（>88%），可以尝试：

1. **更复杂的文本预处理**
   - 词干提取/词形还原
   - 去除HTML标签和特殊字符
   - 处理否定词（如 "not good" → "not_good"）

2. **更多特征工程**
   - 情感词典特征
   - 评论长度特征
   - 大写字母比例
   - 标点符号特征

3. **其他算法**
   - 逻辑回归（通常比朴素贝叶斯更好）
   - SVM with linear kernel
   - 深度学习（LSTM, BERT）

4. **数据增强**
   - 同义词替换
   - 回译（Back Translation）

## 📚 相关知识点

### 为什么朴素贝叶斯适合文本分类？
1. **速度快**: 训练和预测都非常快
2. **高维友好**: 文本特征维度高（数万维），朴素贝叶斯不易过拟合
3. **稀疏数据**: 文本向量化后非常稀疏，朴素贝叶斯处理得当
4. **可解释性**: 可以查看每个词的重要性

### MultinomialNB vs BernoulliNB
- **MultinomialNB**: 适用于词频统计（TF-IDF, CountVectorizer）
- **BernoulliNB**: 适用于二值特征（词是否出现）

### 平滑参数 alpha
- alpha=0: 无平滑（可能遇到零概率问题）
- alpha=1: 拉普拉斯平滑（默认）
- alpha<1: 弱平滑
- alpha>1: 强平滑

## 🎓 学习要点

通过本项目，你将学会：

1. ✅ 使用Keras加载和预处理IMDB数据集
2. ✅ 文本向量化技术（TF-IDF, CountVectorizer）
3. ✅ N-gram特征工程
4. ✅ 朴素贝叶斯分类器的使用和调优
5. ✅ 网格搜索和交叉验证
6. ✅ 特征选择方法（卡方检验）
7. ✅ 模型集成（投票）
8. ✅ 评估指标和可视化
9. ✅ 模型保存和加载

## 📖 参考资料

- [Scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Scikit-learn: Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [IMDB Dataset Paper](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
- [朴素贝叶斯教程](../04_naive_bayes.ipynb)

## ⚠️ 注意事项

1. **首次运行**: 会自动下载IMDB数据集（~84MB），需要网络连接
2. **运行时间**: 完整实验约需10-20分钟（取决于CPU性能）
3. **内存需求**: 建议至少4GB可用内存
4. **Python版本**: 需要Python 3.7+

## 🎉 总结

本项目展示了如何通过系统化的特征工程和模型优化，将朴素贝叶斯在IMDB情感分析任务上的准确率从基础的83%提升至**85%以上**，甚至接近88%。这证明了即使是"简单"的算法，通过合理的优化策略也能达到很好的效果！
