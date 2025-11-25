"""
IMDB电影评论情感分析 - 朴素贝叶斯挑战练习
目标：使用朴素贝叶斯实现情感分类，准确率 > 85%

数据集：IMDB 50K电影评论（25K训练，25K测试）
标签：正面(1)、负面(0)

优化策略：
1. 特征工程：TF-IDF、N-gram、特征选择
2. 模型调优：网格搜索平滑参数
3. 集成方法：组合多种向量化策略
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Keras/TensorFlow用于数据加载
from tensorflow import keras

# Sklearn工具
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti TC', 'SimHei'
]
plt.rcParams['axes.unicode_minus'] = False

# 随机种子
np.random.seed(42)


# ==================== 数据加载与预处理 ====================

def load_imdb_data(num_words=None):
    """
    加载IMDB数据集

    IMDB数据集说明：
    - 包含50,000条电影评论（25,000训练 + 25,000测试）
    - 每条评论已被预处理为整数序列（词索引）
    - 标签：0=负面，1=正面（二分类）
    - 数据会自动下载并缓存到 ~/.keras/datasets/imdb.npz

    Args:
        num_words: 保留最常见的N个词，None表示保留所有
                   例如：num_words=10000表示只保留最常见的10000个词
                   其他词会被标记为2（未知词）

    Returns:
        (x_train, y_train): 训练集（整数序列，标签）
        (x_test, y_test): 测试集（整数序列，标签）
        word_index: 词到索引的映射字典 {单词: 索引}
    """
    print("正在加载IMDB数据集...")

    # 加载数据（会自动缓存到 ~/.keras/datasets/）
    # 返回的x是整数序列列表，每个整数代表一个词的索引
    # 例如：[1, 14, 22, 16, ...] 表示一条评论
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=num_words
    )

    # 获取词索引：字典形式 {单词: 索引}
    # 注意：索引0,1,2是保留的特殊索引
    # 0: padding（填充）
    # 1: start of sequence（序列开始标记）
    # 2: unknown（未知词/超出num_words范围的词）
    # 实际词的索引从3开始
    word_index = keras.datasets.imdb.get_word_index()

    print(f"训练集大小: {len(x_train)}")
    print(f"测试集大小: {len(x_test)}")
    print(f"词汇表大小: {len(word_index)}")
    print(f"正面评论比例: {y_train.mean():.2%}")

    return (x_train, y_train), (x_test, y_test), word_index


def decode_review(encoded_review, word_index):
    """
    将整数序列转换回文本

    解码原理：
    - IMDB数据集中，每条评论被编码为整数序列
    - 例如：[1, 14, 22, 16, 43, 530, ...]
    - 我们需要通过word_index字典将整数映射回单词

    为什么要减3？
    - word_index中的索引从1开始：{word: index}
    - 但Keras保留了0,1,2三个特殊索引
    - 0: padding（填充）
    - 1: start of sequence（序列开始）
    - 2: unknown（未知词）
    - 所以实际词的编码 = word_index中的索引 + 3
    - 解码时需要反向操作：encoded_index - 3 = word_index

    Args:
        encoded_review: 整数序列，如 [1, 14, 22, 16, ...]
        word_index: 词到索引的映射字典 {单词: 索引}

    Returns:
        decoded_text: 解码后的文本字符串
    """
    # 创建反向索引：{索引: 单词}
    # 例如：{'the': 1, 'and': 2} -> {1: 'the', 2: 'and'}
    reverse_word_index = {v: k for k, v in word_index.items()}

    # 解码整数序列为文本
    # - 对每个整数i，减3后在reverse_word_index中查找对应的词
    # - 如果找不到（超出词汇表），用'?'代替
    # - 最后用空格连接所有词
    decoded_text = ' '.join([
        reverse_word_index.get(i - 3, '?') for i in encoded_review
    ])

    return decoded_text


def convert_to_text(x_data, word_index):
    """
    批量转换整数序列为文本

    为什么要转换？
    - Keras的IMDB数据是整数序列，适合深度学习（LSTM等）
    - 但朴素贝叶斯需要文本数据，因为要用TF-IDF/CountVectorizer提取特征
    - 所以需要将整数序列转回文本

    Args:
        x_data: 整数序列列表，如 [[1, 14, 22, ...], [1, 194, 1153, ...], ...]
        word_index: 词索引字典

    Returns:
        texts: 解码后的文本列表，如 ['the movie was great ...', 'worst film ever ...', ...]
    """
    print("正在将整数序列转换为文本...")
    # 对每条评论调用decode_review函数
    texts = [decode_review(review, word_index) for review in x_data]
    return texts


# ==================== 基础模型 ====================

def baseline_model(X_train, y_train, X_test, y_test):
    """
    基础模型：简单TF-IDF + MultinomialNB

    策略说明：
    - 使用TF-IDF将文本转换为数值特征
    - 使用多项式朴素贝叶斯进行分类
    - 这是最简单直接的方法，作为baseline

    Args:
        X_train: 训练集文本列表
        y_train: 训练集标签
        X_test: 测试集文本列表
        y_test: 测试集标签

    Returns:
        model: 训练好的MultinomialNB模型
        vectorizer: TF-IDF向量化器
        accuracy: 测试集准确率
    """
    print("\n" + "="*60)
    print("基础模型：TF-IDF + MultinomialNB")
    print("="*60)

    # ========== TF-IDF向量化 ==========
    # TF-IDF: Term Frequency-Inverse Document Frequency（词频-逆文档频率）
    # 作用：将文本转换为数值特征向量
    vectorizer = TfidfVectorizer(
        max_features=10000,      # 只保留出现频率最高的10000个词作为特征
                                 # 这样可以降维，减少计算量

        stop_words='english',    # 去除英文停用词（the, a, is等无意义词）
                                 # 停用词对情感分析贡献不大

        max_df=0.7,              # 忽略在超过70%文档中出现的词
                                 # 这些词太常见，区分度低（如"movie", "film"）

        min_df=5                 # 忽略出现次数少于5次的词
                                 # 这些词太罕见，可能是噪声或拼写错误
    )

    # fit_transform：学习词汇表并转换训练集
    # 返回稀疏矩阵 (25000, 10000)：25000条评论 × 10000个特征
    X_train_vec = vectorizer.fit_transform(X_train)

    # transform：使用已学习的词汇表转换测试集
    # 注意：测试集不能用fit_transform，必须用相同的词汇表
    X_test_vec = vectorizer.transform(X_test)

    print(f"特征矩阵形状: {X_train_vec.shape}")
    # 输出示例：(25000, 10000) -> 25000条评论，每条用10000维向量表示

    # ========== 训练多项式朴素贝叶斯模型 ==========
    # MultinomialNB：适用于离散特征（如词频、TF-IDF）
    model = MultinomialNB(
        alpha=1.0               # 拉普拉斯平滑参数（加法平滑）
                                # 作用：防止零概率问题
                                # - alpha=1.0: 标准拉普拉斯平滑（默认）
                                # - alpha=0: 不平滑（可能遇到零概率）
                                # - alpha<1: 弱平滑
                                # - alpha>1: 强平滑（更保守）
    )
    model.fit(X_train_vec, y_train)

    # ========== 预测 ==========
    # predict：返回预测类别（0或1）
    y_pred = model.predict(X_test_vec)

    # predict_proba：返回属于每个类别的概率
    # 返回形状：(n_samples, 2)，每行是[P(负面), P(正面)]
    # 例如：[[0.8, 0.2], [0.3, 0.7], ...] 表示第1条评论80%负面，第2条评论70%正面
    # [:, 1] 取第二列，即正类（正面评论）的概率
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    # ========== 评估 ==========
    # accuracy_score：准确率 = (预测正确的样本数) / (总样本数)
    accuracy = accuracy_score(y_test, y_pred)

    # roc_auc_score：ROC曲线下面积（Area Under Curve）
    # 衡量模型区分正负类的能力，取值0-1，越接近1越好
    # 需要概率值而非类别
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred,
                                target_names=['负面', '正面']))

    return model, vectorizer, accuracy


# ==================== 优化模型 ====================

def optimized_model_ngram(X_train, y_train, X_test, y_test):
    """
    优化策略1：使用N-gram特征

    什么是N-gram？
    - Unigram (1-gram): 单个词，如 "good", "bad", "movie"
    - Bigram (2-gram): 两个连续的词，如 "very good", "not bad", "worst movie"
    - Trigram (3-gram): 三个连续的词，如 "not very good"

    为什么N-gram有效？
    - 捕捉词语组合的语义，如 "not good" 和 "good" 意义完全相反
    - 对情感分析特别有用，因为情感常由词组表达
    - 例如："not" + "bad" = 正面，但单看"not"和"bad"都是负面

    Args:
        X_train: 训练集文本
        y_train: 训练集标签
        X_test: 测试集文本
        y_test: 测试集标签

    Returns:
        best_model: 准确率最高的模型
        best_vectorizer: 对应的向量化器
        best_accuracy: 最佳准确率
    """
    print("\n" + "="*60)
    print("优化策略1：N-gram特征")
    print("="*60)

    results = []

    # 测试不同的n-gram组合
    ngram_configs = [
        (1, 1),  # unigram: 只用单词，如 ["good", "movie", "bad"]
        (1, 2),  # unigram + bigram: 单词+词对，如 ["good", "movie", "good movie"]
        (1, 3),  # unigram + bigram + trigram: 单词+词对+三连词
        (2, 2),  # bigram only: 只用词对，如 ["good movie", "very bad"]
    ]

    for ngram_range in ngram_configs:
        print(f"\n测试 {ngram_range}-gram...")

        vectorizer = TfidfVectorizer(
            max_features=15000,       # 增加特征数（因为N-gram会产生更多特征）
            ngram_range=ngram_range,  # N-gram范围
                                      # (1,1): 只有unigram
                                      # (1,2): unigram + bigram
                                      # (2,3): bigram + trigram
            stop_words='english',
            max_df=0.7,
            min_df=5
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # 使用较小的alpha（0.5），因为特征更丰富了
        model = MultinomialNB(alpha=0.5)
        model.fit(X_train_vec, y_train)

        accuracy = model.score(X_test_vec, y_test)
        print(f"{ngram_range}-gram 准确率: {accuracy:.4f}")

        results.append({
            'ngram_range': ngram_range,
            'accuracy': accuracy,
            'model': model,
            'vectorizer': vectorizer
        })

    # 选择准确率最高的配置
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n最佳N-gram配置: {best_result['ngram_range']}")
    print(f"最佳准确率: {best_result['accuracy']:.4f}")

    return best_result['model'], best_result['vectorizer'], best_result['accuracy']


def optimized_model_grid_search(X_train, y_train, X_test, y_test):
    """
    优化策略2：网格搜索参数调优

    什么是网格搜索？
    - 自动尝试参数的所有组合
    - 使用交叉验证评估每组参数
    - 选择表现最好的参数组合

    为什么需要参数调优？
    - alpha（平滑参数）对朴素贝叶斯影响很大
    - 不同数据集的最优参数不同
    - 手动调参效率低，容易遗漏最优值

    Args:
        X_train: 训练集文本
        y_train: 训练集标签
        X_test: 测试集文本
        y_test: 测试集标签

    Returns:
        best_model: 最优参数的模型
        best_vectorizer: 向量化器
        best_accuracy: 测试集准确率
    """
    print("\n" + "="*60)
    print("优化策略2：网格搜索参数调优")
    print("="*60)

    # 向量化（使用N-gram和更多特征）
    vectorizer = TfidfVectorizer(
        max_features=20000,       # 增加到20000个特征
        ngram_range=(1, 2),       # 使用unigram + bigram
        stop_words='english',
        max_df=0.7,
        min_df=3                  # 降低最小词频阈值
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"特征矩阵形状: {X_train_vec.shape}")

    # ========== 参数网格 ==========
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
        # alpha（拉普拉斯平滑参数）:
        # - 0.001: 几乎不平滑，模型更信任训练数据（可能过拟合）
        # - 0.1: 弱平滑，适合大数据集
        # - 1.0: 标准平滑（默认）
        # - 2.0: 强平滑，更保守（防止过拟合）

        'fit_prior': [True, False]
        # fit_prior（是否学习类先验概率）:
        # - True: 从训练数据学习P(y)，如P(正面)=0.5, P(负面)=0.5
        # - False: 使用均匀先验，如P(正面)=P(负面)=0.5
        # - 如果数据不平衡，True通常更好
    }

    # ========== 网格搜索 ==========
    grid_search = GridSearchCV(
        MultinomialNB(),          # 基础模型
        param_grid,               # 参数网格
        cv=5,                     # 5折交叉验证
                                  # 将训练集分成5份，轮流用4份训练、1份验证
        scoring='accuracy',       # 优化目标：准确率
        n_jobs=-1,                # 使用所有CPU核心并行计算
        verbose=1                 # 显示进度
    )
    # 总共测试：6个alpha × 2个fit_prior × 5折CV = 60次训练

    print("\n正在进行网格搜索...")
    print("这将测试 6×2=12 组参数，每组5折交叉验证，共60次训练")
    grid_search.fit(X_train_vec, y_train)

    print(f"\n最优参数: {grid_search.best_params_}")
    print(f"最优交叉验证准确率: {grid_search.best_score_:.4f}")

    # 测试集评估（使用最优参数的模型）
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test_vec, y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")

    return best_model, vectorizer, test_accuracy


def optimized_model_feature_selection(X_train, y_train, X_test, y_test):
    """
    优化策略3：特征选择（卡方检验）

    什么是特征选择？
    - 从大量特征中选出最相关的子集
    - 去除噪声特征，提高模型性能
    - 减少计算量和过拟合风险

    卡方检验（Chi-square test）:
    - 统计方法，衡量特征与标签的相关性
    - 值越大表示特征越重要
    - 适用于分类问题

    为什么要特征选择？
    - 并非所有词都对情感分类有用
    - 例如："movie", "film"太常见，区分度低
    - 保留"excellent", "terrible"这类强情感词

    Args:
        X_train: 训练集文本
        y_train: 训练集标签
        X_test: 测试集文本
        y_test: 测试集标签

    Returns:
        best_model: 最佳模型
        best_vectorizer: 向量化器
        best_accuracy: 最佳准确率
        feature_selector: 特征选择器
    """
    print("\n" + "="*60)
    print("优化策略3：特征选择（卡方检验）")
    print("="*60)

    # 向量化（使用CountVectorizer因为卡方检验需要非负特征）
    # 注意：TF-IDF的值可能为负，不适合卡方检验
    vectorizer = CountVectorizer(
        max_features=30000,       # 先生成30000个候选特征
        ngram_range=(1, 2),       # unigram + bigram
        stop_words='english',
        max_df=0.7,
        min_df=3
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"原始特征数: {X_train_vec.shape[1]}")

    # 测试不同的特征数量
    k_values = [5000, 10000, 15000, 20000]
    results = []

    for k in k_values:
        print(f"\n测试保留 {k} 个最佳特征...")

        # ========== 特征选择 ==========
        # SelectKBest：选择K个最佳特征
        # chi2：卡方检验，计算每个特征的卡方统计量
        selector = SelectKBest(chi2, k=k)

        # fit_transform：计算特征重要性并筛选
        # 从30000个特征中选出k个最相关的
        X_train_selected = selector.fit_transform(X_train_vec, y_train)
        X_test_selected = selector.transform(X_test_vec)

        # 训练模型（使用较小的alpha）
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train_selected, y_train)

        accuracy = model.score(X_test_selected, y_test)
        print(f"准确率: {accuracy:.4f}")

        results.append({
            'k': k,
            'accuracy': accuracy,
            'model': model,
            'selector': selector
        })

    # 选择最佳特征数
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n最佳特征数: {best_result['k']}")
    print(f"最佳准确率: {best_result['accuracy']:.4f}")
    print(f"特征选择前: 30000 → 特征选择后: {best_result['k']}")
    print(f"降维比例: {(1 - best_result['k']/30000)*100:.1f}%")

    return (best_result['model'], vectorizer,
            best_result['accuracy'], best_result['selector'])


def combined_voting_classifier(X_train, y_train, X_test, y_test):
    """
    优化策略4：组合多个模型（投票集成）

    什么是集成学习？
    - 训练多个不同的模型
    - 组合它们的预测结果
    - "三个臭皮匠，胜过诸葛亮"

    为什么要集成？
    - 不同模型有不同的优势和盲点
    - TF-IDF捕捉词权重，Count捕捉词频，Binary捕捉词出现
    - MultinomialNB适合计数特征，BernoulliNB适合二值特征
    - 组合后更稳健

    投票方式：
    - 硬投票：多数投票，如2个模型预测正面就是正面
    - 软投票：概率平均，如(0.7+0.6+0.8)/3=0.7，再判断类别

    Args:
        X_train: 训练集文本
        y_train: 训练集标签
        X_test: 测试集文本
        y_test: 测试集标签

    Returns:
        accuracy: 投票集成的测试准确率
    """
    print("\n" + "="*60)
    print("优化策略4：多模型投票集成")
    print("="*60)

    # ========== 模型1：TF-IDF + MultinomialNB ==========
    # 特点：关注词的重要性（TF-IDF权重）
    vec1 = TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                           stop_words='english', max_df=0.7, min_df=3)
    X_train_1 = vec1.fit_transform(X_train)
    X_test_1 = vec1.transform(X_test)
    model1 = MultinomialNB(alpha=0.1)  # 小alpha，相信TF-IDF的区分度

    # ========== 模型2：Count + MultinomialNB ==========
    # 特点：关注词频（出现次数）
    vec2 = CountVectorizer(max_features=15000, ngram_range=(1, 2),
                          stop_words='english', max_df=0.7, min_df=3)
    X_train_2 = vec2.fit_transform(X_train)
    X_test_2 = vec2.transform(X_test)
    model2 = MultinomialNB(alpha=0.5)  # 中等alpha

    # ========== 模型3：Binary + BernoulliNB ==========
    # 特点：关注词是否出现（忽略频率）
    # binary=True：词出现标记为1，不出现标记为0
    vec3 = CountVectorizer(max_features=15000, ngram_range=(1, 2),
                          stop_words='english', max_df=0.7, min_df=3,
                          binary=True)  # 二值化
    X_train_3 = vec3.fit_transform(X_train)
    X_test_3 = vec3.transform(X_test)
    model3 = BernoulliNB(alpha=0.5)  # BernoulliNB专为二值特征设计

    # ========== 训练各个模型 ==========
    print("\n训练模型1 (TF-IDF + MultinomialNB)...")
    model1.fit(X_train_1, y_train)
    acc1 = model1.score(X_test_1, y_test)
    print(f"模型1准确率: {acc1:.4f}")

    print("\n训练模型2 (Count + MultinomialNB)...")
    model2.fit(X_train_2, y_train)
    acc2 = model2.score(X_test_2, y_test)
    print(f"模型2准确率: {acc2:.4f}")

    print("\n训练模型3 (Binary + BernoulliNB)...")
    model3.fit(X_train_3, y_train)
    acc3 = model3.score(X_test_3, y_test)
    print(f"模型3准确率: {acc3:.4f}")

    # ========== 软投票（Soft Voting）==========
    print("\n使用软投票组合预测...")

    # 获取每个模型的概率预测
    # 形状：(n_samples, 2)，每行是[P(负面), P(正面)]
    y_proba_1 = model1.predict_proba(X_test_1)
    y_proba_2 = model2.predict_proba(X_test_2)
    y_proba_3 = model3.predict_proba(X_test_3)

    # 平均概率
    # 例如：对某条评论
    # 模型1: [0.2, 0.8] (80%正面)
    # 模型2: [0.3, 0.7] (70%正面)
    # 模型3: [0.4, 0.6] (60%正面)
    # 平均: [0.3, 0.7] (70%正面) -> 预测为正面
    y_proba_avg = (y_proba_1 + y_proba_2 + y_proba_3) / 3

    # 根据平均概率判断类别
    # [:, 1]取正面概率，>0.5则预测为正面(1)，否则负面(0)
    y_pred_voting = (y_proba_avg[:, 1] > 0.5).astype(int)

    voting_accuracy = accuracy_score(y_test, y_pred_voting)
    print(f"\n投票集成准确率: {voting_accuracy:.4f}")
    print(f"相比最佳单模型({max(acc1, acc2, acc3):.4f})，")
    print(f"{'提升' if voting_accuracy > max(acc1, acc2, acc3) else '略降'} "
          f"{abs(voting_accuracy - max(acc1, acc2, acc3)):.4f}")

    return voting_accuracy


# ==================== 可视化与分析 ====================

def plot_confusion_matrix(y_true, y_pred, title="混淆矩阵"):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['负面', '正面']
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_top_features(model, vectorizer, n_top=20):
    """
    分析最具区分性的特征词

    如何衡量特征重要性？
    - 朴素贝叶斯学习了每个类别下每个词的log概率
    - P(word|positive) vs P(word|negative)
    - 差异越大，说明这个词越能区分正负面

    示例：
    - "excellent": P(word|positive)高，P(word|negative)低 -> 强正面词
    - "terrible": P(word|negative)高，P(word|positive)低 -> 强负面词
    - "movie": 两个概率都高 -> 无区分度

    Args:
        model: 训练好的MultinomialNB模型
        vectorizer: 向量化器
        n_top: 显示Top N个词
    """
    print("\n" + "="*60)
    print("最具区分性的特征词")
    print("="*60)

    # 获取所有特征名（词汇）
    # 例如：['good', 'bad', 'movie', 'excellent', ...]
    feature_names = vectorizer.get_feature_names_out()

    # ========== 获取特征的log概率 ==========
    # model.feature_log_prob_: 形状 (n_classes, n_features)
    # 对MultinomialNB，这是每个类别下每个特征的log概率
    # log_probs[0]: 负面类的log概率向量 [log P(word1|负面), log P(word2|负面), ...]
    # log_probs[1]: 正面类的log概率向量 [log P(word1|正面), log P(word2|正面), ...]
    log_probs = model.feature_log_prob_

    # ========== 计算特征重要性 ==========
    # 重要性 = log P(word|正面) - log P(word|负面)
    # = log [P(word|正面) / P(word|负面)]
    # - 正值：该词在正面评论中更常见（正面词）
    # - 负值：该词在负面评论中更常见（负面词）
    # - 接近0：该词在两类中差不多（无区分度）
    feature_importance = log_probs[1] - log_probs[0]

    # ========== 找出Top正面词 ==========
    # argsort()：返回排序后的索引
    # [-n_top:]: 取最后n_top个（最大值）
    # [::-1]: 反转，从大到小
    top_positive_indices = feature_importance.argsort()[-n_top:][::-1]
    top_positive_words = [(feature_names[i], feature_importance[i])
                          for i in top_positive_indices]

    # ========== 找出Top负面词 ==========
    # [:n_top]: 取前n_top个（最小值）
    top_negative_indices = feature_importance.argsort()[:n_top]
    top_negative_words = [(feature_names[i], feature_importance[i])
                          for i in top_negative_indices]

    print(f"\nTop {n_top} 正面情感词：")
    for word, score in top_positive_words:
        print(f"  {word:20s} {score:8.4f}")

    print(f"\nTop {n_top} 负面情感词：")
    for word, score in top_negative_words:
        print(f"  {word:20s} {score:8.4f}")

    # ========== 可视化 ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 正面词柱状图
    words_pos = [w for w, s in top_positive_words]
    scores_pos = [s for w, s in top_positive_words]
    ax1.barh(words_pos, scores_pos, color='green', alpha=0.6)
    ax1.set_xlabel('特征重要性分数 (log比值)')
    ax1.set_title(f'Top {n_top} 正面情感词')
    ax1.invert_yaxis()  # 反转y轴，让最重要的在上面

    # 负面词柱状图
    words_neg = [w for w, s in top_negative_words]
    scores_neg = [s for w, s in top_negative_words]
    ax2.barh(words_neg, scores_neg, color='red', alpha=0.6)
    ax2.set_xlabel('特征重要性分数 (log比值)')
    ax2.set_title(f'Top {n_top} 负面情感词')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig('top_features.png', dpi=150, bbox_inches='tight')
    print("\n特征重要性图已保存: top_features.png")
    plt.show()


def plot_results_comparison(results):
    """
    比较不同模型的结果

    Args:
        results: 字典，格式 {模型名称: 准确率}
    """
    print("\n" + "="*60)
    print("模型性能对比")
    print("="*60)

    df = pd.DataFrame(list(results.items()),
                      columns=['模型', '准确率'])
    df = df.sort_values('准确率', ascending=False)

    print(df.to_string(index=False))

    # 可视化
    plt.figure(figsize=(12, 6))
    bars = plt.barh(df['模型'], df['准确率'])

    # 为达到85%的模型标记不同颜色
    colors = ['green' if acc >= 0.85 else 'steelblue'
              for acc in df['准确率']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xlabel('准确率')
    plt.title('不同优化策略的性能对比')
    plt.xlim(0.75, max(df['准确率']) * 1.05)

    # 添加85%基准线
    plt.axvline(x=0.85, color='red', linestyle='--',
                linewidth=2, label='目标: 85%')

    # 添加数值标签
    for i, v in enumerate(df['准确率']):
        plt.text(v + 0.002, i, f'{v:.4f}', va='center')

    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==================== 主函数 ====================

def main():
    """主函数：运行所有实验"""

    print("="*60)
    print("IMDB情感分析 - 朴素贝叶斯挑战练习")
    print("目标：准确率 > 85%")
    print("="*60)

    # 1. 加载数据
    (x_train, y_train), (x_test, y_test), word_index = load_imdb_data()

    # 2. 转换为文本
    X_train_text = convert_to_text(x_train, word_index)
    X_test_text = convert_to_text(x_test, word_index)

    # 3. 展示示例
    print("\n示例评论：")
    print("-" * 60)
    print(f"文本: {X_train_text[0][:200]}...")
    print(f"标签: {'正面' if y_train[0] == 1 else '负面'}")

    # 存储结果
    results = {}

    # 4. 基础模型
    baseline_nb, baseline_vec, baseline_acc = baseline_model(
        X_train_text, y_train, X_test_text, y_test
    )
    results['基础模型 (TF-IDF + MNB)'] = baseline_acc

    # 5. 优化策略1：N-gram
    ngram_nb, ngram_vec, ngram_acc = optimized_model_ngram(
        X_train_text, y_train, X_test_text, y_test
    )
    results['N-gram优化'] = ngram_acc

    # 6. 优化策略2：网格搜索
    grid_nb, grid_vec, grid_acc = optimized_model_grid_search(
        X_train_text, y_train, X_test_text, y_test
    )
    results['网格搜索优化'] = grid_acc

    # 7. 优化策略3：特征选择
    fs_nb, fs_vec, fs_acc, fs_selector = optimized_model_feature_selection(
        X_train_text, y_train, X_test_text, y_test
    )
    results['特征选择优化'] = fs_acc

    # 8. 优化策略4：投票集成
    voting_acc = combined_voting_classifier(
        X_train_text, y_train, X_test_text, y_test
    )
    results['投票集成'] = voting_acc

    # 9. 结果对比
    plot_results_comparison(results)

    # 10. 选择最佳模型进行详细分析
    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]

    print("\n" + "="*60)
    print(f"最佳模型: {best_model_name}")
    print(f"最佳准确率: {best_accuracy:.4f}")
    print(f"是否达标 (>85%): {'✓ 是' if best_accuracy > 0.85 else '✗ 否'}")
    print("="*60)

    # 11. 使用网格搜索模型进行详细分析（通常表现最好）
    X_test_vec = grid_vec.transform(X_test_text)
    y_pred = grid_nb.predict(X_test_vec)

    # 混淆矩阵
    plot_confusion_matrix(y_test, y_pred,
                         title=f"混淆矩阵 - {best_model_name}")

    # 特征分析
    analyze_top_features(grid_nb, grid_vec, n_top=20)

    # 12. 保存最佳模型
    print("\n保存最佳模型...")
    model_path = Path('best_model.pkl')
    vectorizer_path = Path('vectorizer.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(grid_nb, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(grid_vec, f)

    print(f"模型已保存到: {model_path}")
    print(f"向量化器已保存到: {vectorizer_path}")

    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()
