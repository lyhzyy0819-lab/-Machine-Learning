import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# sklearn相关
from sklearn.datasets import load_iris, make_classification, fetch_20newsgroups
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# 文本处理
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import warnings

# 忽略所有警告
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['font.sans-serif'] = [
        'Arial Unicode MS',  # macOS通用
        'PingFang SC',       # macOS系统字体
        'STHeiti',           # 华文黑体
        'Heiti TC',          # 黑体-繁
        'SimHei',            # 黑体
    ]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

np.random.seed(42)

# 加载部分类别的新闻数据（使用本地缓存）
import pickle
from pathlib import Path

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

# 从本地缓存加载数据
data_home = Path("/Users/lyh/scikit_learn_data/20news_home")

# 加载训练集
with open(data_home / "20news-bydate_py3.pkz", 'rb') as f:
    train_data_all = pickle.load(f)

# 加载测试集
with open(data_home / "20news-bydate-test_py3.pkz", 'rb') as f:
    test_data_all = pickle.load(f)


# 筛选指定类别
def filter_categories(data, categories):
    """筛选指定类别的数据"""
    # 获取类别索引
    category_indices = [i for i, name in enumerate(data['target_names']) if name in categories]
    category_map = {old_idx: new_idx for new_idx, old_idx in enumerate(category_indices)}

    # 筛选数据
    filtered_data = []
    filtered_target = []
    for text, label in zip(data['data'], data['target']):
        if label in category_indices:
            filtered_data.append(text)
            filtered_target.append(category_map[label])

    return {
        'data': filtered_data,
        'target': np.array(filtered_target),
        'target_names': [data['target_names'][i] for i in category_indices]
    }


newsgroups_train = filter_categories(train_data_all, categories)
newsgroups_test = filter_categories(test_data_all, categories)

print(f"训练样本数: {len(newsgroups_train['data'])}")
print(f"测试样本数: {len(newsgroups_test['data'])}")
print(f"类别: {newsgroups_train['target_names']}")

# 查看示例
print("\n示例文本：")
print(newsgroups_train['data'][0][:200] + "...")
print(f"类别: {newsgroups_train['target_names'][newsgroups_train['target'][0]]}")