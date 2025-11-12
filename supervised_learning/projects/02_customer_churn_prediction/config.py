"""
配置文件 - 客户流失预测项目
包含所有项目配置参数、路径和超参数
"""

import os
from pathlib import Path

# ==================== 项目路径配置 ====================
# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 模型目录
MODELS_DIR = BASE_DIR / "models"

# 图表目录
FIGURES_DIR = BASE_DIR / "figures"

# 日志目录
LOGS_DIR = BASE_DIR / "logs"

# 确保目录存在
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== 数据文件路径 ====================
# 原始数据文件名
DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# 完整路径
DATA_PATH = RAW_DATA_DIR / DATA_FILE

# 处理后的数据
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "churn_processed.csv"

# ==================== 模型保存路径 ====================
BEST_MODEL_PATH = MODELS_DIR / "churn_model_best.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ==================== 随机种子 ====================
RANDOM_STATE = 42

# ==================== 数据配置 ====================
# 在线数据源（备用）
ONLINE_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# 目标变量
TARGET_COL = 'Churn'

# ID列
ID_COL = 'customerID'

# ==================== 特征配置 ====================
# 数值型特征
NUMERICAL_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

# 分类型特征（二值）
BINARY_FEATURES = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling'
]

# 多分类特征
CATEGORICAL_FEATURES = [
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod'
]

# ==================== 特征工程配置 ====================
# 是否创建交互特征
CREATE_INTERACTION_FEATURES = True

# 在网时长分组
TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_LABELS = ['0-1年', '1-2年', '2-4年', '4年以上']

# 月费用分组
MONTHLY_CHARGES_BINS = [0, 30, 60, 90, 120]
MONTHLY_CHARGES_LABELS = ['低', '中', '高', '很高']

# ==================== 模型训练配置 ====================
# 训练集/验证集/测试集分割比例
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # 从训练集中分出的验证集比例

# 交叉验证折数
CV_FOLDS = 5

# 是否使用样本（用于快速测试）
USE_SAMPLE = False
SAMPLE_SIZE = 5000

# ==================== SMOTE配置 ====================
# 是否使用SMOTE处理类别不平衡
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 'auto'  # 或指定比例如 0.8
SMOTE_K_NEIGHBORS = 5

# ==================== 模型配置 ====================
# 要训练的模型列表
MODELS_TO_TRAIN = [
    'logistic_regression',
    'decision_tree',
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'lightgbm'
]

# ==================== 模型超参数 ====================
# Logistic Regression
LOGISTIC_PARAMS = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l2'],
    'max_iter': [1000]
}

# Decision Tree
DECISION_TREE_PARAMS = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Random Forest
RANDOM_FOREST_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Gradient Boosting
GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

# XGBoost
XGBOOST_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# LightGBM
LIGHTGBM_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'subsample': [0.8, 0.9, 1.0]
}

# ==================== 超参数搜索配置 ====================
# 是否进行超参数调优
TUNE_HYPERPARAMETERS = True

# 搜索方法: 'grid' 或 'random'
SEARCH_METHOD = 'random'

# RandomizedSearchCV参数
RANDOM_SEARCH_ITERATIONS = 20
RANDOM_SEARCH_CV = 3

# ==================== 评估指标 ====================
# 主要评估指标（用于模型选择）
PRIMARY_METRIC = 'roc_auc'

# 所有评估指标
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# 目标性能指标
TARGET_METRICS = {
    'accuracy': 0.80,
    'precision': 0.75,
    'recall': 0.70,
    'f1': 0.72,
    'roc_auc': 0.85
}

# ==================== 可视化配置 ====================
# 图表风格
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# 图表尺寸
FIGURE_SIZE = (12, 6)
LARGE_FIGURE_SIZE = (15, 10)

# 中文字体配置（根据系统选择）
import platform
system = platform.system()
if system == 'Darwin':  # macOS
    CHINESE_FONT = 'Arial Unicode MS'
elif system == 'Windows':
    CHINESE_FONT = 'SimHei'
else:  # Linux
    CHINESE_FONT = 'WenQuanYi Micro Hei'

# 颜色方案
COLOR_PALETTE = 'husl'

# 图表保存DPI
SAVE_DPI = 300

# ==================== 日志配置 ====================
LOG_FILE = LOGS_DIR / 'churn_prediction.log'
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==================== 其他配置 ====================
# 是否显示进度条
SHOW_PROGRESS = True

# 是否保存中间结果
SAVE_INTERMEDIATE = True

# 警告设置
import warnings
warnings.filterwarnings('ignore')

# NumPy打印精度
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# Pandas显示选项
import pandas as pd
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

if __name__ == '__main__':
    # 测试配置
    print("=" * 60)
    print("客户流失预测 - 配置信息")
    print("=" * 60)
    print(f"\n项目根目录: {BASE_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODELS_DIR}")
    print(f"图表目录: {FIGURES_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    print(f"\n随机种子: {RANDOM_STATE}")
    print(f"测试集比例: {TEST_SIZE}")
    print(f"交叉验证折数: {CV_FOLDS}")
    print(f"使用SMOTE: {USE_SMOTE}")
    print(f"\n目标性能:")
    for metric, value in TARGET_METRICS.items():
        print(f"  {metric.upper()}: {value}")
    print("\n配置加载成功！")
