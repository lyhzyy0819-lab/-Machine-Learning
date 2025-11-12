"""
配置文件 - NYC出租车行程时长预测项目
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
TRAIN_FILE = "nyc_taxi_train.csv"
TEST_FILE = "nyc_taxi_test.csv"

# 完整路径
TRAIN_DATA_PATH = RAW_DATA_DIR / TRAIN_FILE
TEST_DATA_PATH = RAW_DATA_DIR / TEST_FILE

# 处理后的数据
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test_processed.csv"

# ==================== 模型保存路径 ====================
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ==================== 随机种子 ====================
RANDOM_STATE = 42

# ==================== 数据配置 ====================
# 纽约市地理边界（用于异常值检测）
NYC_BOUNDS = {
    'min_latitude': 40.5,
    'max_latitude': 41.0,
    'min_longitude': -74.3,
    'max_longitude': -73.7
}

# 行程时长范围（秒）
TRIP_DURATION_LIMITS = {
    'min': 60,        # 最短1分钟
    'max': 7200       # 最长2小时
}

# 乘客数量范围
PASSENGER_COUNT_LIMITS = {
    'min': 1,
    'max': 6
}

# ==================== 特征工程配置 ====================
# 时间特征
TIME_FEATURES = [
    'year', 'month', 'day', 'weekday', 'hour',
    'is_weekend', 'is_rush_hour', 'is_night'
]

# 高峰时段定义
RUSH_HOURS = {
    'morning': (7, 10),    # 早高峰 7:00-10:00
    'evening': (17, 20)    # 晚高峰 17:00-20:00
}

# 深夜时段定义
NIGHT_HOURS = (0, 6)      # 深夜 0:00-6:00

# 地理特征
GEO_FEATURES = [
    'manhattan_distance',
    'euclidean_distance',
    'bearing',
    'center_latitude',
    'center_longitude'
]

# 交互特征
INTERACTION_FEATURES = [
    'distance_per_passenger',
    'distance_rush_hour',
    'hour_distance'
]

# 所有特征列表
ALL_FEATURES = TIME_FEATURES + GEO_FEATURES + INTERACTION_FEATURES + ['passenger_count']

# ==================== 模型训练配置 ====================
# 训练集/测试集分割比例
TEST_SIZE = 0.2

# 交叉验证折数
CV_FOLDS = 5

# 是否使用样本（用于快速测试）
USE_SAMPLE = False
SAMPLE_SIZE = 10000

# ==================== 模型超参数 ====================
# Ridge回归
RIDGE_PARAMS = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

# Lasso回归
LASSO_PARAMS = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

# ElasticNet
ELASTICNET_PARAMS = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9]
}

# 随机森林
RANDOM_FOREST_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

# XGBoost
XGBOOST_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# ==================== 评估指标 ====================
METRICS = ['r2', 'rmse', 'mae', 'mape']

# 目标性能指标
TARGET_METRICS = {
    'r2': 0.75,      # R² > 0.75
    'rmse': 300,     # RMSE < 300秒 (5分钟)
    'mae': 200       # MAE < 200秒 (3分钟)
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
COLOR_PALETTE = 'Set2'

# 图表保存DPI
SAVE_DPI = 300

# ==================== 日志配置 ====================
LOG_FILE = LOGS_DIR / 'training.log'
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==================== Kaggle数据集配置 ====================
# 如果使用Kaggle数据集
KAGGLE_DATASET = "nyc-taxi-trip-duration"
KAGGLE_COMPETITION = None  # 如果是竞赛数据

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
    print("NYC出租车行程时长预测 - 配置信息")
    print("=" * 60)
    print(f"\n项目根目录: {BASE_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODELS_DIR}")
    print(f"图表目录: {FIGURES_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    print(f"\n随机种子: {RANDOM_STATE}")
    print(f"测试集比例: {TEST_SIZE}")
    print(f"交叉验证折数: {CV_FOLDS}")
    print(f"\n目标性能:")
    for metric, value in TARGET_METRICS.items():
        print(f"  {metric.upper()}: {value}")
    print("\n配置加载成功！")