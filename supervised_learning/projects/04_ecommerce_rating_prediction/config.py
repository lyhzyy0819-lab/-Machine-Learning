"""
电商评分预测项目配置文件
包含所有路径、参数和模型配置
"""

from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")


# ==================== 路径配置 ====================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'
FIGURE_DIR = PROJECT_ROOT / 'figures'
LOG_DIR = PROJECT_ROOT / 'logs'

# 数据文件
RAW_DATA_FILE = RAW_DATA_DIR / 'amazon.csv'  # 下载后的数据文件
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / 'processed_data.csv'

# 模型文件
REGRESSION_MODEL_FILE = MODEL_DIR / 'rating_regression_model.pkl'
CLASSIFICATION_MODEL_FILE = MODEL_DIR / 'rating_classification_model.pkl'
SCALER_FILE = MODEL_DIR / 'scaler.pkl'
TFIDF_FILE = MODEL_DIR / 'tfidf_vectorizer.pkl'
METADATA_FILE = MODEL_DIR / 'metadata.json'


# ==================== 数据配置 ====================

# Kaggle数据集信息
KAGGLE_DATASET = 'karkavelrajaj/amazon-sales-dataset'
EXPECTED_COLUMNS = [
    'product_id', 'product_name', 'category',
    'discounted_price', 'actual_price', 'discount_percentage',
    'rating', 'rating_count', 'about_product',
    'user_id', 'user_name', 'review_id',
    'review_title', 'review_content', 'img_link', 'product_link'
]

# 目标变量
TARGET_REGRESSION = 'rating'  # 回归目标：预测评分 (1.0-5.0)
TARGET_CLASSIFICATION = 'high_rating'  # 分类目标：预测高/低评分 (>= 4.0)

# 高评分阈值
HIGH_RATING_THRESHOLD = 4.0


# ==================== 特征工程配置 ====================

# 文本特征
TFIDF_MAX_FEATURES = 100  # TF-IDF提取的最大特征数
TFIDF_MIN_DF = 2  # 最小文档频率
TFIDF_MAX_DF = 0.8  # 最大文档频率
TEXT_COLUMNS = ['review_content', 'review_title', 'about_product']

# 价格分桶
PRICE_BINS = [0, 500, 1000, 2000, 5000, 100000]
PRICE_LABELS = ['very_low', 'low', 'medium', 'high', 'very_high']

# 折扣力度分桶
DISCOUNT_BINS = [0, 20, 40, 60, 100]
DISCOUNT_LABELS = ['low', 'medium', 'high', 'very_high']

# 类别编码
CATEGORY_ENCODING = 'onehot'  # 'onehot' or 'label'


# ==================== 模型训练配置 ====================

# 通用参数
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# 回归模型配置
REGRESSION_MODELS = {
    'linear': {
        'model': LinearRegression(),
        'description': '线性回归（Baseline）'
    },
    'ridge': {
        'model': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'description': 'Ridge回归（L2正则化）',
        'param_grid': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
    },
    'lasso': {
        'model': Lasso(alpha=1.0, random_state=RANDOM_STATE, max_iter=10000),
        'description': 'Lasso回归（L1正则化）',
        'param_grid': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        }
    },
    'random_forest': {
        'model': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'description': '随机森林回归',
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    }
}

# 如果安装了XGBoost，添加XGBoost模型
if HAS_XGBOOST:
    REGRESSION_MODELS['xgboost'] = {
        'model': XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'description': 'XGBoost回归',
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    }

# 分类模型配置
CLASSIFICATION_MODELS = {
    'logistic': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'description': '逻辑回归（Baseline）',
        'param_grid': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
    },
    'svm': {
        'model': SVC(random_state=RANDOM_STATE, probability=True),
        'description': '支持向量机（SVM）',
        'param_grid': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'description': '随机森林分类',
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    }
}

# 如果安装了XGBoost，添加XGBoost分类模型
if HAS_XGBOOST:
    CLASSIFICATION_MODELS['xgboost'] = {
        'model': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'),
        'description': 'XGBoost分类',
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    }


# ==================== 评估指标配置 ====================

# 回归评估指标
REGRESSION_METRICS = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

# 分类评估指标
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']


# ==================== 可视化配置 ====================

# 图表样式
FIGURE_SIZE = (10, 6)
FIGURE_DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# 颜色配置
COLOR_PALETTE = 'Set2'
PRIMARY_COLOR = '#1f77b4'
SECONDARY_COLOR = '#ff7f0e'


# ==================== 日志配置 ====================

# 日志级别
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# 日志格式
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# ==================== 运行模式配置 ====================

# 是否使用样本数据（用于快速测试）
USE_SAMPLE = False
SAMPLE_SIZE = 500

# 是否跳过耗时的模型（如XGBoost）
QUICK_MODE = False

# 是否生成可视化图表
GENERATE_PLOTS = True

# 是否进行超参数调优
TUNE_HYPERPARAMETERS = False


# ==================== 辅助函数 ====================

def create_directories():
    """创建所有必要的目录"""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, FIGURE_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_model_config(task='regression'):
    """
    获取模型配置

    Args:
        task: 'regression' or 'classification'

    Returns:
        模型配置字典
    """
    if task == 'regression':
        return REGRESSION_MODELS
    elif task == 'classification':
        return CLASSIFICATION_MODELS
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'")


if __name__ == '__main__':
    # 创建目录
    create_directories()
    print("✅ 配置加载成功!")
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print(f"📊 数据文件: {RAW_DATA_FILE}")
    print(f"🤖 模型保存目录: {MODEL_DIR}")
    print(f"📈 可用回归模型: {list(REGRESSION_MODELS.keys())}")
    print(f"🎯 可用分类模型: {list(CLASSIFICATION_MODELS.keys())}")
