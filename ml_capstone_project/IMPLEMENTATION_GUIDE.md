# 📋 ML Capstone Project - 完整实施指南

> 本文档是 `ml_capstone_project` 综合实战模块的详细实施蓝图
> 记录所有需要创建的文件、内容要求和实施顺序
> 供 Claude AI 逐步完善整个模块

---

## 📊 项目概览

**模块名称**：机器学习综合实战项目 (ML Capstone Project)
**定位**：监督学习和无监督学习后的桥接模块
**核心目标**：教会学习者面对陌生数据的完整 ML 问题解决思维

**预计工作量**：
- 总文件数：约 80-100 个文件
- 核心文档：10 个 Markdown 文档（重点）
- Notebook：约 15-20 个教学 Notebook
- Python 代码：约 15-20 个 .py 文件
- 配置和说明：约 10-15 个辅助文件

---

## 🎯 实施优先级

### P0 - 核心价值（必须完成）⭐⭐⭐
1. ✅ 目录结构创建
2. ⏳ **IMPLEMENTATION_GUIDE.md**（本文档）
3. ⏳ **ML_WORKFLOW_GUIDE.md** - 完整决策树
4. ⏳ **algorithm_comparison_table.md** - 算法对比表
5. ⏳ **CAPSTONE_CHECKLIST.md** - 能力检查清单
6. ⏳ 综合项目 Phase 1-4（基础工作流）
7. ⏳ main_workflow.py + config.py
8. ⏳ src/ 模块化代码（8个模块）

### P1 - 重要补充（增强价值）⭐⭐
9. ⏳ 综合项目 Phase 5-6（整合与总结）
10. ⏳ 第1-5章理论框架内容
11. ⏳ 代码模板库（第8章）
12. ⏳ 决策模板库（第7章）

### P2 - 可选内容（扩展增强）⭐
13. ⏳ 交互式工具（algorithm_selector.ipynb）
14. ⏳ 未来扩展（第9章）
15. ⏳ 迷你项目案例

---

## 📁 完整文件清单与实施计划

### 一、根目录文件（7个）

#### 1. README.md ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0
**预计篇幅**：300-400 行

**内容要求**：
```markdown
# 机器学习综合实战项目

## 模块简介
- 项目定位和核心价值
- 与其他模块的区别
- 学习者收获

## 目录结构说明
- 9个章节的详细介绍
- 文件组织逻辑

## 快速开始
- 环境准备
- 数据下载
- 使用流程（3种学习路径）

## 学习路线图
- 2-3周学习计划
- 每周学习目标
- 完成标准

## 常见问题 Q&A
```

---

#### 2. START_HERE.md ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0
**预计篇幅**：150-200 行

**内容要求**：
```markdown
# 从这里开始

## 学习前的准备
- [ ] 已完成监督学习模块
- [ ] 已完成无监督学习模块
- [ ] 熟悉 Python 和 scikit-learn
- [ ] 完成至少 3 个实战项目

## 三种学习路径

### 路径1：完整学习（推荐）
Week 1 → Week 2 → Week 3

### 路径2：快速上手
直接进入第6章综合项目

### 路径3：文档参考
按需查阅决策树和算法对比表

## 第一步：阅读决策树
ML_WORKFLOW_GUIDE.md

## 第二步：开始综合项目
06_comprehensive_project/

## 获取帮助
```

---

#### 3. ML_WORKFLOW_GUIDE.md ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0（最重要！）
**预计篇幅**：800-1000 行

**内容要求**：
```markdown
# 机器学习完整工作流程决策树

## 第一部分：数据诊断阶段

### 决策点1：数据有标签吗？
├─ YES → 监督学习路径
│   └─ 决策点2：目标变量类型？
│       ├─ 连续值 → 回归问题
│       │   └─ 决策点3：数据量大小？
│       │       ├─ < 1万 → 线性回归/决策树
│       │       ├─ 1-10万 → 随机森林/XGBoost
│       │       └─ > 10万 → LightGBM/神经网络
│       │
│       └─ 离散值 → 分类问题
│           └─ 决策点4：类别是否平衡？
│               ├─ 平衡 → 标准分类算法
│               └─ 不平衡 → SMOTE/调整权重
│
└─ NO → 无监督学习路径
    └─ 决策点2：目标是什么？
        ├─ 分群 → 聚类算法
        ├─ 降维 → PCA/t-SNE/UMAP
        └─ 异常 → Isolation Forest/One-Class SVM

### 决策点X：能否结合监督+无监督？
场景1：先聚类分群，再对每群分别建模
场景2：PCA降维后再分类/回归
场景3：异常分数作为新特征
场景4：聚类标签作为新特征

## 第二部分：问题定义阶段
- 业务问题 → ML问题映射
- 成功指标选择
- 基准性能设定

## 第三部分：算法选择阶段
（详细的算法选择矩阵）

## 第四部分：数据处理阶段
- 缺失值处理决策树
- 特征工程决策树
- 数据标准化决策树

## 第五部分：模型训练阶段
- 交叉验证策略
- 超参数调优方法
- 模型融合策略

## 第六部分：模型评估阶段
- 指标选择决策树
- 业务价值评估

## 完整案例演示
从陌生数据到最终方案的完整决策过程
```

**实施说明**：
- 这是整个模块最核心的文档
- 需要大量的决策树图（可以用 Mermaid 语法）
- 每个决策点都要有：判断标准、典型案例、常见错误
- 参考医生诊断手册的风格

---

#### 4. CAPSTONE_CHECKLIST.md ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0
**预计篇幅**：200-300 行

**内容要求**：
```markdown
# 综合能力检查清单

## 使用说明
本清单用于自我评估和面试准备

## 1. 技术能力（40分）

### 1.1 数据诊断能力（10分）
- [ ] 能快速了解数据规模、类型、分布（2分）
- [ ] 能识别缺失值、异常值模式（2分）
- [ ] 能判断问题类型（监督/无监督）（3分）
- [ ] 能评估特征质量和数据价值（3分）

### 1.2 特征工程能力（10分）
- [ ] 熟练使用多种编码方法（2分）
- [ ] 能设计交互特征和聚合特征（3分）
- [ ] 能进行特征选择和降维（3分）
- [ ] 理解特征工程的业务意义（2分）

### 1.3 算法选择能力（10分）
...

### 1.4 模型评估能力（10分）
...

## 2. 决策能力（30分）
...

## 3. 工程能力（20分）
...

## 4. 业务能力（10分）
...

## 总分评估
- 90-100分：优秀，可独立负责项目
- 75-89分：良好，可在指导下工作
- 60-74分：及格，需要继续学习
- < 60分：需要重新学习相关内容

## 能力提升建议
```

---

#### 5. requirements.txt ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

**内容**：
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
lightgbm>=3.3.0
imbalanced-learn>=0.9.0
```

---

#### 6. .gitignore ⭐
**状态**：⏳ 待创建
**优先级**：P2

---

#### 7. CHANGELOG.md ⭐
**状态**：⏳ 待创建
**优先级**：P2

---

### 二、第1章：数据诊断框架（3-4个文件）

**目录**：`01_data_diagnosis_framework/`

#### 2.1 data_diagnosis_checklist.md ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1
**预计篇幅**：150-200 行

**内容要求**：
```markdown
# 数据诊断检查清单

## 第一步：数据概览
- [ ] 数据规模：__ 行 × __ 列
- [ ] 内存占用：__ MB/GB
- [ ] 数据来源：________
- [ ] 收集时间：________

## 第二步：特征分析
- [ ] 数值特征数量：__
- [ ] 分类特征数量：__
- [ ] 日期时间特征：__
- [ ] 文本特征数量：__

## 第三步：数据质量
- [ ] 缺失值比例：__%
- [ ] 重复行数量：__
- [ ] 异常值检测：__

## 第四步：目标变量分析
...

## 诊断报告模板
```

---

#### 2.2 data_quality_analysis.ipynb ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1
**类型**：Jupyter Notebook

**Notebook 结构**：
```
1. 导入库和数据
2. 数据概览分析
   - 基本信息展示
   - 内存使用分析
3. 特征类型识别
   - 自动识别数值/分类特征
   - 可视化特征分布
4. 数据质量检查
   - 缺失值分析
   - 异常值检测
   - 重复值检查
5. 相关性分析
6. 生成诊断报告
```

---

#### 2.3 problem_type_identifier.py ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1
**类型**：Python 工具脚本

**功能需求**：
```python
class ProblemTypeIdentifier:
    """自动识别ML问题类型的工具类"""

    def identify(self, df, target_col=None):
        """
        识别问题类型

        Returns:
            dict: {
                'has_target': bool,
                'problem_type': 'regression'/'classification'/'clustering'/'...',
                'target_info': {...},
                'recommendations': [...]
            }
        """
        pass
```

---

### 三、第2章：问题定义指南（3个文件）

**目录**：`02_problem_definition_guide/`

#### 3.1 problem_type_decision_tree.md ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0

---

#### 3.2 business_to_ml_mapping.ipynb ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

#### 3.3 success_metrics_guide.md ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

### 四、第3章：算法选择矩阵（4个文件）⭐⭐⭐

**目录**：`03_algorithm_selection_matrix/`

#### 4.1 algorithm_comparison_table.md ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0（最重要！）
**预计篇幅**：500-700 行

**内容要求**：
```markdown
# 机器学习算法全面对比表

## 一、监督学习算法对比

### 1. 线性回归 (Linear Regression)
**适用场景**：
- 数据量：100+ 样本
- 特征关系：线性关系
- 目标变量：连续值
- 典型应用：房价预测、销售预测

**优点**：
- 简单快速，易于解释
- 对线性关系建模效果好
- 计算复杂度低 O(n*p^2)

**缺点**：
- 只能建模线性关系
- 对异常值敏感
- 容易欠拟合

**参数调优**：
- 无主要超参数

**使用建议**：
- 作为 baseline 模型
- 特征工程后可能效果很好
- 结合正则化(Ridge/Lasso)提升效果

---

### 2. 逻辑回归 (Logistic Regression)
...

（继续列出所有8种监督学习算法）

---

## 二、无监督学习算法对比

### 1. K-Means 聚类
...

（继续列出所有6种无监督学习算法）

---

## 三、算法选择决策矩阵

### 按数据量选择
| 数据量 | 推荐算法 |
|--------|---------|
| < 1K | 逻辑回归、决策树 |
| 1K-10K | 随机森林、SVM |
| 10K-100K | XGBoost、LightGBM |
| > 100K | LightGBM、神经网络 |

### 按问题类型选择
...

### 按性能要求选择
...

## 四、常见场景算法推荐
...
```

---

#### 4.2 algorithm_selector.ipynb ⭐⭐
**状态**：⏳ 待创建
**优先级**：P2

---

#### 4.3 supervised_methods_guide.md ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

#### 4.4 unsupervised_methods_guide.md ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

### 五、第4章：数据处理与特征工程（4个文件）

**目录**：`04_preprocessing_and_features/`

#### 5.1 preprocessing_cookbook.md ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

#### 5.2 feature_engineering_patterns.ipynb ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0

---

#### 5.3 imbalanced_data_guide.md ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

#### 5.4 preprocessing_template.py ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

### 六、第5章：模型评估框架（3个文件）

**目录**：`05_model_evaluation/`

#### 6.1 metrics_selection_guide.md ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

#### 6.2 cross_validation_guide.ipynb ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

#### 6.3 business_value_analysis.md ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

### 七、第6章：综合实战项目（约40个文件）⭐⭐⭐

**目录**：`06_comprehensive_project/`

这是整个模块的核心，文件最多，内容最丰富。

#### 7.1 项目根目录文件（5个）

##### 7.1.1 README.md ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0

##### 7.1.2 WORKFLOW_DEMONSTRATION.md ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0

##### 7.1.3 config.py ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0
**预计篇幅**：200-300 行

**内容要求**：
```python
"""
综合实战项目配置文件
管理所有路径、参数、模型配置
"""
from pathlib import Path
import os

class Config:
    """项目配置类"""

    # ==================== 项目路径配置 ====================
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'

    MODELS_DIR = PROJECT_ROOT / 'models'
    FIGURES_DIR = PROJECT_ROOT / 'figures'
    LOGS_DIR = PROJECT_ROOT / 'logs'

    # ==================== 数据文件配置 ====================
    RAW_DATA_FILE = 'telecom_customer_churn.csv'

    # ==================== 随机种子 ====================
    RANDOM_SEED = 42

    # ==================== 数据分割配置 ====================
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2

    # ==================== 特征配置 ====================
    NUMERICAL_FEATURES = [
        'tenure', 'MonthlyCharges', 'TotalCharges'
    ]

    CATEGORICAL_FEATURES = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    TARGET_COL = 'Churn'

    # ==================== 特征工程配置 ====================
    # 数值分箱配置
    BINNING_CONFIG = {
        'tenure': {
            'bins': [0, 12, 24, 48, 100],
            'labels': ['0-1年', '1-2年', '2-4年', '4年+']
        }
    }

    # 交互特征配置
    INTERACTION_FEATURES = [
        ('MonthlyCharges', 'tenure', 'charges_per_tenure'),
        # ...
    ]

    # ==================== 模型训练配置 ====================
    CV_FOLDS = 5
    N_JOBS = -1
    VERBOSE = 1

    # 模型列表
    MODELS = {
        'logistic_regression': {
            'enabled': True,
            'params': {...}
        },
        'decision_tree': {...},
        'random_forest': {...},
        'xgboost': {...},
        'lightgbm': {...},
    }

    # ==================== 评估指标配置 ====================
    CLASSIFICATION_METRICS = [
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    ]

    # ==================== 可视化配置 ====================
    FIGURE_DPI = 100
    FIGURE_SIZE = (10, 6)

    # ==================== 日志配置 ====================
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 创建必要的目录
for dir_path in [Config.DATA_DIR, Config.RAW_DATA_DIR,
                 Config.PROCESSED_DATA_DIR, Config.MODELS_DIR,
                 Config.FIGURES_DIR, Config.LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
```

---

##### 7.1.4 main_workflow.py ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0
**预计篇幅**：300-400 行

**内容要求**：
```python
"""
综合实战项目主工作流程
完整的端到端ML Pipeline
"""
import argparse
from pathlib import Path
import logging
from config import Config
from src import (
    data_diagnosis,
    data_preprocessing,
    feature_engineering,
    supervised_pipeline,
    unsupervised_pipeline,
    model_evaluation,
    visualization
)

def setup_logging():
    """配置日志"""
    pass

def phase1_data_diagnosis(data_path):
    """
    阶段1：数据诊断
    """
    print("\n" + "="*80)
    print("阶段1：数据诊断")
    print("="*80)

    # 加载数据
    df = data_diagnosis.load_data(data_path)

    # 生成诊断报告
    report = data_diagnosis.generate_report(df)

    # 保存报告
    # ...

    return df, report

def phase2_quick_baseline(df):
    """
    阶段2：快速Baseline
    """
    print("\n" + "="*80)
    print("阶段2：快速Baseline")
    print("="*80)

    # 最小特征工程
    # ...

    # 简单模型训练
    # ...

    return baseline_results

def phase3_supervised_solution(df):
    """
    阶段3：监督学习方案
    """
    pass

def phase4_unsupervised_insights(df):
    """
    阶段4：无监督学习增强
    """
    pass

def phase5_integrated_approach(df, supervised_results, unsupervised_results):
    """
    阶段5：方法整合
    """
    pass

def phase6_final_solution(all_results):
    """
    阶段6：最终方案
    """
    pass

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ML Capstone Project')
    parser.add_argument('--data', type=str,
                       default=Config.RAW_DATA_DIR / Config.RAW_DATA_FILE,
                       help='数据文件路径')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', '1', '2', '3', '4', '5', '6'],
                       help='运行哪个阶段')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（减少模型数量）')

    args = parser.parse_args()

    # 设置日志
    setup_logging()

    # 执行工作流
    if args.phase == 'all' or args.phase == '1':
        df, diagnosis_report = phase1_data_diagnosis(args.data)

    if args.phase == 'all' or args.phase == '2':
        baseline_results = phase2_quick_baseline(df)

    # ... 其他阶段

    print("\n" + "="*80)
    print("✅ 工作流程完成！")
    print("="*80)

if __name__ == '__main__':
    main()
```

---

##### 7.1.5 requirements.txt ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

#### 7.2 src/ 模块化代码（8个模块）⭐⭐⭐

**目录**：`06_comprehensive_project/src/`

##### 7.2.1 __init__.py ⭐
**状态**：⏳ 待创建

##### 7.2.2 data_diagnosis.py ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0

##### 7.2.3 data_preprocessing.py ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0

##### 7.2.4 feature_engineering.py ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0

##### 7.2.5 supervised_pipeline.py ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0

##### 7.2.6 unsupervised_pipeline.py ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0

##### 7.2.7 model_evaluation.py ⭐⭐⭐
**状态**：⏳ 待创建
**优先级**：P0

##### 7.2.8 visualization.py ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

##### 7.2.9 utils.py ⭐⭐
**状态**：⏳ 待创建
**优先级**：P1

---

#### 7.3 Phase 1-6 Notebooks（约15-20个）⭐⭐⭐

详见各 phase 子目录...

---

### 八、第7章：决策模板库（5个文件）

**目录**：`07_decision_templates/`

---

### 九、第8章：代码模板库（5个文件）⭐⭐

**目录**：`08_code_templates/`

---

### 十、第9章：未来扩展（3个文件）

**目录**：`09_future_extensions/`

---

## 📝 实施顺序建议

### 第一轮：核心文档（1-2天）
1. ✅ IMPLEMENTATION_GUIDE.md（本文档）
2. ⏳ ML_WORKFLOW_GUIDE.md
3. ⏳ algorithm_comparison_table.md
4. ⏳ CAPSTONE_CHECKLIST.md

### 第二轮：项目框架（2-3天）
5. ⏳ config.py
6. ⏳ main_workflow.py
7. ⏳ src/ 8个模块

### 第三轮：Phase 1-4（3-4天）
8. ⏳ Phase 1 Notebooks
9. ⏳ Phase 2 Notebooks
10. ⏳ Phase 3 Notebooks
11. ⏳ Phase 4 Notebooks

### 第四轮：Phase 5-6 + 理论章节（2-3天）
12. ⏳ Phase 5-6 Notebooks
13. ⏳ 第1-5章理论内容

### 第五轮：模板和扩展（1-2天）
14. ⏳ 代码模板库
15. ⏳ 决策模板库
16. ⏳ 未来扩展内容

---

## 🎯 质量标准

### 文档质量标准
- [ ] 中文注释详细，每个关键步骤都有解释
- [ ] 包含理论讲解，不仅是代码
- [ ] 有实际案例和可视化
- [ ] 符合 CLAUDE.md 中的教学规范

### 代码质量标准
- [ ] 符合 PEP 8 规范
- [ ] 完整的 docstring
- [ ] 模块化设计，职责分明
- [ ] 可复用性强

### Notebook 质量标准
- [ ] 包含：理论 + 实现 + 可视化 + 练习
- [ ] 循序渐进，由浅入深
- [ ] 输出结果清晰可见

---

## 📊 进度跟踪

**总体进度**：0% (0/100 文件)

### 按优先级统计
- P0 核心文件：0/30 完成
- P1 重要文件：0/40 完成
- P2 可选文件：0/30 完成

### 按模块统计
- ✅ 目录结构：100% 完成
- ⏳ 根目录文档：0% (0/7)
- ⏳ 第1章：0% (0/4)
- ⏳ 第2章：0% (0/3)
- ⏳ 第3章：0% (0/4)
- ⏳ 第4章：0% (0/4)
- ⏳ 第5章：0% (0/3)
- ⏳ 第6章：0% (0/40)
- ⏳ 第7章：0% (0/5)
- ⏳ 第8章：0% (0/5)
- ⏳ 第9章：0% (0/3)

---

## 📌 注意事项

1. **严格遵循教学规范**：参考 CLAUDE.md 中的注释标准
2. **保持风格一致**：参考 `supervised_learning/projects/02_customer_churn_prediction/`
3. **数据使用**：使用 Kaggle Telco Customer Churn 数据集
4. **逐步完善**：优先完成 P0，再完成 P1，最后 P2
5. **测试验证**：每个文件完成后测试运行

---

## 🔄 更新记录

- 2024-11-26：创建本实施指南文档
- 待续...

---

**下一步行动**：开始创建 ML_WORKFLOW_GUIDE.md（核心决策树文档）
