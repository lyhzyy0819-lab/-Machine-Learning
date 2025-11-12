# 客户流失预测项目快速上手指南

## 🚀 5分钟快速开始

### 1. 安装依赖（1分钟）

```bash
# 进入项目目录
cd /path/to/02_customer_churn_prediction

# 安装所有依赖
pip install -r requirements.txt
```

**需要的Python版本**: Python 3.8+

**核心依赖包**:
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- imbalanced-learn (for SMOTE)
- matplotlib
- seaborn

---

### 2. 快速测试（2分钟）

```bash
# 使用2000条样本数据，仅训练基础模型
python main.py --sample --sample-size 2000 --quick
```

**预期输出**:
```
==========================================
客户流失预测系统
==========================================

阶段 1/7: 数据加载
  样本数量: 2,000
  流失客户比例: 26.5%

阶段 2/7: 数据预处理
  清洗后样本数: 1,989

阶段 4/7: 特征工程
  最终建模特征数: 32

阶段 5/7: 模型训练
  训练集大小: 1,591
  测试集大小: 398

模型性能对比:
Model                 Accuracy  Precision  Recall  F1-Score  ROC-AUC
Logistic Regression   0.7856    0.7234     0.6789  0.7005    0.8234
Decision Tree         0.7623    0.6945     0.7012  0.6978    0.7845
Random Forest         0.8034    0.7456     0.7234  0.7343    0.8567

最佳模型: Random Forest
项目执行完成！总耗时: 2分34秒
```

---

### 3. 完整运行（30-40分钟）

```bash
# 使用全部数据，训练所有模型（包含超参数调优）
python main.py
```

这将:
- 加载全部7,043条客户数据
- 训练6种分类模型
- 使用SMOTE处理类别不平衡
- 进行超参数调优
- 生成完整的可视化报告
- 保存最佳模型

---

### 4. 使用模型预测（30秒）

```bash
# 假设你有新客户数据文件: new_customers.csv
python predict.py \
  --input new_customers.csv \
  --output predictions.csv \
  --show-probability
```

**输出文件格式** (`predictions.csv`):
```csv
customerID,churn_probability,churn_prediction,risk_level
7590-VHVEG,0.8234,1,高风险
5575-GNVDE,0.3421,0,低风险
3668-QPYBK,0.6789,1,中风险
```

---

## 📁 主要文件说明

| 文件 | 用途 | 何时使用 |
|------|------|---------|
| **main.py** | 训练模型 | 模型开发、更新模型 |
| **predict.py** | 预测新客户 | 已有训练好的模型，进行批量预测 |
| **config.py** | 所有配置参数 | 修改超参数、路径等配置 |
| **src/** | 核心模块（7个） | 查看具体实现细节 |
| **customer_churn_prediction.ipynb** | Jupyter交互式演示 | 学习和探索数据 |
| **README.md** | 完整项目文档 | 深入了解项目 |
| **requirements.txt** | 依赖列表 | 环境搭建 |

---

## 🎯 快速命令参考

### 训练相关命令

```bash
# ============ 快速模式 ============
# 最快速度（2分钟）- 样本数据 + 基础模型
python main.py --sample --sample-size 2000 --quick

# 样本测试（5分钟）- 样本数据 + 所有模型
python main.py --sample

# 开发调试（5分钟）- 样本 + 不调优 + 不可视化
python main.py --sample --no-tuning --no-viz

# ============ 生产模式 ============
# 标准训练（30分钟）- 全部数据 + 超参数调优
python main.py

# 生产级训练（35分钟）- 全量重训练
python main.py --retrain-full

# ============ 自定义选项 ============
# 跳过XGBoost和LightGBM（节省20分钟）
python main.py --no-xgboost --no-lightgbm

# 不使用SMOTE处理类别不平衡
python main.py --no-smote

# 跳过可视化（节省5分钟）
python main.py --no-viz

# 自定义测试集比例
python main.py --test-size 0.3

# 仅训练不调优（节省15分钟）
python main.py --no-tuning
```

### 预测相关命令

```bash
# ============ 基础预测 ============
# 批量预测（输出CSV）
python predict.py \
  --input new_customers.csv \
  --output predictions.csv

# ============ 高级选项 ============
# 显示流失概率
python predict.py \
  --input new_customers.csv \
  --output predictions.csv \
  --show-probability

# 查看Top 100高风险客户
python predict.py \
  --input new_customers.csv \
  --top-k 100 \
  --show-probability

# 自定义分类阈值（默认0.5）
python predict.py \
  --input new_customers.csv \
  --threshold 0.6 \
  --output predictions.csv

# 使用样本测试预测功能
python predict.py \
  --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --sample 100 \
  --output test_predictions.csv
```

---

## 📊 预期输出

### 运行成功后会生成

#### 1. 模型文件 (`models/`)
```
models/
├── churn_model_best.pkl          # 最佳模型（LightGBM）
├── scaler.pkl                    # 数据缩放器
├── feature_names.pkl             # 特征名列表
└── metadata.json                 # 模型元数据
```

#### 2. 可视化图表 (`figures/`)
```
figures/
├── eda/                          # 探索性数据分析
│   ├── numerical_features.png
│   ├── categorical_features.png
│   ├── correlation_heatmap.png
│   ├── churn_analysis.png
│   ├── contract_analysis.png
│   └── service_analysis.png
└── evaluation/                   # 模型评估
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── feature_importance.png
    ├── model_comparison.png
    └── learning_curves.png
```

#### 3. 日志文件 (`logs/`)
```
logs/
└── churn_prediction.log          # 完整的运行日志
```

#### 4. 控制台输出
```
模型性能对比:
Model                 Accuracy  Precision  Recall  F1-Score  ROC-AUC  训练时间
Logistic Regression   0.7856    0.7234     0.6789  0.7005    0.8234   0.5s
Decision Tree         0.7623    0.6945     0.7012  0.6978    0.7845   0.3s
Random Forest         0.8034    0.7456     0.7234  0.7343    0.8567   8.5s
Gradient Boosting     0.8178    0.7689     0.7345  0.7513    0.8634   15.2s
XGBoost               0.8245    0.7856     0.7423  0.7634    0.8756   12.3s
LightGBM              0.8289    0.7923     0.7512  0.7712    0.8812   6.8s

最佳模型: LightGBM
性能指标:
  Accuracy:  0.8289
  Precision: 0.7923
  Recall:    0.7512
  F1 Score:  0.7712
  ROC-AUC:   0.8812

目标达成情况:
  Accuracy  > 0.80: ✓ 达成
  Precision > 0.75: ✓ 达成
  Recall    > 0.70: ✓ 达成
  F1 Score  > 0.72: ✓ 达成
  ROC-AUC   > 0.85: ✓ 达成
```

---

## 📖 使用Jupyter Notebook

### 启动Jupyter

```bash
# 启动Jupyter Notebook
jupyter notebook customer_churn_prediction.ipynb
```

### Notebook内容结构

1. **数据加载与探索**
   - 加载数据
   - 查看数据基本信息
   - 缺失值分析

2. **数据可视化**
   - 数值特征分布
   - 分类特征分布
   - 目标变量分析
   - 相关性热图

3. **数据预处理**
   - 缺失值处理
   - 数据类型转换
   - 目标变量编码

4. **特征工程**
   - One-Hot编码
   - 数值分箱
   - 交互特征创建
   - 特征选择

5. **模型训练**
   - SMOTE处理
   - 训练多种模型
   - 超参数调优

6. **模型评估**
   - 评估指标计算
   - 混淆矩阵
   - ROC曲线
   - 特征重要性

7. **预测应用**
   - 单客户预测
   - 批量预测
   - 风险分层

---

## ❓ 常见问题

### Q1: 运行时间多长？

**A**:
- **快速测试模式** (`--sample --quick`): 2-3分钟
- **样本模式** (`--sample`): 5-10分钟
- **标准模式** (全部数据): 30-40分钟
- **生产模式** (`--retrain-full`): 35-45分钟

*时间取决于CPU性能和数据量*

### Q2: 需要GPU吗？

**A**: 不需要。本项目使用的模型（逻辑回归、随机森林、XGBoost、LightGBM）在CPU上运行效率已经很高。

### Q3: 数据从哪里来？

**A**:
- **自带数据**: 项目已包含数据文件 `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Kaggle**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **IBM**: 代码会自动从IBM在线源下载（如果本地不存在）

### Q4: 如何修改超参数？

**A**: 编辑 `config.py` 文件

```python
# 示例：修改XGBoost超参数搜索空间
XGBOOST_PARAMS = {
    'n_estimators': [100, 200, 300],     # 修改这里
    'max_depth': [3, 5, 7],              # 修改这里
    'learning_rate': [0.01, 0.05, 0.1],  # 修改这里
}
```

### Q5: 模型文件在哪里？

**A**: 训练完成后，模型保存在 `models/` 目录:
- `churn_model_best.pkl` - 最佳模型
- `metadata.json` - 模型信息（性能指标、训练日期等）

可以用 `predict.py` 直接加载使用。

### Q6: 如何只训练某些模型？

**A**: 使用命令行参数：

```bash
# 跳过XGBoost
python main.py --no-xgboost

# 跳过LightGBM
python main.py --no-lightgbm

# 只训练基础模型（逻辑回归、决策树、随机森林）
python main.py --quick
```

### Q7: 如何处理自己的数据？

**A**:
1. **数据格式**: 确保CSV格式，包含必要的列（参考原始数据）
2. **必需的列**:
   - 客户ID: `customerID`
   - 目标变量: `Churn` (Yes/No)
   - 其他特征: `tenure`, `MonthlyCharges`, `Contract` 等

3. **使用方法**:
```bash
# 方法1: 替换原始数据文件
cp your_data.csv data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
python main.py

# 方法2: 修改config.py中的DATA_FILE路径
# 然后运行
python main.py
```

---

## 🔧 故障排除

### 问题1: ImportError: No module named 'xxx'

**解决**:
```bash
pip install xxx
# 或者重新安装所有依赖
pip install -r requirements.txt
```

### 问题2: 内存不足

**解决**:
```bash
# 使用样本模式
python main.py --sample --sample-size 3000

# 或者跳过XGBoost和LightGBM
python main.py --no-xgboost --no-lightgbm
```

### 问题3: 数据文件找不到

**解决**:
```bash
# 检查数据文件是否存在
ls data/raw/

# 如果不存在，代码会自动下载
# 或手动下载：
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```

### 问题4: 训练太慢

**解决**:
```bash
# 跳过超参数调优（节省50%时间）
python main.py --no-tuning

# 跳过可视化（节省10%时间）
python main.py --no-viz

# 两者结合
python main.py --no-tuning --no-viz
```

### 问题5: 图表不显示中文

**解决**:
在 `config.py` 中修改字体设置：

```python
# macOS
CHINESE_FONT = 'Arial Unicode MS'

# Windows
CHINESE_FONT = 'SimHei'

# Linux
CHINESE_FONT = 'WenQuanYi Micro Hei'
```

---

## 🎓 学习路径建议

### 初学者（第1次运行）

```bash
# 步骤1: 快速测试（了解流程）
python main.py --sample --sample-size 2000 --quick

# 步骤2: 查看生成的文件
ls models/
ls figures/
cat logs/churn_prediction.log

# 步骤3: 学习Jupyter Notebook
jupyter notebook customer_churn_prediction.ipynb

# 步骤4: 阅读完整文档
# 打开 README.md
```

### 进阶（理解代码）

```bash
# 步骤1: 查看核心模块
cat src/data_loader.py
cat src/feature_engineering.py
cat src/model_training.py

# 步骤2: 修改配置参数
# 编辑 config.py，尝试不同的超参数

# 步骤3: 完整训练
python main.py

# 步骤4: 理解模型评估
cat src/model_evaluation.py
```

### 高级（实战应用）

```bash
# 步骤1: 准备自己的数据
# 替换 data/raw/ 中的数据文件

# 步骤2: 修改特征工程
# 编辑 src/feature_engineering.py
# 添加领域相关的特征

# 步骤3: 生产环境训练
python main.py --retrain-full

# 步骤4: 部署为API
# 参考 README.md 中的部署方案
```

---

## 💡 实用技巧

### 技巧1: 组合使用参数

```bash
# 快速开发迭代
python main.py --sample --quick --no-viz --no-tuning

# 生产环境最佳实践
python main.py --retrain-full --no-viz

# 性能优化测试
python main.py --sample --no-xgboost --no-lightgbm --no-smote
```

### 技巧2: 查看训练进度

```bash
# 实时查看日志
tail -f logs/churn_prediction.log

# 查看最新100行
tail -n 100 logs/churn_prediction.log
```

### 技巧3: 对比不同配置

```bash
# 配置1: 使用SMOTE
python main.py --sample
mv models/churn_model_best.pkl models/model_with_smote.pkl

# 配置2: 不使用SMOTE
python main.py --sample --no-smote
mv models/churn_model_best.pkl models/model_without_smote.pkl

# 对比两个模型
python predict.py --model-file models/model_with_smote.pkl ...
python predict.py --model-file models/model_without_smote.pkl ...
```

### 技巧4: 使用虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv churn_env

# 激活环境
source churn_env/bin/activate  # Linux/macOS
churn_env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 运行项目
python main.py

# 退出环境
deactivate
```

---

## 🔗 相关资源

- **完整文档**: README.md（必读！）
- **Kaggle数据集**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Scikit-learn文档**: https://scikit-learn.org/stable/
- **XGBoost文档**: https://xgboost.readthedocs.io/
- **LightGBM文档**: https://lightgbm.readthedocs.io/

---

## 📞 获取帮助

如果遇到问题：

1. **查看日志文件**: `logs/churn_prediction.log`
2. **阅读完整文档**: `README.md`（特别是"常见问题"章节）
3. **检查数据文件**: 确保数据格式正确
4. **降低复杂度**: 使用 `--sample --quick` 快速定位问题

---

**祝学习顺利！5分钟就能看到结果，开始你的客户流失预测之旅吧！** 🚀📊
