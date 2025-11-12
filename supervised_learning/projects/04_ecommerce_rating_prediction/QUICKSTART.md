# 电商评分预测项目 - 快速开始指南

欢迎使用电商评分预测项目！本指南将帮助您快速上手。

## 📋 目录

1. [项目简介](#项目简介)
2. [前置要求](#前置要求)
3. [快速安装](#快速安装)
4. [数据下载](#数据下载)
5. [运行项目](#运行项目)
6. [常见问题](#常见问题)

---

## 🎯 项目简介

本项目使用机器学习技术预测Amazon产品评分，包含：

- **回归任务**: 预测具体评分 (1.0-5.0)
- **分类任务**: 预测高/低评分 (>= 4.0 为高评分)

**项目亮点**:
- 完整的数据预处理和特征工程
- 多种机器学习模型对比（线性回归、随机森林、XGBoost等）
- 详细的可视化分析报告
- 支持超参数调优
- TODO注释引导实现

---

## 🔧 前置要求

### 系统要求
- Python 3.8 或更高版本
- 至少 2GB 可用内存
- 至少 1GB 磁盘空间

### 检查Python版本
```bash
python --version
# 或
python3 --version
```

如果Python版本低于3.8，请先升级Python。

---

## 📦 快速安装

### 步骤1: 克隆/下载项目

如果项目在Git仓库中：
```bash
git clone <repository-url>
cd supervised_learning/projects/04_ecommerce_rating_prediction
```

### 步骤2: 创建虚拟环境（推荐）

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 步骤3: 安装依赖包

**完整安装（推荐）**:
```bash
pip install -r requirements.txt
```

**最小安装（仅核心功能）**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

**验证安装**:
```bash
python -c "import pandas, numpy, sklearn, matplotlib; print('✓ 核心包安装成功!')"
```

---

## 📥 数据下载

### 方法1: 自动下载（推荐）

#### 步骤1: 安装Kaggle CLI
```bash
pip install kaggle
```

#### 步骤2: 配置Kaggle API

1. 登录 [Kaggle](https://www.kaggle.com)
2. 点击右上角头像 → **Account**
3. 滚动到 **API** 部分
4. 点击 **Create New API Token**
5. 下载 `kaggle.json` 文件

#### 步骤3: 放置凭证文件

**Windows**:
```bash
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\
```

**Linux/Mac**:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 步骤4: 运行下载脚本
```bash
python download_data.py
```

### 方法2: 手动下载

1. 访问 [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
2. 点击 **Download** 下载数据集
3. 解压文件，找到 `amazon.csv`
4. 将文件放到 `data/raw/amazon.csv`

**验证数据文件**:
```bash
python download_data.py --verify
```

---

## 🚀 运行项目

### 方式1: 使用Jupyter Notebook（推荐初学者）

```bash
# 启动Jupyter Notebook
jupyter notebook
```

在浏览器中打开Notebook，逐步执行单元格学习。

**优点**:
- 交互式学习
- 可视化结果即时显示
- 适合探索和调试

### 方式2: 运行Python脚本（推荐）

#### 快速测试（使用样本数据）
```bash
python main.py --sample --quick
```
- `--sample`: 使用500条样本数据
- `--quick`: 仅训练基础模型

**预计运行时间**: 1-2分钟

#### 完整训练（回归任务）
```bash
python main.py
```
**预计运行时间**: 5-15分钟（取决于数据大小）

#### 训练分类模型
```bash
python main.py --task classification
```

#### 同时训练回归和分类模型
```bash
python main.py --task both
```

#### 启用超参数调优
```bash
python main.py --tune
```
**注意**: 调优会显著增加运行时间（可能需要30分钟以上）

#### 更多选项
```bash
# 不训练XGBoost（节省时间）
python main.py --no-xgboost

# 跳过可视化（节省时间）
python main.py --no-viz

# 自定义测试集比例
python main.py --test-size 0.3

# 组合使用
python main.py --sample --quick --no-viz
```

### 方式3: 分步运行各个模块

```bash
# 1. 测试数据加载
python src/data_loader.py

# 2. 测试预处理
python src/data_preprocessing.py

# 3. 测试特征工程
python src/feature_engineering.py

# 4. 测试模型训练
python src/model_training.py

# 5. 运行完整流程
python main.py
```

### 预测新数据

训练完成后，使用预测脚本：

```bash
# 使用默认模型预测
python predict.py

# 指定输入和输出文件
python predict.py --input data/raw/new_data.csv --output predictions.csv

# 使用分类模型
python predict.py --task classification

# 快速测试（预测前100条）
python predict.py --sample 100
```

---

## 📂 项目结构

```
04_ecommerce_rating_prediction/
├── config.py                 # 配置文件（路径、参数）
├── main.py                   # 主程序入口
├── predict.py                # 预测脚本
├── download_data.py          # 数据下载脚本
├── requirements.txt          # 依赖包列表
├── QUICKSTART.md            # 本文件
│
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据
│   └── processed/            # 处理后的数据
│
├── src/                      # 源代码目录
│   ├── data_loader.py        # 数据加载
│   ├── data_preprocessing.py # 数据预处理
│   ├── feature_engineering.py # 特征工程
│   ├── model_training.py     # 模型训练
│   ├── model_evaluation.py   # 模型评估
│   ├── visualization.py      # 可视化
│   └── utils.py              # 工具函数
│
├── models/                   # 模型保存目录
│   ├── rating_regression_model.pkl
│   ├── rating_classification_model.pkl
│   ├── scaler.pkl
│   └── metadata.json
│
├── figures/                  # 图表保存目录
└── logs/                     # 日志保存目录
```

---

## ❓ 常见问题

### 1. 安装依赖时出错

**问题**: `pip install` 失败

**解决方案**:
```bash
# 升级pip
pip install --upgrade pip

# 如果是网络问题，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 逐个安装问题包
pip install pandas numpy scikit-learn
```

### 2. XGBoost安装失败

**问题**: `ERROR: Could not find a version that satisfies the requirement xgboost`

**解决方案**:
```bash
# 方法1: 使用conda安装
conda install -c conda-forge xgboost

# 方法2: 跳过XGBoost，运行时使用 --no-xgboost
python main.py --no-xgboost
```

### 3. Kaggle API配置问题

**问题**: `OSError: Could not find kaggle.json`

**解决方案**:
- 确保 `kaggle.json` 文件在正确位置
  - Windows: `C:\Users\<用户名>\.kaggle\kaggle.json`
  - Linux/Mac: `~/.kaggle/kaggle.json`
- 检查文件权限（Linux/Mac需要600）
  ```bash
  chmod 600 ~/.kaggle/kaggle.json
  ```
- 或者使用手动下载方式

### 4. 内存不足

**问题**: `MemoryError` 或系统卡死

**解决方案**:
```bash
# 使用样本数据
python main.py --sample --sample-size 200

# 跳过耗时模型
python main.py --sample --no-xgboost

# 快速模式
python main.py --sample --quick
```

### 5. 数据文件找不到

**问题**: `FileNotFoundError: data/raw/amazon.csv`

**解决方案**:
```bash
# 验证数据文件
python download_data.py --verify

# 手动创建目录
mkdir -p data/raw

# 重新下载
python download_data.py
```

### 6. 中文显示乱码

**问题**: 图表中文显示为方块

**解决方案**:
在 `config.py` 中添加中文字体配置，或在代码中：
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['PingFang HK']  # Mac
```

### 7. Jupyter Notebook无法启动

**问题**: `jupyter: command not found`

**解决方案**:
```bash
# 安装Jupyter
pip install jupyter notebook

# 或使用JupyterLab
pip install jupyterlab
jupyter lab
```

### 8. 导入模块错误

**问题**: `ModuleNotFoundError: No module named 'src'`

**解决方案**:
```bash
# 确保在项目根目录运行
cd /path/to/04_ecommerce_rating_prediction

# 或设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

---

## 📚 学习路径

### 初学者路径
1. 先运行快速测试: `python main.py --sample --quick`
2. 查看生成的日志和图表
3. 阅读各个模块的代码和TODO注释
4. 在Jupyter Notebook中逐步实验

### 进阶路径
1. 实现所有TODO注释
2. 尝试不同的特征工程方法
3. 调整模型超参数
4. 添加新的模型算法
5. 优化性能和准确率

---

## 📞 获取帮助

如果遇到问题：

1. **查看日志文件**: `logs/main.log`
2. **检查错误信息**: 仔细阅读错误堆栈
3. **搜索文档**: 查看scikit-learn、pandas官方文档
4. **在线社区**: Stack Overflow、GitHub Issues

---

## 🎉 开始探索

现在您已经准备好了！按照以下步骤开始：

```bash
# 1. 下载数据
python download_data.py

# 2. 快速测试
python main.py --sample --quick

# 3. 查看结果
ls models/      # 查看保存的模型
ls figures/     # 查看生成的图表
cat logs/main.log  # 查看日志

# 4. 完整训练
python main.py

# 5. 进行预测
python predict.py
```

祝您学习愉快！ 🚀

---

**最后更新**: 2025-01-12
**项目版本**: 1.0
**Python版本**: 3.8+
