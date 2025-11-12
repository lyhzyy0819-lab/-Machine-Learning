# 🚀 快速开始 - 监督学习教程

## 第一步：启动Jupyter Lab

```bash
# 1. 打开终端，进入项目目录
cd "/Users/lyh/Desktop/ Machine Learning/supervised_learning"

# 2. 激活conda环境
conda activate ml_env

# 3. 启动Jupyter Lab
jupyter lab
```

## 第二步：开始第一个Notebook

在Jupyter Lab中打开 **`01_linear_regression.ipynb`**

### 学习建议：
1. **按顺序运行每个单元格**（Shift + Enter）
2. **观察输出结果**（图表、数值）
3. **阅读Markdown说明**
4. **修改代码参数，重新运行**
5. **完成练习题**

---

## 学习路线图

```
第1天 (2-3小时)
  → 01_linear_regression.ipynb
  ✓ 理解线性回归
  ✓ 从零实现
  ✓ 评估模型

第2天 (2-3小时)
  → 02_polynomial_regression_regularization.ipynb
  ✓ 多项式回归
  ✓ 过拟合与正则化
  ✓ Ridge vs Lasso

第3-4天 (3-4小时)
  → 03_house_price_prediction.ipynb
  ✓ 完整实战项目
  ✓ 特征工程
  ✓ 模型对比

第5天 (2-3小时)
  → 04_logistic_regression.ipynb
  ✓ 分类问题
  ✓ 逻辑回归
  ✓ 决策边界

第6天 (2-3小时)
  → 05_svm.ipynb
  ✓ 支持向量机
  ✓ 核技巧

第7天 (3-4小时)
  → 06_tree_ensemble.ipynb
  ✓ 决策树
  ✓ 随机森林
  ✓ XGBoost

第8天 (2小时)
  → 07_model_evaluation.ipynb
  ✓ 模型评估
  ✓ 交叉验证

第9-10天 (4-5小时)
  → 08_titanic_project.ipynb
  ✓ Kaggle竞赛
  ✓ 完整流程

第11-12天 (3-4小时)
  → 09_fraud_detection.ipynb
  ✓ 不平衡数据
  ✓ 异常检测
```

---

## 目前已创建的文件

✅ **01_linear_regression.ipynb** - 线性回归完整教程
✅ **02_polynomial_regression_regularization.ipynb** - 多项式回归与正则化
✅ **README.md** - 完整学习指南

🔄 **其他notebook将陆续创建**

---

## 马上开始！

### 现在就执行：
```bash
conda activate ml_env
cd "/Users/lyh/Desktop/ Machine Learning/supervised_learning"
jupyter lab
```

### 然后打开：
**`01_linear_regression.ipynb`**

---

## 遇到问题？

### 环境问题
```bash
# 检查Python版本
python --version  # 应该是3.8

# 检查包是否安装
pip list | grep scikit-learn
pip list | grep pandas
```

### 安装缺失的包
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

## 学习技巧

### ✅ 推荐做法
- 每天固定时间学习
- 完成一个再开始下一个
- 运行所有代码，观察结果
- 完成练习题
- 做笔记

### ❌ 避免
- 只看不做
- 跳过练习
- 追求完美理解才前进

---

**开始你的机器学习之旅吧！** 🎯
