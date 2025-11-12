# 监督学习实战项目集 (Supervised Learning Projects)

> 企业级端到端机器学习项目实战

本目录是**监督学习教程**的第三部分，包含完整的、符合企业规范的实战项目。每个项目都涵盖从数据获取到模型部署的完整流程，将前面学习的基础知识综合应用到真实业务场景中。

---

## 📂 项目列表

### 1. [房价预测 (House Price Prediction)](./01_house_price_prediction/)

**项目类型**: 回归问题
**业务场景**: 房地产价格评估
**数据集**: California Housing Dataset (20,640条记录)

**项目亮点**:
- ✅ 完整的EDA（探索性数据分析）
- ✅ 地理特征工程（经纬度聚类）
- ✅ 多模型对比（线性回归、Ridge、Lasso、随机森林、XGBoost）
- ✅ 模型融合（Stacking）
- ✅ 超参数调优

**技术栈**:
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost
- K-Means聚类

**学习时长**: 4-6小时
**难度**: ⭐⭐⭐

---

### 2. [客户流失预测 (Customer Churn Prediction)](./02_customer_churn_prediction/)

**项目类型**: 二分类问题
**业务场景**: 电信客户流失预警
**数据集**: Telco Customer Churn (7,043条记录)

**项目亮点**:
- ✅ 完整的分类问题流程
- ✅ 类别不平衡处理（SMOTE）
- ✅ 多种评估指标（ROC-AUC、Precision、Recall、F1）
- ✅ 混淆矩阵详细分析
- ✅ 特征重要性解释
- ✅ 业务洞察与决策建议
- ✅ 生产环境部署准备

**技术栈**:
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost, LightGBM
- Imbalanced-learn (SMOTE)

**学习时长**: 5-7小时
**难度**: ⭐⭐⭐⭐

---

### 3. [NYC出租车行程时长预测 (NYC Taxi Trip Duration Prediction)](./03_nyc_taxi_duration_prediction/)

**项目类型**: 回归问题（Kaggle竞赛）
**业务场景**: 出租车行程时长预测系统
**数据集**: NYC Taxi Trip Duration Dataset (1.5M+条记录)

**项目亮点**:
- ✅ 完整的模块化Python代码（.py + .ipynb双模式）
- ✅ Kaggle竞赛标准流程（train_test_split + 测试集预测）
- ✅ 地理特征工程（Haversine距离、方位角）
- ✅ 时间特征工程（高峰时段、周末、深夜）
- ✅ 8种模型对比（线性→Ridge→Lasso→ElasticNet→RF→XGBoost→Stacking）
- ✅ 全量重训练支持（Kaggle最佳实践）
- ✅ 一键生成submission.csv
- ✅ 完整的评估体系（残差分析、学习曲线、特征重要性）

**技术栈**:
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, XGBoost
- Haversine公式、地理计算
- 模块化架构（8个核心模块）

**代码特色**:
- 🏗️ 企业级模块化设计（src/目录结构）
- 📝 完整的中文注释和Docstring
- 🔧 配置管理（config.py）
- 📊 3种使用方式（Python脚本/一键生成/Jupyter）
- 🎯 Kaggle竞赛就绪

**学习时长**: 6-8小时
**难度**: ⭐⭐⭐⭐⭐

---

## 🎯 项目特点

### 企业级标准

每个项目都遵循企业级机器学习项目的完整流程：

1. **业务理解** - 明确业务目标和价值
2. **数据探索** - 全面的EDA和可视化
3. **数据预处理** - 缺失值、异常值、编码
4. **特征工程** - 创建业务相关的新特征
5. **模型训练** - 对比多种算法
6. **模型评估** - 使用合适的评估指标
7. **模型优化** - 超参数调优
8. **模型解释** - 特征重要性、业务洞察
9. **模型部署** - 保存模型、创建预测接口
10. **业务建议** - 将技术结果转化为行动方案

### 代码质量

- ✅ 详细的中文注释
- ✅ 清晰的代码结构
- ✅ 完整的可视化
- ✅ 可复现的结果（设置随机种子）

### 学习价值

- ✅ 适合作为求职作品集
- ✅ 涵盖常见面试考点
- ✅ 体现工程化思维
- ✅ 包含业务思考

---

## 🚀 如何使用

### 环境准备

```bash
# 激活conda环境
conda activate ml_env

# 安装必要的包（如果还没安装）
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn joblib
```

### 运行项目

```bash
# 进入项目目录
cd 01_house_price_prediction/

# 启动Jupyter Notebook
jupyter notebook house_price_prediction.ipynb

# 或使用Jupyter Lab
jupyter lab
```

### 学习建议

1. **先学基础** - 如果是初学者，建议先完成上级目录中的基础教程（01-06）
2. **按顺序学习** - 建议按项目编号顺序学习（难度递增）
3. **完整运行** - 从头到尾运行所有代码单元格
4. **理解业务** - 关注业务背景和实际应用
5. **动手修改** - 尝试修改参数、特征工程等
6. **做笔记** - 记录关键洞察和学习心得

---

## 📊 项目对比

| 项目 | 类型 | 难度 | 时长 | 主要技能点 | 代码结构 |
|------|------|------|------|------------|----------|
| 房价预测 | 回归 | ⭐⭐⭐ | 4-6h | 特征工程、模型融合、地理聚类 | .ipynb |
| 客户流失预测 | 分类 | ⭐⭐⭐⭐ | 5-7h | 类别不平衡、ROC-AUC、业务洞察 | .ipynb |
| **NYC出租车** | **回归/Kaggle** | **⭐⭐⭐⭐⭐** | **6-8h** | **Haversine、Kaggle流程、模块化** | **.py + .ipynb** |

### 项目特色对比

| 特性 | 项目1 | 项目2 | 项目3 (NYC) |
|------|-------|-------|-------------|
| 代码结构 | Notebook | Notebook | 模块化(.py) + Notebook |
| 数据规模 | 2万 | 7千 | **150万** |
| 特征工程 | K-Means聚类 | 编码处理 | **地理+时间+交互** |
| 模型数量 | 6个 | 5个 | **8个** |
| Kaggle支持 | ❌ | ❌ | **✅ 完整支持** |
| 生产部署 | 基础 | 中级 | **高级（模块化）** |
| 文档完整度 | ⭐⭐⭐ | ⭐⭐⭐⭐ | **⭐⭐⭐⭐⭐** |

---

## 🎓 学习路径

### 推荐学习路径
```
第一阶段: 回归基础 + 实战
├─ ../01_linear_regression.ipynb (基础)
├─ ../02_polynomial_regression_regularization.ipynb (基础)
└─ ./01_house_price_prediction/ (实战应用)

第二阶段: 分类基础 + 实战
├─ ../03_logistic_regression.ipynb (基础)
├─ ../04_svm.ipynb (基础)
├─ ../05_tree_ensemble.ipynb (基础)
├─ ../06_model_evaluation.ipynb (基础)
└─ ./02_customer_churn_prediction/ (实战应用)

第三阶段: 高级回归 + Kaggle竞赛 ⭐ NEW
├─ ./03_nyc_taxi_duration_prediction/ (综合实战)
│   ├─ 模块化Python开发
│   ├─ Kaggle竞赛流程
│   ├─ 地理数据处理
│   └─ 大规模数据集
└─ 参加真实Kaggle竞赛

第四阶段: 持续提升
└─ 学习Top Kagglers方案，优化自己的项目
```

### 进阶路径
```
1. 深入理解每个项目的业务背景
2. 尝试不同的特征工程方法
3. 对比更多模型算法
4. 优化超参数提升性能
5. 学习模型部署和监控
6. 完成自己的项目
```

---

## 💡 项目扩展建议

学完现有项目后，可以尝试：

### 扩展现有项目
- **房价预测**: 尝试使用深度学习、添加更多外部数据
- **客户流失**: 添加时间序列特征、尝试集成学习

### 新项目方向
1. **推荐系统** - 电影/商品推荐
2. **文本分类** - 情感分析、新闻分类
3. **时间序列** - 股票预测、销量预测
4. **图像分类** - MNIST、CIFAR-10
5. **异常检测** - 信用卡欺诈、设备故障

---

## 📚 参考资源

### 数据集来源
- **Kaggle**: https://www.kaggle.com/datasets
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/
- **Scikit-learn内置数据集**: https://scikit-learn.org/stable/datasets.html

### 学习资源
- **Scikit-learn文档**: https://scikit-learn.org/
- **Kaggle Learn**: https://www.kaggle.com/learn
- **机器学习实战社区**: https://www.kaggle.com/competitions

### 优秀项目参考
- **Kaggle Notebooks**: 学习Top选手的解决方案
- **GitHub ML Projects**: 搜索机器学习项目

---

## 🔧 常见问题

### Q1: 这些项目适合什么水平的学习者？

**A**:
- 需要具备Python基础
- 了解基本的机器学习概念
- 建议先完成 `supervised_learning` 基础教程

### Q2: 如何获取数据集？

**A**: 每个项目的notebook中都包含数据下载说明，支持：
- Kaggle API自动下载
- 在线URL加载
- 手动下载链接

### Q3: 项目可以用于求职作品集吗？

**A**: 可以！但建议：
- 理解每一步的原理
- 能够解释你的决策（为什么选择这个模型？）
- 最好做一些个性化修改或改进
- 添加自己的业务思考

### Q4: 运行项目需要GPU吗？

**A**: 不需要。所有项目都可以在CPU上运行（10-30分钟内完成）。

### Q5: 如何提高模型性能？

**A**:
1. 更多的特征工程
2. 尝试不同的算法
3. 超参数调优
4. 模型融合
5. 获取更多/更好的数据

---

## 📝 项目模板结构

每个项目都遵循统一的结构：

```
project_name/
├── project_name.ipynb          # 主notebook文件
├── data/                       # 数据目录
│   └── raw/                    # 原始数据
│   └── processed/              # 处理后的数据
├── models/                     # 训练好的模型
│   ├── model.pkl               # 最终模型
│   ├── scaler.pkl              # 数据预处理器
│   └── metadata.pkl            # 模型元信息
├── figures/                    # 可视化图表（可选）
└── README.md                   # 项目说明
```

---

## ✅ 学习检查清单

完成每个项目后，检查是否掌握：

### 通用技能
- [ ] 能够独立进行EDA
- [ ] 理解数据预处理的重要性
- [ ] 会进行特征工程
- [ ] 能够选择合适的模型
- [ ] 理解评估指标的含义
- [ ] 会进行超参数调优
- [ ] 能够解释模型结果

### 房价预测项目
- [ ] 理解回归问题
- [ ] 掌握地理特征处理
- [ ] 会使用模型融合
- [ ] 理解R²、RMSE等指标

### 客户流失预测项目
- [ ] 理解分类问题
- [ ] 掌握类别不平衡处理
- [ ] 理解ROC-AUC、Precision、Recall
- [ ] 会分析混淆矩阵
- [ ] 能够提出业务建议

### NYC出租车项目
- [ ] 掌握模块化Python开发
- [ ] 理解Kaggle竞赛流程
- [ ] 会使用Haversine公式计算距离
- [ ] 掌握地理和时间特征工程
- [ ] 理解全量重训练的意义
- [ ] 会生成Kaggle提交文件
- [ ] 理解train_test_split在竞赛中的作用
- [ ] 能够阅读和编写模块化代码

---

## 🎯 下一步

完成这些项目后，你可以：

1. **参加Kaggle竞赛**
   - 应用所学技能
   - 与全球ML工程师竞争
   - 获得真实项目经验

2. **深入某个领域**
   - 计算机视觉
   - 自然语言处理
   - 推荐系统
   - 时间序列

3. **学习深度学习**
   - PyTorch/TensorFlow
   - 神经网络基础
   - CNN、RNN、Transformer

4. **实际项目应用**
   - 找真实业务问题
   - 完整的MLOps流程
   - 模型部署上线

---

## 📞 反馈与贡献

如果你：
- 发现代码错误
- 有改进建议
- 想要贡献新项目
- 需要帮助或有问题

欢迎反馈！

---

**祝学习顺利！记住：最好的学习方式是动手实践！** 🚀
