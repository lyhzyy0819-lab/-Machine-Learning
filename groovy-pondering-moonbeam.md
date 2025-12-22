# 神经网络与深度学习基础模块 - 实施计划

## 📋 项目概述

**目标**: 为机器学习教学项目创建完整的神经网络模块,从感知器到深度学习的系统性学习路径

**用户需求**:
- ✅ 完整路径: 从感知器→MLP→深度学习
- ✅ 先补充数学基础: 矩阵微分和反向传播数学推导
- ✅ 三种实现对比: NumPy从零实现 → PyTorch → sklearn
- ✅ 多样化项目: 经典入门(MNIST/CIFAR) + Kaggle竞赛 + 垂直领域应用

**关键约束**:
- 遵循CLAUDE.md中的教学规范 (高密度注释、从零实现优先、详细可视化)
- 保持与现有模块一致的代码风格和文档格式
- 每个notebook必须包含: 理论→从零实现→sklearn/PyTorch对比→可视化→实战→练习题

---

## 🎯 阶段划分

### 阶段1: 数学基础补充 (3-4天)
**优先级**: 最高 ⭐⭐⭐
**位置**: `/math_foundations/`

补充神经网络所需的核心数学知识

### 阶段2: 神经网络核心模块 (2-3周)
**优先级**: 高 ⭐⭐⭐
**位置**: `/neural_networks/`

创建完整的神经网络教学体系

### 阶段3: 实战项目 (1-2周)
**优先级**: 中 ⭐⭐
**位置**: `/neural_networks/projects/`

多样化的实战项目,覆盖不同应用场景

---

## 📝 详细实施计划

## 阶段1: 数学基础补充

### 1.1 创建矩阵微分教程
**文件**: `/math_foundations/05_matrix_calculus.ipynb`

**内容结构** (预计800-1000行):
1. **基础矩阵导数** (20%)
   - 标量对向量/矩阵的导数
   - 向量对向量的导数(Jacobian矩阵)
   - 矩阵对标量的导数
   - 常用矩阵求导规则速查表

2. **神经网络中的梯度计算** (40%)
   - 损失函数对权重的梯度: ∂L/∂W
   - 损失函数对偏置的梯度: ∂L/∂b
   - 链式法则的矩阵形式
   - 向量化批量梯度计算

3. **实践应用** (40%)
   - 线性层梯度的从零实现
   - 激活函数的导数计算
   - 数值梯度检验(gradient checking)
   - 简单网络的完整梯度计算示例

**可视化要求**:
- 梯度方向的几何意义(3D曲面+梯度向量)
- 数值梯度 vs 解析梯度对比图
- 不同学习率下的参数更新轨迹

### 1.2 创建反向传播数学推导教程
**文件**: `/math_foundations/06_backpropagation_math.ipynb`

**内容结构** (预计1000-1200行):
1. **单层网络的反向传播** (30%)
   - 前向传播: z = Wx + b, a = σ(z)
   - 损失函数: L = loss(a, y)
   - 反向传播完整推导: ∂L/∂W, ∂L/∂b, ∂L/∂x
   - 常见激活函数的导数(Sigmoid, Tanh, ReLU)

2. **多层网络的链式法则** (40%)
   - 两层网络的完整推导
   - 误差从输出层反向传播到隐藏层
   - 梯度消失/爆炸的数学根源
   - 残差连接的数学解释

3. **向量化实现** (30%)
   - 批量数据的反向传播
   - 广播机制在梯度计算中的应用
   - 从零实现2层网络的前向和反向传播
   - 与自动微分对比验证

**可视化要求**:
- 计算图的可视化(节点+边+梯度流)
- 梯度消失问题演示(不同深度的梯度范数)
- 激活函数导数曲线对比

### 1.3 更新数学基础模块文档
**文件**: `/math_foundations/README.md`

- 添加新章节的学习指南
- 更新学习路线图
- 添加神经网络数学知识导航

---

## 阶段2: 神经网络核心模块

### 2.1 创建模块基础结构

**目录结构**:
```
neural_networks/
├── README.md                              # 模块总览
├── START_HERE.md                          # 学习指南
├── 01_perceptron_and_history.ipynb       # 感知器与历史
├── 02_mlp_and_forward_propagation.ipynb  # 多层感知器与前向传播
├── 03_backpropagation_algorithm.ipynb    # 反向传播算法
├── 04_activation_functions.ipynb         # 激活函数详解
├── 05_loss_functions_and_optimizers.ipynb # 损失函数与优化器
├── 06_regularization_techniques.ipynb    # 正则化技术
├── 07_batch_normalization.ipynb          # 批标准化
├── 08_deep_learning_frameworks.ipynb     # PyTorch实践
├── 09_hyperparameter_tuning.ipynb        # 超参数调优
├── 10_advanced_topics.ipynb              # 高级主题(权重初始化等)
│
└── projects/
    ├── README.md
    ├── 01_mnist_digit_recognition/        # MNIST手写数字识别
    ├── 02_cifar10_image_classification/   # CIFAR-10图像分类
    ├── 03_kaggle_competition/              # Kaggle竞赛项目
    └── 04_vertical_domain_application/     # 垂直领域应用
```

### 2.2 核心教程详细设计

#### Notebook 1: 感知器与历史 (01_perceptron_and_history.ipynb)
**预计行数**: 1000-1200行

**内容大纲**:
1. **生物神经元启发** (10%)
   - 神经元结构图解
   - 从生物到人工的简化过程

2. **单层感知器** (30%)
   - McCulloch-Pitts神经元
   - Rosenblatt感知器算法
   - 感知器学习规则的数学推导
   - 从零实现感知器类
   - 线性可分问题的可视化

3. **XOR问题与局限性** (25%)
   - Minsky的批判
   - 线性不可分问题演示
   - XOR问题无法解决的证明
   - 多层网络的必要性

4. **实践项目** (35%)
   - AND/OR/NOT逻辑门实现
   - 可视化决策边界
   - 感知器学习过程动画
   - 练习: 尝试用单层感知器解决XOR(失败案例)

**从零实现要求**:
```python
class PerceptronFromScratch:
    """
    从零实现的感知器算法

    参数说明、数学公式、学习规则详解
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        pass

    def fit(self, X, y):
        """逐步展示权重更新过程"""
        pass

    def predict(self, X):
        """激活函数: step function"""
        pass
```

**可视化要求**:
- 3个子图: AND门、OR门、XOR门的决策边界
- 学习曲线: 训练误差随迭代次数的变化
- 权重向量的演化动画

---

#### Notebook 2: 多层感知器与前向传播 (02_mlp_and_forward_propagation.ipynb)
**预计行数**: 1500-1800行

**内容大纲**:
1. **MLP架构** (20%)
   - 输入层、隐藏层、输出层
   - 全连接网络结构
   - 参数数量计算
   - 万能近似定理简介

2. **前向传播详解** (30%)
   - 单个神经元的计算
   - 层级计算的矩阵表示
   - 批处理的向量化实现
   - 从零实现完整的前向传播

3. **激活函数初步** (20%)
   - Sigmoid函数及其导数
   - 为什么需要非线性激活函数
   - 可视化不同激活函数的效果

4. **实践应用** (30%)
   - 从零实现2层MLP
   - 解决XOR问题(证明多层网络的能力)
   - 非线性函数拟合实验
   - 决策边界可视化

**从零实现要求**:
```python
class MLPFromScratch:
    """
    从零实现的多层感知器

    完整的前向传播实现,暂不包含反向传播
    """
    def __init__(self, layer_sizes, activation='sigmoid'):
        """
        layer_sizes: [input_size, hidden1_size, hidden2_size, ..., output_size]
        """
        self.weights = []  # 权重矩阵列表
        self.biases = []   # 偏置向量列表
        self._initialize_parameters()

    def forward(self, X):
        """
        详细的前向传播步骤
        返回所有中间激活值(用于反向传播)
        """
        pass
```

**可视化要求**:
- XOR问题的解决: 输入空间 → 隐藏层表示 → 输出决策边界
- 函数拟合: 原始函数 vs 网络近似
- 网络结构图: 节点+连接+权重标注

---

#### Notebook 3: 反向传播算法 (03_backpropagation_algorithm.ipynb)
**预计行数**: 2000-2500行 (最核心的notebook)

**内容大纲**:
1. **理论基础回顾** (15%)
   - 链式法则的矩阵形式
   - 计算图的概念
   - 前向模式 vs 反向模式自动微分

2. **单层网络反向传播** (25%)
   - 完整的数学推导(参考06_backpropagation_math.ipynb)
   - 代码逐行实现
   - 中间变量的形状标注
   - 梯度检查验证正确性

3. **多层网络反向传播** (30%)
   - 误差的逐层反向传播
   - 权重和偏置的梯度计算
   - 完整的MLPWithBackprop类实现
   - 训练循环: epoch → batch → forward → backward → update

4. **实践验证** (30%)
   - MNIST简化数据集(100个样本)训练
   - 损失曲线、准确率曲线
   - 数值梯度 vs 反向传播梯度对比
   - 梯度消失/爆炸问题演示
   - 练习: 实现3层网络的反向传播

**从零实现要求**:
```python
class MLPWithBackpropagation:
    """
    完整的MLP实现,包含反向传播
    """
    def backward(self, X, y):
        """
        反向传播算法

        步骤:
        1. 前向传播保存所有激活值
        2. 计算输出层误差 δ^L
        3. 反向传播误差: δ^(l-1) = (W^l)^T δ^l ⊙ σ'(z^(l-1))
        4. 计算梯度: ∂L/∂W^l = δ^l (a^(l-1))^T
        5. 返回所有权重和偏置的梯度
        """
        pass

    def train(self, X, y, epochs, learning_rate, batch_size):
        """
        完整的训练循环
        """
        pass

    def gradient_check(self, X, y, epsilon=1e-7):
        """
        数值梯度检验
        """
        pass
```

**可视化要求**:
- 计算图可视化(前向和反向的数据流)
- 训练过程: 损失、准确率、梯度范数
- 权重直方图: 训练前 vs 训练后
- 激活值分布: 各层的激活值统计

---

#### Notebook 4: 激活函数详解 (04_activation_functions.ipynb)
**预计行数**: 1200-1500行

**内容大纲**:
1. **为什么需要激活函数** (15%)
   - 无激活函数 = 线性变换
   - 数学证明: 多层线性网络等价于单层

2. **经典激活函数** (40%)
   - Sigmoid: 饱和问题、梯度消失
   - Tanh: 相比Sigmoid的优势
   - ReLU: Dead ReLU问题
   - Leaky ReLU / PReLU / ELU
   - 每个函数的数学定义+导数+代码实现

3. **现代激活函数** (25%)
   - Swish / GELU
   - Mish
   - 使用场景和选择建议

4. **实验对比** (20%)
   - 相同网络结构,不同激活函数的训练对比
   - 梯度消失问题演示(Sigmoid vs ReLU)
   - 收敛速度对比

**可视化要求**:
- 3x3子图: 9种激活函数的曲线
- 导数曲线对比
- 训练曲线对比(损失下降速度)
- 梯度流可视化(不同深度层的梯度范数)

---

#### Notebook 5: 损失函数与优化器 (05_loss_functions_and_optimizers.ipynb)
**预计行数**: 1800-2000行

**内容大纲**:
1. **损失函数** (30%)
   - MSE: 回归问题
   - 交叉熵: 二分类和多分类
   - 损失函数的信息论解释
   - 从零实现各种损失函数

2. **梯度下降家族** (30%)
   - Batch GD vs Mini-batch GD vs SGD
   - 收敛性分析
   - 学习率的影响

3. **动量优化** (20%)
   - Momentum: 物理直觉
   - Nesterov Accelerated Gradient
   - 从零实现

4. **自适应优化器** (20%)
   - AdaGrad / RMSprop / Adam
   - Adam的实现细节和超参数
   - AdamW vs Adam
   - 从零实现Adam优化器

**从零实现要求**:
```python
class AdamOptimizer:
    """
    从零实现Adam优化器
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        pass

    def update(self, params, grads):
        """
        更新参数

        步骤:
        1. 计算一阶矩估计 m_t
        2. 计算二阶矩估计 v_t
        3. 偏差修正
        4. 参数更新
        """
        pass
```

**可视化要求**:
- 优化器轨迹对比(在2D损失曲面上)
- 学习率调度策略: step decay, cosine annealing
- 不同优化器的收敛速度对比

---

#### Notebook 6: 正则化技术 (06_regularization_techniques.ipynb)
**预计行数**: 1500-1700行

**内容大纲**:
1. **过拟合分析** (20%)
   - 偏差-方差权衡
   - 过拟合的可视化演示
   - 训练集 vs 测试集性能差异

2. **L1/L2正则化** (25%)
   - 数学推导和贝叶斯解释
   - 对权重的影响可视化
   - 从零实现带正则化的训练

3. **Dropout** (30%)
   - Dropout原理和实现
   - 训练模式 vs 推理模式
   - Dropout的集成学习解释
   - 从零实现Dropout层

4. **其他技术** (25%)
   - Early Stopping
   - Data Augmentation基础
   - Batch Normalization预告

**从零实现要求**:
```python
class DropoutLayer:
    """
    从零实现Dropout层
    """
    def forward(self, X, training=True, dropout_rate=0.5):
        """
        训练时: 随机丢弃神经元
        推理时: 缩放输出
        """
        pass

    def backward(self, dout):
        """
        反向传播: 只传递未被丢弃的梯度
        """
        pass
```

**可视化要求**:
- 过拟合演示: 训练误差持续下降,测试误差上升
- L1 vs L2的权重分布对比
- Dropout的效果: 有无Dropout的训练曲线对比

---

#### Notebook 7: 批标准化 (07_batch_normalization.ipynb)
**预计行数**: 1200-1400行

**内容大纲**:
1. **Internal Covariate Shift** (20%)
   - 问题定义和影响
   - 为什么需要标准化

2. **Batch Normalization原理** (35%)
   - 前向传播算法
   - 反向传播推导
   - 训练模式 vs 推理模式的差异
   - 移动平均的更新

3. **从零实现** (30%)
   - BatchNorm层的完整实现
   - 集成到MLP中
   - 训练和推理的正确使用

4. **效果验证** (15%)
   - 有无BN的训练速度对比
   - 学习率敏感性降低
   - Layer Norm等变种简介

**可视化要求**:
- 激活值分布: 训练过程中的变化
- 有无BN的收敛速度对比
- 不同学习率下的稳定性对比

---

#### Notebook 8: PyTorch实践 (08_deep_learning_frameworks.ipynb)
**预计行数**: 2000-2500行

**内容大纲**:
1. **PyTorch基础** (25%)
   - Tensor操作
   - 自动微分: autograd
   - 与NumPy的对比

2. **nn.Module设计模式** (30%)
   - 定义自定义网络
   - Sequential vs 自定义forward
   - 参数初始化
   - 模型保存与加载

3. **训练循环** (25%)
   - DataLoader使用
   - 标准训练循环模板
   - GPU加速基础
   - 训练监控

4. **重新实现之前的模型** (20%)
   - 用PyTorch实现MLP
   - 与NumPy版本的对比
   - MNIST完整训练流程
   - 练习: 用PyTorch实现自定义激活函数

**代码模板**:
```python
import torch
import torch.nn as nn

class MLPPyTorch(nn.Module):
    """
    PyTorch版本的MLP

    对比NumPy实现:
    - 自动微分
    - GPU加速
    - 更高效的内存管理
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        # 定义层
        pass

    def forward(self, x):
        # 前向传播
        pass

# 标准训练循环
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    一个epoch的训练
    """
    pass
```

**可视化要求**:
- PyTorch vs NumPy性能对比
- GPU vs CPU训练速度对比
- TensorBoard可视化集成

---

#### Notebook 9: 超参数调优 (09_hyperparameter_tuning.ipynb)
**预计行数**: 1500-1700行

**内容大纲**:
1. **关键超参数** (20%)
   - 学习率、batch size、网络结构
   - 正则化参数、优化器选择
   - 每个参数的影响分析

2. **调优策略** (30%)
   - Grid Search
   - Random Search
   - Bayesian Optimization
   - 学习率范围测试

3. **实验管理** (25%)
   - 实验记录的重要性
   - 使用TensorBoard或Weights & Biases
   - 可复现性(随机种子)

4. **实战调优** (25%)
   - MNIST网络调优案例
   - 从零到最优的完整流程
   - 调优技巧总结

**可视化要求**:
- 超参数网格搜索热图
- 学习率范围测试曲线
- 多次实验的对比分析

---

#### Notebook 10: 高级主题 (10_advanced_topics.ipynb)
**预计行数**: 1500-1800行

**内容大纲**:
1. **权重初始化** (30%)
   - Xavier初始化推导
   - He初始化(ReLU网络)
   - 不同初始化的实验对比

2. **梯度问题深入** (25%)
   - 梯度消失的根本原因
   - 梯度爆炸的检测与处理
   - 梯度裁剪技术
   - 残差连接原理

3. **学习率调度** (20%)
   - Step Decay
   - Exponential Decay
   - Cosine Annealing
   - Warm Restarts

4. **模型诊断** (25%)
   - 训练不收敛的诊断
   - 过拟合/欠拟合的识别
   - 调试技巧

**可视化要求**:
- 不同初始化方法的激活值分布
- 梯度范数随深度的变化
- 学习率调度策略对比

---

### 2.3 文档创建

#### README.md (模块总览)
**预计行数**: 400-500行

**内容结构**:
1. 模块简介和学习目标
2. 先修知识(数学基础模块)
3. 学习路线图(10个notebook的关系)
4. 学习时间估计(2-3周)
5. 教程对比表(难度、重点、时长)
6. 使用说明和注意事项
7. 常见问题FAQ
8. 参考资源

#### START_HERE.md (学习指南)
**预计行数**: 300-400行

**内容结构**:
1. 快速开始(前3天学什么)
2. 学习路径建议
3. 每个notebook的核心要点
4. 检查清单(学完每个主题后的自测)
5. 项目实践建议

---

## 阶段3: 实战项目

### 3.1 项目1: MNIST手写数字识别
**位置**: `/neural_networks/projects/01_mnist_digit_recognition/`

**文件结构**:
```
01_mnist_digit_recognition/
├── README.md                              # 项目说明
├── 01_numpy_implementation.ipynb          # NumPy完整实现
├── 02_pytorch_implementation.ipynb        # PyTorch实现
├── 03_sklearn_mlp.ipynb                   # sklearn对比
├── 04_error_analysis.ipynb                # 错误分析和可视化
├── data/                                  # MNIST数据(自动下载)
├── models/                                # 保存的模型
└── outputs/                               # 可视化结果
```

**项目目标**:
- 从零实现MLP达到95%+准确率
- 对比三种实现方式的性能
- 深入分析错误样本
- 超参数调优实验

**预计时长**: 4-6小时

---

### 3.2 项目2: CIFAR-10图像分类
**位置**: `/neural_networks/projects/02_cifar10_image_classification/`

**文件结构**:
```
02_cifar10_image_classification/
├── README.md
├── 01_data_exploration.ipynb              # 数据分析
├── 02_mlp_baseline.ipynb                  # MLP基线(较差)
├── 03_improved_mlp.ipynb                  # 改进的MLP
├── 04_comparison.ipynb                    # 对比分析
├── data/
├── models/
└── outputs/
```

**项目目标**:
- 理解MLP在图像分类中的局限性
- 数据增强技术
- 为后续CNN学习铺垫

**预计时长**: 5-7小时

---

### 3.3 项目3: Kaggle竞赛项目
**位置**: `/neural_networks/projects/03_kaggle_competition/`

**候选竞赛**:
- Titanic生存预测(结构化数据+MLP)
- Digit Recognizer(MNIST进阶)
- Fashion MNIST(更难的图像分类)

**文件结构**:
```
03_kaggle_competition/
├── README.md                              # 竞赛介绍和目标
├── 01_data_analysis.ipynb                 # EDA
├── 02_feature_engineering.ipynb           # 特征工程
├── 03_model_training.ipynb                # 模型训练
├── 04_ensemble_methods.ipynb              # 集成学习
├── 05_submission.ipynb                    # 提交结果
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
└── submissions/
```

**项目目标**:
- 完整的Kaggle竞赛流程
- 从EDA到提交的端到端实践
- 目标: 进入Top 25%

**预计时长**: 8-10小时

---

### 3.4 项目4: 垂直领域应用
**位置**: `/neural_networks/projects/04_vertical_domain_application/`

**候选领域**:
- 医疗: 糖尿病预测(Pima Indians Diabetes)
- 金融: 信用卡违约预测(UCI Credit Card)
- 时间序列: 股票价格预测(简化版)

**文件结构**:
```
04_vertical_domain_application/
├── README.md
├── 01_domain_understanding.ipynb          # 领域知识
├── 02_data_preprocessing.ipynb            # 数据预处理
├── 03_model_development.ipynb             # 模型开发
├── 04_model_evaluation.ipynb              # 模型评估
├── 05_deployment_preparation.ipynb        # 部署准备
├── data/
├── models/
└── reports/
```

**项目目标**:
- 真实业务场景的建模
- 类别不平衡处理
- 模型可解释性

**预计时长**: 10-12小时

---

## 🔧 技术实现细节

### 代码风格规范

**注释密度标准** (参考CLAUDE.md):
```python
# ❌ 不符合要求
z = np.dot(X, W) + b

# ✅ 符合要求
# 计算线性变换: z = X @ W + b
# 形状说明:
#   X: (batch_size, input_size) - 输入数据
#   W: (input_size, output_size) - 权重矩阵
#   b: (output_size,) - 偏置向量(广播)
#   z: (batch_size, output_size) - 线性输出
z = np.dot(X, W) + b
```

**从零实现模板**:
```python
class AlgorithmFromScratch:
    """
    [算法名称] 从零实现

    理论基础:
        [数学公式和原理]

    参数:
    -----
    param1 : type
        参数说明

    属性:
    -----
    attr1 : type
        属性说明
    """

    def __init__(self, param1=default):
        # 初始化说明
        self.param1 = param1

    def fit(self, X, y):
        """
        训练模型

        步骤:
        1. [步骤1说明]
        2. [步骤2说明]
        ...
        """
        pass
```

**可视化规范**:
```python
# 3子图对比模板
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 左图: 原始/方法1
axes[0].set_title('标题1', fontsize=14, fontweight='bold')
axes[0].set_xlabel('X轴标签')
axes[0].set_ylabel('Y轴标签')
axes[0].grid(True, alpha=0.3)

# 中图: 对比/方法2
# ...

# 右图: 结果/方法3
# ...

plt.tight_layout()
plt.show()
```

---

## 📅 时间安排估算

| 阶段 | 任务 | 预计时间 | 优先级 |
|-----|------|---------|--------|
| **阶段1** | 数学基础补充 | 3-4天 | ⭐⭐⭐ |
| 1.1 | 矩阵微分notebook | 1-1.5天 | 最高 |
| 1.2 | 反向传播数学notebook | 1.5-2天 | 最高 |
| 1.3 | 更新文档 | 0.5天 | 高 |
| **阶段2** | 核心教程(10个notebook) | 14-18天 | ⭐⭐⭐ |
| 2.1 | 感知器与历史 | 1-1.5天 | 高 |
| 2.2 | MLP与前向传播 | 1.5-2天 | 高 |
| 2.3 | 反向传播算法 | 2-3天 | 最高 |
| 2.4 | 激活函数 | 1-1.5天 | 中 |
| 2.5 | 损失函数与优化器 | 1.5-2天 | 高 |
| 2.6 | 正则化技术 | 1.5-2天 | 高 |
| 2.7 | 批标准化 | 1-1.5天 | 中 |
| 2.8 | PyTorch实践 | 2-2.5天 | 高 |
| 2.9 | 超参数调优 | 1.5-2天 | 中 |
| 2.10 | 高级主题 | 1.5-2天 | 中 |
| 2.11 | 模块文档(README等) | 1天 | 高 |
| **阶段3** | 实战项目 | 8-10天 | ⭐⭐ |
| 3.1 | MNIST项目 | 2-2.5天 | 高 |
| 3.2 | CIFAR-10项目 | 2-2.5天 | 中 |
| 3.3 | Kaggle竞赛 | 2-3天 | 中 |
| 3.4 | 垂直领域应用 | 2-2.5天 | 中 |
| **总计** | | **25-32天** | |

---

## 🎓 学习路径建议

### 快速路径(2周)
1. 数学基础补充(3天)
2. Notebook 1-3, 8 (核心+框架, 6天)
3. MNIST项目(2天)
4. 选修其他notebook

### 标准路径(3-4周)
1. 数学基础补充(4天)
2. 全部10个核心notebook(15天)
3. 2个实战项目(5天)

### 深度路径(5-6周)
1. 数学基础补充(4天)
2. 全部10个核心notebook(18天)
3. 全部4个实战项目(10天)
4. 完成所有练习题

---

## ✅ 质量检查清单

### 每个Notebook完成后检查:
- [ ] 有完整的docstring和注释
- [ ] 核心算法有从零实现
- [ ] 包含sklearn或PyTorch对比
- [ ] 至少3个可视化图表
- [ ] 包含真实数据集应用
- [ ] 有3-4个不同难度的练习题
- [ ] 中文注释无语法错误
- [ ] 代码可以顺利运行
- [ ] 遵循项目的代码风格
- [ ] 数学公式清晰正确

### 项目完成后检查:
- [ ] README完整详细
- [ ] START_HERE指南清晰
- [ ] 所有文件组织规范
- [ ] 与其他模块风格一致
- [ ] 学习时间估计准确
- [ ] 参考资源完善

---

## 📚 参考资源

### 必读书籍:
- 《神经网络与深度学习》邱锡鹏
- 《Deep Learning》Ian Goodfellow
- 《动手学深度学习》李沐

### 在线资源:
- CS231n (Stanford)
- 3Blue1Brown 的神经网络系列
- PyTorch官方教程

### 数据集来源:
- MNIST: torchvision.datasets
- CIFAR-10: torchvision.datasets
- Kaggle竞赛官网
- UCI Machine Learning Repository

---

## 🚀 实施顺序建议

### 第一周: 数学基础
1. 创建05_matrix_calculus.ipynb
2. 创建06_backpropagation_math.ipynb
3. 更新math_foundations的README

### 第二周: 核心基础
4. 创建neural_networks目录结构
5. 完成Notebook 1-3 (感知器→MLP→反向传播)
6. 创建模块README和START_HERE

### 第三周: 深化理解
7. 完成Notebook 4-7 (激活函数→优化器→正则化→BN)

### 第四周: 框架实践
8. 完成Notebook 8-10 (PyTorch→调优→高级)
9. 开始MNIST项目

### 第五周: 实战项目
10. 完成CIFAR-10项目
11. 开始Kaggle竞赛或垂直领域项目

---

## 备注

- 所有代码必须在Jupyter Notebook中测试通过
- 数据集下载失败时提供备用方案
- 每个阶段完成后进行代码审查
- 保持与用户的定期沟通,根据反馈调整计划
