# CNN 完整知识点速查手册

> 基于 `deep_learning/4.1_cnn/` 全部 12 个教程整理

---

## 目录

1. [为什么用卷积](#1-为什么用卷积)
2. [卷积数学原理](#2-卷积数学原理)
3. [卷积从零实现](#3-卷积从零实现)
4. [池化层](#4-池化层)
5. [CNN 各层详解](#5-cnn-各层详解)
6. [CNN 反向传播](#6-cnn-反向传播)
7. [经典架构](#7-经典架构)
8. [迁移学习](#8-迁移学习)
9. [特征可视化](#9-特征可视化)
10. [轻量级架构](#10-轻量级架构)
11. [注意力机制](#11-注意力机制)
12. [检测与分割](#12-检测与分割)

---

## 1. 为什么用卷积

> 来源：01_why_convolution.ipynb

### 1.1 全连接的问题

**参数爆炸**：一张 224×224×3 的图像，连接到 1000 个神经元：
- 全连接参数：224 × 224 × 3 × 1000 = **1.5 亿**参数
- 3×3 卷积：3 × 3 × 3 × 64 = **1,728** 参数

### 1.2 图像的三大特性

| 特性 | 说明 | 卷积如何利用 |
|------|------|-------------|
| **局部性** | 相邻像素相关性高 | 卷积核只看局部区域 |
| **平移不变性** | 猫在左边还是右边都是猫 | 权值共享 |
| **层次性** | 边缘→纹理→部件→物体 | 多层堆叠 |

### 1.3 卷积的三大优势

```
1. 局部连接：每个输出只连接 K×K 区域，而非全图
2. 权值共享：同一卷积核在所有位置复用
3. 平移不变性：特征在哪都能检测到
```

### 1.4 参数量对比

```python
# 全连接
fc_params = H × W × C_in × C_out  # 巨大

# 卷积
conv_params = K × K × C_in × C_out  # 小得多
```

---

## 2. 卷积数学原理

> 来源：02_convolution_math.ipynb

### 2.1 符号约定

| 符号 | 含义 |
|------|------|
| H, W | 输入高度、宽度 |
| C_in, C_out | 输入/输出通道数 |
| K | 卷积核大小 |
| P | Padding（填充） |
| S | Stride（步幅） |

### 2.2 卷积运算定义

**2D 卷积**（实际是互相关）：
$$O[i,j] = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} I[i+m, j+n] \cdot K[m,n]$$

**多通道卷积**：
$$O[i,j] = \sum_{c=0}^{C_{in}-1} \sum_{m} \sum_{n} I[c, i+m, j+n] \cdot K[c, m, n] + b$$

### 2.3 输出尺寸公式 ⭐

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1$$

**常用配置速查**：

| 配置 | 效果 |
|------|------|
| `K=3, P=1, S=1` | 尺寸不变 |
| `K=3, P=1, S=2` | 尺寸减半 |
| `K=5, P=2, S=1` | 尺寸不变 |
| `K=7, P=3, S=2` | 尺寸减半 |

**Same Padding 计算**：
```python
P = K // 2  # 对于 K=3，P=1
```

### 2.4 常见参数详解

| 参数 | 作用 | PyTorch |
|------|------|---------|
| **stride** | 滑动步幅，控制下采样 | `stride=2` |
| **padding** | 边缘填充 | `padding=1` |
| **dilation** | 空洞卷积，增大感受野 | `dilation=2` |
| **groups** | 分组卷积 | `groups=C_in` 为 Depthwise |

---

## 3. 卷积从零实现

> 来源：03_convolution_from_scratch.ipynb

### 3.1 五个实现版本

| 版本 | 特点 | 复杂度 |
|------|------|--------|
| V1 | 4 层循环，最直观 | O(H×W×K²) |
| V2 | 2 层循环，NumPy 切片 | 更快 |
| V3 | 支持 Padding/Stride | 通用 |
| V4 | 多通道输入 | RGB 图像 |
| V5 | 完整批量处理 | 生产级 |

### 3.2 核心代码

```python
# 最基础的卷积实现
def conv2d_basic(image, kernel):
    H, W = image.shape
    K = kernel.shape[0]
    H_out, W_out = H - K + 1, W - K + 1
    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            # 提取感受野
            receptive_field = image[i:i+K, j:j+K]
            # 点积求和
            output[i, j] = np.sum(receptive_field * kernel)

    return output
```

### 3.3 多通道卷积

```python
# 多输入通道：在通道维度累加
for c in range(C_in):
    output += conv2d(input[c], kernel[c])

# 多输出通道：每个输出通道一个卷积核组
for c_out in range(C_out):
    output[c_out] = multi_channel_conv(input, kernels[c_out])
```

---

## 4. 池化层

> 来源：04_pooling_layers.ipynb

### 4.1 池化类型对比

| 类型 | 操作 | 特点 | 适用场景 |
|------|------|------|----------|
| **Max Pooling** | 取最大值 | 保留最强特征 | 分类任务常用 |
| **Avg Pooling** | 取平均值 | 保留整体信息 | 语义分割 |
| **Global Avg Pool** | 整个通道平均 | 替代 FC 层 | 现代网络常用 |

### 4.2 池化的作用

1. **降维**：减少特征图尺寸（通常减半）
2. **增大感受野**：后续层"看到"更大范围
3. **提供不变性**：小幅平移不影响输出
4. **防止过拟合**：减少参数

### 4.3 池化实现

```python
def max_pool2d(image, pool_size=2, stride=2):
    H, W = image.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            window = image[i*stride:i*stride+pool_size,
                          j*stride:j*stride+pool_size]
            output[i, j] = np.max(window)  # 或 np.mean()

    return output
```

### 4.4 全局平均池化

```python
# 替代 Flatten + FC
nn.AdaptiveAvgPool2d(1)  # 输出 (C, 1, 1)
```

---

## 5. CNN 各层详解

> 来源：05_cnn_architecture.ipynb

### 5.1 完整流水线

```
输入图像 → [Conv → BN → ReLU → Pool] × N → Flatten → [FC → ReLU → Dropout] × M → Softmax → 输出
```

### 5.2 各层作用

| 层 | 作用 | 公式/说明 |
|------|------|----------|
| **Conv** | 特征提取 | 卷积运算 |
| **BatchNorm** | 稳定训练 | 标准化 + 缩放 |
| **ReLU** | 非线性 | `max(0, x)` |
| **Pool** | 降维 | Max/Avg |
| **Flatten** | 展平 | 3D → 1D |
| **FC** | 分类 | 矩阵乘法 |
| **Dropout** | 正则化 | 随机丢弃 |
| **Softmax** | 概率输出 | 归一化指数 |

### 5.3 关键公式

**ReLU**：
$$f(x) = \max(0, x)$$

**Dropout（Inverted）**：
$$x' = \frac{x \cdot \text{mask}}{1-p}$$

**Softmax**：
$$\sigma(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**BatchNorm**：
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

### 5.4 形状追踪示例（MNIST）

```
输入: (1, 28, 28)
  ↓ Conv(32, 3×3, p=1)
(32, 28, 28)
  ↓ MaxPool(2)
(32, 14, 14)
  ↓ Conv(64, 3×3, p=1)
(64, 14, 14)
  ↓ MaxPool(2)
(64, 7, 7)
  ↓ Flatten
(3136,)
  ↓ FC(128)
(128,)
  ↓ FC(10)
(10,)
```

---

## 6. CNN 反向传播

> 来源：06_cnn_backprop.ipynb

### 6.1 卷积层梯度

**对输入的梯度**：将上游梯度与**旋转180°的卷积核**做卷积

**对卷积核的梯度**：输入与上游梯度做卷积

```python
# 对权重的梯度
dW = conv2d(input, dout)

# 对输入的梯度
dX = conv2d(dout, rotate180(W))
```

### 6.2 池化层梯度

**Max Pooling**：梯度只流向最大值位置

```python
# 记录前向时的最大值位置
mask = (input == max_value)
# 反向时只在该位置传梯度
dX = dout * mask
```

**Average Pooling**：梯度平均分配

```python
dX = dout / (pool_size * pool_size)
```

### 6.3 梯度检验

```python
# 数值梯度
grad_numerical = (f(x+h) - f(x-h)) / (2*h)

# 解析梯度
grad_analytical = backward()

# 相对误差应 < 1e-5
error = |grad_numerical - grad_analytical| / max(|grad|)
```

---

## 7. 经典架构

> 来源：07_classic_architectures.ipynb, 07a, 07b

### 7.1 架构演进时间线

```
1998       2012        2014        2014        2015        2017
 │          │           │           │           │           │
LeNet → AlexNet → VGGNet → Inception → ResNet → DenseNet
 │          │           │           │           │           │
开创      复兴       深度探索    多尺度      残差连接    密集连接
```

### 7.2 LeNet-5 (1998)

**贡献**：CNN 开山之作，手写数字识别

**网络架构**：
```
输入: 32×32×1 (灰度图)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ C1: Conv 5×5, 6个卷积核 → 28×28×6                            │
│     ↓                                                        │
│ S2: AvgPool 2×2, stride=2 → 14×14×6                         │
│     ↓                                                        │
│ C3: Conv 5×5, 16个卷积核 → 10×10×16                          │
│     ↓                                                        │
│ S4: AvgPool 2×2, stride=2 → 5×5×16                          │
│     ↓                                                        │
│ C5: Conv 5×5, 120个卷积核 → 1×1×120 (等效FC)                 │
│     ↓                                                        │
│ F6: FC 120→84, Tanh                                          │
│     ↓                                                        │
│ Output: FC 84→10, Softmax                                    │
└─────────────────────────────────────────────────────────────┘

总参数量: ~60K
特点: Tanh激活, Average Pooling
```

### 7.3 AlexNet (2012)

**贡献**：开启深度学习时代，ImageNet 冠军

**网络架构**：
```
输入: 224×224×3 (RGB)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Conv1: 11×11, 96, stride=4, ReLU → 55×55×96                 │
│     ↓ MaxPool 3×3, stride=2 → 27×27×96                      │
│     ↓                                                        │
│ Conv2: 5×5, 256, pad=2, ReLU → 27×27×256                    │
│     ↓ MaxPool 3×3, stride=2 → 13×13×256                     │
│     ↓                                                        │
│ Conv3: 3×3, 384, pad=1, ReLU → 13×13×384                    │
│     ↓                                                        │
│ Conv4: 3×3, 384, pad=1, ReLU → 13×13×384                    │
│     ↓                                                        │
│ Conv5: 3×3, 256, pad=1, ReLU → 13×13×256                    │
│     ↓ MaxPool 3×3, stride=2 → 6×6×256                       │
│     ↓                                                        │
│ Flatten → 9216                                               │
│     ↓                                                        │
│ FC1: 9216→4096, ReLU, Dropout(0.5)                          │
│     ↓                                                        │
│ FC2: 4096→4096, ReLU, Dropout(0.5)                          │
│     ↓                                                        │
│ FC3: 4096→1000, Softmax                                      │
└─────────────────────────────────────────────────────────────┘

总参数量: ~60M (大部分在FC层)
```

**核心创新**：
1. **ReLU**：解决梯度消失
2. **Dropout**：防止过拟合
3. **GPU 训练**：双 GPU 并行
4. **数据增强**：裁剪、翻转、颜色抖动
5. **LRN**：局部响应归一化（现已弃用）

### 7.4 VGGNet (2014)

**贡献**：证明深度的重要性

**核心思想**：只用 3×3 小卷积堆叠

**网络架构 (VGG-16)**：
```
输入: 224×224×3
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Block 1: [Conv3-64] × 2 → MaxPool → 112×112×64              │
│     ↓                                                        │
│ Block 2: [Conv3-128] × 2 → MaxPool → 56×56×128              │
│     ↓                                                        │
│ Block 3: [Conv3-256] × 3 → MaxPool → 28×28×256              │
│     ↓                                                        │
│ Block 4: [Conv3-512] × 3 → MaxPool → 14×14×512              │
│     ↓                                                        │
│ Block 5: [Conv3-512] × 3 → MaxPool → 7×7×512                │
│     ↓                                                        │
│ Flatten → 25088                                              │
│     ↓                                                        │
│ FC: 25088→4096→4096→1000                                    │
└─────────────────────────────────────────────────────────────┘

VGG-16: 13个卷积层 + 3个FC层 = 16层
VGG-19: 16个卷积层 + 3个FC层 = 19层
总参数量: ~138M
```

**为什么 3×3 更好？**
```
感受野等效：
2 × (3×3) = 1 × (5×5)  感受野都是 5
3 × (3×3) = 1 × (7×7)  感受野都是 7

参数对比：
2 × 3² = 18 < 5² = 25
3 × 3² = 27 < 7² = 49

额外好处：更多非线性（每层一个 ReLU）
```

**VGG 配置**：
```python
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}
```

### 7.5 ResNet (2015) ⭐

**贡献**：解决深度网络退化问题

**核心思想**：残差连接

**网络架构 (ResNet-50)**：
```
输入: 224×224×3
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Conv1: 7×7, 64, stride=2, BN, ReLU → 112×112×64             │
│     ↓ MaxPool 3×3, stride=2 → 56×56×64                      │
│     ↓                                                        │
│ Stage 1: Bottleneck × 3 → 56×56×256                         │
│          [1×1,64 → 3×3,64 → 1×1,256]                        │
│     ↓                                                        │
│ Stage 2: Bottleneck × 4 → 28×28×512                         │
│          [1×1,128 → 3×3,128 → 1×1,512]                      │
│     ↓                                                        │
│ Stage 3: Bottleneck × 6 → 14×14×1024                        │
│          [1×1,256 → 3×3,256 → 1×1,1024]                     │
│     ↓                                                        │
│ Stage 4: Bottleneck × 3 → 7×7×2048                          │
│          [1×1,512 → 3×3,512 → 1×1,2048]                     │
│     ↓                                                        │
│ GlobalAvgPool → 1×1×2048                                     │
│     ↓                                                        │
│ FC: 2048→1000                                                │
└─────────────────────────────────────────────────────────────┘

ResNet 配置:
┌──────────┬───────┬───────┬───────┬───────┬────────────┐
│ 层       │ 18    │ 34    │ 50    │ 101   │ 152        │
├──────────┼───────┼───────┼───────┼───────┼────────────┤
│ Block    │ Basic │ Basic │ Bottle│ Bottle│ Bottle     │
│ Stage1   │ 2     │ 3     │ 3     │ 3     │ 3          │
│ Stage2   │ 2     │ 4     │ 4     │ 4     │ 8          │
│ Stage3   │ 2     │ 6     │ 6     │ 23    │ 36         │
│ Stage4   │ 2     │ 3     │ 3     │ 3     │ 3          │
│ 参数量   │ 11M   │ 21M   │ 25M   │ 44M   │ 60M        │
└──────────┴───────┴───────┴───────┴───────┴────────────┘
```

**残差块结构**：
```
        ┌─────────────────────┐
        │                     │ 恒等映射 (shortcut)
输入 x ─┼─→ Conv → BN → ReLU →├─→ (+) → ReLU → 输出
        │       → Conv → BN   │
        │         F(x)        │
        └─────────────────────┘

输出 H(x) = F(x) + x
```

**为什么有效？**
$$\frac{\partial H}{\partial x} = \frac{\partial F}{\partial x} + 1$$

梯度至少为 1，不会消失！

**两种残差块**：
```
BasicBlock (ResNet-18/34):     Bottleneck (ResNet-50+):
Conv 3×3 → BN → ReLU           Conv 1×1 → BN → ReLU (降维)
Conv 3×3 → BN                  Conv 3×3 → BN → ReLU
(+x) → ReLU                    Conv 1×1 → BN (升维)
                               (+x) → ReLU
```

### 7.6 Inception/GoogLeNet (2014)

**贡献**：多尺度特征提取

**核心思想**：多分支并行，1×1 降维

**网络架构 (GoogLeNet/Inception-V1)**：
```
输入: 224×224×3
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stem:                                                        │
│   Conv 7×7, 64, stride=2 → 112×112×64                       │
│     ↓ MaxPool 3×3, stride=2 → 56×56×64                      │
│   Conv 1×1, 64 → Conv 3×3, 192 → 56×56×192                  │
│     ↓ MaxPool 3×3, stride=2 → 28×28×192                     │
│     ↓                                                        │
│ Inception 3a-3b: → 28×28×256 → 28×28×480                    │
│     ↓ MaxPool → 14×14×480                                   │
│     ↓                                                        │
│ Inception 4a-4e: → 14×14×512 → ... → 14×14×832              │
│     ↓ MaxPool → 7×7×832                                     │
│     ↓                                                        │
│ Inception 5a-5b: → 7×7×832 → 7×7×1024                       │
│     ↓                                                        │
│ GlobalAvgPool → 1×1×1024                                     │
│     ↓ Dropout(0.4)                                           │
│ FC: 1024→1000                                                │
└─────────────────────────────────────────────────────────────┘

总层数: 22层 (含FC)
参数量: ~5M (比VGG少27倍!)
```

**Inception 模块结构**：
```
              ┌── 1×1 Conv ──────────────┐
              │                          │
输入 ─────────┼── 1×1 → 3×3 Conv ────────┼── Concat → 输出
              │                          │
              ├── 1×1 → 5×5 Conv ────────┤
              │                          │
              └── MaxPool → 1×1 Conv ───┘
```

**1×1 卷积的作用**：
1. 降维减少计算量
2. 跨通道信息融合
3. 增加非线性

### 7.7 DenseNet (2017)

**贡献**：特征复用，参数高效

**核心思想**：每层连接所有前序层（拼接而非相加）

**网络架构 (DenseNet-121)**：
```
输入: 224×224×3
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stem: Conv 7×7, 64, stride=2 → MaxPool → 56×56×64           │
│     ↓                                                        │
│ DenseBlock 1: 6 层 (k=32) → 56×56×256                       │
│     ↓ Transition 1: Conv1×1 + AvgPool → 28×28×128           │
│     ↓                                                        │
│ DenseBlock 2: 12 层 (k=32) → 28×28×512                      │
│     ↓ Transition 2: Conv1×1 + AvgPool → 14×14×256           │
│     ↓                                                        │
│ DenseBlock 3: 24 层 (k=32) → 14×14×1024                     │
│     ↓ Transition 3: Conv1×1 + AvgPool → 7×7×512             │
│     ↓                                                        │
│ DenseBlock 4: 16 层 (k=32) → 7×7×1024                       │
│     ↓                                                        │
│ GlobalAvgPool → 1024                                         │
│     ↓                                                        │
│ FC: 1024→1000                                                │
└─────────────────────────────────────────────────────────────┘

DenseNet 配置:
┌───────────┬─────────────────────────────────────┬────────────┐
│ 版本      │ DenseBlock 层数 [1,2,3,4]           │ 参数量     │
├───────────┼─────────────────────────────────────┼────────────┤
│ DenseNet-121 │ [6, 12, 24, 16]                 │ 8M         │
│ DenseNet-169 │ [6, 12, 32, 32]                 │ 14M        │
│ DenseNet-201 │ [6, 12, 48, 32]                 │ 20M        │
│ DenseNet-264 │ [6, 12, 64, 48]                 │ 34M        │
└───────────┴─────────────────────────────────────┴────────────┘
```

**Dense 连接模式**：
```
DenseBlock 内部：
x0 → x1 = H1(x0)
     x2 = H2([x0, x1])
     x3 = H3([x0, x1, x2])
     ...

每层输出通道 = 增长率 k (如 32)
经过 L 层后通道数 = k0 + L × k
```

**ResNet vs DenseNet**：

| 特性 | ResNet | DenseNet |
|------|--------|----------|
| 连接方式 | 相加 (+) | 拼接 (concat) |
| 特征传递 | 前一层 | 所有前序层 |
| 参数效率 | 中等 | 更高 |
| 增长率 | - | k (如 32) |

---

## 8. 迁移学习

> 来源：08_transfer_learning.ipynb

### 8.1 为什么迁移学习有效？

- **浅层特征通用**：边缘、纹理、颜色（可复用）
- **深层特征任务相关**：需要微调

### 8.2 两种策略

| 策略 | 做法 | 适用场景 |
|------|------|----------|
| **特征提取** | 冻结所有层，只训练分类头 | 数据少，任务相似 |
| **微调** | 解冻部分/全部层，小学习率 | 数据较多 |

### 8.3 分层学习率

```python
# 浅层 → 深层：学习率递增
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},   # 最浅
    {'params': model.layer2.parameters(), 'lr': 5e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 5e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},       # 最深
])
```

### 8.4 PyTorch 迁移学习代码

```python
# 加载预训练模型
model = torchvision.models.resnet50(weights='IMAGENET1K_V1')

# 方式1：特征提取（冻结）
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
model.fc = nn.Linear(2048, num_classes)

# 方式2：微调（解冻后几层）
for param in model.layer4.parameters():
    param.requires_grad = True
```

---

## 9. 特征可视化

> 来源：09_feature_visualization.ipynb

### 9.1 可视化方法

| 方法 | 原理 | 作用 |
|------|------|------|
| **特征图可视化** | 直接显示中间层输出 | 观察各层学到什么 |
| **Grad-CAM** | 梯度加权激活 | 定位重要区域 |
| **Occlusion** | 遮挡敏感性分析 | 找关键区域 |

### 9.2 特征层次

```
浅层（layer1）: 边缘、颜色、纹理
中层（layer2-3）: 纹理组合、部件
深层（layer4）: 完整物体、语义特征
```

### 9.3 Grad-CAM 原理

$$L^c_{Grad-CAM} = ReLU\left(\sum_k \alpha^c_k A^k\right)$$

其中：
- $A^k$ 是第 k 个特征图
- $\alpha^c_k = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}$ 是梯度的全局平均

```python
# Grad-CAM 关键步骤
gradients = torch.autograd.grad(outputs[class_idx], activations)
weights = gradients.mean(dim=[2, 3])  # 全局平均池化
cam = (weights * activations).sum(dim=1)  # 加权求和
cam = F.relu(cam)  # 只保留正贡献
```

---

## 10. 轻量级架构

> 来源：10_modern_efficient_architectures.ipynb

### 10.1 深度可分离卷积 ⭐

将标准卷积分解为两步：

```
标准卷积：
输入 (C_in, H, W) → [Conv K×K] → 输出 (C_out, H', W')
参数：C_out × C_in × K × K

深度可分离卷积：
输入 (C_in, H, W) → [Depthwise K×K] → (C_in, H', W') → [Pointwise 1×1] → (C_out, H', W')
参数：C_in × K × K + C_out × C_in
```

**压缩比**：
$$\text{压缩比} = \frac{1}{C_{out}} + \frac{1}{K^2} \approx \frac{1}{K^2} \approx \frac{1}{9}$$

### 10.2 PyTorch 实现

```python
# Depthwise 卷积：groups = in_channels
nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)

# Pointwise 卷积：1×1
nn.Conv2d(in_ch, out_ch, kernel_size=1)
```

### 10.3 MobileNet V1

```
基础块：Depthwise 3×3 → BN → ReLU → Pointwise 1×1 → BN → ReLU
```

### 10.4 MobileNet V2：倒残差 ⭐

```
传统残差（ResNet）：          倒残差（MobileNet V2）：
  宽 → 窄 → 宽                  窄 → 宽 → 窄

输入 (256)                    输入 (24)
    ↓                             ↓
1×1 降维 → 64                 1×1 升维 → 144 (×6)
    ↓                             ↓
3×3 卷积                      Depthwise 3×3
    ↓                             ↓
1×1 升维 → 256                1×1 降维 → 24 (无激活!)
    ↓                             ↓
(+) 残差                      (+) 残差
```

**为什么倒过来？**
1. 高维空间做 Depthwise，信息损失小
2. 最后不加激活（线性瓶颈），避免 ReLU 损失信息
3. Depthwise 参数不随通道数增加

**ReLU6**：
$$\text{ReLU6}(x) = \min(\max(0, x), 6)$$
- 限制输出范围 [0, 6]
- 量化友好

### 10.5 EfficientNet：复合缩放

```
depth = α^φ
width = β^φ
resolution = γ^φ

约束：α × β² × γ² ≈ 2
```

φ 增加 1，FLOPs 翻倍。

---

## 11. 注意力机制

> 来源：11_attention_mechanisms.ipynb

### 11.1 SE-Net（通道注意力）⭐

**Squeeze-and-Excitation**：让网络学习"关注哪些通道"

```
输入 (C, H, W)
    ↓
Global AvgPool → (C, 1, 1)     ← Squeeze（压缩）
    ↓
FC(C/r) → ReLU → FC(C) → Sigmoid   ← Excitation（激励）
    ↓
通道加权 × 原特征              ← Scale（缩放）
    ↓
输出 (C, H, W)
```

**公式**：
$$z_c = \frac{1}{H \times W} \sum_i \sum_j X[c, i, j]$$
$$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))$$
$$\tilde{X} = s \otimes X$$

**PyTorch 实现**：
```python
class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.squeeze(x).view(b, c)
        s = self.excitation(z).view(b, c, 1, 1)
        return x * s
```

### 11.2 空间注意力

```
输入 (C, H, W)
    ↓
通道维度 MaxPool + AvgPool → (2, H, W)
    ↓
Conv 7×7 → Sigmoid → (1, H, W)
    ↓
空间加权 × 原特征
```

### 11.3 CBAM

组合通道注意力 + 空间注意力：

```
输入 → 通道注意力 → 空间注意力 → 输出
```

---

## 12. 检测与分割

> 来源：12_detection_segmentation_intro.ipynb

### 12.1 任务对比

| 任务 | 输入 | 输出 | 粒度 |
|------|------|------|------|
| **分类** | 图像 | 类别标签 | 图像级 |
| **检测** | 图像 | 边界框 + 类别 | 物体级 |
| **分割** | 图像 | 像素级标签 | 像素级 |

### 12.2 IoU（交并比）

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{交集面积}}{\text{并集面积}}$$

- IoU > 0.5 通常认为检测正确

```python
def calculate_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    return inter / (area1 + area2 - inter)
```

### 12.3 NMS（非极大值抑制）

去除重复检测框：

```python
def nms(boxes, scores, threshold=0.5):
    # 1. 按分数排序
    indices = scores.argsort()[::-1]
    keep = []

    while len(indices) > 0:
        # 2. 保留最高分的框
        i = indices[0]
        keep.append(i)

        # 3. 计算与其他框的 IoU
        ious = [calculate_iou(boxes[i], boxes[j]) for j in indices[1:]]

        # 4. 去除 IoU > threshold 的框
        indices = indices[1:][np.array(ious) < threshold]

    return keep
```

### 12.4 YOLO 思想

```
图像 → 划分 S×S 网格 → 每个格子预测 B 个框 + C 类概率
输出形状：S × S × (B × 5 + C)
        = S × S × (x, y, w, h, conf, class1, class2, ...)
```

### 12.5 U-Net（语义分割）

**编码器-解码器 + 跳跃连接**：

```
编码器（下采样）：
(3, 256, 256) → Conv → Pool → (64, 128, 128)
              → Conv → Pool → (128, 64, 64)
              → Conv → Pool → (256, 32, 32)
              → Conv → Pool → (512, 16, 16)

解码器（上采样）：
(512, 16, 16) → UpConv + Skip → (256, 32, 32)
              → UpConv + Skip → (128, 64, 64)
              → UpConv + Skip → (64, 128, 128)
              → UpConv + Skip → (num_classes, 256, 256)
```

**跳跃连接**：将编码器特征拼接到解码器，保留细节信息

---

## 📊 公式速查表

### 卷积相关

| 公式 | 说明 |
|------|------|
| $H_{out} = \frac{H_{in} + 2P - K}{S} + 1$ | 输出尺寸 |
| $\text{Params} = C_{out} \times C_{in} \times K^2 + C_{out}$ | 卷积参数量 |
| $\text{MACs} = C_{out} \times H_{out} \times W_{out} \times C_{in} \times K^2$ | 计算量 |

### 深度可分离卷积

$$\text{压缩比} = \frac{1}{C_{out}} + \frac{1}{K^2} \approx \frac{1}{9}$$

### 感受野

$$RF_n = RF_{n-1} + (K_n - 1) \times \prod_{i=1}^{n-1} S_i$$

---

## 🎯 核心记忆口诀

1. **卷积三优势**：局部连接、权值共享、平移不变
2. **VGG 精髓**：小卷积堆叠，深度制胜
3. **ResNet 精髓**：残差连接，学习 F(x) 而非 H(x)
4. **Inception 精髓**：多尺度并行，1×1 降维
5. **MobileNet 精髓**：深度可分离，倒残差结构
6. **DenseNet 精髓**：密集连接，特征复用
7. **SE-Net 精髓**：Squeeze → Excitation → Scale

---

## 🔧 PyTorch 代码速查

```python
# 标准卷积
nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1)

# Depthwise 卷积
nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)

# Pointwise 卷积（1×1）
nn.Conv2d(in_ch, out_ch, kernel_size=1)

# 全局平均池化
nn.AdaptiveAvgPool2d(1)

# BatchNorm
nn.BatchNorm2d(num_features)

# 残差块核心
out = F.relu(self.bn(self.conv(x)) + x)

# 加载预训练模型
model = torchvision.models.resnet50(weights='IMAGENET1K_V1')

# 冻结参数
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
model.fc = nn.Linear(2048, num_classes)
```

---

*最后更新：2025-01*
*基于 deep_learning/4.1_cnn/ 全部 12 个教程整理*
