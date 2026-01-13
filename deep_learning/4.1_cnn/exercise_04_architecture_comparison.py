"""
练习 4: 自定义数据集对比

目标：
1. 在 CIFAR-100 数据集上对比三种 CNN 架构
2. 理解不同架构的参数效率和准确率权衡
3. 学习 EfficientNet 的迁移学习方法

对比的三种架构：
- SimpleCNN: 标准卷积堆叠
- MobileNetV2_Small: 深度可分离卷积 + 倒残差
- EfficientNet-B0: 复合缩放 + MBConv + SE注意力

关键对比维度：
- 参数量（模型大小）
- 训练时间（效率）
- 测试准确率（性能）
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 配置
# ============================================

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子（保证实验可重复）
torch.manual_seed(42)
np.random.seed(42)

# 训练配置
EPOCHS = 10          # 训练轮数
BATCH_SIZE = 128     # 批次大小
LEARNING_RATE = 0.001  # 学习率


# ============================================
# Part 1: 数据加载
# ============================================

def load_cifar100():
    """
    加载 CIFAR-100 数据集

    CIFAR-100 vs CIFAR-10 的区别：
    ┌─────────────┬───────────────┬───────────────┐
    │   特性       │   CIFAR-10    │   CIFAR-100   │
    ├─────────────┼───────────────┼───────────────┤
    │ 类别数       │      10       │      100      │
    │ 每类样本数   │    6000       │      600      │
    │ 总样本数     │   60000       │    60000      │
    │ 图像尺寸     │   32×32×3     │   32×32×3     │
    │ 难度         │     较低      │      较高     │
    └─────────────┴───────────────┴───────────────┘

    CIFAR-100 的挑战：
    1. 类别更多（100 vs 10），每类样本更少（600 vs 6000）
    2. 存在层级结构：20 个超类，每个超类包含 5 个细分类
    3. 类间差异更小，类内差异更大

    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # ========================================
    # 数据增强策略
    #
    # 训练时使用数据增强来：
    # 1. 增加数据多样性，防止过拟合
    # 2. 提高模型的泛化能力
    # ========================================
    transform_train = transforms.Compose([
        # 随机裁剪：先填充 4 像素，再随机裁剪回 32×32
        # 这模拟了图像的小幅平移
        transforms.RandomCrop(32, padding=4),

        # 随机水平翻转：以 50% 概率翻转
        # 适用于左右对称的目标（如动物、车辆）
        transforms.RandomHorizontalFlip(),

        # 转换为 Tensor，并将像素值从 [0, 255] 归一化到 [0, 1]
        transforms.ToTensor(),

        # 标准化：减去均值，除以标准差
        # 这些值是 CIFAR-100 训练集的统计量
        # 标准化后数据分布接近 N(0, 1)，有助于训练稳定
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),  # RGB 三通道均值
            std=(0.2675, 0.2565, 0.2761)    # RGB 三通道标准差
        )
    ])

    # 测试时不使用数据增强，只做标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    # ========================================
    # 加载数据集
    # download=True 会自动下载（如果不存在）
    # ========================================
    print("加载 CIFAR-100 数据集...")
    train_data = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    test_data = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # ========================================
    # 创建数据加载器
    # num_workers: 使用多进程加载数据
    # pin_memory: 将数据放入固定内存，加速 GPU 传输
    # ========================================
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,      # 训练时打乱顺序
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=100,
        shuffle=False,     # 测试时不打乱
        num_workers=2,
        pin_memory=True
    )

    print(f"训练集: {len(train_data)} 样本")
    print(f"测试集: {len(test_data)} 样本")
    print(f"类别数: 100")

    return train_loader, test_loader


# ============================================
# Part 2: 模型定义
# ============================================

# ----------------------------------------
# 模型 1: SimpleCNN（标准卷积）
# ----------------------------------------

class SimpleCNN(nn.Module):
    """
    简单 CNN 模型 - 使用标准卷积

    架构设计理念：
    1. 逐层增加通道数：3 → 32 → 64 → 128 → 256
    2. 使用 MaxPooling 逐步降低空间分辨率
    3. 每个卷积后使用 BatchNorm + ReLU

    架构图：
    输入 (3, 32, 32)
        ↓
    Conv 3×3 → 32 通道 → BN → ReLU
        ↓
    Conv 3×3 → 64 通道 → BN → ReLU → MaxPool 2×2  → (64, 16, 16)
        ↓
    Conv 3×3 → 128 通道 → BN → ReLU
        ↓
    Conv 3×3 → 128 通道 → BN → ReLU → MaxPool 2×2 → (128, 8, 8)
        ↓
    Conv 3×3 → 256 通道 → BN → ReLU
        ↓
    Conv 3×3 → 256 通道 → BN → ReLU → AdaptiveAvgPool → (256, 1, 1)
        ↓
    Flatten → Linear → 100 类

    参数量分析：
    - 主要参数在卷积层：C_out × C_in × K × K
    - 3×3 卷积参数较多，但特征提取能力强
    """

    def __init__(self, num_classes=100):
        super().__init__()

        # 特征提取部分
        self.features = nn.Sequential(
            # Block 1: 3 → 32 通道
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 32 → 64 通道 + 下采样
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32×32 → 16×16

            # Block 3: 64 → 128 通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: 128 → 128 通道 + 下采样
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16×16 → 8×8

            # Block 5: 128 → 256 通道
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 6: 256 → 256 通道 + 全局池化
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 8×8 → 1×1
        )

        # 分类头
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch, 3, 32, 32)
        x = self.features(x)      # → (batch, 256, 1, 1)
        x = x.view(x.size(0), -1) # → (batch, 256)
        x = self.classifier(x)    # → (batch, num_classes)
        return x


# ----------------------------------------
# 模型 2: MobileNetV2_Small（深度可分离卷积）
# ----------------------------------------

class InvertedResidual(nn.Module):
    """
    倒残差块（Inverted Residual Block）- MobileNet V2 的核心

    与传统残差块的对比：

    传统残差块（ResNet）：         倒残差块（MobileNet V2）：
    输入 (256 通道)                输入 (24 通道)
        ↓                              ↓
    Conv 1×1 降维 → 64             Conv 1×1 升维 → 144 (×6 扩展)
        ↓                              ↓
    Conv 3×3 提特征                Depthwise 3×3（每通道独立）
        ↓                              ↓
    Conv 1×1 升维 → 256            Conv 1×1 降维 → 24
        ↓                              ↓
    (+) ← 残差连接                 (+) ← 残差连接

    「宽 → 窄 → 宽」                「窄 → 宽 → 窄」

    为什么倒过来？
    1. 在高维空间做 Depthwise 卷积，信息损失更小
    2. ReLU 在低维空间会损失信息，所以最后不加激活
    3. Depthwise 卷积参数与通道数无关，高维计算量增加有限

    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步幅（1 保持尺寸，2 下采样）
        expand_ratio: 扩展比例（通常为 6）
    """

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()

        self.stride = stride
        # 残差连接条件：步幅为 1 且输入输出通道相同
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # 扩展后的通道数
        hidden_dim = in_channels * expand_ratio

        layers = []

        # ========================================
        # Expand 层（升维）
        # 使用 1×1 卷积将通道数扩展 expand_ratio 倍
        # ========================================
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)  # ReLU6: min(max(0, x), 6)，量化友好
            ])

        # ========================================
        # Depthwise 卷积（空间特征提取）
        # groups=hidden_dim 表示每个通道独立卷积
        # 参数量：hidden_dim × 3 × 3（与标准卷积的 C_out × C_in × 3 × 3 相比大幅减少）
        # ========================================
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=3, stride=stride, padding=1,
                groups=hidden_dim,  # 关键：分组数 = 通道数 → Depthwise
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # ========================================
        # Project 层（降维，线性投影）
        # 注意：这里没有激活函数！
        # 原因：在低维空间使用 ReLU 会损失信息
        # ========================================
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
            # 没有激活函数
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)  # 残差连接
        else:
            return self.conv(x)


class MobileNetV2_Small(nn.Module):
    """
    简化版 MobileNet V2 - 适配 CIFAR-100 (32×32)

    核心设计：
    1. 深度可分离卷积：Depthwise + Pointwise
    2. 倒残差结构：窄 → 宽 → 窄
    3. 线性瓶颈：最后的 Pointwise 不加激活

    架构：
    Stem: Conv 3×3 → 32 通道
        ↓
    Inverted Residuals × 10（通道数逐渐增加）
        ↓
    Conv 1×1 → 1280 通道
        ↓
    Global AvgPool → Flatten → FC → 100 类

    参数效率：
    - 深度可分离卷积约节省 9 倍参数（相比标准 3×3）
    - 整体参数量约 1.2M，远小于 VGG 等传统网络
    """

    def __init__(self, num_classes=100):
        super().__init__()

        # Stem: 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # ========================================
        # 倒残差块配置（来自 MobileNet V2 论文）
        # ========================================
        #
        # 参数说明: InvertedResidual(in_ch, out_ch, stride, expand_ratio)
        #   - in_ch: 输入通道数
        #   - out_ch: 输出通道数
        #   - stride: 步幅（1=保持尺寸，2=下采样）
        #   - expand_ratio: 扩展比例（中间层通道数 = in_ch × expand_ratio）
        #
        # 这些数字来自 MobileNet V2 论文的 Table 2，是通过大量实验确定的最优配置
        #
        # 设计原则：
        # 1. 通道数逐层增加：16 → 24 → 32 → 64 → 96 → 160 → 320
        #    原因：更深的层需要更多通道来表示复杂的高级特征
        #
        # 2. 每次下采样时通道数翻倍（近似）
        #    例如：24→32（stride=2），32→64（stride=2）
        #    原因：空间分辨率减半，用更多通道补偿信息损失
        #
        # 3. expand_ratio=6 是论文实验得出的最佳值
        #    首层用 1 是因为输入通道已经较多（32），不需要扩展
        #
        # 4. 相同通道数的连续块形成"阶段"
        #    例如：24→24, 32→32, 64→64
        #    原因：重复块可以提取更丰富的特征
        #
        # 原始 MobileNet V2 配置表（简化版）:
        # ┌─────────┬────────┬─────────┬────────┬─────────┬──────────────┐
        # │  输入   │  输出  │ stride  │ expand │  重复   │   空间尺寸   │
        # ├─────────┼────────┼─────────┼────────┼─────────┼──────────────┤
        # │   32    │   16   │    1    │   1    │    1    │   32 → 32    │
        # │   16    │   24   │    1    │   6    │    2    │   32 → 32    │
        # │   24    │   32   │    2    │   6    │    3    │   32 → 16    │
        # │   32    │   64   │    2    │   6    │    4    │   16 → 8     │
        # │   64    │   96   │    1    │   6    │    3    │    8 → 8     │
        # │   96    │  160   │    2    │   6    │    3    │    8 → 4     │
        # │  160    │  320   │    1    │   6    │    1    │    4 → 4     │
        # └─────────┴────────┴─────────┴────────┴─────────┴──────────────┘
        #
        # 本实现是简化版，减少了重复次数以适应 CIFAR-100 的小尺寸图像
        #
        self.blocks = nn.Sequential(
            # 阶段 1: 降维，不扩展（因为 stem 已经是 32 通道）
            InvertedResidual(32, 16, stride=1, expand_ratio=1),

            # 阶段 2: 16 → 24 通道，2 个块
            InvertedResidual(16, 24, stride=1, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),  # 残差连接生效

            # 阶段 3: 24 → 32 通道，stride=2 下采样 (32×32 → 16×16)
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),  # 残差连接生效

            # 阶段 4: 32 → 64 通道，stride=2 下采样 (16×16 → 8×8)
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),  # 残差连接生效

            # 阶段 5: 64 → 96 通道
            InvertedResidual(64, 96, stride=1, expand_ratio=6),

            # 阶段 6: 96 → 160 通道，stride=2 下采样 (8×8 → 4×4)
            InvertedResidual(96, 160, stride=2, expand_ratio=6),

            # 阶段 7: 160 → 320 通道
            InvertedResidual(160, 320, stride=1, expand_ratio=6),
        )

        # 最后的 1×1 卷积：扩展到 1280 通道
        self.last_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Dropout(0.2),          # 防止过拟合
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)       # (batch, 32, 32, 32)
        x = self.blocks(x)     # (batch, 320, 4, 4)
        x = self.last_conv(x)  # (batch, 1280, 4, 4)
        x = self.classifier(x) # (batch, num_classes)
        return x


# ----------------------------------------
# 模型 3: EfficientNet-B0（预训练 + 迁移学习）
# ----------------------------------------

def create_efficientnet_b0(num_classes=100):
    """
    创建适配 CIFAR-100 的 EfficientNet-B0

    EfficientNet 的核心创新：复合缩放（Compound Scaling）

    传统的网络缩放只调整一个维度：
    - 宽度缩放：增加每层通道数
    - 深度缩放：增加网络层数
    - 分辨率缩放：使用更大的输入图像

    EfficientNet 同时调整三个维度：
    - depth = α^φ
    - width = β^φ
    - resolution = γ^φ
    约束：α × β² × γ² ≈ 2

    EfficientNet-B0 是基线模型，通过 φ=0,1,2,... 得到 B1-B7

    迁移学习策略：
    1. 加载 ImageNet 预训练权重
    2. 替换分类头（1000 类 → 100 类）
    3. 可选：冻结部分层只训练分类头（本例不冻结，全部微调）

    注意：
    - EfficientNet 原始输入是 224×224，CIFAR 是 32×32
    - 预训练权重仍然有效，因为低层特征（边缘、纹理）是通用的
    - 但效果可能不如在相似分辨率数据上预训练的模型

    参数:
        num_classes: 输出类别数

    返回:
        model: 适配后的 EfficientNet-B0 模型
    """
    # ========================================
    # 加载预训练模型
    # weights=EfficientNet_B0_Weights.DEFAULT 使用最新的预训练权重
    # ========================================
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # ========================================
    # 替换分类头
    # 原始：Linear(1280, 1000) 用于 ImageNet 1000 类
    # 替换：Linear(1280, 100) 用于 CIFAR-100
    # ========================================
    in_features = model.classifier[1].in_features  # 获取输入特征数 = 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    # ========================================
    # 可选：冻结特征提取层（只训练分类头）
    #
    # 如果数据集很小或与 ImageNet 差异大，可以冻结前面的层：
    # for param in model.features.parameters():
    #     param.requires_grad = False
    #
    # 本例中我们全部微调，因为 CIFAR-100 有 50000 张训练图片
    # ========================================

    return model


# ============================================
# Part 3: 训练函数
# ============================================

def train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    训练模型并记录历史

    训练流程：
    1. 前向传播：计算预测值
    2. 计算损失：交叉熵损失
    3. 反向传播：计算梯度
    4. 更新参数：Adam 优化器
    5. 学习率调度：余弦退火

    参数:
        model: PyTorch 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        lr: 初始学习率

    返回:
        history: 包含训练/测试准确率和时间的字典
    """
    # 将模型移到 GPU/CPU
    model = model.to(device)

    # ========================================
    # 优化器：Adam
    # Adam 结合了 Momentum 和 RMSprop 的优点
    # - 自适应学习率
    # - 适合大多数任务
    # ========================================
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ========================================
    # 损失函数：交叉熵
    # 用于多分类任务
    # 内部已包含 Softmax，所以模型输出是 logits
    # ========================================
    criterion = nn.CrossEntropyLoss()

    # ========================================
    # 学习率调度器：余弦退火
    # 学习率从初始值平滑下降到接近 0
    # 比固定学习率或阶梯下降效果更好
    # ========================================
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 记录训练历史
    history = {
        'train_acc': [],
        'test_acc': [],
        'train_time': []
    }

    for epoch in range(epochs):
        # ========================================
        # 训练阶段
        # ========================================
        model.train()  # 设置为训练模式（启用 Dropout、BatchNorm 训练行为）
        correct, total = 0, 0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            # 将数据移到设备
            x, y = x.to(device), y.to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(x)

            # 计算损失
            loss = criterion(outputs, y)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 统计准确率
            predicted = outputs.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_time = time.time() - start_time
        train_acc = 100 * correct / total

        # ========================================
        # 测试阶段
        # ========================================
        model.eval()  # 设置为评估模式（禁用 Dropout、使用 BatchNorm 的移动平均）
        correct, total = 0, 0

        with torch.no_grad():  # 不计算梯度，节省内存
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        test_acc = 100 * correct / total

        # 记录历史
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_time'].append(train_time)

        # 更新学习率
        scheduler.step()

        # 打印进度
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"Train={train_acc:.1f}%, Test={test_acc:.1f}%, "
              f"Time={train_time:.1f}s")

    return history


# ============================================
# Part 4: 可视化
# ============================================

def plot_comparison(results, model_params):
    """
    可视化对比结果

    生成三个子图：
    1. 测试准确率曲线 - 观察收敛速度和最终性能
    2. 参数量对比 - 观察模型大小
    3. 效率分析 - 准确率/参数量比

    参数:
        results: 各模型的训练历史 {model_name: history}
        model_params: 各模型的参数量 {model_name: param_count}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 颜色配置
    colors = {
        'SimpleCNN': 'steelblue',
        'MobileNetV2': 'coral',
        'EfficientNet-B0': 'forestgreen'
    }

    # ----------------------------------------
    # 图 1: 测试准确率曲线
    # ----------------------------------------
    for name, history in results.items():
        epochs = range(1, len(history['test_acc']) + 1)
        axes[0].plot(epochs, history['test_acc'],
                     label=name, color=colors[name],
                     linewidth=2, marker='o', markersize=4)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('测试准确率曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ----------------------------------------
    # 图 2: 参数量对比
    # ----------------------------------------
    names = list(results.keys())
    params = [model_params[n] / 1e6 for n in names]  # 转换为 M
    bars = axes[1].bar(names, params, color=[colors[n] for n in names])

    axes[1].set_ylabel('参数量 (M)')
    axes[1].set_title('模型大小对比')

    # 在柱状图上标注数值
    for bar, p in zip(bars, params):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{p:.2f}M', ha='center', fontsize=10)

    # ----------------------------------------
    # 图 3: 效率分析（准确率 vs 参数量）
    # ----------------------------------------
    final_accs = [results[n]['test_acc'][-1] for n in names]
    params_m = [model_params[n] / 1e6 for n in names]

    for name, acc, p in zip(names, final_accs, params_m):
        axes[2].scatter(p, acc, s=100, c=colors[name], label=name)
        axes[2].annotate(name, (p, acc), textcoords="offset points",
                         xytext=(5, 5), fontsize=9)

    axes[2].set_xlabel('参数量 (M)')
    axes[2].set_ylabel('Test Accuracy (%)')
    axes[2].set_title('效率分析：准确率 vs 参数量')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('CIFAR-100 架构对比实验', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('exercise_04_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n图表已保存为 exercise_04_comparison.png")


# ============================================
# Part 5: 主函数
# ============================================

def main():
    """
    主函数：执行完整的对比实验

    步骤：
    1. 加载 CIFAR-100 数据
    2. 创建三种模型
    3. 分别训练
    4. 对比分析
    """
    print("=" * 60)
    print("练习 4: CIFAR-100 架构对比实验")
    print("=" * 60)
    print(f"\n使用设备: {device}")
    print(f"训练轮数: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")

    # ========================================
    # Step 1: 加载数据
    # ========================================
    print("\n" + "-" * 40)
    train_loader, test_loader = load_cifar100()

    # ========================================
    # Step 2: 创建模型
    # ========================================
    print("\n" + "-" * 40)
    print("创建模型...")

    models = {
        'SimpleCNN': SimpleCNN(num_classes=100),
        'MobileNetV2': MobileNetV2_Small(num_classes=100),
        'EfficientNet-B0': create_efficientnet_b0(num_classes=100)
    }

    # 统计参数量
    model_params = {}
    print("\n模型参数量：")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        model_params[name] = params
        print(f"  {name}: {params:,} ({params/1e6:.2f}M)")

    # ========================================
    # Step 3: 训练对比
    # ========================================
    results = {}

    for name, model in models.items():
        print("\n" + "=" * 50)
        print(f"训练 {name}")
        print("=" * 50)
        results[name] = train_model(model, train_loader, test_loader)

    # ========================================
    # Step 4: 结果分析
    # ========================================
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    print(f"\n{'模型':<20} {'参数量':>12} {'最终准确率':>12} {'平均Epoch时间':>15}")
    print("-" * 60)

    for name in models.keys():
        params = model_params[name]
        final_acc = results[name]['test_acc'][-1]
        avg_time = np.mean(results[name]['train_time'])
        print(f"{name:<20} {params/1e6:>11.2f}M {final_acc:>11.1f}% {avg_time:>14.1f}s")

    # ========================================
    # Step 5: 可视化
    # ========================================
    print("\n" + "-" * 40)
    print("生成对比图表...")
    plot_comparison(results, model_params)

    # ========================================
    # 分析总结
    # ========================================
    print("\n" + "=" * 60)
    print("分析总结")
    print("=" * 60)
    print("""
    1. SimpleCNN:
       - 使用标准 3×3 卷积
       - 参数量适中（~1.1M）
       - 实现简单，容易理解

    2. MobileNetV2:
       - 使用深度可分离卷积 + 倒残差结构
       - 参数量相近（~1.2M）但设计更高效
       - 移动端部署的首选

    3. EfficientNet-B0:
       - 使用复合缩放 + MBConv + SE注意力
       - 参数量较大（~5.3M）但准确率最高
       - 预训练权重带来显著提升

    关键发现：
    - 预训练权重对准确率提升很大
    - 深度可分离卷积在相似参数量下效果更好
    - 网络架构设计比单纯增加参数更重要
    """)


if __name__ == "__main__":
    main()
