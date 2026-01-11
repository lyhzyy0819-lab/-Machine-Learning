"""
CNN 综合项目练习 - MNIST 手写数字分类器

=============================================================
前置知识（请确保已学习）：
- 05_cnn_architecture.ipynb（CNN 各层的作用）
- 06_cnn_backprop.ipynb（反向传播原理）
=============================================================

项目目标：
使用 PyTorch 从零构建并训练 MNIST 分类器

完成标准：
- 所有 TODO 部分正确实现
- 测试准确率达到 98% 以上

使用方法：
    python exercise_cnn_mnist.py

提示：
- 如果遇到困难，可以参考 project_mnist_classifier.py
- 但建议先自己尝试，这样学习效果更好！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 配置
# ============================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Part 1: 定义 CNN 模型
# ============================================================
#
# 回顾 05 章学到的内容：
# - Conv2d: 卷积层，提取局部特征
# - ReLU: 激活函数，引入非线性
# - MaxPool2d: 池化层，下采样
# - Linear: 全连接层，做最终分类
# - Dropout: 防止过拟合
#
# 你的任务：实现一个 CNN，架构如下：
#
#   输入: (N, 1, 28, 28)
#   ↓
#   Conv(1→32, 3×3, padding=1) → ReLU → MaxPool(2×2)
#   输出: (N, 32, 14, 14)
#   ↓
#   Conv(32→64, 3×3, padding=1) → ReLU → MaxPool(2×2)
#   输出: (N, 64, 7, 7)
#   ↓
#   Flatten
#   输出: (N, 64*7*7) = (N, 3136)
#   ↓
#   FC(3136→128) → ReLU → Dropout(0.5)
#   输出: (N, 128)
#   ↓
#   FC(128→10)
#   输出: (N, 10)
#
# ============================================================

class MNISTClassifier(nn.Module):
    """
    MNIST 分类 CNN

    架构说明：
    - 2 个卷积层：提取图像特征
    - 2 个池化层：下采样，减少计算量
    - 2 个全连接层：综合特征做分类
    - Dropout：防止过拟合

    输入：(N, 1, 28, 28) - N张 28×28 的灰度图
    输出：(N, 10) - 10个类别的分数（logits）
    """

    def __init__(self):
        super(MNISTClassifier, self).__init__()

        # ========================================
        # TODO: 定义第一个卷积层
        # 输入通道: 1（灰度图）
        # 输出通道: 32
        # 卷积核大小: 3×3
        # padding: 1（保持尺寸不变）
        # ========================================
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )  # TODO: 替换为 nn.Conv2d(...)

        # ========================================
        # TODO: 定义第二个卷积层
        # 输入通道: 32
        # 输出通道: 64
        # 卷积核大小: 3×3
        # padding: 1
        # ========================================
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )  # TODO: 替换为 nn.Conv2d(...)

        # ========================================
        # TODO: 定义池化层
        # 池化窗口: 2×2
        # 步幅: 2
        # ========================================
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )  # TODO: 替换为 nn.MaxPool2d(...)

        # ========================================
        # TODO: 定义第一个全连接层
        # 输入维度: 64 * 7 * 7 = 3136
        # 输出维度: 128
        # ========================================
        self.fc1 = nn.Linear(
            in_features=64 * 7 * 7,
            out_features=128,
        )  # TODO: 替换为 nn.Linear(...)

        # ========================================
        # TODO: 定义第二个全连接层
        # 输入维度: 128
        # 输出维度: 10（10个类别）
        # ========================================
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10,
        )  # TODO: 替换为 nn.Linear(...)

        # ========================================
        # TODO: 定义 Dropout 层
        # 丢弃概率: 0.5
        # ========================================
        self.dropout = nn.Dropout(p=0.5)  # TODO: 替换为 nn.Dropout(...)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入图像，shape (N, 1, 28, 28)

        返回:
            logits: 类别分数，shape (N, 10)
        """
        # ========================================
        # TODO: 实现前向传播
        #
        # 步骤:
        # 1. Conv1 → ReLU → Pool
        # 2. Conv2 → ReLU → Pool
        # 3. Flatten（展平）
        # 4. FC1 → ReLU → Dropout
        # 5. FC2
        #
        # 提示:
        # - 使用 F.relu() 作为激活函数
        # - 使用 x.view(-1, 64*7*7) 展平
        # ========================================

        # 第一个卷积块: Conv1 → ReLU → Pool
        # TODO: 实现
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 第二个卷积块: Conv2 → ReLU → Pool
        # TODO: 实现
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # 展平
        # TODO: 实现
        x = x.view(x.size(0), -1)

        # 全连接层: FC1 → ReLU → Dropout
        # TODO: 实现
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 输出层: FC2
        # TODO: 实现
        x = self.fc2(x)

        return x  # TODO: 确保返回正确的输出


# ============================================================
# Part 2: 实现训练函数
# ============================================================
#
# 回顾 06 章学到的内容：
# - 前向传播：输入 → 网络 → 预测
# - 计算损失：预测 vs 真实标签
# - 反向传播：loss.backward()
# - 更新权重：optimizer.step()
#
# ============================================================

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个 epoch

    参数:
        model: CNN 模型
        train_loader: 训练数据加载器
        optimizer: 优化器（如 Adam）
        criterion: 损失函数（如 CrossEntropyLoss）
        device: 设备（CPU 或 GPU）

    返回:
        avg_loss: 平均损失
        accuracy: 准确率（百分比）
    """
    model.train()  # 设置为训练模式（启用 Dropout）
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # ========================================
        # TODO: 实现训练循环
        #
        # 步骤:
        # 1. 将数据移动到设备（data.to(device), target.to(device)）
        # 2. 清零梯度（optimizer.zero_grad()）
        # 3. 前向传播（output = model(data)）
        # 4. 计算损失（loss = criterion(output, target)）
        # 5. 反向传播（loss.backward()）
        # 6. 更新权重（optimizer.step()）
        # 7. 统计损失和准确率
        # ========================================

        # 1. 将数据移动到设备
        # TODO: 实现
        data, target = data.to(device), target.to(device)

        # 2. 清零梯度（为什么需要？因为 PyTorch 会累积梯度）
        # TODO: 实现
        optimizer.zero_grad()

        # 3. 前向传播
        # TODO: 实现
        output = model(data)

        # 4. 计算损失
        # TODO: 实现
        loss = criterion(output, target)


        # 5. 反向传播（这一步计算所有参数的梯度）
        # TODO: 实现
        loss.backward()

        # 6. 更新权重（使用梯度更新参数）
        # TODO: 实现
        optimizer.step()

        # 7. 统计
        # total_loss += loss.item()
        # pred = output.argmax(dim=1)
        # correct += pred.eq(target).sum().item()
        # total += target.size(0)
        # pass  # TODO: 删除这行，实现上面的步骤
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ============================================================
# Part 3: 实现评估函数
# ============================================================

def evaluate(model, test_loader, criterion, device):
    """
    在测试集上评估模型

    参数:
        model: CNN 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备

    返回:
        avg_loss: 平均损失
        accuracy: 准确率（百分比）
    """
    model.eval()  # 设置为评估模式（禁用 Dropout）
    total_loss = 0
    correct = 0
    total = 0

    # ========================================
    # TODO: 实现评估循环
    #
    # 提示:
    # - 使用 torch.no_grad() 禁用梯度计算（节省内存）
    # - 不需要反向传播和更新权重
    # ========================================

    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            # TODO: 实现评估逻辑
            # 1. 将数据移动到设备
            data, target = data.to(device), target.to(device)
            # 2. 前向传播
            output = model(data)
            # 3. 计算损失
            loss = criterion(output, target)
            # 4. 统计准确率
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ============================================================
# Part 4: 主函数 - 完整训练流程
# ============================================================

def load_data():
    """
    加载 MNIST 数据集

    这部分已经实现好了，你不需要修改
    """
    # 数据转换：转为张量并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
    ])

    # 下载并加载数据集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader


def visualize_predictions(model, test_loader, device, num_samples=10):
    """
    可视化模型预测结果

    这部分已经实现好了，你不需要修改
    """
    model.eval()
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)

    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i, ax in enumerate(axes.flatten()):
        if i < num_samples:
            img = data[i].cpu().numpy().squeeze()
            true_label = target[i].item()
            pred_label = pred[i].item()

            ax.imshow(img, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        ax.axis('off')

    plt.suptitle('Model Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：完整的训练和评估流程
    """
    print("=" * 60)
    print("MNIST 手写数字分类器 - 综合练习")
    print("=" * 60)
    print(f"\n使用设备: {DEVICE}")

    # ========================================
    # Step 1: 加载数据
    # ========================================
    print("\n[1/5] 加载数据...")
    train_loader, test_loader = load_data()
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    # ========================================
    # Step 2: 创建模型
    # ========================================
    print("\n[2/5] 创建模型...")

    # TODO: 创建模型并移动到设备
    # model = MNISTClassifier().to(DEVICE)
    # model = None  # TODO: 替换
    model = MNISTClassifier().to(DEVICE)

    if model is not None:
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n总参数量: {total_params:,}")
    else:
        print("错误：模型未创建！请检查 MNISTClassifier 类的实现")
        return

    # ========================================
    # Step 3: 定义损失函数和优化器
    # ========================================
    print("\n[3/5] 定义损失函数和优化器...")

    # TODO: 创建损失函数（交叉熵损失）
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()  # TODO: 替换

    # TODO: 创建优化器（Adam）
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # TODO: 替换

    if criterion is None or optimizer is None:
        print("错误：损失函数或优化器未创建！")
        return

    # ========================================
    # Step 4: 训练
    # ========================================
    print("\n[4/5] 开始训练...")
    print("-" * 60)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )

        # 评估
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, DEVICE
        )

        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch}/{EPOCHS}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

    print("-" * 60)

    # ========================================
    # Step 5: 结果
    # ========================================
    print("\n[5/5] 训练完成！")

    final_acc = test_accs[-1]
    print(f"\n最终测试准确率: {final_acc:.2f}%")

    if final_acc >= 98:
        print("\n恭喜！你的模型达到了 98% 以上的准确率！")
        print("你已经成功完成了这个练习！")
    elif final_acc >= 95:
        print("\n不错！你的模型准确率超过了 95%")
        print("尝试调整超参数或增加训练轮数来进一步提升")
    else:
        print("\n模型还需要改进，请检查以下几点：")
        print("1. MNISTClassifier 的 __init__ 方法是否正确定义了所有层？")
        print("2. forward 方法是否正确实现了前向传播？")
        print("3. train_one_epoch 函数是否正确实现了训练循环？")

    # 可视化预测结果
    print("\n可视化预测结果...")
    visualize_predictions(model, test_loader, DEVICE)

    print("\n" + "=" * 60)
    print("练习完成！")
    print("=" * 60)


# ============================================================
# 测试代码（用于验证你的实现）
# ============================================================

def test_model_structure():
    """
    测试模型结构是否正确
    """
    print("\n" + "=" * 60)
    print("测试模型结构...")
    print("=" * 60)

    try:
        model = MNISTClassifier()

        # 检查层是否存在
        assert model.conv1 is not None, "conv1 未定义！"
        assert model.conv2 is not None, "conv2 未定义！"
        assert model.pool is not None, "pool 未定义！"
        assert model.fc1 is not None, "fc1 未定义！"
        assert model.fc2 is not None, "fc2 未定义！"
        assert model.dropout is not None, "dropout 未定义！"

        # 测试前向传播
        x = torch.randn(2, 1, 28, 28)
        output = model(x)

        assert output.shape == (2, 10), f"输出形状错误！期望 (2, 10)，得到 {output.shape}"

        print("✓ 模型结构测试通过！")
        return True

    except Exception as e:
        print(f"✗ 模型结构测试失败: {e}")
        return False


def test_training_step():
    """
    测试训练步骤是否正确
    """
    print("\n" + "=" * 60)
    print("测试训练步骤...")
    print("=" * 60)

    try:
        # 创建简单的模型和数据
        model = MNISTClassifier()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # 模拟一个 batch
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))

        # 记录初始权重
        initial_weight = model.conv1.weight.clone() if model.conv1 is not None else None

        if initial_weight is None:
            print("✗ 模型未正确初始化")
            return False

        # 执行一次训练步骤
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # 检查权重是否更新
        if torch.equal(initial_weight, model.conv1.weight):
            print("✗ 权重未更新！请检查训练循环")
            return False

        print("✓ 训练步骤测试通过！")
        return True

    except Exception as e:
        print(f"✗ 训练步骤测试失败: {e}")
        return False


if __name__ == '__main__':
    # 运行测试
    model_ok = test_model_structure()
    training_ok = test_training_step()

    if model_ok and training_ok:
        print("\n所有测试通过！开始完整训练...\n")
        main()
    else:
        print("\n请先修复上面的错误，然后重新运行")
        print("\n提示：")
        print("1. 完成 MNISTClassifier 类中所有的 TODO")
        print("2. 完成 train_one_epoch 函数中的 TODO")
        print("3. 完成 evaluate 函数中的 TODO")
        print("4. 完成 main 函数中的 TODO")
