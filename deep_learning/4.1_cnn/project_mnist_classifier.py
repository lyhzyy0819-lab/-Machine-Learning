"""
MNIST 手写数字分类器

项目目标：使用 CNN 对 MNIST 手写数字进行分类

使用方法：
    python project_mnist_classifier.py

要求：
    - PyTorch
    - torchvision
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 配置
# ============================================================

# 超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


# ============================================================
# 数据准备
# ============================================================

def load_data():
    """
    加载 MNIST 数据集

    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
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

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    return train_loader, test_loader


# ============================================================
# CNN 模型
# ============================================================

class MNISTNet(nn.Module):
    """
    MNIST 分类 CNN

    架构:
        输入: (1, 28, 28)
        Conv(1→32, 3×3, p=1) → ReLU → MaxPool(2×2)  → (32, 14, 14)
        Conv(32→64, 3×3, p=1) → ReLU → MaxPool(2×2) → (64, 7, 7)
        Flatten → (3136)
        FC(3136→128) → ReLU → Dropout(0.5)
        FC(128→10)

    参数:
        - 输入: (N, 1, 28, 28)
        - 输出: (N, 10) 未归一化的类别分数
    """

    def __init__(self):
        super(MNISTNet, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(
            in_channels=1,    # MNIST 是灰度图
            out_channels=32,  # 32 个滤波器
            kernel_size=3,    # 3×3 卷积核
            padding=1         # same padding
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 28 → 14 (pool1) → 7 (pool2)
        # 展平后: 64 × 7 × 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入图像 (N, 1, 28, 28)

        返回:
            logits: 类别分数 (N, 10)
        """
        # Conv1 → ReLU → Pool
        x = self.pool(F.relu(self.conv1(x)))  # (N, 32, 14, 14)

        # Conv2 → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))  # (N, 64, 7, 7)

        # 展平
        x = x.view(-1, 64 * 7 * 7)  # (N, 3136)

        # FC1 → ReLU → Dropout
        x = self.dropout(F.relu(self.fc1(x)))  # (N, 128)

        # FC2 (输出层)
        x = self.fc2(x)  # (N, 10)

        return x


# ============================================================
# 训练函数
# ============================================================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个 epoch

    返回:
        平均损失, 准确率
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 移动到设备
        data, target = data.to(device), target.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        output = model(data)

        # 计算损失
        loss = criterion(output, target)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    在测试集上评估模型

    返回:
        平均损失, 准确率
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ============================================================
# 可视化函数
# ============================================================

def visualize_predictions(model, test_loader, device, num_samples=10):
    """可视化模型预测结果"""
    model.eval()

    # 获取一批数据
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)

    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)

    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for i, ax in enumerate(axes.flatten()):
        if i < num_samples:
            img = data[i].cpu().numpy().squeeze()
            true_label = target[i].item()
            pred_label = pred[i].item()

            ax.imshow(img, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'真实: {true_label}, 预测: {pred_label}', color=color)
        ax.axis('off')

    plt.suptitle('模型预测结果 (绿色=正确, 红色=错误)', fontsize=14)
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    plt.show()
    print("预测结果已保存到 predictions.png")


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    # 损失曲线
    axes[0].plot(epochs, train_losses, 'b-', label='训练损失')
    axes[0].plot(epochs, test_losses, 'r-', label='测试损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(epochs, train_accs, 'b-', label='训练准确率')
    axes[1].plot(epochs, test_accs, 'r-', label='测试准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('准确率曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("训练历史已保存到 training_history.png")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    print("=" * 60)
    print("MNIST 手写数字分类器")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    train_loader, test_loader = load_data()

    # 2. 创建模型
    print("\n[2/4] 创建模型...")
    model = MNISTNet().to(DEVICE)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练
    print("\n[3/4] 开始训练...")
    print("-" * 60)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss, train_acc = train_epoch(
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
              f"训练损失={train_loss:.4f}, 训练准确率={train_acc:.2f}%, "
              f"测试损失={test_loss:.4f}, 测试准确率={test_acc:.2f}%")

    print("-" * 60)

    # 5. 可视化
    print("\n[4/4] 可视化结果...")
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    visualize_predictions(model, test_loader, DEVICE)

    # 6. 保存模型
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("\n模型已保存到 mnist_cnn.pth")

    print("\n" + "=" * 60)
    print(f"训练完成！最终测试准确率: {test_accs[-1]:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
