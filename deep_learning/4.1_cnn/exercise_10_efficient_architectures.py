"""
10 现代轻量架构练习题解答

练习 1：参数压缩比计算
练习 2：完整 separable_conv2d 实现
练习 3：宽度乘数实验
练习 4：架构对比（概念框架）

运行方法：
    python exercise_10_efficient_architectures.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 练习 1：参数压缩比计算
# ============================================================

def exercise_1_compression_ratio():
    """
    计算深度可分离卷积的参数压缩比

    问题：
    1. 输入 128 通道，输出 256 通道，5×5 卷积核
    2. 输入 512 通道，输出 512 通道，3×3 卷积核

    压缩比公式：
        标准卷积参数: C_out × C_in × K × K
        深度可分离参数: C_in × K × K + C_out × C_in
        压缩比 = 深度可分离 / 标准 = 1/C_out + 1/K²
    """
    print("=" * 60)
    print("练习 1：参数压缩比计算")
    print("=" * 60)

    def calculate_compression(C_in, C_out, K):
        """计算压缩比"""
        # 标准卷积参数量
        std_params = C_out * C_in * K * K

        # 深度可分离卷积参数量
        # = Depthwise (C_in × K × K) + Pointwise (C_out × C_in)
        dw_params = C_in * K * K
        pw_params = C_out * C_in
        sep_params = dw_params + pw_params

        # 压缩比
        ratio = sep_params / std_params
        compression = std_params / sep_params

        return std_params, sep_params, ratio, compression

    # 案例 1：128 → 256, 5×5
    print("\n案例 1: 128 通道 → 256 通道, 5×5 卷积核")
    print("-" * 50)
    std, sep, ratio, comp = calculate_compression(128, 256, 5)
    print(f"  标准卷积参数: {std:,}")
    print(f"  深度可分离参数: {sep:,}")
    print(f"  压缩比: 1/{comp:.1f} = {ratio:.4f}")
    print(f"  理论值: 1/{256} + 1/{25} = {1/256 + 1/25:.4f}")

    # 案例 2：512 → 512, 3×3
    print("\n案例 2: 512 通道 → 512 通道, 3×3 卷积核")
    print("-" * 50)
    std, sep, ratio, comp = calculate_compression(512, 512, 3)
    print(f"  标准卷积参数: {std:,}")
    print(f"  深度可分离参数: {sep:,}")
    print(f"  压缩比: 1/{comp:.1f} = {ratio:.4f}")
    print(f"  理论值: 1/{512} + 1/{9} = {1/512 + 1/9:.4f}")

    print("\n💡 结论：")
    print("   - 5×5 卷积压缩约 25 倍")
    print("   - 3×3 卷积压缩约 9 倍")
    print("   - 通道数越多，越接近理论极限 1/K²")


# ============================================================
# 练习 2：完整 separable_conv2d 实现
# ============================================================

def depthwise_conv2d(input_tensor, kernels, stride=1, padding=0):
    """
    深度卷积的 NumPy 实现

    参数:
        input_tensor: 输入, shape (C, H, W)
        kernels: 卷积核, shape (C, K, K)
        stride: 步幅
        padding: 填充

    返回:
        output: 输出, shape (C, H_out, W_out)
    """
    C, H, W = input_tensor.shape
    _, K, _ = kernels.shape

    # 添加 padding
    if padding > 0:
        input_padded = np.pad(
            input_tensor,
            ((0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        input_padded = input_tensor

    # 计算输出尺寸
    H_out = (H + 2*padding - K) // stride + 1
    W_out = (W + 2*padding - K) // stride + 1

    # 初始化输出
    output = np.zeros((C, H_out, W_out))

    # 对每个通道独立卷积
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                region = input_padded[c, h_start:h_start+K, w_start:w_start+K]
                output[c, i, j] = np.sum(region * kernels[c])

    return output


def pointwise_conv2d(input_tensor, kernels):
    """
    点卷积（1×1 卷积）的 NumPy 实现

    参数:
        input_tensor: 输入, shape (C_in, H, W)
        kernels: 1×1 卷积核, shape (C_out, C_in)

    返回:
        output: 输出, shape (C_out, H, W)
    """
    C_in, H, W = input_tensor.shape
    C_out, _ = kernels.shape

    # 重塑为矩阵乘法
    input_flat = input_tensor.reshape(C_in, -1)
    output_flat = kernels @ input_flat
    output = output_flat.reshape(C_out, H, W)

    return output


def separable_conv2d(input_tensor, dw_kernels, pw_kernels, stride=1, padding=0):
    """
    完整的深度可分离卷积实现（支持 stride）

    参数:
        input_tensor: 输入, shape (C_in, H, W)
        dw_kernels: Depthwise 核, shape (C_in, K, K)
        pw_kernels: Pointwise 核, shape (C_out, C_in)
        stride: 步幅（用于 Depthwise）
        padding: 填充（用于 Depthwise）

    返回:
        output: 输出, shape (C_out, H_out, W_out)
    """
    # Step 1: Depthwise 卷积
    dw_output = depthwise_conv2d(input_tensor, dw_kernels, stride, padding)

    # Step 2: Pointwise 卷积
    output = pointwise_conv2d(dw_output, pw_kernels)

    return output


def exercise_2_separable_conv():
    """验证 separable_conv2d 实现"""
    print("\n" + "=" * 60)
    print("练习 2：完整 separable_conv2d 实现")
    print("=" * 60)

    # 测试参数
    np.random.seed(42)
    C_in, C_out = 32, 64
    H, W = 16, 16
    K = 3
    stride = 2
    padding = 1

    # 创建输入和权重
    x = np.random.randn(C_in, H, W).astype(np.float32)
    dw_kernels = np.random.randn(C_in, K, K).astype(np.float32) * 0.1
    pw_kernels = np.random.randn(C_out, C_in).astype(np.float32) * 0.1

    # NumPy 实现
    output_np = separable_conv2d(x, dw_kernels, pw_kernels, stride, padding)

    print(f"输入形状: {x.shape}")
    print(f"Depthwise 核: {dw_kernels.shape}")
    print(f"Pointwise 核: {pw_kernels.shape}")
    print(f"stride={stride}, padding={padding}")
    print(f"输出形状: {output_np.shape}")

    # 使用 PyTorch 验证
    x_torch = torch.from_numpy(x).unsqueeze(0)  # (1, C_in, H, W)

    # PyTorch Depthwise
    dw_conv = nn.Conv2d(C_in, C_in, K, stride, padding, groups=C_in, bias=False)
    dw_conv.weight.data = torch.from_numpy(dw_kernels).unsqueeze(1)

    # PyTorch Pointwise
    pw_conv = nn.Conv2d(C_in, C_out, 1, bias=False)
    pw_conv.weight.data = torch.from_numpy(pw_kernels).unsqueeze(-1).unsqueeze(-1)

    with torch.no_grad():
        dw_out = dw_conv(x_torch)
        output_torch = pw_conv(dw_out).squeeze(0).numpy()

    # 比较结果
    diff = np.abs(output_np - output_torch).max()
    print(f"\n与 PyTorch 结果的最大差异: {diff:.2e}")
    print(f"验证结果: {'✓ 通过' if diff < 1e-4 else '✗ 失败'}")


# ============================================================
# 练习 3：宽度乘数实验
# ============================================================

class MobileNetV2_WithAlpha(nn.Module):
    """
    带宽度乘数的 MobileNet V2

    参数:
        alpha: 宽度乘数，控制每层通道数
        num_classes: 分类数
    """

    def __init__(self, alpha=1.0, num_classes=10):
        super().__init__()

        self.alpha = alpha

        # 根据 alpha 调整通道数
        def ch(c):
            return max(8, int(c * alpha))

        # 简化的网络结构
        self.features = nn.Sequential(
            # 初始层
            nn.Conv2d(3, ch(32), 3, padding=1, bias=False),
            nn.BatchNorm2d(ch(32)),
            nn.ReLU6(inplace=True),

            # 几个倒残差块（简化版）
            self._inverted_residual(ch(32), ch(16), 1, 1),
            self._inverted_residual(ch(16), ch(24), 2, 6),
            self._inverted_residual(ch(24), ch(32), 2, 6),
            self._inverted_residual(ch(32), ch(64), 2, 6),
            self._inverted_residual(ch(64), ch(96), 1, 6),
            self._inverted_residual(ch(96), ch(160), 2, 6),

            # 最后的 1×1 卷积
            nn.Conv2d(ch(160), ch(1280), 1, bias=False),
            nn.BatchNorm2d(ch(1280)),
            nn.ReLU6(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch(1280), num_classes)
        )

    def _inverted_residual(self, inp, oup, stride, expand_ratio):
        """简化的倒残差块"""
        hidden_dim = inp * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def exercise_3_width_multiplier():
    """宽度乘数实验"""
    print("\n" + "=" * 60)
    print("练习 3：宽度乘数实验")
    print("=" * 60)

    alphas = [0.5, 0.75, 1.0, 1.25]

    print(f"\n{'Alpha':<10} {'参数量':>15} {'相对 α=1.0':>15}")
    print("-" * 42)

    base_params = None
    for alpha in alphas:
        model = MobileNetV2_WithAlpha(alpha=alpha)
        params = sum(p.numel() for p in model.parameters())

        if alpha == 1.0:
            base_params = params

        ratio = params / base_params if base_params else 1.0
        print(f"{alpha:<10} {params:>15,} {ratio:>14.2f}x")

    print("\n💡 结论：")
    print("   - alpha=0.5 时参数量约为原来的 25%")
    print("   - alpha=0.75 时参数量约为原来的 56%")
    print("   - 宽度乘数是控制模型大小的有效方式")


# ============================================================
# 练习 4：架构对比框架
# ============================================================

def exercise_4_architecture_comparison():
    """架构对比的概念框架"""
    print("\n" + "=" * 60)
    print("练习 4：架构对比（概念框架）")
    print("=" * 60)

    print("""
在自定义数据集上对比不同架构的步骤：

1. 数据准备
   ─────────
   - 加载数据集（如 CIFAR-100、自定义数据集）
   - 数据增强（RandomCrop, HorizontalFlip, ColorJitter 等）
   - 划分训练集/验证集/测试集

2. 模型定义
   ─────────
   - SimpleCNN（基线）
   - MobileNet V2
   - EfficientNet-B0（使用 torchvision 预训练权重）
   - 可选：ResNet-18 作为参考

3. 训练配置
   ─────────
   - 优化器：Adam 或 SGD with momentum
   - 学习率调度：CosineAnnealingLR
   - 损失函数：CrossEntropyLoss
   - 训练轮数：根据数据集大小调整

4. 评估指标
   ─────────
   - 准确率（Top-1, Top-5）
   - 参数量
   - 推理时间（FPS）
   - FLOPs

5. 结果可视化
   ─────────
   - 准确率 vs Epoch 曲线
   - 参数量 vs 准确率 散点图
   - 推理速度对比柱状图
""")

    # 示例代码框架
    print("\n示例代码框架：")
    print("-" * 50)
    print("""
# 1. 定义模型字典
models = {
    'SimpleCNN': SimpleCNN(num_classes=100),
    'MobileNetV2': MobileNetV2_Small(num_classes=100),
    'EfficientNet-B0': create_efficientnet_b0(num_classes=100),
}

# 2. 训练循环
results = {}
for name, model in models.items():
    print(f"训练 {name}...")
    history = train_model(model, train_loader, val_loader, epochs=50)
    results[name] = history

# 3. 评估和可视化
plot_comparison(results)
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "  10 现代轻量架构练习题解答  ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    exercise_1_compression_ratio()
    exercise_2_separable_conv()
    exercise_3_width_multiplier()
    exercise_4_architecture_comparison()

    print("\n" + "=" * 60)
    print("所有练习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
