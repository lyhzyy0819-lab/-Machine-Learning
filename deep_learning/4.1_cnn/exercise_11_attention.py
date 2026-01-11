"""
11 注意力机制练习题解答

练习 1：SE 参数量计算
练习 2：不同 reduction ratio 的 SE 模块
练习 3：CBAM 集成到 VGG（概念框架）
练习 4：自定义注意力模块设计

运行方法：
    python exercise_11_attention.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 练习 1：SE 参数量计算
# ============================================================

def exercise_1_se_params():
    """
    计算 SE 模块增加的参数量

    SE 模块参数 = FC1 参数 + FC2 参数
               = C × (C/r) + (C/r) × C
               = 2 × C² / r

    问题：
    1. 通道数 256，reduction=16
    2. 通道数 512，reduction=32
    """
    print("=" * 60)
    print("练习 1：SE 参数量计算")
    print("=" * 60)

    def calculate_se_params(C, r):
        """计算 SE 模块参数量"""
        fc1_params = C * (C // r)  # 降维层
        fc2_params = (C // r) * C  # 升维层
        total = fc1_params + fc2_params
        return total, fc1_params, fc2_params

    # 案例 1
    print("\n案例 1: C=256, r=16")
    print("-" * 50)
    total, fc1, fc2 = calculate_se_params(256, 16)
    print(f"  FC1 (256 → 16): {fc1:,}")
    print(f"  FC2 (16 → 256): {fc2:,}")
    print(f"  总参数: {total:,}")
    print(f"  理论值: 2 × 256² / 16 = {2 * 256**2 // 16:,}")

    # 案例 2
    print("\n案例 2: C=512, r=32")
    print("-" * 50)
    total, fc1, fc2 = calculate_se_params(512, 32)
    print(f"  FC1 (512 → 16): {fc1:,}")
    print(f"  FC2 (16 → 512): {fc2:,}")
    print(f"  总参数: {total:,}")
    print(f"  理论值: 2 × 512² / 32 = {2 * 512**2 // 32:,}")

    # 与主干网络对比
    print("\n💡 与 ResNet 的对比：")
    print("   ResNet-50 总参数: ~25M")
    print("   SE-ResNet-50 增加: ~2.5M (约 10%)")
    print("   但准确率提升 ~1-2%，性价比很高！")


# ============================================================
# 练习 2：不同 reduction ratio 的 SE 模块
# ============================================================

class SEModule(nn.Module):
    """SE 模块实现"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, H, W = x.shape
        z = self.squeeze(x).view(N, C)
        s = self.excitation(z).view(N, C, 1, 1)
        return x * s


def exercise_2_reduction_ratio():
    """不同 reduction ratio 实验"""
    print("\n" + "=" * 60)
    print("练习 2：不同 Reduction Ratio 的 SE 模块")
    print("=" * 60)

    channels = 256
    reductions = [4, 8, 16, 32]

    print(f"\n通道数: {channels}")
    print(f"\n{'Reduction':>10} {'中间维度':>10} {'参数量':>12} {'相对 r=16':>12}")
    print("-" * 46)

    base_params = None
    for r in reductions:
        se = SEModule(channels, reduction=r)
        params = sum(p.numel() for p in se.parameters())

        if r == 16:
            base_params = params

        mid_dim = channels // r
        ratio = params / base_params if base_params else 1.0
        print(f"{r:>10} {mid_dim:>10} {params:>12,} {ratio:>11.2f}x")

    print("\n💡 权衡分析：")
    print("   - r=4: 参数多，表达能力强，可能过拟合")
    print("   - r=8: 参数适中，常见选择")
    print("   - r=16: 原论文默认值，平衡性好")
    print("   - r=32: 参数少，适合资源受限场景")

    # 实际测试前向传播
    print("\n前向传播测试：")
    x = torch.randn(2, channels, 16, 16)
    for r in reductions:
        se = SEModule(channels, reduction=r)
        y = se(x)
        assert y.shape == x.shape, "形状不匹配"
    print("   ✓ 所有 reduction ratio 的 SE 模块测试通过")


# ============================================================
# 练习 3：CBAM 集成到 VGG
# ============================================================

class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class ChannelAttention(nn.Module):
    """通道注意力模块（CBAM 版本）"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(N, C))
        max_out = self.mlp(self.max_pool(x).view(N, C))
        attention = self.sigmoid(avg_out + max_out).view(N, C, 1, 1)
        return x * attention


class CBAM(nn.Module):
    """CBAM 模块"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def exercise_3_cbam_vgg():
    """CBAM 集成到 VGG 的概念框架"""
    print("\n" + "=" * 60)
    print("练习 3：CBAM 集成到 VGG")
    print("=" * 60)

    print("""
VGG with CBAM 的设计思路：

1. VGG 原始结构
   ─────────────
   VGG-16 包含 5 个卷积块：
   - Block 1: 2 × Conv(64)  → MaxPool
   - Block 2: 2 × Conv(128) → MaxPool
   - Block 3: 3 × Conv(256) → MaxPool
   - Block 4: 3 × Conv(512) → MaxPool
   - Block 5: 3 × Conv(512) → MaxPool

2. 添加 CBAM 的位置
   ─────────────────
   方案 A: 每个卷积块后添加 CBAM
   方案 B: 每个 Conv-ReLU 后添加 CBAM
   方案 C: 只在特定块后添加（如 Block 3, 4, 5）

3. 推荐方案：每个块的最后一个卷积后添加
   这样可以在保持效率的同时获得注意力增强。
""")

    # 简化版 VGG Block with CBAM
    class VGGBlockWithCBAM(nn.Module):
        """带 CBAM 的 VGG 卷积块"""

        def __init__(self, in_channels, out_channels, num_convs):
            super().__init__()

            layers = []
            for i in range(num_convs):
                layers.append(nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels, 3, padding=1
                ))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))

            self.convs = nn.Sequential(*layers)
            self.cbam = CBAM(out_channels)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.convs(x)
            x = self.cbam(x)  # 在卷积后应用 CBAM
            x = self.pool(x)
            return x

    # 测试
    block = VGGBlockWithCBAM(64, 128, num_convs=2)
    x = torch.randn(1, 64, 32, 32)
    y = block(x)

    print(f"VGG Block with CBAM 测试：")
    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print(f"  参数量: {sum(p.numel() for p in block.parameters()):,}")


# ============================================================
# 练习 4：自定义注意力模块设计
# ============================================================

class ParallelAttention(nn.Module):
    """
    并行注意力模块（自定义设计）

    与 CBAM 的串联不同，这里并行计算通道和空间注意力，然后相加。

    结构:
        x → ChannelAttention → ─┐
                                 ├→ 相加 → 输出
        x → SpatialAttention → ─┘
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()

        # 通道注意力分支
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力分支
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

        # 融合权重（可学习）
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        N, C, H, W = x.shape

        # 通道注意力
        ca = self.channel_att(x).view(N, C, 1, 1)
        x_ca = x * ca

        # 空间注意力
        sa = self.spatial_att(x)
        x_sa = x * sa

        # 加权融合
        out = self.alpha * x_ca + (1 - self.alpha) * x_sa

        return out


class MultiScaleAttention(nn.Module):
    """
    多尺度注意力模块（自定义设计）

    使用不同大小的卷积核捕获多尺度的空间关系。
    """

    def __init__(self, channels):
        super().__init__()

        # 多尺度卷积
        self.conv3 = nn.Conv2d(channels, channels // 4, 3, padding=1, groups=channels // 4)
        self.conv5 = nn.Conv2d(channels, channels // 4, 5, padding=2, groups=channels // 4)
        self.conv7 = nn.Conv2d(channels, channels // 4, 7, padding=3, groups=channels // 4)

        # 融合
        self.fuse = nn.Conv2d(channels * 3 // 4, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 多尺度特征
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)

        # 拼接并融合
        concat = torch.cat([f3, f5, f7], dim=1)
        attention = self.sigmoid(self.fuse(concat))

        return x * attention


def exercise_4_custom_attention():
    """自定义注意力模块设计"""
    print("\n" + "=" * 60)
    print("练习 4：自定义注意力模块设计")
    print("=" * 60)

    print("\n设计 1: ParallelAttention（并行注意力）")
    print("-" * 50)

    pa = ParallelAttention(channels=64)
    x = torch.randn(2, 64, 16, 16)
    y = pa(x)

    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print(f"  参数量: {sum(p.numel() for p in pa.parameters()):,}")
    print(f"  融合权重 alpha: {pa.alpha.item():.3f}")

    print("\n设计 2: MultiScaleAttention（多尺度注意力）")
    print("-" * 50)

    msa = MultiScaleAttention(channels=64)
    y2 = msa(x)

    print(f"  输入: {x.shape}")
    print(f"  输出: {y2.shape}")
    print(f"  参数量: {sum(p.numel() for p in msa.parameters()):,}")

    print("\n💡 自定义注意力模块的设计思路：")
    print("   1. 考虑不同维度（通道、空间、时间）")
    print("   2. 考虑不同尺度（多尺度卷积）")
    print("   3. 考虑不同融合方式（串联、并行、加权）")
    print("   4. 保持轻量化（避免引入过多参数）")


# ============================================================
# 主函数
# ============================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "  11 注意力机制练习题解答  ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    exercise_1_se_params()
    exercise_2_reduction_ratio()
    exercise_3_cbam_vgg()
    exercise_4_custom_attention()

    print("\n" + "=" * 60)
    print("所有练习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
