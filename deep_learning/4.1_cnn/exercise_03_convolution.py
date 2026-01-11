"""
03 卷积练习题解答

练习 1：转置卷积 (Transposed Convolution)
练习 2：空洞卷积 (Dilated Convolution)
练习 3：im2col 优化

运行方法：
    python exercise_03_convolution.py
"""

import numpy as np
import time

np.random.seed(42)


# ============================================================
# 练习 1：转置卷积 (Transposed Convolution)
# ============================================================
#
# 转置卷积也叫 "反卷积"（但这个名字不准确）
# 用途：上采样，将小特征图放大（如 2×2 → 4×4）
#
# 原理：
#   普通卷积：大图 → 小图（通过滑动窗口聚合）
#   转置卷积：小图 → 大图（将每个输入值"散布"到输出）
#
# 公式：
#   输出尺寸 = (输入尺寸 - 1) × stride - 2 × padding + kernel_size
#
# 示意图 (stride=2, kernel=3×3):
#
#   输入 (2×2):          输出 (5×5):
#   ┌───┬───┐            ┌───┬───┬───┬───┬───┐
#   │ a │ b │            │   │   │   │   │   │
#   ├───┼───┤    →       ├───┼───┼───┼───┼───┤
#   │ c │ d │            │   │ a×K │   │ b×K │
#   └───┴───┘            ├───┼───┼───┼───┼───┤
#                        │   │   │   │   │   │
#                        ├───┼───┼───┼───┼───┤
#                        │   │ c×K │   │ d×K │
#                        ├───┼───┼───┼───┼───┤
#                        │   │   │   │   │   │
#                        └───┴───┴───┴───┴───┘
#
#   每个输入值 × 核，放置到输出的对应位置，重叠部分相加
# ============================================================

def transposed_conv2d(input_feature, kernel, stride=1, padding=0):
    """
    转置卷积（反卷积）实现

    参数:
        input_feature: 输入特征图, shape (H_in, W_in)
        kernel: 卷积核, shape (k_h, k_w)
        stride: 步幅, 控制输出放大倍数
        padding: 输出裁剪量

    返回:
        output: 上采样后的特征图

    输出尺寸公式:
        H_out = (H_in - 1) × stride + k_h - 2 × padding
        W_out = (W_in - 1) × stride + k_w - 2 × padding
    """
    H_in, W_in = input_feature.shape
    k_h, k_w = kernel.shape

    # ========================================
    # 第1步：计算输出尺寸
    # ========================================
    H_out = (H_in - 1) * stride + k_h - 2 * padding
    W_out = (W_in - 1) * stride + k_w - 2 * padding

    # 创建一个更大的中间结果（包含 padding 区域）
    H_full = (H_in - 1) * stride + k_h
    W_full = (W_in - 1) * stride + k_w

    output_full = np.zeros((H_full, W_full))

    # ========================================
    # 第2步：将每个输入值 × 核，累加到输出
    # ========================================
    for i in range(H_in):
        for j in range(W_in):
            # 计算这个输入值对应的输出位置（左上角）
            i_out = i * stride
            j_out = j * stride

            # 将 input[i,j] × kernel 累加到输出的对应区域
            # 关键：是累加，不是覆盖！重叠区域会相加
            output_full[i_out:i_out+k_h, j_out:j_out+k_w] += (
                input_feature[i, j] * kernel
            )

    # ========================================
    # 第3步：裁剪 padding（如果有）
    # ========================================
    if padding > 0:
        output = output_full[padding:-padding, padding:-padding]
    else:
        output = output_full

    return output


def test_transposed_conv():
    """测试转置卷积"""
    print("=" * 60)
    print("练习 1：转置卷积 (Transposed Convolution)")
    print("=" * 60)

    # 测试1：简单上采样
    print("\n【测试1】2×2 → 4×4 (stride=2, kernel=2×2)")

    input_small = np.array([
        [1, 2],
        [3, 4]
    ], dtype=float)

    # 简单的上采样核（可以用来实现最近邻插值）
    kernel_2x2 = np.array([
        [1, 1],
        [1, 1]
    ], dtype=float)

    output = transposed_conv2d(input_small, kernel_2x2, stride=2, padding=0)

    print(f"输入 ({input_small.shape}):")
    print(input_small)
    print(f"\n核 ({kernel_2x2.shape}):")
    print(kernel_2x2)
    print(f"\n输出 ({output.shape}):")
    print(output)

    # 测试2：与普通卷积的关系
    print("\n" + "-" * 40)
    print("【测试2】验证：转置卷积是普通卷积的'转置'")

    # 3×3 输入，2×2 核，普通卷积后变成 2×2
    input_3x3 = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)

    kernel_2x2 = np.array([
        [1, 0],
        [0, 1]
    ], dtype=float)

    # 普通卷积：3×3 → 2×2
    def simple_conv(img, k):
        h, w = img.shape
        kh, kw = k.shape
        out = np.zeros((h-kh+1, w-kw+1))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = np.sum(img[i:i+kh, j:j+kw] * k)
        return out

    conv_output = simple_conv(input_3x3, kernel_2x2)
    print(f"\n普通卷积: 3×3 → 2×2")
    print(f"输入:\n{input_3x3}")
    print(f"输出:\n{conv_output}")

    # 转置卷积：2×2 → 3×3
    transposed_output = transposed_conv2d(conv_output, kernel_2x2, stride=1, padding=0)
    print(f"\n转置卷积: 2×2 → 3×3")
    print(f"输出:\n{transposed_output}")

    print("\n💡 注意：转置卷积不是卷积的逆运算！")
    print("   它只是在形状上是'反向'的")


# ============================================================
# 练习 2：空洞卷积 (Dilated Convolution)
# ============================================================
#
# 也叫扩张卷积、膨胀卷积 (Atrous Convolution)
#
# 核心思想：
#   在卷积核元素之间插入"空洞"（零），增大感受野
#   而不增加参数量和计算量
#
# 示意图 (3×3 核，dilation=2):
#
#   普通 3×3 核:          空洞 3×3 核 (dilation=2):
#   ┌─┬─┬─┐              ┌─┬─┬─┬─┬─┐
#   │*│*│*│              │*│0│*│0│*│
#   ├─┼─┼─┤              ├─┼─┼─┼─┼─┤
#   │*│*│*│   →          │0│0│0│0│0│
#   ├─┼─┼─┤              ├─┼─┼─┼─┼─┤
#   │*│*│*│              │*│0│*│0│*│
#   └─┴─┴─┘              ├─┼─┼─┼─┼─┤
#                        │0│0│0│0│0│
#   感受野: 3×3           ├─┼─┼─┼─┼─┤
#                        │*│0│*│0│*│
#                        └─┴─┴─┴─┴─┘
#                        感受野: 5×5 (更大！)
#
# 公式：
#   有效核大小 = kernel_size + (kernel_size - 1) × (dilation - 1)
#             = dilation × (kernel_size - 1) + 1
# ============================================================

def dilated_conv2d(image, kernel, dilation=1, padding=0, stride=1):
    """
    空洞卷积（膨胀卷积）实现

    参数:
        image: 输入图像, shape (H, W)
        kernel: 卷积核, shape (k_h, k_w)
        dilation: 膨胀率, 默认 1（普通卷积）
                  dilation=2 表示核元素间隔 1 个像素
        padding: 零填充
        stride: 步幅

    返回:
        output: 卷积结果

    有效核大小:
        k_eff = dilation × (k - 1) + 1

    输出尺寸:
        H_out = (H + 2×padding - k_eff) // stride + 1
    """
    H, W = image.shape
    k_h, k_w = kernel.shape

    # ========================================
    # 第1步：计算有效核大小
    # 例如：3×3 核，dilation=2 → 有效大小 5×5
    # ========================================
    k_h_eff = dilation * (k_h - 1) + 1
    k_w_eff = dilation * (k_w - 1) + 1

    # ========================================
    # 第2步：添加 padding
    # ========================================
    if padding > 0:
        image_padded = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        image_padded = image

    H_padded, W_padded = image_padded.shape

    # ========================================
    # 第3步：计算输出尺寸（使用有效核大小）
    # ========================================
    out_h = (H_padded - k_h_eff) // stride + 1
    out_w = (W_padded - k_w_eff) // stride + 1

    output = np.zeros((out_h, out_w))

    # ========================================
    # 第4步：卷积计算
    # 关键：采样时使用 dilation 间隔
    # ========================================
    for i in range(out_h):
        for j in range(out_w):
            i_start = i * stride
            j_start = j * stride

            # 累加器
            total = 0.0

            # 遍历核的每个元素
            for m in range(k_h):
                for n in range(k_w):
                    # 关键：输入位置要乘以 dilation
                    # 这就是"空洞"的来源
                    img_i = i_start + m * dilation
                    img_j = j_start + n * dilation

                    total += image_padded[img_i, img_j] * kernel[m, n]

            output[i, j] = total

    return output


def test_dilated_conv():
    """测试空洞卷积"""
    print("\n" + "=" * 60)
    print("练习 2：空洞卷积 (Dilated Convolution)")
    print("=" * 60)

    # 创建测试图像
    image = np.arange(1, 50).reshape(7, 7).astype(float)

    # 3×3 拉普拉斯核（边缘检测）
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=float)

    print("\n输入图像 (7×7):")
    print(image)
    print("\n拉普拉斯核 (3×3):")
    print(kernel)

    # 测试不同 dilation
    print("\n" + "-" * 40)
    for d in [1, 2, 3]:
        k_eff = d * (3 - 1) + 1  # 有效核大小
        output = dilated_conv2d(image, kernel, dilation=d)
        print(f"\ndilation={d}: 有效核大小={k_eff}×{k_eff}, 输出形状={output.shape}")
        print(output)

    print("\n💡 观察：")
    print("   - dilation=1: 感受野 3×3, 输出 5×5")
    print("   - dilation=2: 感受野 5×5, 输出 3×3")
    print("   - dilation=3: 感受野 7×7, 输出 1×1")
    print("   空洞卷积在不增加参数的情况下扩大了感受野！")


# ============================================================
# 练习 3：im2col 优化
# ============================================================
#
# 核心思想：
#   将卷积操作转换为矩阵乘法，利用高度优化的 BLAS 库
#
# 步骤：
#   1. im2col：将输入的每个感受野展开成一列
#   2. 矩阵乘法：展开后的输入 × 核
#   3. reshape：将结果变回特征图形状
#
# 示意图：
#
#   输入 (4×4):         核 (2×2):
#   ┌─┬─┬─┬─┐           ┌─┬─┐
#   │a│b│c│d│           │w│x│
#   ├─┼─┼─┼─┤           ├─┼─┤
#   │e│f│g│h│           │y│z│
#   ├─┼─┼─┼─┤           └─┴─┘
#   │i│j│k│l│
#   ├─┼─┼─┼─┤
#   │m│n│o│p│
#   └─┴─┴─┴─┘
#
#   im2col 展开（9个2×2窗口，每个展开成4元素列）：
#
#   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐
#   │ a │ b │ c │ e │ f │ g │ i │ j │ k │
#   │ b │ c │ d │ f │ g │ h │ j │ k │ l │
#   │ e │ f │ g │ i │ j │ k │ m │ n │ o │
#   │ f │ g │ h │ j │ k │ l │ n │ o │ p │
#   └───┴───┴───┴───┴───┴───┴───┴───┴───┘
#     ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
#    窗口1 2  3  4  5  6  7  8  9
#
#   核展开成行向量：[w, x, y, z]
#
#   矩阵乘法：[w,x,y,z] × im2col_matrix = [o1,o2,...,o9]
#
#   reshape 成 (3×3) 输出
# ============================================================

def im2col(image, kernel_h, kernel_w, stride=1, padding=0):
    """
    将图像按感受野展开成矩阵

    参数:
        image: 输入图像, shape (H, W)
        kernel_h, kernel_w: 核的高和宽
        stride: 步幅
        padding: 填充

    返回:
        col: 展开后的矩阵, shape (k_h × k_w, out_h × out_w)
             每一列是一个感受野展开后的向量
    """
    H, W = image.shape

    # 添加 padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    H_padded, W_padded = image.shape

    # 计算输出尺寸
    out_h = (H_padded - kernel_h) // stride + 1
    out_w = (W_padded - kernel_w) // stride + 1

    # 初始化 col 矩阵
    # 每列是一个感受野（k_h × k_w 个元素）
    # 共有 out_h × out_w 个感受野
    col = np.zeros((kernel_h * kernel_w, out_h * out_w))

    col_idx = 0  # 列索引

    for i in range(out_h):
        for j in range(out_w):
            # 提取感受野
            i_start = i * stride
            j_start = j * stride
            receptive_field = image[i_start:i_start+kernel_h,
                                    j_start:j_start+kernel_w]

            # 展开成列向量
            col[:, col_idx] = receptive_field.flatten()
            col_idx += 1

    return col, (out_h, out_w)


def conv2d_im2col(image, kernel, stride=1, padding=0):
    """
    使用 im2col 的卷积实现

    将卷积转换为矩阵乘法:
        output = kernel_row × im2col_matrix

    参数:
        image: 输入图像, shape (H, W)
        kernel: 卷积核, shape (k_h, k_w)
        stride: 步幅
        padding: 填充

    返回:
        output: 卷积结果
    """
    k_h, k_w = kernel.shape

    # ========================================
    # 第1步：im2col - 将图像展开成矩阵
    # ========================================
    col, (out_h, out_w) = im2col(image, k_h, k_w, stride, padding)

    # ========================================
    # 第2步：将核展开成行向量
    # ========================================
    kernel_row = kernel.flatten().reshape(1, -1)  # (1, k_h×k_w)

    # ========================================
    # 第3步：矩阵乘法
    # (1, k_h×k_w) × (k_h×k_w, out_h×out_w) = (1, out_h×out_w)
    # ========================================
    output_flat = np.dot(kernel_row, col)  # (1, out_h×out_w)

    # ========================================
    # 第4步：reshape 成输出形状
    # ========================================
    output = output_flat.reshape(out_h, out_w)

    return output


def test_im2col():
    """测试 im2col 优化"""
    print("\n" + "=" * 60)
    print("练习 3：im2col 优化")
    print("=" * 60)

    # 创建测试数据
    image = np.arange(1, 17).reshape(4, 4).astype(float)
    kernel = np.array([[1, 0], [0, 1]], dtype=float)

    print("\n输入图像 (4×4):")
    print(image)
    print("\n卷积核 (2×2):")
    print(kernel)

    # im2col 展开
    col, (out_h, out_w) = im2col(image, 2, 2)
    print(f"\nim2col 展开结果 (形状: {col.shape}):")
    print(f"  - 行数 = k_h × k_w = 2 × 2 = 4")
    print(f"  - 列数 = out_h × out_w = {out_h} × {out_w} = 9")
    print("\n展开矩阵（每列是一个感受野）:")
    print(col)

    # 使用 im2col 的卷积
    output_im2col = conv2d_im2col(image, kernel)
    print(f"\nim2col 卷积输出 ({output_im2col.shape}):")
    print(output_im2col)

    # 对比普通实现
    def conv2d_naive(img, k):
        h, w = img.shape
        kh, kw = k.shape
        out = np.zeros((h-kh+1, w-kw+1))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = np.sum(img[i:i+kh, j:j+kw] * k)
        return out

    output_naive = conv2d_naive(image, kernel)
    print(f"\n普通卷积输出 ({output_naive.shape}):")
    print(output_naive)

    print(f"\n结果是否一致: {np.allclose(output_im2col, output_naive)}")

    # 性能对比
    print("\n" + "-" * 40)
    print("性能对比 (64×64 图像, 3×3 核)")

    large_img = np.random.randn(64, 64)
    large_kernel = np.random.randn(3, 3)

    # 普通实现
    start = time.time()
    for _ in range(50):
        _ = conv2d_naive(large_img, large_kernel)
    time_naive = (time.time() - start) / 50 * 1000

    # im2col 实现
    start = time.time()
    for _ in range(50):
        _ = conv2d_im2col(large_img, large_kernel)
    time_im2col = (time.time() - start) / 50 * 1000

    print(f"\n普通实现: {time_naive:.2f} ms")
    print(f"im2col:   {time_im2col:.2f} ms")
    print(f"加速比:    {time_naive/time_im2col:.2f}x")

    print("\n💡 说明：")
    print("   - im2col 将卷积转为矩阵乘法")
    print("   - 矩阵乘法可以利用 BLAS 库高度优化")
    print("   - 这是 CNN 框架的核心优化技术之一")
    print("   - 代价是需要更多内存（展开后的矩阵很大）")


# ============================================================
# 主函数：运行所有测试
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║" + "  03 卷积练习题解答  ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    # 练习1：转置卷积
    test_transposed_conv()

    # 练习2：空洞卷积
    test_dilated_conv()

    # 练习3：im2col 优化
    test_im2col()

    print("\n" + "=" * 60)
    print("所有练习完成！")
    print("=" * 60)

    print("\n📚 总结：")
    print("┌────────────┬──────────────────────────────────┐")
    print("│ 技术       │ 用途                             │")
    print("├────────────┼──────────────────────────────────┤")
    print("│ 转置卷积   │ 上采样，用于分割、生成网络       │")
    print("│ 空洞卷积   │ 扩大感受野，用于语义分割         │")
    print("│ im2col     │ 性能优化，CNN框架核心技术        │")
    print("└────────────┴──────────────────────────────────┘")
