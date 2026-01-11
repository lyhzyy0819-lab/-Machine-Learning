"""
12 目标检测与分割练习题解答

练习 1：手动计算 IoU
练习 2：多类别 NMS
练习 3：U-Net 深度实验
练习 4：U-Net 训练框架
练习 5：检测训练循环框架

运行方法：
    python exercise_12_detection_segmentation.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 练习 1：手动计算 IoU
# ============================================================

def compute_iou(box1, box2):
    """
    计算两个边界框的 IoU

    参数:
        box1, box2: [x1, y1, x2, y2] 格式的边界框

    返回:
        iou: 交并比
    """
    # 交集坐标
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    # 交集面积
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # 各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 并集面积
    union_area = area1 + area2 - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)

    return iou


def exercise_1_manual_iou():
    """手动计算 IoU"""
    print("=" * 60)
    print("练习 1：手动计算 IoU")
    print("=" * 60)

    box_a = [10, 10, 50, 50]  # 40×40 = 1600
    box_b = [30, 30, 70, 70]  # 40×40 = 1600

    print(f"\nBox A: {box_a}")
    print(f"Box B: {box_b}")

    print("\n手动计算步骤：")
    print("-" * 50)

    # Step 1: 计算交集坐标
    inter_x1 = max(box_a[0], box_b[0])  # max(10, 30) = 30
    inter_y1 = max(box_a[1], box_b[1])  # max(10, 30) = 30
    inter_x2 = min(box_a[2], box_b[2])  # min(50, 70) = 50
    inter_y2 = min(box_a[3], box_b[3])  # min(50, 70) = 50
    print(f"1. 交集坐标: ({inter_x1}, {inter_y1}) → ({inter_x2}, {inter_y2})")

    # Step 2: 计算交集面积
    inter_width = inter_x2 - inter_x1  # 50 - 30 = 20
    inter_height = inter_y2 - inter_y1  # 50 - 30 = 20
    inter_area = inter_width * inter_height  # 20 × 20 = 400
    print(f"2. 交集面积: {inter_width} × {inter_height} = {inter_area}")

    # Step 3: 计算各自面积
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])  # 40 × 40 = 1600
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # 40 × 40 = 1600
    print(f"3. Box A 面积: {area_a}")
    print(f"   Box B 面积: {area_b}")

    # Step 4: 计算并集面积
    union_area = area_a + area_b - inter_area  # 1600 + 1600 - 400 = 2800
    print(f"4. 并集面积: {area_a} + {area_b} - {inter_area} = {union_area}")

    # Step 5: 计算 IoU
    iou = inter_area / union_area  # 400 / 2800 ≈ 0.1429
    print(f"5. IoU: {inter_area} / {union_area} = {iou:.4f}")

    # 验证
    iou_computed = compute_iou(box_a, box_b)
    print(f"\n函数计算结果: {iou_computed:.4f}")
    print(f"验证: {'✓ 正确' if abs(iou - iou_computed) < 1e-6 else '✗ 错误'}")


# ============================================================
# 练习 2：多类别 NMS
# ============================================================

def nms_single_class(boxes, scores, iou_threshold=0.5):
    """单类别 NMS"""
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        current = order[0]
        keep.append(current)

        if len(order) == 1:
            break

        current_box = boxes[current]
        other_boxes = boxes[order[1:]]
        ious = np.array([compute_iou(current_box, box) for box in other_boxes])

        mask = ious <= iou_threshold
        order = order[1:][mask]

    return keep


def nms_multi_class(boxes, scores, labels, iou_threshold=0.5):
    """
    多类别 NMS

    不同类别之间的框不应该互相抑制。

    参数:
        boxes: 边界框, shape (N, 4)
        scores: 置信度, shape (N,)
        labels: 类别标签, shape (N,)
        iou_threshold: IoU 阈值

    返回:
        keep: 保留的索引列表
    """
    # 获取所有唯一类别
    unique_labels = np.unique(labels)

    keep_all = []

    # 对每个类别分别进行 NMS
    for label in unique_labels:
        # 获取当前类别的索引
        mask = labels == label
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        # 当前类别的框和分数
        class_boxes = boxes[indices]
        class_scores = scores[indices]

        # 对当前类别进行 NMS
        class_keep = nms_single_class(class_boxes, class_scores, iou_threshold)

        # 将局部索引转换为全局索引
        keep_all.extend(indices[class_keep])

    return keep_all


def exercise_2_multiclass_nms():
    """多类别 NMS 实现"""
    print("\n" + "=" * 60)
    print("练习 2：多类别 NMS")
    print("=" * 60)

    # 模拟检测结果：两个类别（猫=0，狗=1）
    boxes = np.array([
        [100, 100, 200, 200],  # 猫 1
        [105, 105, 205, 205],  # 猫 2（与猫 1 重叠）
        [300, 300, 400, 400],  # 狗 1
        [305, 305, 405, 405],  # 狗 2（与狗 1 重叠）
        [310, 100, 410, 200],  # 猫 3（与猫 1 不重叠）
    ])

    scores = np.array([0.9, 0.85, 0.95, 0.7, 0.8])
    labels = np.array([0, 0, 1, 1, 0])  # 0=猫, 1=狗

    print("\n输入：")
    print(f"  框数量: {len(boxes)}")
    print(f"  类别: 猫={np.sum(labels==0)}个, 狗={np.sum(labels==1)}个")

    # 错误做法：不区分类别的 NMS
    wrong_keep = nms_single_class(boxes, scores, 0.5)
    print(f"\n错误做法（不区分类别）: 保留 {len(wrong_keep)} 个框")
    print(f"  索引: {wrong_keep}")

    # 正确做法：多类别 NMS
    correct_keep = nms_multi_class(boxes, scores, labels, 0.5)
    print(f"\n正确做法（多类别 NMS）: 保留 {len(correct_keep)} 个框")
    print(f"  索引: {correct_keep}")
    print(f"  类别: 猫={sum(labels[i]==0 for i in correct_keep)}个, 狗={sum(labels[i]==1 for i in correct_keep)}个")

    print("\n💡 关键区别：")
    print("   - 单类别 NMS 可能错误地抑制不同类别的框")
    print("   - 多类别 NMS 保证不同类别之间不互相抑制")


# ============================================================
# 练习 3：U-Net 深度实验
# ============================================================

class DoubleConv(nn.Module):
    """两次卷积"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetFlexible(nn.Module):
    """
    可变深度的 U-Net

    参数:
        n_channels: 输入通道数
        n_classes: 输出类别数
        depth: 编码器层数（不包括 bottleneck）
        base_ch: 基础通道数
    """

    def __init__(self, n_channels=3, n_classes=2, depth=4, base_ch=64):
        super().__init__()

        self.depth = depth

        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = n_channels
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(DoubleConv(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = DoubleConv(in_ch, in_ch * 2)

        # 解码器
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        in_ch = in_ch * 2
        for i in range(depth - 1, -1, -1):
            out_ch = base_ch * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            self.decoders.append(DoubleConv(out_ch * 2, out_ch))
            in_ch = out_ch

        # 输出层
        self.outc = nn.Conv2d(base_ch, n_classes, 1)

    def forward(self, x):
        # 编码器
        encoder_features = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            encoder_features.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # 解码器
        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = encoder_features[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.outc(x)


def exercise_3_unet_depth():
    """U-Net 深度实验"""
    print("\n" + "=" * 60)
    print("练习 3：U-Net 深度实验")
    print("=" * 60)

    depths = [2, 3, 4, 5]
    input_size = (1, 3, 256, 256)

    print(f"\n输入尺寸: {input_size}")
    print(f"\n{'深度':>6} {'参数量':>15} {'输出尺寸':>20} {'最小分辨率':>12}")
    print("-" * 55)

    for depth in depths:
        model = UNetFlexible(depth=depth)
        params = sum(p.numel() for p in model.parameters())

        x = torch.randn(input_size)
        with torch.no_grad():
            y = model(x)

        min_res = 256 // (2 ** depth)
        print(f"{depth:>6} {params:>15,} {str(y.shape):>20} {min_res:>12}")

    print("\n💡 深度选择建议：")
    print("   - depth=2: 轻量，适合小图像或快速推理")
    print("   - depth=3: 平衡，适合中等分辨率")
    print("   - depth=4: 原版 U-Net，适合大多数场景")
    print("   - depth=5: 更大感受野，适合大图像和大目标")


# ============================================================
# 练习 4：U-Net 训练框架
# ============================================================

def exercise_4_unet_training():
    """U-Net 训练框架"""
    print("\n" + "=" * 60)
    print("练习 4：U-Net 训练框架")
    print("=" * 60)

    print("""
U-Net 训练的完整流程：

1. 数据准备
   ─────────
   class SegmentationDataset(Dataset):
       def __init__(self, images_dir, masks_dir, transform=None):
           self.images = sorted(glob(images_dir + '/*.png'))
           self.masks = sorted(glob(masks_dir + '/*.png'))
           self.transform = transform

       def __getitem__(self, idx):
           image = Image.open(self.images[idx])
           mask = Image.open(self.masks[idx])
           if self.transform:
               image, mask = self.transform(image, mask)
           return image, mask

2. 损失函数
   ─────────
   # 二分类：BCE + Dice Loss
   class DiceLoss(nn.Module):
       def forward(self, pred, target):
           pred = torch.sigmoid(pred)
           intersection = (pred * target).sum()
           dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
           return 1 - dice

   # 多分类：CrossEntropy + Dice
   criterion = nn.CrossEntropyLoss()

3. 训练循环
   ─────────
   def train_unet(model, train_loader, val_loader, epochs=50, lr=1e-4):
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
           optimizer, mode='max', patience=5
       )

       for epoch in range(epochs):
           model.train()
           for images, masks in train_loader:
               optimizer.zero_grad()
               outputs = model(images)
               loss = criterion(outputs, masks)
               loss.backward()
               optimizer.step()

           # 验证
           val_iou = evaluate(model, val_loader)
           scheduler.step(val_iou)

4. 评估指标
   ─────────
   def compute_iou_segmentation(pred, target, num_classes):
       ious = []
       for cls in range(num_classes):
           pred_cls = (pred == cls)
           target_cls = (target == cls)
           intersection = (pred_cls & target_cls).sum()
           union = (pred_cls | target_cls).sum()
           iou = intersection / (union + 1e-6)
           ious.append(iou)
       return np.mean(ious)  # mIoU

5. 数据增强
   ─────────
   # 图像和 mask 需要同步变换！
   transforms = A.Compose([
       A.HorizontalFlip(p=0.5),
       A.VerticalFlip(p=0.5),
       A.RandomRotate90(p=0.5),
       A.ColorJitter(p=0.3),
   ])
""")


# ============================================================
# 练习 5：检测训练循环框架
# ============================================================

def exercise_5_detection_training():
    """检测训练循环框架"""
    print("\n" + "=" * 60)
    print("练习 5：检测训练循环框架")
    print("=" * 60)

    print("""
简化版 YOLO 训练框架：

1. 数据格式
   ─────────
   # 每张图像的标注格式：
   # [class_id, x_center, y_center, width, height]
   # 坐标都是相对于图像尺寸的归一化值 [0, 1]

   class DetectionDataset(Dataset):
       def __init__(self, images_dir, labels_dir):
           self.images = sorted(glob(images_dir + '/*.jpg'))
           self.labels = sorted(glob(labels_dir + '/*.txt'))

       def __getitem__(self, idx):
           image = load_image(self.images[idx])
           labels = load_labels(self.labels[idx])  # [N, 5]
           return image, labels

2. 损失函数设计
   ─────────────
   YOLO Loss = λ_coord × 定位损失 + λ_obj × 置信度损失 + λ_cls × 分类损失

   class YOLOLoss(nn.Module):
       def __init__(self, S=7, B=2, C=20):
           super().__init__()
           self.S = S
           self.B = B
           self.C = C
           self.lambda_coord = 5.0
           self.lambda_noobj = 0.5

       def forward(self, pred, target):
           # pred: (N, S, S, B*5 + C)
           # target: (N, S, S, 5 + C)  # 简化版

           # 解析预测
           pred_boxes = pred[..., :B*5].reshape(-1, S, S, B, 5)
           pred_cls = pred[..., B*5:]

           # 解析标签
           target_boxes = target[..., :5]
           target_cls = target[..., 5:]
           obj_mask = target[..., 4] > 0  # 有物体的格子

           # 定位损失（只计算有物体的格子）
           coord_loss = F.mse_loss(
               pred_boxes[obj_mask, :4],
               target_boxes[obj_mask, :4]
           )

           # 置信度损失
           obj_loss = F.binary_cross_entropy_with_logits(
               pred_boxes[obj_mask, 4],
               target_boxes[obj_mask, 4]
           )
           noobj_loss = F.binary_cross_entropy_with_logits(
               pred_boxes[~obj_mask, 4],
               torch.zeros_like(pred_boxes[~obj_mask, 4])
           )

           # 分类损失
           cls_loss = F.cross_entropy(pred_cls[obj_mask], target_cls[obj_mask])

           # 总损失
           loss = (self.lambda_coord * coord_loss +
                   obj_loss +
                   self.lambda_noobj * noobj_loss +
                   cls_loss)

           return loss

3. 训练循环
   ─────────
   def train_yolo(model, train_loader, epochs=100, lr=1e-3):
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer, T_max=epochs
       )
       criterion = YOLOLoss()

       for epoch in range(epochs):
           model.train()
           total_loss = 0

           for images, targets in train_loader:
               optimizer.zero_grad()
               outputs = model(images)
               loss = criterion(outputs, targets)
               loss.backward()
               optimizer.step()
               total_loss += loss.item()

           scheduler.step()

           if epoch % 10 == 0:
               print(f'Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}')

4. 后处理
   ───────
   def decode_predictions(pred, conf_threshold=0.5, nms_threshold=0.5):
       # 1. 过滤低置信度预测
       # 2. 转换坐标格式
       # 3. 应用 NMS
       # 4. 返回最终检测结果
       pass

5. 评估指标
   ─────────
   # mAP (mean Average Precision)
   def compute_map(predictions, ground_truths, iou_threshold=0.5):
       # 1. 对每个类别计算 AP
       # 2. 取所有类别的平均值
       pass
""")


# ============================================================
# 主函数
# ============================================================

def main():
    print("╔" + "═" * 58 + "╗")
    print("║" + "  12 目标检测与分割练习题解答  ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    exercise_1_manual_iou()
    exercise_2_multiclass_nms()
    exercise_3_unet_depth()
    exercise_4_unet_training()
    exercise_5_detection_training()

    print("\n" + "=" * 60)
    print("所有练习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
