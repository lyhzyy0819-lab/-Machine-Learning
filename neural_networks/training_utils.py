"""
训练工具模块 (Training Utilities)

本模块提供可复用的深度学习训练工具，包括：
- 学习率调度器（Step Decay, Exponential, Cosine Annealing, Warm Restart）
- 梯度裁剪（Norm Clipping, Value Clipping）
- 梯度累积器
- 早停机制
- 训练诊断工具
- 训练监控器（集成所有功能）

使用示例:
--------
from training_utils import TrainingMonitor, CosineAnnealingScheduler

# 创建训练监控器
monitor = TrainingMonitor(
    lr_scheduler=CosineAnnealingScheduler(initial_lr=0.01, min_lr=1e-6, T_max=100),
    gradient_clip_norm=1.0,
    early_stopping_patience=10,
    diagnose_every=10
)

# 在训练循环中使用
for epoch in range(max_epochs):
    monitor.on_epoch_start(epoch)

    for batch in batches:
        gradients = compute_gradients(batch)
        gradients = monitor.on_batch_end(gradients)  # 梯度裁剪
        update_parameters(gradients)

    if monitor.on_epoch_end(train_loss, val_loss, model):
        break  # 早停触发

monitor.plot_history()
monitor.summary()

作者: Machine Learning 学习项目
日期: 2024年12月
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import copy
import warnings

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
#                           学习率调度器
# =============================================================================

class LearningRateScheduler:
    """
    学习率调度器基类

    所有调度器都继承此类，实现 step() 方法来计算当前 epoch 的学习率

    属性:
    -----
    initial_lr : float
        初始学习率
    current_lr : float
        当前学习率
    epoch : int
        当前 epoch 数
    history : List[float]
        学习率历史记录
    """

    def __init__(self, initial_lr: float):
        """
        参数:
        -----
        initial_lr : float
            初始学习率（η_0）
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.epoch = 0
        self.history = []

    def step(self) -> float:
        """
        计算当前 epoch 的学习率（子类实现）

        返回:
        -----
        float
            当前学习率
        """
        raise NotImplementedError("子类必须实现 step() 方法")

    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.current_lr

    def update(self) -> float:
        """
        更新学习率（每个 epoch 调用一次）

        返回:
        -----
        float
            更新后的学习率
        """
        self.current_lr = self.step()
        self.history.append(self.current_lr)
        self.epoch += 1
        return self.current_lr

    def reset(self):
        """重置调度器状态"""
        self.current_lr = self.initial_lr
        self.epoch = 0
        self.history = []


class StepDecayScheduler(LearningRateScheduler):
    """
    阶梯衰减调度器

    公式: η_t = η_0 × γ^(⌊t/T⌋)

    每 T 个 epoch 将学习率乘以 γ（如 0.1），实现阶梯式下降

    参数:
    -----
    initial_lr : float
        初始学习率（η_0）
    decay_rate : float
        衰减率（γ），每次衰减后学习率变为原来的 γ 倍，默认 0.1
    decay_steps : int
        衰减步数（T），每 T 个 epoch 衰减一次，默认 30

    示例:
    -----
    scheduler = StepDecayScheduler(initial_lr=0.1, decay_rate=0.1, decay_steps=30)
    # Epoch 0-29:  η = 0.1
    # Epoch 30-59: η = 0.01
    # Epoch 60-89: η = 0.001
    """

    def __init__(self, initial_lr: float, decay_rate: float = 0.1, decay_steps: int = 30):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def step(self) -> float:
        # ⌊t/T⌋: 计算当前在第几个衰减周期
        decay_factor = self.epoch // self.decay_steps
        return self.initial_lr * (self.decay_rate ** decay_factor)


class ExponentialDecayScheduler(LearningRateScheduler):
    """
    指数衰减调度器

    公式: η_t = η_0 × e^(-λt)

    学习率按指数曲线平滑下降

    参数:
    -----
    initial_lr : float
        初始学习率（η_0）
    decay_rate : float
        衰减率（λ），控制衰减速度，默认 0.05

    示例:
    -----
    scheduler = ExponentialDecayScheduler(initial_lr=0.1, decay_rate=0.05)
    # Epoch 0:  η = 0.100
    # Epoch 10: η = 0.061
    # Epoch 20: η = 0.037
    """

    def __init__(self, initial_lr: float, decay_rate: float = 0.05):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def step(self) -> float:
        return self.initial_lr * np.exp(-self.decay_rate * self.epoch)


class CosineAnnealingScheduler(LearningRateScheduler):
    """
    余弦退火调度器

    公式: η_t = η_min + 0.5(η_max - η_min)(1 + cos(t/T_max × π))

    学习率按余弦曲线从 η_max 平滑下降到 η_min

    参数:
    -----
    initial_lr : float
        最大学习率（η_max）
    min_lr : float
        最小学习率（η_min），默认 0.0
    T_max : int
        周期长度，默认 100

    示例:
    -----
    scheduler = CosineAnnealingScheduler(initial_lr=0.1, min_lr=0.001, T_max=100)
    # Epoch 0:   η = 0.1 (最大)
    # Epoch 50:  η ≈ 0.05 (中间)
    # Epoch 100: η = 0.001 (最小)
    """

    def __init__(self, initial_lr: float, min_lr: float = 0.0, T_max: int = 100):
        super().__init__(initial_lr)
        self.max_lr = initial_lr
        self.min_lr = min_lr
        self.T_max = T_max

    def step(self) -> float:
        # cos(t/T_max × π): 从 1 下降到 -1
        # (1 + cos(...)): 从 2 下降到 0
        # 0.5 × (1 + cos(...)): 从 1 下降到 0
        cosine = np.cos(np.pi * (self.epoch % self.T_max) / self.T_max)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + cosine)


class WarmRestartScheduler(LearningRateScheduler):
    """
    带重启的余弦退火（SGDR: Stochastic Gradient Descent with Warm Restarts）

    公式: η_t = η_min + 0.5(η_max - η_min)(1 + cos(t_epoch/T_i × π))

    周期性重启到 η_max，帮助跳出局部最优

    参数:
    -----
    initial_lr : float
        最大学习率（η_max）
    min_lr : float
        最小学习率（η_min），默认 0.0
    T_0 : int
        第一个周期的长度，默认 10
    T_mult : float
        周期长度的乘数，默认 2.0
        - T_mult > 1: 周期逐渐变长
        - T_mult = 1: 周期固定

    示例:
    -----
    scheduler = WarmRestartScheduler(initial_lr=0.1, T_0=10, T_mult=2)
    # 第1个周期: Epoch 0-9  (长度10)
    # 第2个周期: Epoch 10-29 (长度20)
    # 第3个周期: Epoch 30-69 (长度40)
    """

    def __init__(self, initial_lr: float, min_lr: float = 0.0,
                 T_0: int = 10, T_mult: float = 2.0):
        super().__init__(initial_lr)
        self.max_lr = initial_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult

        self.T_current = T_0  # 当前周期长度
        self.epoch_in_cycle = 0  # 当前周期内的 epoch
        self.cycle_count = 0  # 已完成的周期数

    def step(self) -> float:
        # 在当前周期内使用余弦退火
        cosine = np.cos(np.pi * self.epoch_in_cycle / self.T_current)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + cosine)

        # 更新周期信息
        self.epoch_in_cycle += 1

        # 如果当前周期结束，重启到下一个周期
        if self.epoch_in_cycle >= self.T_current:
            self.cycle_count += 1
            self.epoch_in_cycle = 0
            # 下一个周期的长度
            self.T_current = int(self.T_0 * (self.T_mult ** self.cycle_count))

        return lr

    def reset(self):
        """重置调度器状态"""
        super().reset()
        self.T_current = self.T_0
        self.epoch_in_cycle = 0
        self.cycle_count = 0


# =============================================================================
#                           梯度裁剪
# =============================================================================

def clip_gradient_norm(gradients: List[np.ndarray],
                       max_norm: float = 1.0) -> Tuple[List[np.ndarray], float]:
    """
    梯度范数裁剪（推荐方法）

    将所有梯度的 L2 范数限制在 max_norm 以内，保持梯度方向不变

    公式:
        g ← (max_norm / ||g||) × g  if ||g|| > max_norm

    参数:
    -----
    gradients : List[np.ndarray]
        梯度列表 [dW1, db1, dW2, db2, ...]
    max_norm : float
        最大梯度范数（阈值 θ），默认 1.0

    返回:
    -----
    Tuple[List[np.ndarray], float]
        (裁剪后的梯度列表, 原始梯度范数)

    示例:
    -----
    gradients = [dW1, db1, dW2, db2]
    clipped_grads, original_norm = clip_gradient_norm(gradients, max_norm=1.0)
    """
    # 步骤1: 计算所有梯度的 L2 范数
    # ||g|| = sqrt(sum(g_i^2)) for all parameters
    total_norm = 0.0
    for grad in gradients:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    # 步骤2: 计算裁剪系数
    clip_coef = max_norm / (total_norm + 1e-6)

    # 步骤3: 如果需要裁剪，所有梯度乘以相同系数
    if clip_coef < 1:
        clipped_gradients = [grad * clip_coef for grad in gradients]
        return clipped_gradients, total_norm
    else:
        return gradients, total_norm


def clip_gradient_value(gradients: List[np.ndarray],
                        clip_value: float = 1.0) -> List[np.ndarray]:
    """
    梯度值裁剪（逐元素裁剪）

    将每个梯度元素限制在 [-clip_value, clip_value] 范围内

    注意: 这会改变梯度方向，通常推荐使用 clip_gradient_norm

    参数:
    -----
    gradients : List[np.ndarray]
        梯度列表
    clip_value : float
        裁剪阈值，默认 1.0

    返回:
    -----
    List[np.ndarray]
        裁剪后的梯度列表
    """
    return [np.clip(grad, -clip_value, clip_value) for grad in gradients]


# =============================================================================
#                           梯度累积器
# =============================================================================

class GradientAccumulator:
    """
    梯度累积器

    用多个小批量累积梯度，模拟大批量训练

    原理:
    -----
    想用 batch_size=128 但内存只够 batch_size=32？
    用 4 个 batch_size=32 累积梯度，效果等同于 batch_size=128！

    公式: g_accum = Σ g_i / n_accumulation_steps

    参数:
    -----
    accumulation_steps : int
        累积步数，默认 4
        等效 batch_size = 实际 batch_size × accumulation_steps

    使用示例:
    --------
    accumulator = GradientAccumulator(accumulation_steps=4)

    for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
        gradients = compute_gradients(X_batch, y_batch)
        accumulator.accumulate(gradients)

        if accumulator.should_update():
            avg_gradients = accumulator.get_averaged_gradients()
            update_parameters(avg_gradients)
            accumulator.reset()
    """

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        self.accumulated_gradients = None

    def accumulate(self, gradients: List[np.ndarray]) -> None:
        """累积梯度"""
        if self.accumulated_gradients is None:
            self.accumulated_gradients = [grad.copy() for grad in gradients]
        else:
            for i, grad in enumerate(gradients):
                self.accumulated_gradients[i] += grad
        self.step_count += 1

    def should_update(self) -> bool:
        """判断是否该更新参数"""
        return self.step_count >= self.accumulation_steps

    def get_averaged_gradients(self) -> List[np.ndarray]:
        """获取平均后的梯度"""
        if self.accumulated_gradients is None:
            raise ValueError("没有累积的梯度！请先调用 accumulate()")
        return [grad / self.accumulation_steps for grad in self.accumulated_gradients]

    def reset(self) -> None:
        """重置累积器"""
        self.accumulated_gradients = None
        self.step_count = 0


# =============================================================================
#                           早停
# =============================================================================

class EarlyStopping:
    """
    早停（Early Stopping）

    监控验证集性能，当连续 patience 个 epoch 没有提升时停止训练

    参数:
    -----
    patience : int
        容忍的 epoch 数，默认 10
    min_delta : float
        最小改进量，小于此值视为没有改进，默认 0.0
    mode : str
        'min'（监控 loss）或 'max'（监控 accuracy），默认 'min'
    verbose : bool
        是否打印信息，默认 True

    属性:
    -----
    best_score : float
        最佳验证性能
    best_model : object
        最佳模型的深拷贝
    counter : int
        没有改进的 epoch 计数
    early_stop : bool
        是否应该停止训练

    使用示例:
    --------
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(...)
        val_loss = validate(...)

        if early_stopping.step(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            break

    best_model = early_stopping.load_best_model()
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = 'min', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.val_score_history = []

        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        elif mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def step(self, val_score: float, model=None) -> bool:
        """
        检查是否应该早停

        参数:
        -----
        val_score : float
            当前 epoch 的验证性能
        model : object, optional
            模型对象（如果提供则保存最佳模型）

        返回:
        -----
        bool
            是否应该停止训练
        """
        self.val_score_history.append(val_score)

        if self.best_score is None:
            self.best_score = val_score
            if model is not None:
                self._save_model(model)
            if self.verbose:
                print(f"  初始验证性能: {val_score:.4f}")
            return False

        if self.is_better(val_score, self.best_score):
            improvement = abs(val_score - self.best_score)
            self.best_score = val_score
            self.counter = 0
            if model is not None:
                self._save_model(model)
            if self.verbose:
                print(f"  验证性能提升 {improvement:.4f} -> 新最佳: {val_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  验证性能未提升 ({self.counter}/{self.patience}) "
                      f"当前: {val_score:.4f} vs 最佳: {self.best_score:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  早停触发！连续 {self.patience} 个 epoch 无改进")
                return True
        return False

    def _save_model(self, model):
        """保存模型的深拷贝"""
        self.best_model = copy.deepcopy(model)

    def load_best_model(self):
        """加载最佳模型"""
        if self.best_model is None:
            raise ValueError("没有保存的最佳模型")
        return self.best_model

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.val_score_history = []


# =============================================================================
#                           训练诊断
# =============================================================================

def diagnose_training(train_loss_history: List[float],
                      val_loss_history: List[float],
                      train_acc_history: List[float] = None,
                      val_acc_history: List[float] = None,
                      verbose: bool = True) -> Dict[str, Any]:
    """
    训练诊断

    根据训练和验证 loss/acc 的表现，诊断训练问题并给出建议

    参数:
    -----
    train_loss_history : List[float]
        训练集 loss 历史
    val_loss_history : List[float]
        验证集 loss 历史
    train_acc_history : List[float], optional
        训练集准确率历史
    val_acc_history : List[float], optional
        验证集准确率历史
    verbose : bool
        是否打印诊断报告，默认 True

    返回:
    -----
    Dict[str, Any]
        诊断结果字典，包含:
        - status: 训练状态 ('good', 'overfitting', 'underfitting', 'exploding', 'stagnant')
        - train_loss: 最终训练 loss
        - val_loss: 最终验证 loss
        - loss_gap: loss 差距
        - train_trend: 训练 loss 趋势
        - val_trend: 验证 loss 趋势
        - suggestions: 建议列表
    """
    result = {}

    final_train_loss = train_loss_history[-1]
    final_val_loss = val_loss_history[-1]
    loss_gap = abs(final_val_loss - final_train_loss)

    recent_epochs = min(10, len(train_loss_history) // 2)
    if recent_epochs < 2:
        recent_epochs = 2
    train_trend = np.mean(np.diff(train_loss_history[-recent_epochs:]))
    val_trend = np.mean(np.diff(val_loss_history[-recent_epochs:]))

    result['train_loss'] = final_train_loss
    result['val_loss'] = final_val_loss
    result['loss_gap'] = loss_gap
    result['train_trend'] = train_trend
    result['val_trend'] = val_trend
    result['suggestions'] = []

    # 诊断逻辑
    if np.isnan(final_train_loss) or np.isinf(final_train_loss):
        result['status'] = 'exploding'
        result['suggestions'] = [
            "降低学习率",
            "使用梯度裁剪",
            "检查权重初始化",
            "使用 Batch Normalization"
        ]
    elif loss_gap > 0.5 and final_train_loss < 0.5:
        result['status'] = 'overfitting'
        result['suggestions'] = [
            "增加正则化（L2、Dropout）",
            "使用数据增强",
            "减小模型容量",
            "提前停止训练",
            "收集更多训练数据"
        ]
    elif loss_gap > 0.2 and val_trend > 0:
        result['status'] = 'overfitting_mild'
        result['suggestions'] = [
            "添加 Dropout",
            "增加 L2 正则化",
            "考虑提前停止"
        ]
    elif final_train_loss > 1.0 and train_trend > -0.001:
        result['status'] = 'underfitting'
        result['suggestions'] = [
            "增加模型容量（更多层/神经元）",
            "训练更多 epoch",
            "降低正则化强度",
            "使用更强的优化器（如 Adam）",
            "检查学习率是否过小"
        ]
    elif abs(train_trend) < 0.001 and final_train_loss > 0.5:
        result['status'] = 'stagnant'
        result['suggestions'] = [
            "降低学习率（使用学习率调度）",
            "检查是否陷入局部最优",
            "尝试 Warm Restart",
            "增加 batch size"
        ]
    else:
        result['status'] = 'good'
        if train_trend < -0.001:
            result['suggestions'] = ["继续训练，可能还能进一步优化"]
        else:
            result['suggestions'] = ["已接近收敛，可以考虑停止训练"]

    if verbose:
        print("\n" + "=" * 60)
        print("  训练诊断报告")
        print("=" * 60)
        print(f"\n数据概览:")
        print(f"  训练集 Loss: {final_train_loss:.4f}")
        print(f"  验证集 Loss: {final_val_loss:.4f}")
        print(f"  Loss 差距: {loss_gap:.4f}")
        print(f"\n趋势分析（最近 {recent_epochs} 个 epoch）:")
        print(f"  训练 Loss 趋势: {train_trend:+.5f} {'下降' if train_trend < 0 else '上升'}")
        print(f"  验证 Loss 趋势: {val_trend:+.5f} {'下降' if val_trend < 0 else '上升'}")
        print(f"\n诊断结果: {result['status']}")
        print("建议:")
        for i, suggestion in enumerate(result['suggestions'], 1):
            print(f"  {i}. {suggestion}")
        print("=" * 60)

    return result


# =============================================================================
#                           训练监控器（核心类）
# =============================================================================

class TrainingMonitor:
    """
    训练监控器 - 集成所有训练技巧的一站式解决方案

    功能:
    -----
    - 学习率调度（可选多种策略）
    - 梯度裁剪（可开关）
    - 早停（可配置）
    - 训练诊断（定期自动诊断）
    - 历史记录（loss、lr、梯度范数）
    - 可视化（学习曲线、学习率曲线）

    参数:
    -----
    lr_scheduler : LearningRateScheduler, optional
        学习率调度器，如 CosineAnnealingScheduler
    gradient_clip_norm : float, optional
        梯度裁剪阈值，None 表示不裁剪
    early_stopping_patience : int, optional
        早停的耐心值，None 表示不使用早停
    early_stopping_min_delta : float
        早停的最小改进量，默认 0.0
    diagnose_every : int
        每隔多少个 epoch 进行一次诊断，默认 10
    verbose : bool
        是否打印详细信息，默认 True

    使用示例:
    --------
    # 创建监控器
    monitor = TrainingMonitor(
        lr_scheduler=CosineAnnealingScheduler(0.01, 1e-6, 100),
        gradient_clip_norm=1.0,
        early_stopping_patience=10,
        diagnose_every=10
    )

    # 训练循环
    for epoch in range(max_epochs):
        monitor.on_epoch_start(epoch)

        for batch in batches:
            gradients = compute_gradients(batch)
            gradients = monitor.on_batch_end(gradients)
            update_parameters(gradients, lr=monitor.get_current_lr())

        train_loss = compute_train_loss()
        val_loss = compute_val_loss()

        if monitor.on_epoch_end(train_loss, val_loss, model):
            break

    # 可视化和总结
    monitor.plot_history()
    monitor.summary()
    """

    def __init__(self,
                 lr_scheduler: Optional[LearningRateScheduler] = None,
                 gradient_clip_norm: Optional[float] = None,
                 early_stopping_patience: Optional[int] = None,
                 early_stopping_min_delta: float = 0.0,
                 diagnose_every: int = 10,
                 verbose: bool = True):

        self.lr_scheduler = lr_scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.diagnose_every = diagnose_every
        self.verbose = verbose

        # 早停
        if early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode='min',
                verbose=False  # 由 TrainingMonitor 统一管理输出
            )
        else:
            self.early_stopping = None

        # 历史记录
        self.train_loss_history = []
        self.val_loss_history = []
        self.lr_history = []
        self.gradient_norm_history = []

        # 状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model = None
        self.stopped_early = False
        self.stop_epoch = None

    def on_epoch_start(self, epoch: int) -> None:
        """
        epoch 开始时调用

        参数:
        -----
        epoch : int
            当前 epoch 编号（从 0 开始）
        """
        self.current_epoch = epoch
        if self.verbose:
            lr = self.get_current_lr()
            print(f"\nEpoch {epoch + 1} | 学习率: {lr:.6f}")

    def on_batch_end(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        批次结束时调用，执行梯度裁剪

        参数:
        -----
        gradients : List[np.ndarray]
            当前批次的梯度

        返回:
        -----
        List[np.ndarray]
            处理后的梯度（可能被裁剪）
        """
        if self.gradient_clip_norm is not None:
            gradients, grad_norm = clip_gradient_norm(gradients, self.gradient_clip_norm)
            self.gradient_norm_history.append(grad_norm)
            return gradients
        return gradients

    def on_epoch_end(self, train_loss: float, val_loss: float,
                     model=None) -> bool:
        """
        epoch 结束时调用

        参数:
        -----
        train_loss : float
            训练 loss
        val_loss : float
            验证 loss
        model : object, optional
            模型对象（用于保存最佳模型）

        返回:
        -----
        bool
            是否应该停止训练（早停触发）
        """
        # 记录历史
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)

        # 记录学习率
        current_lr = self.get_current_lr()
        self.lr_history.append(current_lr)

        # 更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.update()

        # 保存最佳模型
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if model is not None:
                self.best_model = copy.deepcopy(model)

        # 打印进度
        if self.verbose:
            print(f"  训练 Loss: {train_loss:.4f} | 验证 Loss: {val_loss:.4f}", end="")
            if self.early_stopping is not None:
                if val_loss < self.early_stopping.best_score if self.early_stopping.best_score else True:
                    print(" [最佳]", end="")
                else:
                    print(f" [无改进 {self.early_stopping.counter + 1}/{self.early_stopping.patience}]", end="")
            print()

        # 定期诊断
        if self.diagnose_every > 0 and (self.current_epoch + 1) % self.diagnose_every == 0:
            if len(self.train_loss_history) >= 4:
                diagnose_training(
                    self.train_loss_history,
                    self.val_loss_history,
                    verbose=self.verbose
                )

        # 早停检查
        if self.early_stopping is not None:
            if self.early_stopping.step(val_loss, model):
                self.stopped_early = True
                self.stop_epoch = self.current_epoch + 1
                if self.verbose:
                    print(f"\n早停触发！在 Epoch {self.stop_epoch} 停止")
                    print(f"最佳验证 Loss: {self.early_stopping.best_score:.4f}")
                return True

        return False

    def get_current_lr(self) -> float:
        """获取当前学习率"""
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_lr()
        return 0.01  # 默认学习率

    def diagnose(self) -> Dict[str, Any]:
        """执行训练诊断"""
        if len(self.train_loss_history) < 2:
            print("训练历史太短，无法诊断")
            return {}
        return diagnose_training(
            self.train_loss_history,
            self.val_loss_history,
            verbose=self.verbose
        )

    def load_best_model(self):
        """加载最佳模型"""
        if self.best_model is None:
            if self.early_stopping is not None and self.early_stopping.best_model is not None:
                return self.early_stopping.load_best_model()
            raise ValueError("没有保存的最佳模型")
        return self.best_model

    def plot_history(self, figsize: Tuple[int, int] = (14, 5),
                     save_path: Optional[str] = None) -> None:
        """
        绘制训练历史

        参数:
        -----
        figsize : Tuple[int, int]
            图像大小，默认 (14, 5)
        save_path : str, optional
            保存路径，None 表示不保存
        """
        n_plots = 2 if self.lr_scheduler is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        if n_plots == 1:
            axes = [axes]

        # 绘制 Loss 曲线
        ax = axes[0]
        epochs = range(1, len(self.train_loss_history) + 1)
        ax.plot(epochs, self.train_loss_history, 'b-', label='训练 Loss', linewidth=2)
        ax.plot(epochs, self.val_loss_history, 'r-', label='验证 Loss', linewidth=2)

        # 标记最佳点
        best_epoch = np.argmin(self.val_loss_history) + 1
        best_loss = min(self.val_loss_history)
        ax.scatter([best_epoch], [best_loss], c='green', s=100, zorder=5, label=f'最佳 (Epoch {best_epoch})')

        # 标记早停点
        if self.stopped_early:
            ax.axvline(x=self.stop_epoch, color='red', linestyle='--',
                      label=f'早停 (Epoch {self.stop_epoch})')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('训练曲线', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 绘制学习率曲线
        if n_plots > 1:
            ax = axes[1]
            ax.plot(epochs, self.lr_history, 'g-', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('学习率', fontsize=12)
            ax.set_title('学习率变化', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")

        plt.show()

    def summary(self) -> None:
        """打印训练总结"""
        print("\n" + "=" * 60)
        print("  训练总结")
        print("=" * 60)

        total_epochs = len(self.train_loss_history)
        print(f"\n训练 Epochs: {total_epochs}")

        if self.stopped_early:
            print(f"早停: 是 (Epoch {self.stop_epoch})")
        else:
            print("早停: 否")

        print(f"\n最终训练 Loss: {self.train_loss_history[-1]:.4f}")
        print(f"最终验证 Loss: {self.val_loss_history[-1]:.4f}")
        print(f"最佳验证 Loss: {self.best_val_loss:.4f} (Epoch {np.argmin(self.val_loss_history) + 1})")

        if self.lr_scheduler is not None:
            print(f"\n学习率范围: {min(self.lr_history):.6f} - {max(self.lr_history):.6f}")

        if self.gradient_norm_history:
            print(f"\n梯度范数范围: {min(self.gradient_norm_history):.2f} - {max(self.gradient_norm_history):.2f}")
            if max(self.gradient_norm_history) > self.gradient_clip_norm:
                clip_count = sum(1 for g in self.gradient_norm_history if g > self.gradient_clip_norm)
                print(f"梯度裁剪次数: {clip_count}")

        print("\n" + "=" * 60)

    def reset(self) -> None:
        """重置监控器状态"""
        self.train_loss_history = []
        self.val_loss_history = []
        self.lr_history = []
        self.gradient_norm_history = []
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model = None
        self.stopped_early = False
        self.stop_epoch = None

        if self.lr_scheduler is not None:
            self.lr_scheduler.reset()
        if self.early_stopping is not None:
            self.early_stopping.reset()


# =============================================================================
#                           工厂函数
# =============================================================================

def create_scheduler(scheduler_type: str, initial_lr: float,
                     **kwargs) -> LearningRateScheduler:
    """
    创建学习率调度器的工厂函数

    参数:
    -----
    scheduler_type : str
        调度器类型: 'step', 'exponential', 'cosine', 'warm_restart'
    initial_lr : float
        初始学习率
    **kwargs
        调度器特定参数

    返回:
    -----
    LearningRateScheduler
        创建的调度器实例

    示例:
    -----
    scheduler = create_scheduler('cosine', 0.01, min_lr=1e-6, T_max=100)
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == 'step':
        return StepDecayScheduler(
            initial_lr=initial_lr,
            decay_rate=kwargs.get('decay_rate', 0.1),
            decay_steps=kwargs.get('decay_steps', 30)
        )
    elif scheduler_type == 'exponential':
        return ExponentialDecayScheduler(
            initial_lr=initial_lr,
            decay_rate=kwargs.get('decay_rate', 0.05)
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingScheduler(
            initial_lr=initial_lr,
            min_lr=kwargs.get('min_lr', 0.0),
            T_max=kwargs.get('T_max', 100)
        )
    elif scheduler_type == 'warm_restart':
        return WarmRestartScheduler(
            initial_lr=initial_lr,
            min_lr=kwargs.get('min_lr', 0.0),
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2.0)
        )
    else:
        raise ValueError(f"未知的调度器类型: {scheduler_type}")


# =============================================================================
#                           模块测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Training Utils 模块测试")
    print("=" * 60)

    # 测试学习率调度器
    print("\n1. 测试学习率调度器")
    print("-" * 40)

    scheduler = CosineAnnealingScheduler(initial_lr=0.1, min_lr=0.001, T_max=50)
    lrs = []
    for _ in range(50):
        lrs.append(scheduler.update())
    print(f"   Cosine Annealing: {lrs[0]:.4f} -> {lrs[-1]:.4f}")

    # 测试梯度裁剪
    print("\n2. 测试梯度裁剪")
    print("-" * 40)

    gradients = [np.random.randn(10, 20) * 100]
    clipped, original_norm = clip_gradient_norm(gradients, max_norm=1.0)
    clipped_norm = np.sqrt(sum(np.sum(g**2) for g in clipped))
    print(f"   原始范数: {original_norm:.2f} -> 裁剪后: {clipped_norm:.2f}")

    # 测试早停
    print("\n3. 测试早停")
    print("-" * 40)

    early_stopping = EarlyStopping(patience=3, verbose=False)
    losses = [1.0, 0.8, 0.6, 0.5, 0.55, 0.6, 0.65]
    for i, loss in enumerate(losses):
        if early_stopping.step(loss):
            print(f"   早停在 Epoch {i + 1} 触发")
            break

    # 测试工厂函数
    print("\n4. 测试工厂函数")
    print("-" * 40)

    for stype in ['step', 'exponential', 'cosine', 'warm_restart']:
        s = create_scheduler(stype, 0.01)
        print(f"   {stype}: {type(s).__name__}")

    print("\n" + "=" * 60)
    print("  所有测试通过！")
    print("=" * 60)
