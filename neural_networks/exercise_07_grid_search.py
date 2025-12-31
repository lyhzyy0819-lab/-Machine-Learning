"""
练习4：超参数网格搜索

本练习实现系统化的超参数搜索，找出最佳配置组合

搜索空间:
- 学习率: [0.1, 0.01, 0.001]
- 批量大小: [32, 64, 128]
- 学习率调度: ['step', 'cosine']
- L2正则化: [0.0, 0.01, 0.001]

使用 sklearn.datasets.load_digits 数据集

运行方式:
    python exercise_07_grid_search.py

作者: Machine Learning 学习项目
日期: 2024年12月
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import product
import time
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

# 导入自定义训练工具
from training_utils import (
    TrainingMonitor,
    create_scheduler,
    CosineAnnealingScheduler,
    StepDecayScheduler
)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)


# =============================================================================
#                           简单神经网络
# =============================================================================

class SimpleNeuralNetwork:
    """
    简单的两层神经网络

    与 exercise_06 相同的实现，用于网格搜索
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 l2_reg: float = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l2_reg = l2_reg

        # He 初始化
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ self.W2 + self.b2
        exp_Z2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        A2 = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)
        self.cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算损失"""
        batch_size = y_true.shape[0]
        log_probs = -np.log(y_pred[range(batch_size), y_true] + 1e-8)
        ce_loss = np.mean(log_probs)
        l2_loss = 0.5 * self.l2_reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return ce_loss + l2_loss

    def backward(self, y_true: np.ndarray) -> list:
        """反向传播"""
        batch_size = y_true.shape[0]
        X, Z1, A1, Z2, A2 = (self.cache['X'], self.cache['Z1'],
                             self.cache['A1'], self.cache['Z2'], self.cache['A2'])

        dZ2 = A2.copy()
        dZ2[range(batch_size), y_true] -= 1
        dZ2 /= batch_size

        dW2 = A1.T @ dZ2 + self.l2_reg * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)

        dW1 = X.T @ dZ1 + self.l2_reg * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return [dW1, db1, dW2, db2]

    def update_parameters(self, gradients: list, learning_rate: float) -> None:
        """更新参数"""
        dW1, db1, dW2, db2 = gradients
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return np.mean(self.predict(X) == y)

    def save(self, filepath: str) -> None:
        """
        保存模型参数到文件

        参数:
        -----
        filepath : str
            保存路径（不需要 .npz 后缀，会自动添加）

        使用示例:
        --------
        model.save('best_model')  # 会生成 best_model.npz
        """
        np.savez(filepath,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 config={'input_size': self.input_size,
                        'hidden_size': self.hidden_size,
                        'output_size': self.output_size,
                        'l2_reg': self.l2_reg})
        print(f"模型已保存到: {filepath}.npz")

    @classmethod
    def load(cls, filepath: str) -> 'SimpleNeuralNetwork':
        """
        从文件加载模型

        参数:
        -----
        filepath : str
            模型文件路径（需要 .npz 后缀）

        返回:
        -----
        SimpleNeuralNetwork
            加载的模型实例

        使用示例:
        --------
        model = SimpleNeuralNetwork.load('best_model.npz')
        predictions = model.predict(X_test)
        """
        data = np.load(filepath, allow_pickle=True)
        config = data['config'].item()
        model = cls(**config)
        model.W1 = data['W1']
        model.b1 = data['b1']
        model.W2 = data['W2']
        model.b2 = data['b2']
        print(f"模型已从 {filepath} 加载")
        return model


# =============================================================================
#                           网格搜索
# =============================================================================

@dataclass
class SearchResult:
    """单次实验结果"""
    params: Dict[str, Any]
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    training_time: float
    epochs_run: int


class GridSearchCV:
    """
    超参数网格搜索

    功能:
    -----
    - 系统化搜索超参数空间
    - 记录每个配置的性能
    - 找出最佳配置
    - 可视化搜索结果

    参数:
    -----
    param_grid : Dict[str, List]
        参数搜索空间
    n_epochs : int
        每个配置训练的 epoch 数
    verbose : bool
        是否打印详细信息

    使用示例:
    --------
    param_grid = {
        'learning_rate': [0.1, 0.01, 0.001],
        'batch_size': [32, 64],
        'lr_scheduler': ['step', 'cosine'],
        'l2_reg': [0.0, 0.01]
    }

    searcher = GridSearchCV(param_grid, n_epochs=10)
    results = searcher.fit(X_train, y_train, X_val, y_val)
    searcher.summary()
    searcher.plot_results()
    """

    def __init__(self, param_grid: Dict[str, List],
                 n_epochs: int = 10,
                 verbose: bool = True):
        self.param_grid = param_grid
        self.n_epochs = n_epochs
        self.verbose = verbose

        self.results: List[SearchResult] = []
        self.best_result: SearchResult = None
        self.best_params: Dict[str, Any] = None

    def _generate_configs(self) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        configs = []
        for combo in product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)

        return configs

    def _train_single_config(self, config: Dict[str, Any],
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> SearchResult:
        """使用单个配置进行训练"""
        # 重置随机种子以保证可重复性
        np.random.seed(42)

        # 创建模型
        model = SimpleNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_size=64,
            output_size=10,
            l2_reg=config['l2_reg']
        )

        # 创建学习率调度器
        if config['lr_scheduler'] == 'step':
            scheduler = StepDecayScheduler(
                initial_lr=config['learning_rate'],
                decay_rate=0.5,
                decay_steps=5
            )
        else:  # cosine
            scheduler = CosineAnnealingScheduler(
                initial_lr=config['learning_rate'],
                min_lr=config['learning_rate'] * 0.01,
                T_max=self.n_epochs
            )

        # 创建监控器（不使用早停，固定 epochs）
        monitor = TrainingMonitor(
            lr_scheduler=scheduler,
            gradient_clip_norm=1.0,
            early_stopping_patience=None,  # 不使用早停
            diagnose_every=0,  # 不诊断
            verbose=False
        )

        # 训练
        start_time = time.time()
        batch_size = config['batch_size']

        for epoch in range(self.n_epochs):
            monitor.on_epoch_start(epoch)

            # 训练一个 epoch
            train_losses = []
            indices = np.random.permutation(len(X_train))

            for start_idx in range(0, len(X_train), batch_size):
                end_idx = min(start_idx + batch_size, len(X_train))
                batch_indices = indices[start_idx:end_idx]
                X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

                y_pred = model.forward(X_batch)
                loss = model.compute_loss(y_pred, y_batch)
                train_losses.append(loss)

                gradients = model.backward(y_batch)
                gradients = monitor.on_batch_end(gradients)
                model.update_parameters(gradients, monitor.get_current_lr())

            train_loss = np.mean(train_losses)

            # 验证
            val_pred = model.forward(X_val)
            val_loss = model.compute_loss(val_pred, y_val)

            monitor.on_epoch_end(train_loss, val_loss)

        training_time = time.time() - start_time

        # 最终评估
        final_train_loss = train_loss
        final_val_loss = val_loss
        train_acc = model.accuracy(X_train, y_train)
        val_acc = model.accuracy(X_val, y_val)

        return SearchResult(
            params=config.copy(),
            train_loss=final_train_loss,
            val_loss=final_val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            training_time=training_time,
            epochs_run=self.n_epochs
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> List[SearchResult]:
        """
        执行网格搜索

        参数:
        -----
        X_train, y_train : np.ndarray
            训练数据
        X_val, y_val : np.ndarray
            验证数据

        返回:
        -----
        List[SearchResult]
            所有配置的结果
        """
        configs = self._generate_configs()
        n_configs = len(configs)

        if self.verbose:
            print("=" * 70)
            print("  超参数网格搜索")
            print("=" * 70)
            print(f"\n搜索空间:")
            for key, values in self.param_grid.items():
                print(f"  {key}: {values}")
            print(f"\n总配置数: {n_configs}")
            print(f"每个配置训练 {self.n_epochs} epochs")
            print("-" * 70)

        self.results = []
        best_val_acc = 0

        for i, config in enumerate(configs):
            if self.verbose:
                print(f"\n[{i+1}/{n_configs}] 测试配置:")
                for k, v in config.items():
                    print(f"    {k}: {v}")

            result = self._train_single_config(config, X_train, y_train, X_val, y_val)
            self.results.append(result)

            if self.verbose:
                print(f"  结果: val_acc={result.val_acc:.2%}, "
                      f"val_loss={result.val_loss:.4f}, "
                      f"time={result.training_time:.1f}s")

            # 更新最佳结果
            if result.val_acc > best_val_acc:
                best_val_acc = result.val_acc
                self.best_result = result
                self.best_params = config.copy()

                if self.verbose:
                    print(f"  ** 新的最佳配置! **")

        if self.verbose:
            print("\n" + "-" * 70)
            print("搜索完成!")

        return self.results

    def summary(self) -> None:
        """打印搜索结果总结"""
        if not self.results:
            print("没有搜索结果，请先运行 fit()")
            return

        print("\n" + "=" * 70)
        print("  网格搜索结果总结")
        print("=" * 70)

        # 按验证准确率排序
        sorted_results = sorted(self.results, key=lambda x: x.val_acc, reverse=True)

        print("\n所有配置结果（按验证准确率排序）:")
        print("-" * 70)
        print(f"{'排名':<4} {'学习率':<10} {'批量':<6} {'调度器':<8} {'L2正则':<8} {'验证Acc':<10} {'验证Loss':<10}")
        print("-" * 70)

        for i, result in enumerate(sorted_results[:10]):  # 显示前10个
            params = result.params
            print(f"{i+1:<4} {params['learning_rate']:<10.4f} "
                  f"{params['batch_size']:<6} "
                  f"{params['lr_scheduler']:<8} "
                  f"{params['l2_reg']:<8.4f} "
                  f"{result.val_acc:<10.2%} "
                  f"{result.val_loss:<10.4f}")

        print("-" * 70)

        # 最佳配置
        print("\n最佳配置:")
        print("-" * 40)
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print("-" * 40)
        print(f"  验证准确率: {self.best_result.val_acc:.2%}")
        print(f"  验证 Loss: {self.best_result.val_loss:.4f}")
        print(f"  训练准确率: {self.best_result.train_acc:.2%}")

        print("\n" + "=" * 70)

    def plot_results(self, save_path: str = None) -> None:
        """可视化搜索结果"""
        if not self.results:
            print("没有搜索结果，请先运行 fit()")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 学习率 vs 验证准确率
        ax = axes[0, 0]
        for lr in self.param_grid['learning_rate']:
            accs = [r.val_acc for r in self.results if r.params['learning_rate'] == lr]
            ax.scatter([lr] * len(accs), accs, alpha=0.6, s=50)
        ax.set_xlabel('学习率', fontsize=12)
        ax.set_ylabel('验证准确率', fontsize=12)
        ax.set_title('学习率 vs 验证准确率', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # 2. 批量大小 vs 验证准确率
        ax = axes[0, 1]
        batch_sizes = self.param_grid['batch_size']
        for bs in batch_sizes:
            accs = [r.val_acc for r in self.results if r.params['batch_size'] == bs]
            ax.scatter([bs] * len(accs), accs, alpha=0.6, s=50)
        ax.set_xlabel('批量大小', fontsize=12)
        ax.set_ylabel('验证准确率', fontsize=12)
        ax.set_title('批量大小 vs 验证准确率', fontsize=14, fontweight='bold')
        ax.set_xticks(batch_sizes)
        ax.grid(True, alpha=0.3)

        # 3. 调度器对比
        ax = axes[1, 0]
        schedulers = self.param_grid['lr_scheduler']
        scheduler_accs = {s: [r.val_acc for r in self.results if r.params['lr_scheduler'] == s]
                          for s in schedulers}

        positions = range(len(schedulers))
        for i, (sched, accs) in enumerate(scheduler_accs.items()):
            bp = ax.boxplot([accs], positions=[i], widths=0.5)
            ax.scatter([i] * len(accs), accs, alpha=0.6, s=30)

        ax.set_xticks(positions)
        ax.set_xticklabels(schedulers)
        ax.set_ylabel('验证准确率', fontsize=12)
        ax.set_title('学习率调度器对比', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. L2正则化 vs 验证准确率
        ax = axes[1, 1]
        l2_regs = self.param_grid['l2_reg']
        for reg in l2_regs:
            accs = [r.val_acc for r in self.results if r.params['l2_reg'] == reg]
            ax.scatter([reg] * len(accs), accs, alpha=0.6, s=50,
                      label=f'L2={reg}')

        ax.set_xlabel('L2 正则化系数', fontsize=12)
        ax.set_ylabel('验证准确率', fontsize=12)
        ax.set_title('L2正则化 vs 验证准确率', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n图像已保存到: {save_path}")

        plt.show()

    def plot_heatmap(self, x_param: str, y_param: str,
                     save_path: str = None) -> None:
        """
        绘制两个参数的热力图

        参数:
        -----
        x_param : str
            x轴参数名
        y_param : str
            y轴参数名
        """
        if not self.results:
            print("没有搜索结果，请先运行 fit()")
            return

        x_values = sorted(set(self.param_grid[x_param]))
        y_values = sorted(set(self.param_grid[y_param]))

        # 创建准确率矩阵（取平均）
        acc_matrix = np.zeros((len(y_values), len(x_values)))

        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                accs = [r.val_acc for r in self.results
                       if r.params[x_param] == x_val and r.params[y_param] == y_val]
                acc_matrix[i, j] = np.mean(accs) if accs else 0

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(acc_matrix, cmap='YlGn', aspect='auto')

        # 添加数值标签
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                text = ax.text(j, i, f'{acc_matrix[i, j]:.1%}',
                              ha='center', va='center', fontsize=10)

        ax.set_xticks(range(len(x_values)))
        ax.set_yticks(range(len(y_values)))
        ax.set_xticklabels([str(v) for v in x_values])
        ax.set_yticklabels([str(v) for v in y_values])
        ax.set_xlabel(x_param, fontsize=12)
        ax.set_ylabel(y_param, fontsize=12)
        ax.set_title(f'{x_param} vs {y_param} 验证准确率热力图',
                    fontsize=14, fontweight='bold')

        plt.colorbar(im, label='验证准确率')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n热力图已保存到: {save_path}")

        plt.show()

    def save_best_params(self, filepath: str) -> None:
        """
        保存最佳参数到 JSON 文件

        参数:
        -----
        filepath : str
            保存路径（建议使用 .json 后缀）

        使用示例:
        --------
        searcher.save_best_params('best_params.json')
        """
        if not self.best_params:
            print("没有最佳参数，请先运行 fit()")
            return

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': self.best_params,
                'best_val_acc': float(self.best_result.val_acc),
                'best_val_loss': float(self.best_result.val_loss),
                'best_train_acc': float(self.best_result.train_acc),
                'search_space': self.param_grid,
                'n_epochs': self.n_epochs
            }, f, indent=2, ensure_ascii=False)
        print(f"最佳参数已保存到: {filepath}")

    def save_all_results(self, filepath: str) -> None:
        """
        保存所有搜索结果到 JSON 文件

        参数:
        -----
        filepath : str
            保存路径（建议使用 .json 后缀）

        使用示例:
        --------
        searcher.save_all_results('grid_search_results.json')
        """
        if not self.results:
            print("没有搜索结果，请先运行 fit()")
            return

        all_results = []
        for result in self.results:
            all_results.append({
                'params': result.params,
                'train_loss': float(result.train_loss),
                'val_loss': float(result.val_loss),
                'train_acc': float(result.train_acc),
                'val_acc': float(result.val_acc),
                'training_time': float(result.training_time),
                'epochs_run': result.epochs_run
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'search_space': self.param_grid,
                'n_epochs': self.n_epochs,
                'results': all_results,
                'best_params': self.best_params
            }, f, indent=2, ensure_ascii=False)
        print(f"所有搜索结果已保存到: {filepath}")


# =============================================================================
#                           数据准备
# =============================================================================

def load_and_prepare_data():
    """加载并准备数据"""
    print("=" * 60)
    print("  数据准备")
    print("=" * 60)

    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"\n数据集: sklearn.datasets.load_digits")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  类别数: {len(np.unique(y))}")

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)}")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)}")

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
#                           主程序
# =============================================================================

def main():
    """主程序"""
    print("\n" + "=" * 70)
    print("  练习4：超参数网格搜索")
    print("=" * 70)

    # 1. 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()

    # 2. 定义搜索空间
    param_grid = {
        'learning_rate': [0.1, 0.01, 0.001],
        'batch_size': [32, 64, 128],
        'lr_scheduler': ['step', 'cosine'],
        'l2_reg': [0.0, 0.01, 0.001]
    }

    # 3. 执行网格搜索
    searcher = GridSearchCV(
        param_grid=param_grid,
        n_epochs=10,  # 每个配置训练10个epoch
        verbose=True
    )

    start_time = time.time()
    results = searcher.fit(X_train, y_train, X_val, y_val)
    total_time = time.time() - start_time

    print(f"\n总搜索时间: {total_time:.1f} 秒")

    # 4. 打印总结
    searcher.summary()

    # 5. 可视化
    searcher.plot_results(save_path='exercise_07_grid_search_results.png')

    # 6. 绘制热力图
    searcher.plot_heatmap('learning_rate', 'batch_size',
                          save_path='exercise_07_heatmap_lr_bs.png')

    # 7. 使用最佳配置在测试集上评估
    print("\n" + "=" * 70)
    print("  使用最佳配置在测试集上评估")
    print("=" * 70)

    best_params = searcher.best_params
    print(f"\n最佳配置: {best_params}")

    # 用最佳配置训练完整模型
    np.random.seed(42)
    best_model = SimpleNeuralNetwork(
        input_size=64,
        hidden_size=64,
        output_size=10,
        l2_reg=best_params['l2_reg']
    )

    if best_params['lr_scheduler'] == 'step':
        scheduler = StepDecayScheduler(
            initial_lr=best_params['learning_rate'],
            decay_rate=0.5,
            decay_steps=10
        )
    else:
        scheduler = CosineAnnealingScheduler(
            initial_lr=best_params['learning_rate'],
            min_lr=best_params['learning_rate'] * 0.01,
            T_max=50
        )

    # 训练更多 epochs
    for epoch in range(50):
        indices = np.random.permutation(len(X_train))
        for start_idx in range(0, len(X_train), best_params['batch_size']):
            end_idx = min(start_idx + best_params['batch_size'], len(X_train))
            batch_indices = indices[start_idx:end_idx]
            X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

            y_pred = best_model.forward(X_batch)
            gradients = best_model.backward(y_batch)
            best_model.update_parameters(gradients, scheduler.get_lr())

        scheduler.update()

    test_acc = best_model.accuracy(X_test, y_test)
    print(f"\n测试集准确率: {test_acc:.2%}")

    # 8. 保存最佳参数和模型
    print("\n" + "=" * 70)
    print("  保存搜索结果")
    print("=" * 70)

    # 保存最佳参数
    searcher.save_best_params('exercise_07_best_params.json')

    # 保存所有搜索结果（可用于后续分析）
    searcher.save_all_results('exercise_07_all_results.json')

    # 保存最佳模型
    best_model.save('exercise_07_best_model')

    print("\n保存的文件:")
    print("  - exercise_07_best_params.json  (最佳超参数)")
    print("  - exercise_07_all_results.json  (所有搜索结果)")
    print("  - exercise_07_best_model.npz    (最佳模型权重)")

    # 演示如何加载和使用保存的模型
    print("\n" + "-" * 40)
    print("加载模型示例:")
    print("-" * 40)
    print(">>> loaded_model = SimpleNeuralNetwork.load('exercise_07_best_model.npz')")
    print(">>> predictions = loaded_model.predict(X_test)")
    print(">>> accuracy = loaded_model.accuracy(X_test, y_test)")

    print("\n" + "=" * 70)
    print("  练习4 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
