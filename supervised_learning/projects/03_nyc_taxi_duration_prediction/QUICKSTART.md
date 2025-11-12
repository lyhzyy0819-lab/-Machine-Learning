# NYC Taxi 项目快速上手指南

## 🚀 5分钟快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 快速测试（使用模拟数据）
```bash
python main.py --sample --sample-size 2000 --quick
```

### 3. 完整运行
```bash
python main.py --sample
```

## 📁 主要文件说明

- **main.py**: 主程序入口，运行完整ML流程
- **config.py**: 所有配置参数
- **src/**: 核心模块（8个模块）
- **README.md**: 完整项目文档
- **nyc_taxi_analysis.ipynb**: Jupyter交互式演示

## 🎯 快速命令

```bash
# 只训练线性模型（最快）
python main.py --sample --quick

# 跳过XGBoost（加速）
python main.py --sample --no-xgboost

# 使用Jupyter
jupyter notebook nyc_taxi_analysis.ipynb
```

## 📊 预期输出

运行成功后会生成：
- `models/`: 训练好的模型文件
- `figures/`: 可视化图表（15+张）
- `logs/`: 训练日志
- 控制台输出模型对比表格

## ❓ 常见问题

**Q: 运行时间多长？**
A: 样本模式2000条数据约2-3分钟，完整数据约30-60分钟

**Q: 需要GPU吗？**
A: 不需要，CPU即可

**Q: 数据从哪里来？**
A: 代码会自动生成模拟数据，也可下载Kaggle真实数据

详细说明请查看 README.md
