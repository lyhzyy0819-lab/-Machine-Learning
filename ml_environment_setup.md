# 机器学习环境配置指南

本文档指导如何使用Conda创建Python 3.8环境并安装机器学习相关包。

---

## 第一步：创建Conda环境

### 创建新环境
```bash
# 创建名为 ml_env 的Python 3.8环境
conda create -n ml_env python=3.8

# 激活环境
conda activate ml_env

# 验证Python版本
python --version
```

---

## 第二步：安装基础数据科学包

### NumPy、Pandas、Matplotlib
```bash
# 基础数值计算和数据处理
conda install numpy pandas

# 数据可视化
conda install matplotlib seaborn

# 或者使用pip一次性安装
pip install numpy pandas matplotlib seaborn
```

### Jupyter Notebook
```bash
# 安装Jupyter
conda install jupyter notebook

# 或者安装JupyterLab（推荐）
conda install jupyterlab

# 启动Jupyter
jupyter lab
```

---

## 第三步：安装机器学习框架

### Scikit-learn（传统机器学习）
```bash
# 包含常用的机器学习算法
conda install scikit-learn

# 或使用pip
pip install scikit-learn
```

### XGBoost（梯度提升）
```bash
# 高性能梯度提升库
pip install xgboost
```

### LightGBM
```bash
pip install lightgbm
```

---

## 第四步：安装深度学习框架

### PyTorch（推荐）

#### CPU版本
```bash
# 适合学习和小规模实验
pip install torch torchvision torchaudio
```

#### GPU版本（如果有NVIDIA显卡）
```bash
# 先查看CUDA版本
nvidia-smi

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### TensorFlow（可选）
```bash
# CPU版本
pip install tensorflow

# GPU版本（需要CUDA和cuDNN）
pip install tensorflow[and-cuda]
```

---

## 第五步：安装其他常用工具

### 计算机视觉
```bash
# OpenCV
pip install opencv-python

# Pillow（图像处理）
pip install Pillow

# Albumentations（数据增强）
pip install albumentations
```

### 自然语言处理
```bash
# NLTK
pip install nltk

# spaCy
pip install spacy

# Hugging Face Transformers
pip install transformers

# Tokenizers
pip install tokenizers
```

### 实验管理和可视化
```bash
# TensorBoard（PyTorch/TensorFlow可视化）
pip install tensorboard

# Weights & Biases（实验跟踪）
pip install wandb

# MLflow（模型管理）
pip install mlflow
```

### 模型部署
```bash
# Gradio（快速构建Web界面）
pip install gradio

# Streamlit
pip install streamlit

# Flask
pip install flask

# FastAPI
pip install fastapi uvicorn
```

### Kaggle竞赛工具
```bash
# Kaggle API
pip install kaggle

# 配置API（需要先下载kaggle.json）
# mkdir ~/.kaggle
# mv kaggle.json ~/.kaggle/
# chmod 600 ~/.kaggle/kaggle.json
```

---

## 第六步：一键安装脚本

如果想一次性安装所有常用包，可以创建 `requirements.txt`：

```txt
# 基础数据科学
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0

# 机器学习
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0

# 深度学习（PyTorch CPU版）
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# 计算机视觉
opencv-python>=4.5.0
Pillow>=9.0.0
albumentations>=1.3.0

# 自然语言处理
transformers>=4.20.0
tokenizers>=0.13.0

# 工具
jupyter>=1.0.0
jupyterlab>=3.4.0
tensorboard>=2.10.0
tqdm>=4.64.0

# 部署
gradio>=3.40.0
streamlit>=1.25.0

# 其他
requests>=2.28.0
```

然后运行：
```bash
pip install -r requirements.txt
```

---

## 第七步：验证安装

创建一个测试脚本 `test_installation.py`：

```python
#!/usr/bin/env python3
"""验证机器学习环境是否安装成功"""

def test_imports():
    print("Testing package imports...")

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")

    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas: {e}")

    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")

    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn: {e}")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")

    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")

    print("\n✓ Environment setup complete!")

if __name__ == "__main__":
    test_imports()
```

运行验证：
```bash
python test_installation.py
```

---

## 常用命令

### 环境管理
```bash
# 查看所有环境
conda env list

# 激活环境
conda activate ml_env

# 退出环境
conda deactivate

# 删除环境
conda remove -n ml_env --all

# 导出环境
conda env export > environment.yml

# 从配置文件创建环境
conda env create -f environment.yml
```

### 包管理
```bash
# 查看已安装的包
pip list
conda list

# 更新包
pip install --upgrade package_name
conda update package_name

# 卸载包
pip uninstall package_name
conda remove package_name
```

---

## 推荐的最小安装

如果想快速开始，只需安装核心包：

```bash
# 创建环境
conda create -n ml_env python=3.8
conda activate ml_env

# 核心包
pip install numpy pandas matplotlib scikit-learn jupyter

# PyTorch
pip install torch torchvision

# 可视化界面
pip install gradio

# 验证
python -c "import torch; print(torch.__version__)"
```

---

## 故障排除

### 问题1：Conda速度慢
```bash
# 使用清华镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

### 问题2：pip安装慢
```bash
# 使用阿里云镜像
pip install -i https://mirrors.aliyun.com/pypi/simple/ package_name

# 或设置为默认
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

### 问题3：PyTorch CUDA不可用
- 检查CUDA版本是否匹配
- 重新安装对应CUDA版本的PyTorch
- 验证NVIDIA驱动是否正确安装

### 问题4：Jupyter找不到环境
```bash
# 在ml_env环境中安装ipykernel
conda activate ml_env
pip install ipykernel
python -m ipykernel install --user --name=ml_env
```

---

## 下一步

环境配置完成后，可以开始：

1. **运行第一个示例**
   ```bash
   jupyter lab
   # 创建新notebook测试代码
   ```

2. **克隆学习资源**
   ```bash
   git clone https://github.com/pytorch/examples.git
   cd examples
   ```

3. **开始Kaggle竞赛**
   - 访问 https://www.kaggle.com/
   - 尝试Titanic竞赛

4. **探索Hugging Face**
   - 访问 https://huggingface.co/
   - 尝试预训练模型

---

## 参考资源

- [Conda官方文档](https://docs.conda.io/)
- [PyTorch安装指南](https://pytorch.org/get-started/locally/)
- [TensorFlow安装指南](https://www.tensorflow.org/install)
- [Scikit-learn文档](https://scikit-learn.org/)

---

**环境配置完成！开始你的机器学习之旅吧！**
