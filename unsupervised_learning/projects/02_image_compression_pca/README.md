# 🎯 项目2: PCA图像压缩

> 使用主成分分析(PCA)实现图像压缩

---

## 📋 项目概述

**业务场景**：
图像存储和传输需要大量空间和带宽。PCA可以在保持图像主要特征的同时减少数据量。

**项目目标**：
1. 使用PCA对图像进行降维压缩
2. 分析不同压缩率对图像质量的影响
3. 计算压缩比和重构误差
4. 可视化压缩效果

**技术栈**：
- PCA (sklearn.decomposition.PCA)
- 方差解释率
- MSE / PSNR (图像质量评估)
- Matplotlib可视化

---

## 📊 数据集

### 推荐数据集
1. **Olivetti Faces** (sklearn自带)
   - 400张64x64人脸图像
   - 灰度图像

2. **自定义图像**
   - 可以使用自己的图片

---

## 🎯 项目步骤

### Step 1: 加载和探索数据
- [ ] 加载Olivetti Faces数据集
- [ ] 查看数据形状和信息
- [ ] 可视化部分图像样本

### Step 2: 理解PCA用于图像压缩
- [ ] 理解图像如何表示为矩阵
- [ ] 理解PCA如何降维
- [ ] 计算原始图像的数据量

### Step 3: 应用PCA压缩
- [ ] 对图像数据应用PCA
- [ ] 尝试不同的主成分数量
  - 10, 50, 100, 150, 200, ... 主成分
- [ ] 重构压缩后的图像

### Step 4: 评估压缩效果
- [ ] 计算方差解释率
- [ ] 计算重构误差(MSE)
- [ ] 计算压缩比
- [ ] 可视化不同压缩率的结果

### Step 5: 方差解释率分析
- [ ] 绘制累积方差解释率曲线
- [ ] 确定保留95%、99%方差所需的主成分数

### Step 6: 可视化对比
- [ ] 并排显示原始图像和不同压缩率的图像
- [ ] 绘制压缩比 vs 图像质量曲线

---

## 📈 期望输出

### 1. 数据探索
- 数据集基本信息
- 部分图像样本展示

### 2. PCA分析
- 不同主成分数量的压缩效果
- 方差解释率曲线
- 累积方差解释率

### 3. 压缩效果对比
原始图像 vs 不同压缩率：
- n_components = 10 (压缩率: ~98%)
- n_components = 50 (压缩率: ~87%)
- n_components = 100 (压缩率: ~75%)
- n_components = 200 (压缩率: ~50%)

### 4. 质量评估
- MSE vs 主成分数量
- 压缩比 vs 重构质量
- 最佳压缩率推荐

---

## 💡 项目提示

### 计算公式

**压缩比**：
```
压缩比 = 1 - (压缩后大小 / 原始大小)
     = 1 - (n_components / n_features)
```

**均方误差(MSE)**：
```python
mse = np.mean((原始图像 - 重构图像) ** 2)
```

**峰值信噪比(PSNR)**：
```python
psnr = 10 * np.log10(255**2 / mse)
```

### 代码示例

```python
from sklearn.decomposition import PCA

# 应用PCA
n_components = 50
pca = PCA(n_components=n_components)
X_compressed = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_compressed)

# 计算方差解释率
variance_ratio = pca.explained_variance_ratio_
cumsum_variance = np.cumsum(variance_ratio)

# 计算压缩比
compression_ratio = 1 - (n_components / X.shape[1])
print(f"压缩比: {compression_ratio:.2%}")
```

### 可视化技巧

```python
# 并排显示原始和重构图像
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    axes[0, i].imshow(original_images[i], cmap='gray')
    axes[0, i].set_title('Original')
    axes[1, i].imshow(reconstructed_images[i], cmap='gray')
    axes[1, i].set_title(f'Reconstructed (n={n_components})')
```

---

## 📊 分析问题

### 问题1: 方差解释率分析
- 保留90%方差需要多少主成分？
- 保留95%方差需要多少主成分？
- 保留99%方差需要多少主成分？

### 问题2: 压缩率权衡
- 什么样的压缩率能保持图像可识别性？
- 压缩率和图像质量之间的权衡点在哪里？

### 问题3: 主成分可视化
- 前几个主成分代表什么？
- 可视化前10个主成分（特征脸）

### 问题4: 实际应用
- PCA图像压缩与JPEG等压缩算法相比的优缺点？
- PCA更适合什么类型的图像？

---

## ✅ 检查清单

完成项目后，确保：
- [ ] 成功加载和可视化数据
- [ ] 理解PCA压缩原理
- [ ] 尝试多个压缩率（至少5个）
- [ ] 计算方差解释率和MSE
- [ ] 绘制累积方差解释率曲线
- [ ] 可视化不同压缩率的图像对比
- [ ] 分析压缩比与质量的权衡
- [ ] 可视化特征脸（前几个主成分）
- [ ] 代码注释清晰
- [ ] 有完整的分析和结论

---

## 🔬 进阶挑战

1. **彩色图像压缩**
   - 对RGB三通道分别应用PCA
   - 对比与灰度图像的压缩效果

2. **与其他方法对比**
   - 对比PCA与SVD压缩
   - 对比PCA与Autoencoder压缩

3. **压缩时间分析**
   - 测量PCA压缩和重构的时间
   - 分析时间复杂度

---

## 📚 参考资源

- [PCA for Image Compression](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html)
- [Eigenfaces Tutorial](https://en.wikipedia.org/wiki/Eigenface)

---

**开始时间**: __________
**完成时间**: __________
**项目耗时**: __________

**💪 完成这个项目后，你将深入理解PCA的实际应用！**
