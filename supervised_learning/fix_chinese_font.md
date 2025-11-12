# 修复Matplotlib中文显示问题

## 问题描述
运行notebook时出现中文字符无法显示的警告：
```
UserWarning: Glyph 29983 (\N{CJK UNIFIED IDEOGRAPH-751F}) missing from current font.
```

---

## 🚀 快速解决方案

### 方法1：在每个Notebook开头添加（推荐）

在导入库的cell中添加以下代码：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============ 添加这部分代码 ============
# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
# =====================================

# 其他配置...
```

---

### 方法2：使用配置文件（一次配置，所有notebook生效）

1. **导入配置模块**

在每个notebook的第一个代码cell中：

```python
# 导入中文字体配置
from matplotlib_config import setup_chinese_font
setup_chinese_font()

# 然后正常导入其他库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

### 方法3：修改matplotlib全局配置（永久）

创建matplotlib配置文件：

```bash
# 1. 找到配置文件位置
python -c "import matplotlib; print(matplotlib.matplotlib_fname())"

# 2. 编辑配置文件（或创建用户配置）
mkdir -p ~/.matplotlib
nano ~/.matplotlib/matplotlibrc
```

在文件中添加：

```
font.sans-serif: Arial Unicode MS, PingFang SC, STHeiti
axes.unicode_minus: False
```

保存后重启Jupyter。

---

## 🔍 检查可用的中文字体

运行以下代码查看系统中可用的中文字体：

```python
import matplotlib.font_manager as fm

# 查找所有中文字体
chinese_fonts = []
for font in fm.fontManager.ttflist:
    if 'CJK' in font.name or 'Chinese' in font.name or \
       any(cn in font.name for cn in ['Arial Unicode', 'PingFang', 'Heiti', 'STHeiti', 'Songti', 'SimHei']):
        chinese_fonts.append(font.name)

print("系统中可用的中文字体:")
for font in set(chinese_fonts):
    print(f"  - {font}")
```

---

## macOS常用中文字体

按推荐顺序：

1. **Arial Unicode MS** - 最通用，包含几乎所有字符
2. **PingFang SC** - 苹方，macOS默认中文字体
3. **STHeiti** - 华文黑体
4. **Heiti TC** - 黑体-繁体
5. **Songti SC** - 宋体

---

## 📝 更新后的Notebook模板

完整的导入cell示例：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ========== 中文显示配置 ==========
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# ================================

# 设置随机种子
np.random.seed(42)

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

print('✓ 所有库导入成功')
print(f'✓ 中文字体: {plt.rcParams["font.sans-serif"][0]}')
```

---

## 测试中文显示

运行以下代码测试：

```python
import matplotlib.pyplot as plt

# 配置中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 测试图表
plt.figure(figsize=(8, 5))
plt.plot([1, 2, 3], [1, 4, 9], 'o-')
plt.title('测试中文标题')
plt.xlabel('横坐标（中文）')
plt.ylabel('纵坐标（中文）')
plt.grid(True)
plt.show()

print('如果看到中文，说明配置成功！')
```

---

## ⚠️ 注意事项

1. **必须在绘图之前设置字体**，在导入matplotlib后立即配置
2. **每次重启notebook都需要重新配置**（除非使用全局配置）
3. **如果一个字体不工作，尝试列表中的其他字体**
4. **Windows和Linux用户需要使用不同的字体名称**

---

## 现在就修复

### 立即在当前notebook中修复：

找到导入库的cell（通常是第一个代码cell），在这两行之后：

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

添加：

```python
# 修复中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```

然后**重新运行这个cell和后面所有cell**（Kernel → Restart & Run All）

---

## 完成！

配置后中文就能正常显示了，不会再有警告信息。
