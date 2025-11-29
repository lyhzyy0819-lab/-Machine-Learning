# 📊 第一章：数据诊断框架

> **核心理念**："Know Your Data" - 诊断决定方向
>
> **目标**：拿到陌生数据后，15分钟内完成快速诊断，知道要处理什么问题

## 📂 章节内容

| 文档 | 定位 | 适用场景 | 阅读时间 |
|------|------|----------|----------|
| **data_diagnosis_quick_reference.md** ⭐ | 速查表 | 快速诊断陌生数据 | 15分钟 |
| **data_problem_to_solution_mapping.md** | 问题→方案映射 | 已知问题，找处理方法 | 30分钟 |
| **data_diagnosis_decision_tree.md** | 可视化决策树 | 系统化完整诊断 | 45分钟 |
| **common_data_issues.md** | 深入参考 | 了解问题详细方案 | 1-2小时 |
| **diagnosis_checklist.md** | 检查清单 | 确保不遗漏关键项 | 20分钟 |

---

## 🚀 三种使用模式

### 模式1：快速诊断（15-30分钟）⭐

**适合**：拿到新数据，需要快速了解问题

**流程**：
```
Step 1: 快速扫描数据（5分钟）
   → 运行 df.info(), df.describe(), df.isnull().sum()

Step 2: 使用速查表诊断（10分钟）
   → 打开 data_diagnosis_quick_reference.md
   → 根据数据特征快速定位问题

Step 3: 确定处理方案（5-10分钟）
   → 打开 data_problem_to_solution_mapping.md
   → 查表得到处理方案
```

**实战示例**：
```
场景：拿到客户流失数据，50000行×25列

5分钟扫描 → 发现：缺失值3500个，流失率10%
10分钟查表 → 诊断：中等数据集，轻度缺失，中度不平衡
5分钟方案 → 决策：简单填充+SMOTE或类权重

总计20分钟完成诊断！
```

---

### 模式2：系统化诊断（1-2小时）

**适合**：重要项目，需要全面诊断

**流程**：
```
Step 1: 使用决策树（30-45分钟）
   → 打开 data_diagnosis_decision_tree.md
   → 按14个步骤逐步诊断

Step 2: 深入了解问题（30-60分钟）
   → 打开 common_data_issues.md
   → 针对发现的问题查看详细方案

Step 3: 检查清单验证（15分钟）
   → 打开 diagnosis_checklist.md
   → 确保不遗漏关键检查项
```

---

### 模式3：问题驱动查找（5-10分钟）

**适合**：遇到具体问题，需要快速解决

**流程**：
```
遇到问题                     查阅文档
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
缺失值太多，不知道怎么办    → data_problem_to_solution_mapping.md
                           → 查看"缺失值处理"章节
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
异常值不确定是否删除        → common_data_issues.md
                           → 查看"异常值问题"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
类别严重不平衡              → data_problem_to_solution_mapping.md
                           → 查看"类别不平衡"
```

---

## 🔗 与其他章节的连接

本章是整个ML工作流的第一步，诊断结果会直接影响后续决策：

```
【01_data_diagnosis】← 当前章节
   ↓ 诊断完成，发现问题
【02_problem_definition】问题定义
   ↓ 明确问题类型（分类/回归/聚类）
【03_algorithm_selection】算法选择
   ↓ 根据数据特征选定算法
【04_preprocessing_features】预处理（处理01中发现的问题）
   ↓ 数据准备完成
【05_model_evaluation】模型评估
```

**关键关联**：
- **诊断发现缺失值** → 04章查找缺失值处理方法
- **诊断发现类别不平衡** → 03章选择支持不平衡的算法 + 04章处理
- **诊断发现数据量小** → 03章避免复杂模型，选择简单算法
- **诊断发现异常值多** → 03章选择鲁棒算法（如树模型）

---

## 💡 核心诊断维度

### 1. 数据量级诊断

| 样本量 | 特征量 | 诊断结果 | 对后续影响 |
|--------|--------|---------|-----------|
| <1K | <20 | 小数据集 | 03章：算法选择受限（简单模型）<br>过拟合风险高，必须交叉验证 |
| 1K-50K | 20-100 | 中等数据集 | 03章：大部分算法适用<br>标准ML流程 |
| 50K-500K | 100-1000 | 大数据集 | 03章：需要高效算法（LightGBM）<br>可考虑采样加速 |
| >500K | >1000 | 超大数据集 | 03章：必须高效算法<br>04章：采样或分布式处理 |

### 2. 数据质量诊断

| 问题类型 | 严重程度判断 | 查阅文档 | 影响 |
|---------|------------|---------|------|
| 缺失值 | 缺失率 >50% = 严重 | data_problem_to_solution_mapping | 04章：选择填充方法 |
| 异常值 | 异常率 >5% = 需处理 | common_data_issues | 03章：选择鲁棒算法 |
| 重复值 | 重复率 >10% = 问题 | data_problem_to_solution_mapping | 04章：删除重复行 |
| 类别不平衡 | 最大类 >80% = 严重 | common_data_issues | 03章+05章：调整算法和评估指标 |

### 3. 数据分布诊断

| 分布特征 | 影响 | 对后续影响 |
|---------|------|-----------|
| 正态 vs 偏态 | 算法性能 | 03章：偏态数据避免线性模型<br>04章：可能需要变换 |
| 线性 vs 非线性 | 模型类型 | 03章：非线性关系选树模型/SVM |
| 高维稀疏 | 维度灾难 | 04章：必须降维或特征选择 |

详见：`data_diagnosis_quick_reference.md`

---

## 📊 快速导航

### 按使用场景导航

| 我的场景 | 从哪里开始 | 预计时间 |
|----------|-----------|---------|
| **拿到新数据，快速诊断** | 👉 [data_diagnosis_quick_reference.md](data_diagnosis_quick_reference.md) | 15-30分钟 |
| **已知问题，找解决方案** | 👉 [data_problem_to_solution_mapping.md](data_problem_to_solution_mapping.md) | 5-10分钟 |
| **重要项目，全面诊断** | 👉 [data_diagnosis_decision_tree.md](data_diagnosis_decision_tree.md) | 1-2小时 |
| **深入了解某个问题** | 👉 [common_data_issues.md](common_data_issues.md) | 按需查阅 |

### 按问题类型导航

| 问题类型 | 快速诊断 | 解决方案 | 深入了解 |
|---------|---------|---------|---------|
| 缺失值 | quick_reference | problem_to_solution | common_data_issues |
| 异常值 | quick_reference | problem_to_solution | common_data_issues |
| 类别不平衡 | quick_reference | problem_to_solution | common_data_issues |
| 数据泄漏 | quick_reference | problem_to_solution | common_data_issues |
| 重复值 | quick_reference | problem_to_solution | - |
| 数据类型 | quick_reference | problem_to_solution | - |

---

## 🎯 数据诊断的重要性

### 为什么数据诊断如此关键？

**"垃圾进，垃圾出" (Garbage In, Garbage Out)**

数据质量直接决定模型性能的上限，诊断不足会导致：

| 问题 | 未诊断的后果 | 诊断后的收益 |
|------|-----------|------------|
| **缺失值未处理** | 模型训练失败或性能下降30% | 正确填充可恢复大部分信息 |
| **异常值未处理** | 模型被极值主导，泛化差 | 鲁棒处理提升10-20%性能 |
| **类别不平衡未知** | Accuracy高但预测无效 | 调整算法+指标，实际可用 |
| **数据泄漏未发现** | 测试集虚高，上线失败 | 提前发现避免重大损失 |

### 诊断能解决什么问题？

1. **提前发现问题**
   - 避免在错误的数据上浪费时间
   - 及早发现数据收集/处理中的错误

2. **指导后续决策**
   - 确定需要哪些预处理步骤（04章）
   - 选择合适的算法（03章）
   - 设定正确的评估指标（05章）

3. **避免常见陷阱**
   - 数据泄漏
   - 过拟合
   - 类别不平衡
   - 多重共线性

---

## 🛠️ 实战代码工具

### 快速诊断脚本

```python
import pandas as pd
import numpy as np

def quick_diagnosis(df, target_col=None):
    """
    15分钟快速诊断脚本

    用法:
        df = pd.read_csv('data.csv')
        quick_diagnosis(df, target_col='label')
    """
    print("="*60)
    print("📊 快速数据诊断")
    print("="*60)

    # 1. 基础信息
    print(f"\n1. 基础信息")
    print(f"   样本数: {len(df):,}")
    print(f"   特征数: {df.shape[1]}")
    print(f"   内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 2. 数据质量
    print(f"\n2. 数据质量")
    missing = df.isnull().sum().sum()
    print(f"   缺失值: {missing:,} ({missing/(len(df)*df.shape[1])*100:.2f}%)")
    print(f"   重复行: {df.duplicated().sum():,}")

    # 3. 数据类型
    print(f"\n3. 数据类型")
    print(f"   数值型: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   类别型: {len(df.select_dtypes(include=['object', 'category']).columns)}")

    # 4. 目标变量（如果提供）
    if target_col and target_col in df.columns:
        print(f"\n4. 目标变量: {target_col}")
        if df[target_col].dtype in [np.int64, np.float64] and df[target_col].nunique() > 10:
            print(f"   类型: 回归问题")
            print(f"   范围: [{df[target_col].min():.2f}, {df[target_col].max():.2f}]")
        else:
            print(f"   类型: 分类问题")
            print(f"   类别数: {df[target_col].nunique()}")
            value_counts = df[target_col].value_counts()
            max_ratio = value_counts.max() / len(df)
            print(f"   最大类占比: {max_ratio*100:.1f}%")
            if max_ratio > 0.8:
                print(f"   ⚠️  严重不平衡！")

    print("\n" + "="*60)
    print("下一步:")
    print("1. 查看 data_diagnosis_quick_reference.md 的详细诊断表")
    print("2. 根据问题查看 data_problem_to_solution_mapping.md")
    print("="*60)

# 使用示例
# df = pd.read_csv('your_data.csv')
# quick_diagnosis(df, target_col='target')
```

### 使用综合项目的诊断模块

```python
import sys
sys.path.append('../06_comprehensive_project')

from src import data_diagnosis

# 完整诊断报告
report = data_diagnosis.generate_diagnosis_report(df, target='target_column')

# 缺失值分析
data_diagnosis.missing_value_analysis(df)
data_diagnosis.visualize_missing_values(df)

# 异常值检测
outliers = data_diagnosis.outlier_analysis(df, method='iqr')

# 相关性分析
data_diagnosis.correlation_analysis(df, threshold=0.7)
```

---

## 💬 常见问题 (FAQ)

### Q1: 数据诊断需要多长时间？

**A**: 取决于使用模式：
- **快速模式**（模式1）：15-30分钟 - 适合日常使用
- **系统化模式**（模式2）：1-2小时 - 适合重要项目
- **问题驱动**（模式3）：5-10分钟 - 遇到问题时查找

### Q2: 每次建模都要做完整诊断吗？

**A**:
- ✅ **新数据集**：必须做完整诊断（模式2）
- ✅ **重要项目**：必须做完整诊断
- ⏭️ **熟悉数据**：快速诊断即可（模式1）
- ⏭️ **已诊断数据**：跳过，直接建模

### Q3: 诊断和预处理有什么区别？

**A**:
- **01章（诊断）**：**发现**问题，不解决问题
- **04章（预处理）**：**解决**问题，实际处理数据

**流程**：01诊断 → 知道有什么问题 → 04预处理 → 实际处理

### Q4: 诊断后发现很多问题怎么办？

**A**: 按优先级处理：
1. **P0 - 必须处理**：数据泄漏、严重缺失（>50%）
2. **P1 - 建议处理**：中度缺失、异常值、类别不平衡
3. **P2 - 可选处理**：轻度缺失、弱相关特征

使用 `data_problem_to_solution_mapping.md` 快速找到解决方案。

### Q5: 诊断发现问题，但不知道怎么处理？

**A**: 按以下顺序查找：
1. **快速方案**：`data_problem_to_solution_mapping.md` - 直接给代码
2. **详细方案**：`common_data_issues.md` - 深入理解
3. **完整处理**：`04_preprocessing_and_features/` - 系统化处理

---

## 🎓 使用建议

### 第一次使用

1. 阅读本 README.md（10分钟）
2. 在你的数据上运行快速诊断脚本（5分钟）
3. 根据结果查阅 `data_diagnosis_quick_reference.md`（10分钟）
4. 总计：25分钟完成第一次诊断！

### 日常使用

- **模式1**（最常用）：拿到新数据 → 打开速查表 → 15分钟完成
- **模式3**（遇到问题）：遇到具体问题 → 打开mapping表 → 5分钟解决

### 重要项目

- **模式2**（全面）：使用决策树 → 系统化诊断 → 1-2小时确保不遗漏

---

## 📖 延伸阅读

### 本项目相关

- [ML_WORKFLOW_GUIDE.md](../ML_WORKFLOW_GUIDE.md) - 完整的6部分工作流
- [02_problem_definition_guide](../02_problem_definition_guide/) - 下一章：问题定义
- [04_preprocessing_and_features](../04_preprocessing_and_features/) - 实际处理数据问题

### 实战应用

- [06_comprehensive_project/Phase 1](../06_comprehensive_project/) - 完整诊断案例

---

**最后更新**：2024年11月
**建议时间分配**：
- 快速诊断：15-30分钟（80%场景）
- 系统化诊断：1-2小时（20%场景）

**下一步**：根据你的场景，选择合适的使用模式开始诊断！
