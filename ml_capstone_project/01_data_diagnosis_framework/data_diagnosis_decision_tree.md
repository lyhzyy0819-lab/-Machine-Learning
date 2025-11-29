# 🌳 数据诊断决策树

> 系统化的数据诊断流程，从基础检查到深度分析

## 📋 使用说明

### 两种使用模式

**快速模式（30-45分钟）**：
- 适合：日常数据诊断，快速了解数据概况
- 方法：浏览决策树流程图，重点检查关键节点（标记⭐的步骤）
- 输出：问题清单和处理方向

**完整模式（1-2小时）**：
- 适合：重要项目，需要全面诊断不遗漏任何问题
- 方法：按14个步骤逐步诊断，使用配套代码检查每个环节
- 输出：详细诊断报告和完整处理方案

### 决策树结构

本决策树采用**层级递进式**设计：
1. 按顺序检查每个节点
2. 根据检查结果选择分支
3. 记录发现的问题
4. 最终得到处理建议

### 与其他文档的配合

- **快速入门** → 先查看 [data_diagnosis_quick_reference.md](data_diagnosis_quick_reference.md)（15分钟）
- **问题解决** → 发现问题后查看 [data_problem_to_solution_mapping.md](data_problem_to_solution_mapping.md)
- **深入理解** → 每个问题的详细说明见 [common_data_issues.md](common_data_issues.md)

---

## 🎯 决策树完整流程

```
                           【开始数据诊断】
                                 │
                                 ↓
                        ┌────────────────┐
                        │ 1. 加载数据成功？│
                        └────────────────┘
                         │              │
                       成功            失败
                         │              │
                         ↓              └→ 检查文件路径、格式、编码
                                           修复后重新加载

┌────────────────────────────────────────────────────────────────┐
│                    第一层：基础信息检查                        │
└────────────────────────────────────────────────────────────────┘

                        ┌────────────────┐
                        │ 2. 数据集大小？ │
                        └────────────────┘
                    │           │           │
                  <1000     1K-100K      >100K
                    │           │           │
                    ↓           ↓           ↓
              数据量过小    正常范围      大数据集
              考虑：       继续诊断     考虑采样/
              • 收集更多                分布式处理
              • 数据增强
              • 迁移学习
                    │           │           │
                    └───────────┴───────────┘
                                 │
                                 ↓
                        ┌────────────────┐
                        │ 3. 特征类型分布？│
                        └────────────────┘
                           记录：
                           • 数值型特征数
                           • 类别型特征数
                           • 文本特征数
                           • 时间特征数
                                 │
                                 ↓

┌────────────────────────────────────────────────────────────────┐
│                    第二层：数据质量检查                        │
└────────────────────────────────────────────────────────────────┘

                        ┌────────────────┐
                        │ 4. 缺失值检查   │
                        └────────────────┘
                                 │
                        是否存在缺失值？
                         │            │
                        是            否
                         │            │
                         ↓            ↓
                  计算缺失率     继续下一步
                         │
                 ┌───────┴───────┐
                 │               │
            缺失率<5%        缺失率>5%
                 │               │
                 ↓               ↓
            简单填充          分析缺失模式
            • 均值/中位数    ┌─────────┴─────────┐
            • 众数          │                   │
            • 常数        MCAR              MAR/MNAR
                         (随机缺失)        (非随机缺失)
                             │                   │
                             ↓                   ↓
                        删除/简单填充      建模填充/
                                          特殊处理
                             │                   │
                             └────────┬──────────┘
                                      ↓

                        ┌────────────────┐
                        │ 5. 重复值检查   │
                        └────────────────┘
                                 │
                        是否存在重复行？
                         │            │
                        是            否
                         │            │
                         ↓            ↓
                  分析重复原因   继续下一步
                         │
                 ┌───────┴───────┐
                 │               │
            数据录入错误      真实重复
                 │               │
                 ↓               ↓
              直接删除        保留或加权
                 │               │
                 └───────┬───────┘
                         ↓

                        ┌────────────────┐
                        │ 6. 异常值检测   │
                        └────────────────┘
                                 │
                   选择检测方法（IQR/Z-Score）
                                 │
                        是否存在异常值？
                         │            │
                        是            否
                         │            │
                         ↓            ↓
                  判断异常性质   继续下一步
                         │
                 ┌───────┴───────┐
                 │               │
              数据错误        真实极值
                 │               │
                 ↓               ↓
          删除/修正            保留/截断/
                              转换
                 │               │
                 └───────┬───────┘
                         ↓

┌────────────────────────────────────────────────────────────────┐
│                    第三层：分布分析                            │
└────────────────────────────────────────────────────────────────┘

                        ┌────────────────┐
                        │ 7. 目标变量分析 │
                        └────────────────┘
                                 │
                        确定问题类型
                 ┌───────┴───────┐
                 │               │
              分类问题        回归问题
                 │               │
                 ↓               ↓
         检查类别平衡？      检查值域范围
                 │               │
         ┌───────┴───────┐       │
         │               │       │
      平衡(4:6)      不平衡       ↓
         │           │       是否需要对数
         ↓           ↓       或Box-Cox变换？
     正常使用    考虑：           │
     准确率      • SMOTE     ┌───┴───┐
                 • 类权重     │       │
                 • 特殊算法   是      否
                 • F1/AUC     │       │
         │           │        ↓       ↓
         └───────┬───┘    规划变换  继续
                 ↓

                        ┌────────────────┐
                        │ 8. 数值特征分布 │
                        └────────────────┘
                                 │
                     计算偏度(skewness)
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
            偏度<-0.5       -0.5~0.5        >0.5
            (左偏)        (对称分布)        (右偏)
                 │               │               │
                 ↓               ↓               ↓
            考虑右偏变换    无需变换      考虑左偏变换
            • 平方          继续         • log(x)
            • 指数                       • sqrt(x)
                 │               │               │
                 └───────────────┴───────────────┘
                                 │
                                 ↓

                        ┌────────────────┐
                        │ 9. 类别特征分布 │
                        └────────────────┘
                                 │
                      检查唯一值数量
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
            唯一值=1         2-20            >20
            (常量)        (类别型)       (高基数)
                 │               │               │
                 ↓               ↓               ↓
              删除该列      One-Hot或     考虑：
                         Label Encoding  • Target Encoding
                                        • 分组
                                        • 作为数值处理
                 │               │               │
                 └───────────────┴───────────────┘
                                 │
                                 ↓

┌────────────────────────────────────────────────────────────────┐
│                    第四层：特征关系分析                        │
└────────────────────────────────────────────────────────────────┘

                        ┌────────────────┐
                        │ 10. 相关性分析  │
                        └────────────────┘
                                 │
                   计算特征间相关系数矩阵
                                 │
                   是否存在高相关特征对？
                   (|r| > 0.9)
                         │            │
                        是            否
                         │            │
                         ↓            ↓
                 识别高相关特征对  继续下一步
                         │
                 选择保留策略：
                 • 业务重要性
                 • 与目标相关性
                 • 特征重要性
                         │
                         ↓

                        ┌────────────────┐
                        │11. 特征与目标   │
                        │    关系分析     │
                        └────────────────┘
                                 │
                     ┌───────────┴───────────┐
                     │                       │
                  分类问题                回归问题
                     │                       │
                     ↓                       ↓
              卡方检验/              Pearson/
              互信息                Spearman相关
                     │                       │
                     └───────────┬───────────┘
                                 │
                      找出弱相关特征
                      (考虑删除或特征工程)
                                 │
                                 ↓

┌────────────────────────────────────────────────────────────────┐
│                    第五层：高级诊断                            │
└────────────────────────────────────────────────────────────────┘

                        ┌────────────────┐
                        │ 12. 多重共线性  │
                        └────────────────┘
                                 │
                    (仅线性模型需要检查)
                                 │
                      计算VIF (方差膨胀因子)
                                 │
                        是否 VIF > 10？
                         │            │
                        是            否
                         │            │
                         ↓            ↓
                   存在共线性     无共线性问题
                         │
                   删除高VIF特征
                   或使用正则化
                         │
                         ↓

                        ┌────────────────┐
                        │ 13. 数据泄漏检查│
                        └────────────────┘
                                 │
                      检查以下情况：
                      □ 是否有ID列被用作特征？
                      □ 是否有未来信息？
                      □ 目标变量的变种？
                      □ 测试集特有信息？
                                 │
                        发现泄漏？
                         │            │
                        是            否
                         │            │
                         ↓            ↓
                   移除泄漏特征    继续
                         │
                         ↓

┌────────────────────────────────────────────────────────────────┐
│                       最终：生成报告                           │
└────────────────────────────────────────────────────────────────┘

                        ┌────────────────┐
                        │ 14. 总结诊断结果│
                        └────────────────┘
                                 │
                   生成诊断报告包含：
                   ✓ 数据基本信息
                   ✓ 发现的问题列表
                   ✓ 问题严重程度
                   ✓ 建议处理方案
                   ✓ 潜在风险提示
                   ✓ 下一步行动计划
                                 │
                                 ↓
                        【诊断完成】
```

---

## 🛠️ 决策树实战代码

### 完整诊断流程实现

```python
import pandas as pd
import numpy as np
import sys
sys.path.append('../06_comprehensive_project')
from src import data_diagnosis

class DataDiagnosisFlow:
    """数据诊断决策流程"""

    def __init__(self, df, target_col=None):
        self.df = df
        self.target_col = target_col
        self.issues = []  # 记录发现的问题
        self.recommendations = []  # 记录建议

    def step1_basic_info(self):
        """第一步：基础信息"""
        print("\n" + "="*60)
        print("第一步：基础信息检查")
        print("="*60)

        n_samples, n_features = self.df.shape

        # 数据集大小判断
        if n_samples < 1000:
            self.issues.append("⚠️  数据量过小（<1000行）")
            self.recommendations.append("考虑收集更多数据或使用数据增强")
        elif n_samples > 100000:
            self.issues.append("ℹ️  大数据集（>10万行）")
            self.recommendations.append("考虑采样进行快速分析")

        # 特征类型统计
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        print(f"样本数: {n_samples:,}")
        print(f"特征数: {n_features}")
        print(f"数值型特征: {len(numeric_cols)}")
        print(f"类别型特征: {len(categorical_cols)}")

        return True

    def step2_missing_values(self):
        """第二步：缺失值检查"""
        print("\n" + "="*60)
        print("第二步：缺失值检查")
        print("="*60)

        missing_stats = data_diagnosis.missing_value_analysis(self.df, threshold=0.0)

        if len(missing_stats) > 0:
            # 严重缺失
            severe = missing_stats[missing_stats['缺失比例(%)'] > 50]
            if len(severe) > 0:
                self.issues.append(f"❌ {len(severe)}个特征缺失率>50%")
                self.recommendations.append("建议删除严重缺失的特征")

            # 中度缺失
            moderate = missing_stats[(missing_stats['缺失比例(%)'] > 5) &
                                    (missing_stats['缺失比例(%)'] <= 50)]
            if len(moderate) > 0:
                self.issues.append(f"⚠️  {len(moderate)}个特征缺失率5-50%")
                self.recommendations.append("使用KNN填充或建模填充")

        return True

    def step3_duplicates(self):
        """第三步：重复值检查"""
        print("\n" + "="*60)
        print("第三步：重复值检查")
        print("="*60)

        n_duplicates = self.df.duplicated().sum()

        if n_duplicates > 0:
            dup_ratio = n_duplicates / len(self.df) * 100
            self.issues.append(f"⚠️  发现{n_duplicates}行重复数据（{dup_ratio:.1f}%）")
            self.recommendations.append("检查重复原因后删除")
            print(f"重复行数: {n_duplicates} ({dup_ratio:.1f}%)")
        else:
            print("✅ 无重复行")

        return True

    def step4_outliers(self):
        """第四步：异常值检测"""
        print("\n" + "="*60)
        print("第四步：异常值检测")
        print("="*60)

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) > 0:
            outlier_report = data_diagnosis.outlier_analysis(
                self.df, method='iqr', columns=numeric_cols.tolist()
            )

            # 统计异常值较多的特征
            high_outlier_features = [
                col for col, info in outlier_report.items()
                if info['outlier_ratio'] > 5
            ]

            if high_outlier_features:
                self.issues.append(f"⚠️  {len(high_outlier_features)}个特征异常值>5%")
                self.recommendations.append("考虑异常值截断或删除")

        return True

    def step5_target_analysis(self):
        """第五步：目标变量分析"""
        if self.target_col is None or self.target_col not in self.df.columns:
            print("\n跳过目标变量分析（未指定目标列）")
            return True

        print("\n" + "="*60)
        print("第五步：目标变量分析")
        print("="*60)

        target = self.df[self.target_col]

        # 判断问题类型
        if target.nunique() <= 20:  # 分类问题
            print("问题类型: 分类")

            # 检查类别平衡
            value_counts = target.value_counts()
            max_ratio = value_counts.max() / len(target)

            if max_ratio > 0.8:
                self.issues.append(f"❌ 严重类别不平衡（最大类占比{max_ratio:.1%}）")
                self.recommendations.append("使用SMOTE、类权重调整或特殊算法")
            elif max_ratio > 0.7:
                self.issues.append(f"⚠️  中度类别不平衡（最大类占比{max_ratio:.1%}）")
                self.recommendations.append("注意使用F1-Score和AUC评估")

        else:  # 回归问题
            print("问题类型: 回归")

            # 检查分布偏度
            skewness = target.skew()
            if abs(skewness) > 0.5:
                self.issues.append(f"⚠️  目标变量偏度{skewness:.2f}")
                self.recommendations.append("考虑对数变换或Box-Cox变换")

        return True

    def step6_correlations(self):
        """第六步：相关性分析"""
        print("\n" + "="*60)
        print("第六步：特征相关性分析")
        print("="*60)

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr().abs()

            # 找出高相关特征对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        high_corr_pairs.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j])
                        )

            if high_corr_pairs:
                self.issues.append(f"⚠️  发现{len(high_corr_pairs)}对高相关特征")
                self.recommendations.append("考虑删除冗余特征以避免多重共线性")

        return True

    def generate_report(self):
        """生成最终报告"""
        print("\n" + "="*60)
        print("📋 数据诊断报告")
        print("="*60)

        print(f"\n数据集: {self.df.shape[0]:,} 行 × {self.df.shape[1]} 列")

        print(f"\n🔍 发现问题 ({len(self.issues)}个):")
        if self.issues:
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
        else:
            print("✅ 未发现明显问题")

        print(f"\n💡 处理建议 ({len(self.recommendations)}个):")
        if self.recommendations:
            for i, rec in enumerate(self.recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✅ 数据质量良好，可直接进入建模阶段")

        print("\n" + "="*60)

    def run_full_diagnosis(self):
        """运行完整诊断流程"""
        print("\n" + "🌳"*30)
        print("开始完整数据诊断流程")
        print("🌳"*30)

        # 依次执行所有步骤
        self.step1_basic_info()
        self.step2_missing_values()
        self.step3_duplicates()
        self.step4_outliers()
        self.step5_target_analysis()
        self.step6_correlations()

        # 生成报告
        self.generate_report()

        print("\n✅ 诊断流程完成！")


# 使用示例
if __name__ == '__main__':
    # 加载数据
    df = pd.read_csv('your_data.csv')

    # 创建诊断流程
    diagnosis = DataDiagnosisFlow(df, target_col='target')

    # 运行完整诊断
    diagnosis.run_full_diagnosis()
```

---

## 📊 决策节点详解

### 关键决策点说明

#### 1. 缺失率阈值选择（5%, 20%, 50%）

| 阈值 | 含义 | 依据 |
|------|------|------|
| 5% | 轻度缺失 | 简单方法足够，不影响模型 |
| 20% | 中度缺失 | 需要谨慎处理，可能影响模型 |
| 50% | 严重缺失 | 信息严重不足，建议删除 |

#### 2. 异常值判断标准

**IQR方法：**
```
下界 = Q1 - 1.5 * IQR
上界 = Q3 + 1.5 * IQR
```
- k=1.5: 温和异常值（常用）
- k=3.0: 极端异常值（严格）

**Z-Score方法：**
```
z = (x - μ) / σ
```
- |z| > 3: 异常值（99.7%规则）
- |z| > 2: 可疑值（95%规则）

#### 3. 类别不平衡阈值

| 比例 | 判断 | 处理 |
|------|------|------|
| 4:6 ~ 5:5 | 平衡 | 无需处理 |
| 3:7 | 轻度不平衡 | 调整评估指标 |
| 2:8 | 中度不平衡 | SMOTE/权重 |
| 1:9 | 严重不平衡 | 特殊算法 |

---

## 🎯 实战案例

### 案例1：电商用户流失预测

```python
# 数据诊断发现的问题
issues = [
    "缺失值: 用户年龄缺失30%",
    "异常值: 购买金额存在负值",
    "不平衡: 流失用户仅占5%",
    "相关性: 浏览次数与点击次数高度相关(r=0.95)"
]

# 处理方案
solutions = {
    "缺失值": "使用KNN填充年龄（基于购买行为相似度）",
    "异常值": "负值购买金额为数据错误，删除该记录",
    "不平衡": "使用SMOTE过采样+类权重调整",
    "相关性": "删除点击次数，保留浏览次数"
}
```

### 案例2：房价预测

```python
# 数据诊断发现的问题
issues = [
    "右偏分布: 房价严重右偏(skewness=2.1)",
    "异常值: 存在极端豪宅价格",
    "多重共线性: 建筑面积与房间数高度相关"
]

# 处理方案
solutions = {
    "右偏分布": "对房价取对数: log(price)",
    "异常值": "保留极端值（真实数据），使用RobustScaler标准化",
    "多重共线性": "使用Ridge回归添加正则化"
}
```

---

## ✅ 决策树使用检查清单

- [ ] 每个步骤都仔细执行了吗？
- [ ] 记录了所有发现的问题吗？
- [ ] 为每个问题制定了处理方案吗？
- [ ] 考虑了业务背景和数据来源吗？
- [ ] 生成了完整的诊断报告吗？
- [ ] 向团队/导师汇报了诊断结果吗？

---

**下一步：** 查看 [诊断检查清单](diagnosis_checklist.md)，确保不遗漏任何关键检查项！
