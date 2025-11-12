"""
Matplotlib中文字体配置
在每个notebook开头导入这个模块即可解决中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

def setup_chinese_font():
    """
    配置matplotlib支持中文显示
    """
    # macOS系统中文字体配置
    fonts = [
        'Arial Unicode MS',  # macOS通用
        'PingFang SC',       # macOS系统字体
        'STHeiti',           # 华文黑体
        'Heiti TC',          # 黑体-繁
        'SimHei',            # 黑体
    ]

    # 设置字体
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 设置其他参数
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100

    print('✓ 中文字体配置成功')
    print(f'✓ 当前字体: {plt.rcParams["font.sans-serif"][0]}')

# 自动执行配置
setup_chinese_font()
