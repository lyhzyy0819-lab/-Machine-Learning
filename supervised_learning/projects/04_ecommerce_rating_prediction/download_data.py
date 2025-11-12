"""
数据下载脚本
自动从Kaggle下载Amazon销售数据集
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

import config


def check_kaggle_installation():
    """
    检查Kaggle CLI是否已安装

    Returns:
        bool: 是否已安装
    """
    try:
        result = subprocess.run(['kaggle', '--version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            print(f"✓ Kaggle CLI 已安装: {result.stdout.strip()}")
            return True
        else:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def print_kaggle_setup_instructions():
    """
    打印Kaggle API设置说明
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "Kaggle API 设置指南")
    print("=" * 80)
    print("\n📋 步骤1: 安装 Kaggle CLI")
    print("-" * 80)
    print("运行以下命令安装:")
    print("  pip install kaggle")

    print("\n📋 步骤2: 获取 Kaggle API 凭证")
    print("-" * 80)
    print("1. 登录到 Kaggle 网站: https://www.kaggle.com")
    print("2. 点击右上角头像 → Account")
    print("3. 滚动到 'API' 部分")
    print("4. 点击 'Create New API Token'")
    print("5. 会下载一个 kaggle.json 文件")

    print("\n📋 步骤3: 配置 API 凭证")
    print("-" * 80)

    if os.name == 'nt':  # Windows
        kaggle_dir = Path.home() / '.kaggle'
        print(f"将 kaggle.json 文件放到: {kaggle_dir}")
        print("\n或者在命令行中运行:")
        print(f"  mkdir {kaggle_dir}")
        print(f"  move kaggle.json {kaggle_dir}\\")
    else:  # Linux/Mac
        kaggle_dir = Path.home() / '.kaggle'
        print(f"将 kaggle.json 文件放到: {kaggle_dir}")
        print("\n在终端中运行:")
        print(f"  mkdir -p {kaggle_dir}")
        print(f"  mv ~/Downloads/kaggle.json {kaggle_dir}/")
        print(f"  chmod 600 {kaggle_dir}/kaggle.json")

    print("\n📋 步骤4: 验证设置")
    print("-" * 80)
    print("运行以下命令验证:")
    print("  kaggle datasets list")
    print("\n如果显示数据集列表，则配置成功！")

    print("\n📋 步骤5: 再次运行此脚本")
    print("-" * 80)
    print("  python download_data.py")

    print("\n" + "=" * 80)


def check_kaggle_credentials():
    """
    检查Kaggle API凭证是否已配置

    Returns:
        bool: 是否已配置
    """
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

    if kaggle_json.exists():
        print(f"✓ Kaggle API 凭证已配置: {kaggle_json}")
        return True
    else:
        print(f"✗ Kaggle API 凭证未找到")
        print(f"  期望位置: {kaggle_json}")
        return False


def download_dataset():
    """
    从Kaggle下载数据集

    Returns:
        bool: 下载是否成功
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "下载 Amazon 销售数据集")
    print("=" * 80)

    # 确保数据目录存在
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n数据集: {config.KAGGLE_DATASET}")
    print(f"保存到: {config.RAW_DATA_DIR}")

    try:
        # 下载数据集
        print("\n正在下载数据集...")
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', config.KAGGLE_DATASET,
            '-p', str(config.RAW_DATA_DIR),
            '--unzip'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        if result.returncode == 0:
            print("✓ 数据集下载成功!")
            print(result.stdout)
            return True
        else:
            print("✗ 数据集下载失败!")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("✗ 下载超时! 请检查网络连接或手动下载")
        return False
    except Exception as e:
        print(f"✗ 下载出错: {str(e)}")
        return False


def verify_downloaded_data():
    """
    验证下载的数据文件

    Returns:
        bool: 数据文件是否存在且有效
    """
    print("\n" + "=" * 80)
    print(" " * 30 + "验证数据文件")
    print("=" * 80)

    if not config.RAW_DATA_FILE.exists():
        print(f"✗ 数据文件不存在: {config.RAW_DATA_FILE}")

        # 列出下载目录中的所有文件
        print(f"\n下载目录中的文件:")
        if config.RAW_DATA_DIR.exists():
            files = list(config.RAW_DATA_DIR.glob('*'))
            if files:
                for f in files:
                    print(f"  - {f.name}")
            else:
                print("  (空)")

        return False

    # 检查文件大小
    file_size_mb = config.RAW_DATA_FILE.stat().st_size / (1024 * 1024)
    print(f"✓ 数据文件存在: {config.RAW_DATA_FILE}")
    print(f"  文件大小: {file_size_mb:.2f} MB")

    # 尝试读取前几行验证格式
    try:
        import pandas as pd
        df_sample = pd.read_csv(config.RAW_DATA_FILE, nrows=5)
        print(f"  数据列数: {df_sample.shape[1]}")
        print(f"  列名: {', '.join(df_sample.columns[:5])}...")
        print("\n✓ 数据文件格式验证通过!")
        return True
    except Exception as e:
        print(f"✗ 数据文件格式验证失败: {str(e)}")
        return False


def print_manual_download_instructions():
    """
    打印手动下载说明（如果自动下载失败）
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "手动下载说明")
    print("=" * 80)

    print("\n如果自动下载失败，请手动下载:")
    print("\n步骤1: 访问Kaggle数据集页面")
    print(f"  https://www.kaggle.com/datasets/{config.KAGGLE_DATASET}")

    print("\n步骤2: 点击 'Download' 按钮")
    print("  （可能需要先登录并接受竞赛规则）")

    print("\n步骤3: 解压下载的文件")
    print("  找到 'amazon.csv' 文件")

    print("\n步骤4: 移动文件到项目目录")
    print(f"  目标位置: {config.RAW_DATA_FILE}")

    if os.name == 'nt':  # Windows
        print(f"\n在命令行中运行:")
        print(f"  move amazon.csv \"{config.RAW_DATA_FILE}\"")
    else:  # Linux/Mac
        print(f"\n在终端中运行:")
        print(f"  mv amazon.csv \"{config.RAW_DATA_FILE}\"")

    print("\n步骤5: 验证文件")
    print("  python download_data.py --verify")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='下载Amazon销售数据集')
    parser.add_argument('--verify', action='store_true',
                       help='仅验证数据文件是否存在')
    parser.add_argument('--force', action='store_true',
                       help='强制重新下载（即使文件已存在）')

    args = parser.parse_args()

    print("=" * 80)
    print(" " * 20 + "Amazon 销售数据集下载工具")
    print("=" * 80)

    # 如果只是验证
    if args.verify:
        verify_downloaded_data()
        return

    # 检查文件是否已存在
    if config.RAW_DATA_FILE.exists() and not args.force:
        print("\n✓ 数据文件已存在!")
        verify_downloaded_data()
        print("\n提示: 如需重新下载，使用 --force 参数")
        return

    # 检查Kaggle CLI
    print("\n检查环境...")
    if not check_kaggle_installation():
        print("✗ Kaggle CLI 未安装")
        print_kaggle_setup_instructions()
        return

    # 检查API凭证
    if not check_kaggle_credentials():
        print_kaggle_setup_instructions()
        return

    # 下载数据集
    if download_dataset():
        # 验证下载的数据
        if verify_downloaded_data():
            print("\n" + "=" * 80)
            print("✓ 数据下载完成！可以开始训练模型了")
            print("=" * 80)
            print("\n下一步:")
            print("  # 快速测试")
            print("  python main.py --sample --quick")
            print("\n  # 完整训练")
            print("  python main.py")
        else:
            print("\n数据文件验证失败，请检查下载的文件")
    else:
        print_manual_download_instructions()


if __name__ == '__main__':
    main()
