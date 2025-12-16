#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能文档问答系统 - 依赖安装脚本
自动安装所需的Python库和依赖项
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package_name, import_name=None):
    """安装Python包"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"🔧 正在安装 {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} 安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {package_name} 安装失败: {e}")
            return False

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    else:
        print(f"✅ Python版本: {sys.version}")
        return True

def check_pip():
    """检查pip是否可用"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        print("✅ pip 可用")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip 不可用")
        return False

def create_env_file():
    """创建.env文件模板"""
    env_content = """# 智能文档问答系统 - 环境变量配置

# API密钥配置
GLM_API_KEY=your_api_key_here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/

# 其他配置
DEBUG=False
LOG_LEVEL=INFO
"""

    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(env_content)
        print("✅ 已创建 .env 配置文件")
        print("⚠️ 请编辑 .env 文件，设置您的API密钥")
    else:
        print("✅ .env 配置文件已存在")

def create_sample_documents():
    """创建示例文档文件"""
    sample_content = """人工智能技术概述

人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。AI系统可以学习、推理、感知、理解语言，并做出决策。

机器学习
机器学习是人工智能的一个子集，它使计算机能够在没有被明确编程的情况下学习和改进。机器学习算法通过分析数据来识别模式，并使用这些模式来对新数据做出预测或决策。

深度学习
深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。深度学习在图像识别、自然语言处理和语音识别等领域取得了重大突破。

自然语言处理
自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。NLP使计算机能够理解、解释和生成人类语言。

计算机视觉
计算机视觉是人工智能的一个分支，使计算机能够从数字图像或视频中获取有意义的信息。计算机视觉系统可以识别物体、检测面孔、分析场景等。

应用领域
人工智能技术在各个领域都有广泛应用，包括医疗诊断、自动驾驶、金融分析、推荐系统、智能助手等。随着技术的发展，AI将继续改变我们的生活方式和工作方式。
"""

    doc_file = Path("local_documents.txt")
    if not doc_file.exists():
        with open(doc_file, "w", encoding="utf-8") as f:
            f.write(sample_content)
        print("✅ 已创建示例文档文件 'local_documents.txt'")
    else:
        print("✅ 示例文档文件已存在")

def main():
    """主安装流程"""
    print("=" * 60)
    print("智能文档问答系统 - 依赖安装器")
    print("=" * 60)
    print()

    # 检查Python版本
    if not check_python_version():
        input("按Enter键退出...")
        return

    # 检查pip
    if not check_pip():
        print("请先安装pip后再运行此脚本")
        input("按Enter键退出...")
        return

    print("\n📦 开始安装依赖包...")
    print("-" * 40)

    # 必需的包及其导入名称映射
    required_packages = [
        ("customtkinter", "customtkinter"),
        ("langchain-openai", "langchain_openai"),
        ("langchain-core", "langchain_core"),
        ("python-dotenv", "dotenv"),
        ("openai", "openai"),
        ("tiktoken", "tiktoken"),
    ]

    # 可选的包
    optional_packages = [
        ("Pillow", "PIL"),  # 图像处理
        ("matplotlib", "matplotlib"),  # 绘图
        ("networkx", "networkx"),  # 图形分析
        ("numpy", "numpy"),  # 数值计算
        ("pandas", "pandas"),  # 数据处理
    ]

    # 安装必需包
    print("\n🔧 安装必需依赖...")
    success_count = 0
    for package, import_name in required_packages:
        if install_package(package, import_name):
            success_count += 1

    print(f"\n必需包安装完成: {success_count}/{len(required_packages)}")

    # 安装可选包
    print("\n🎯 安装可选依赖...")
    optional_success = 0
    for package, import_name in optional_packages:
        if install_package(package, import_name):
            optional_success += 1

    print(f"\n可选包安装完成: {optional_success}/{len(optional_packages)}")

    # 创建配置文件
    print("\n⚙️ 创建配置文件...")
    create_env_file()
    create_sample_documents()

    print("\n" + "=" * 60)
    print("🎉 安装完成!")
    print("=" * 60)
    print()

    if success_count == len(required_packages):
        print("✅ 所有必需依赖安装成功!")
        print()
        print("📋 下一步:")
        print("1. 编辑 .env 文件，设置您的API密钥")
        print("2. 运行 GUI 应用: python document_qa_gui.py")
        print("3. 或运行命令行版本: python 文档问答系统.py")
        print()
        if optional_success < len(optional_packages):
            print(f"⚠️ 部分可选功能不可用 ({optional_success}/{len(optional_packages)})")
            print("运行以下命令安装完整功能:")
            print("python install_requirements.py --optional")
    else:
        print("❌ 部分必需依赖安装失败!")
        print("请检查网络连接或手动安装失败的包")

    print("\n💡 使用帮助:")
    print("- 问题描述越具体，回答越准确")
    print("- 支持多轮对话，可以引用之前讨论的内容")
    print("- 可以通过右侧监控面板查看系统状态")
    print("- 支持 Ctrl+Enter 快捷发送消息")

    input("\n按Enter键退出...")

if __name__ == "__main__":
    # 支持命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--optional":
        print("🎯 仅安装可选依赖...")
        # 只安装可选包的逻辑
        optional_packages = [
            ("Pillow", "PIL"),
            ("matplotlib", "matplotlib"),
            ("networkx", "networkx"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
        ]
        for package, import_name in optional_packages:
            install_package(package, import_name)
    else:
        main()