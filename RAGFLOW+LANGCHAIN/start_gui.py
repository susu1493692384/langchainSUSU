#!/usr/bin/env python3
"""
RAGFlow GUI 快速启动脚本
自动检查依赖并启动GUI
"""

import sys
import os
import subprocess
import importlib

def check_dependencies():
    """检查并安装必要的依赖"""
    required_packages = [
        'tkinter',
        'langchain',
        'requests',
        'python-dotenv',
        'Pillow'
    ]

    missing_packages = []
    optional_packages = ['faiss-cpu', 'chromadb']  # 可选的向量数据库

    print("🔍 检查依赖包...")

    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'Pillow':
                import PIL
            else:
                importlib.import_module(package)
            print(f"✅ {package} - 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 未安装")

    # 检查可选包
    for package in optional_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package} - 已安装")
        except ImportError:
            print(f"⚠️ {package} - 未安装 (可选)")

    if missing_packages:
        print(f"\n❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        response = input("是否自动安装缺少的依赖? (y/n): ").lower().strip()

        if response in ['y', 'yes', '是']:
            print("📦 正在安装缺少的依赖包...")
            for package in missing_packages:
                try:
                    install_name = {
                        'Pillow': 'Pillow',
                        'python-dotenv': 'python-dotenv'
                    }.get(package, package)

                    subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
                    print(f"✅ 成功安装 {package}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ 安装 {package} 失败: {e}")
                    return False
        else:
            print("⚠️ 请手动安装缺少的依赖包后再运行GUI")
            return False

    print("✅ 依赖检查完成!")
    return True

def check_ragflow_integration():
    """检查RAGFlow集成模块"""
    print("\n🔍 检查RAGFlow集成模块...")

    integration_file = "ragflow_langchain_integration.py"
    if os.path.exists(integration_file):
        print(f"✅ {integration_file} - 存在")
        return True
    else:
        print(f"❌ {integration_file} - 不存在")
        print("请确保 ragflow_langchain_integration.py 文件在当前目录下")
        return False

def check_env_file():
    """检查环境变量配置"""
    print("\n🔍 检查环境变量配置...")

    if os.path.exists(".env"):
        print("✅ .env 文件 - 存在")
        return True
    elif os.path.exists(".env.example"):
        print("⚠️ .env 文件 - 不存在，但发现 .env.example")
        response = input("是否从 .env.example 创建 .env 文件? (y/n): ").lower().strip()

        if response in ['y', 'yes', '是']:
            try:
                with open(".env.example", 'r', encoding='utf-8') as src:
                    content = src.read()
                with open(".env", 'w', encoding='utf-8') as dst:
                    dst.write(content)
                print("✅ 已创建 .env 文件，请编辑其中的配置")
                return True
            except Exception as e:
                print(f"❌ 创建 .env 文件失败: {e}")
                return False
        else:
            print("⚠️ 请创建 .env 文件并配置相关参数")
            return False
    else:
        print("⚠️ .env 文件 - 不存在")
        response = input("是否创建示例 .env 文件? (y/n): ").lower().strip()

        if response in ['y', 'yes', '是']:
            try:
                example_env = """# RAGFlow API 配置
RAGFLOW_API_URL=http://localhost:9380
RAGFLOW_API_KEY=your_ragflow_api_key_here

# LLM 配置
# OpenAI 配置
OPENAI_API_KEY=your_openai_api_key_here

# GLM 配置
GLM_API_KEY=your_glm_api_key_here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-4.5

# 嵌入模型配置
EMBEDDING_MODEL=embedding-2

# GUI 配置
GUI_THEME=modern
GUI_FONT_SIZE=10
"""
                with open(".env", 'w', encoding='utf-8') as f:
                    f.write(example_env)
                print("✅ 已创建 .env 文件，请编辑其中的配置")
                return True
            except Exception as e:
                print(f"❌ 创建 .env 文件失败: {e}")
                return False
        else:
            print("⚠️ 请创建 .env 文件并配置相关参数")
            return False

def start_gui():
    """启动GUI"""
    print("\n🚀 启动RAGFlow GUI...")
    try:
        from ragflow_modern_gui import main
        main()
    except Exception as e:
        print(f"❌ 启动GUI失败: {e}")
        return False
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("🚀 RAGFlow + LangChain GUI 启动器")
    print("=" * 50)

    # 检查依赖
    if not check_dependencies():
        input("\n按回车键退出...")
        return

    # 检查集成模块
    if not check_ragflow_integration():
        input("\n按回车键退出...")
        return

    # 检查环境变量
    if not check_env_file():
        input("\n按回车键退出...")
        return

    print("\n" + "=" * 50)
    print("✅ 所有检查完成，准备启动GUI")
    print("=" * 50)

    # 启动GUI
    start_gui()

if __name__ == "__main__":
    main()