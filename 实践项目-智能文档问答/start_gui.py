#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动GUI应用的脚本
包含错误处理和依赖检查
"""

import sys
import os

def check_dependencies():
    """检查依赖是否安装"""
    required_modules = [
        ('customtkinter', 'customtkinter'),
        ('langchain_openai', 'langchain_openai'),
        ('langchain_core', 'langchain_core'),
        ('dotenv', 'dotenv'),
        ('openai', 'openai'),
    ]

    missing_modules = []

    for module_name, import_name in required_modules:
        try:
            __import__(import_name)
            print(f"✅ {module_name} 已安装")
        except ImportError:
            print(f"❌ {module_name} 未安装")
            missing_modules.append(module_name)

    if missing_modules:
        print(f"\n请安装缺失的模块: {', '.join(missing_modules)}")
        print("运行: pip install " + " ".join(missing_modules))
        return False

    return True

def check_api_config():
    """检查API配置"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GLM_API_KEY")
    base_url = os.getenv("GLM_BASE_URL")

    if not api_key:
        print("⚠️ 警告: 未设置GLM_API_KEY，请检查.env文件")
        return False

    if not base_url:
        print("⚠️ 警告: 未设置GLM_BASE_URL，请检查.env文件")
        return False

    print("✅ API配置检查通过")
    return True

def main():
    """主启动函数"""
    print("=" * 50)
    print("智能文档问答系统 GUI启动器")
    print("=" * 50)

    # 检查Python版本
    if sys.version_info < (3, 8):
        print(f"❌ Python版本过低: {sys.version}")
        print("需要Python 3.8或更高版本")
        input("按Enter键退出...")
        return

    print(f"✅ Python版本: {sys.version}")

    # 检查依赖
    print("\n检查依赖...")
    if not check_dependencies():
        input("按Enter键退出...")
        return

    # 检查API配置
    print("\n检查API配置...")
    if not check_api_config():
        print("\n提示: 即使API配置缺失，GUI仍会启动，但问答功能不可用")
        response = input("是否继续启动? (y/n): ").lower()
        if response != 'y':
            return

    # 启动GUI
    print("\n启动GUI应用...")
    try:
        # 导入并运行GUI
        import customtkinter as ctk
        from document_qa_gui import DocumentQAGUI

        print("✅ GUI模块导入成功")
        print("正在启动界面...")

        # 创建并运行应用
        app = DocumentQAGUI()
        app.run()

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有文件都在同一目录下")
        input("按Enter键退出...")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        input("按Enter键退出...")

if __name__ == "__main__":
    main()