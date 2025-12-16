#!/usr/bin/env python3
"""
LangChain 进阶学习快速启动脚本
一键运行所有进阶示例
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """打印欢迎横幅"""
    print("=" * 70)
    print("🚀 LangChain 进阶学习快速启动器 🚀")
    print("=" * 70)
    print("本脚本将带您体验LangChain的进阶功能")
    print("包括：智能体、评估调试、生产部署")
    print("=" * 70)

def check_environment():
    """检查环境配置"""
    print("\n📋 检查环境配置...")

    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    else:
        print(f"✅ Python版本: {sys.version.split()[0]}")

    # 检查必要文件
    required_files = [
        ".env",
        "05_agents_tools.py",
        "06_evaluation_debugging.py",
        "07_production_deployment.py"
    ]

    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ 缺少文件: {file}")
            return False
        else:
            print(f"✅ 找到文件: {file}")

    # 检查环境变量
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  警告: 未找到ANTHROPIC_API_KEY")
        print("   请在.env文件中配置您的API密钥")
    else:
        print("✅ API密钥配置正确")

    return True

def install_dependencies():
    """安装必要依赖"""
    print("\n📦 检查并安装依赖...")

    dependencies = [
        "langchain",
        "langchain-openai",
        "langchain-community",
        "fastapi",
        "uvicorn[standard]",
        "redis",
        "aioredis",
        "python-dotenv",
        "pydantic"
    ]

    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep} 已安装")
        except ImportError:
            print(f"📥 安装 {dep}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✅ {dep} 安装成功")
            except subprocess.CalledProcessError:
                print(f"❌ {dep} 安装失败")
                return False

    return True

def run_example(example_name: str, file_path: str):
    """运行示例"""
    print(f"\n{'='*20} {example_name} {'='*20}")

    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=120  # 2分钟超时
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {example_name} 运行成功 (耗时: {execution_time:.1f}秒)")
            if result.stdout:
                print("输出:")
                print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"❌ {example_name} 运行失败")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)

    except subprocess.TimeoutExpired:
        print(f"⏰ {example_name} 运行超时")
    except Exception as e:
        print(f"❌ 运行 {example_name} 时出错: {e}")

def show_menu():
    """显示选择菜单"""
    print("\n" + "="*70)
    print("📚 选择要运行的进阶示例:")
    print("="*70)
    print("1. 🤖 智能体和工具 (05_agents_tools.py)")
    print("2. 📊 评估和调试 (06_evaluation_debugging.py)")
    print("3. 🌐 生产部署 (07_production_deployment.py)")
    print("4. 🔄 运行所有示例")
    print("5. 📋 查看学习指南")
    print("6. ⚙️  环境检查")
    print("0. 🚪 退出")
    print("="*70)

def main():
    """主函数"""
    print_banner()

    if not check_environment():
        print("\n❌ 环境检查失败，请解决上述问题后重试")
        return

    if not install_dependencies():
        print("\n❌ 依赖安装失败，请检查网络连接和权限")
        return

    while True:
        show_menu()

        try:
            choice = input("\n请选择 (0-6): ").strip()

            if choice == "0":
                print("\n👋 感谢使用LangChain进阶学习工具！")
                break
            elif choice == "1":
                run_example("智能体和工具", "05_agents_tools.py")
            elif choice == "2":
                run_example("评估和调试", "06_evaluation_debugging.py")
            elif choice == "3":
                run_example("生产部署", "07_production_deployment.py")
            elif choice == "4":
                print("\n🚀 开始运行所有示例...")
                examples = [
                    ("智能体和工具", "05_agents_tools.py"),
                    ("评估和调试", "06_evaluation_debugging.py"),
                    ("生产部署", "07_production_deployment.py")
                ]

                for name, file in examples:
                    run_example(name, file)
                    time.sleep(2)  # 间隔2秒

                print("\n🎉 所有示例运行完成！")
            elif choice == "5":
                show_learning_guide()
            elif choice == "6":
                check_environment()
            else:
                print("❌ 无效选择，请输入0-6之间的数字")

        except KeyboardInterrupt:
            print("\n\n👋 感谢使用，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

def show_learning_guide():
    """显示学习指南"""
    print("\n" + "="*70)
    print("📖 LangChain 进阶学习指南")
    print("="*70)

    print("\n🎯 学习路径:")
    print("1. 智能体和工具 → 创建能使用外部工具的AI")
    print("2. 评估和调试 → 确保AI应用的性能和质量")
    print("3. 生产部署 → 将应用部署到生产环境")

    print("\n📚 推荐学习顺序:")
    print("• 第1周: 掌握智能体开发")
    print("• 第2周: 学习质量保证技术")
    print("• 第3周: 实践生产部署")

    print("\n🛠️ 实践项目建议:")
    print("• 企业知识库问答系统")
    print("• 智能客服系统")
    print("• 代码助手应用")

    print("\n📖 详细指南请查看: ADVANCED_GUIDE.md")

    print("\n🔗 有用资源:")
    print("• 官方文档: https://python.langchain.com/")
    print("• API参考: https://api.python.langchain.com/")
    print("• 社区论坛: https://github.com/langchain-ai/langchain")

if __name__ == "__main__":
    main()