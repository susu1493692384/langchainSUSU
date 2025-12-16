#!/usr/bin/env python3
"""
LangChain 快速入门脚本
这个脚本可以帮助您快速验证环境配置并体验LangChain的基本功能
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def check_environment():
    """检查环境配置"""
    print("检查环境配置...\n")

    # 检查Python版本
    print(f"Python版本: {sys.version}")

    # 检查必要的包
    required_packages = [
        'langchain',
        'langchain_openai',
        'langchain_community',
        'python-dotenv'
    ]

    print("\n检查包安装状态:")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"[OK] {package} - 已安装")
        except ImportError:
            print(f"[ERROR] {package} - 未安装")
            print(f"请运行: pip install {package}")
            return False

    # 检查环境变量
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("\n[WARNING] 未找到OPENAI_API_KEY")
        print("请:")
        print("1. 复制 .env.example 为 .env")
        print("2. 在 .env 文件中添加您的OpenAI API密钥")
        return False
    else:
        print("[OK] OPENAI_API_KEY - 已配置")

    print("\n[OK] 环境检查通过！")
    return True

def quick_hello_world():
    """LangChain Hello World"""
    print("\n🤖 LangChain Hello World 示例\n")

    try:
        # 创建LLM实例
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=50
        )

        # 创建提示模板
        prompt = PromptTemplate(
            input_variables=["name"],
            template="你好 {name}！我是LangChain助手，很高兴认识你！请用一句话介绍一下你自己。"
        )

        # 创建链
        chain = prompt | llm | StrOutputParser()

        # 执行链
        result = chain.invoke({"name": "新朋友"})

        print(f"AI助手: {result}")

    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False

    return True

def simple_qa_example():
    """简单问答示例"""
    print("\n💬 简单问答示例\n")

    try:
        # 创建LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

        # 创建问答提示模板
        qa_template = PromptTemplate(
            input_variables=["question"],
            template="请用简洁的中文回答这个问题：{question}"
        )

        # 创建链
        qa_chain = qa_template | llm | StrOutputParser()

        # 测试问题
        questions = [
            "什么是LangChain？",
            "LangChain能做什么？"
        ]

        for question in questions:
            print(f"问题: {question}")
            answer = qa_chain.invoke({"question": question})
            print(f"回答: {answer}\n")

    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False

    return True

def creative_example():
    """创意示例"""
    print("\n✨ 创意示例 - AI诗人\n")

    try:
        # 创建LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

        # 创建诗歌提示模板
        poem_template = PromptTemplate(
            input_variables=["topic", "style"],
            template="""请写一首关于{topic}的{style}风格诗歌，要求：
1. 不少于4行
2. 语言优美
3. 富有想象力

诗歌：
"""
        )

        # 创建链
        poem_chain = poem_template | llm | StrOutputParser()

        # 生成诗歌
        topics = [
            {"topic": "科技", "style": "现代"},
            {"topic": "星空", "style": "古典"}
        ]

        for item in topics:
            print(f"主题: {item['topic']} | 风格: {item['style']}")
            poem = poem_chain.invoke(item)
            print(poem)
            print("-" * 40)

    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False

    return True

def interactive_test():
    """交互式测试"""
    print("\n🎮 交互式测试")
    print("您可以输入任何问题，AI助手会尽力回答。")
    print("输入 'quit' 或 'exit' 退出测试。\n")

    try:
        # 创建LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

        # 创建简单的提示模板
        simple_prompt = PromptTemplate(
            input_variables=["question"],
            template="{question}"
        )

        # 创建链
        chat_chain = simple_prompt | llm | StrOutputParser()

        while True:
            try:
                user_input = input("您: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break

                if not user_input:
                    continue

                print("思考中...")
                response = chat_chain.invoke({"question": user_input})
                print(f"AI助手: {response}\n")

            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 处理输入时出错: {e}")

    except Exception as e:
        print(f"❌ 初始化交互测试时出错: {e}")
        return False

    return True

def show_next_steps():
    """显示后续步骤"""
    print("\n🎉 恭喜！您已经成功运行了第一个LangChain应用！\n")
    print("📚 接下来建议您学习：")
    print("1. 运行 python 01_basic_llm.py - 学习基础LLM调用")
    print("2. 运行 python 02_chains.py - 学习链式调用")
    print("3. 运行 python 03_templates_memory.py - 学习模板和记忆")
    print("4. 运行 python 04_vector_storage.py - 学习向量存储和检索")
    print("5. 阅读 README.md - 了解完整的学习路径")
    print("\n💡 提示：建议按顺序运行所有示例来全面了解LangChain的功能！")

def main():
    """主函数"""
    print("LangChain 快速入门测试")
    print("=" * 50)

    # 检查环境
    if not check_environment():
        print("\n❌ 环境配置有问题，请检查后重试。")
        return

    # 运行示例
    examples = [
        ("Hello World", quick_hello_world),
        ("简单问答", simple_qa_example),
        ("创意生成", creative_example)
    ]

    for name, func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            if func():
                print(f"✅ {name} 示例执行成功！")
            else:
                print(f"❌ {name} 示例执行失败。")
        except Exception as e:
            print(f"❌ {name} 示例执行出错: {e}")

    # 交互式测试（可选）
    try:
        user_wants_interactive = input("\n是否进行交互式测试？(y/n): ").strip().lower()
        if user_wants_interactive in ['y', 'yes', '是']:
            interactive_test()
    except KeyboardInterrupt:
        print("\n")

    # 显示后续步骤
    show_next_steps()

if __name__ == "__main__":
    main()