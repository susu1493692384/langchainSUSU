#!/usr/bin/env python3
"""
LangChain 基础示例 - 简单LLM调用
这是LangChain的入门示例，展示如何使用LangChain调用大语言模型
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()
    
def basic_llm_example():
    """基础LLM调用示例"""
    print("=== LangChain 基础LLM调用示例 ===\n")

    # 创建ChatAnthropic实例
    # 注意：您需要在.env文件中设置GLM_API_KEY
    llm = ChatOpenAI(
        model="glm-4.5",
        temperature=0.1,
        max_tokens=100,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建消息
    message = HumanMessage(content="你好！请用中文介绍一下你自己。")


    # 调用模型
    try:
        response = llm.invoke([message])
        print(f"用户: {message.content}")
        print(f"AI助手: {response.content}\n")
    except Exception as e:
        print(f"调用模型时出错: {e}")
        print("请确保您已经设置了有效的GLM API密钥。")

def multiple_questions_example():
    """多个问题示例"""
    print("=== 多个问题示例 ===\n")

    # 创建ChatAnthropic实例
    llm = ChatOpenAI(
        name ="glm-4",  # 智谱AI支持的模型
        verbose= True,
        temperature=0.7,  # 控制输出随机性，0-1之间
        max_completion_tokens= 150,    # 限制输出长度
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 准备多个问题
    questions = [
        "什么是LangChain？",
        "LangChain有哪些主要功能？",
        "如何开始使用LangChain？"
    ]

    for i, question in enumerate(questions, 1):
        message = HumanMessage(content=question)
        try:
            response = llm.invoke([message])
            print(f"问题 {i}: {question}")
            print(f"回答 {i}: {response.content}\n")
        except Exception as e:
            print(f"处理问题 {i} 时出错: {e}")

def custom_parameters_example():
    """自定义参数示例"""
    print("=== 自定义参数示例 ===\n")

    # 同一个问题，不同的temperature设置
    question = "请用创意的方式描述一下编程的乐趣。"

    # 低temperature - 更确定性的回答
    llm_deterministic = ChatOpenAI(
        model="glm-4",
        temperature=0.1,
        max_tokens=100,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 高temperature - 更有创意的回答
    llm_creative = ChatOpenAI(
        model="glm-4",
        temperature=1.0,
        max_tokens=100,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    message = HumanMessage(content=question)

    try:
        print("问题:", question)
        print("\n低temperature (0.1) - 确定性回答:")
        response1 = llm_deterministic.invoke([message])
        print(response1.content)

        print("\n高temperature (1.0) - 创意回答:")
        response2 = llm_creative.invoke([message])
        print(response2.content)

    except Exception as e:
        print(f"调用模型时出错: {e}")

if __name__ == "__main__":
    import sys
    import io
    # 设置UTF-8编码输出
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("🤖 欢迎来到LangChain学习世界！\n")

    # 运行基础示例
    basic_llm_example()

    print("\n" + "="*50 + "\n")

    # 运行多个问题示例
    #multiple_questions_example()

    print("\n" + "="*50 + "\n")

    # 运行自定义参数示例
    #custom_parameters_example()

    print("\n✨ 示例完成！您已经学会了如何在LangChain中进行基础的LLM调用。")