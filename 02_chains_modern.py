#!/usr/bin/env python3
"""
LangChain 现代链式调用示例 (使用LCEL)
展示如何使用最新的LangChain Expression Language (LCEL) 构建链
 02_chains_modern.py 包含了7个现代化的链示例:

  1. 基础LCEL链 - 使用 | 管道操作符
  2. 并行链 - 同时执行多个任务
  3. 条件链 - 根据条件选择不同路径
  4. 顺序链 - 步骤间的数据传递
  5. JSON输出链 - 结构化数据输出
  6. 自定义函数链 - 集成自定义Python函数
  7. 聊天模板链 - 使用ChatPromptTemplate
这是一个现代化的版本，解决了原版本中的导入问题
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch
)
from langchain_core.messages import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()

def basic_lcel_chain():
    """基础LCEL链 - 使用管道操作符"""
    print("=== 基础LCEL链示例 ===\n")

    # 创建LLM (使用智谱AI配置)
    llm = ChatOpenAI(
        model="glm-4.6",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建提示模板
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="请用中文写一段关于{topic}的简短介绍，大约50个字。"
    )

    # 创建输出解析器
    output_parser = StrOutputParser()

#使用LCEL语法创建链：prompt | llm | output_parser
#LCEL链中，数据按照从左到右的顺序依次通过每个组件。
#数据流动规则：
#单向流动：数据只能从左向右流动
#传递格式：前一个组件的输出作为下一个组件的输入
#类型匹配：相邻组件的输入输出类型必须兼容
#管道操作：| 操作符表示数据的顺序传递
    chain = prompt | llm | output_parser

    # 测试不同的主题
    topics = ["人工智能", "机器学习", "深度学习"]

    for topic in topics:
        try:
            result = chain.invoke({"topic": topic})
            print(f"主题: {topic}")
            print(f"介绍: {result}\n")
        except Exception as e:
            print(f"处理主题 '{topic}' 时出错: {e}")

def parallel_chain_example():
    """并行链示例 - 同时执行多个任务"""
    print("=== 并行链示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 定义多个提示模板
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="总结以下文本的要点：\n{text}\n\n要点："
    )

    translation_prompt = PromptTemplate(
        input_variables=["text"],
        template="将以下文本翻译成英文：\n{text}\n\nTranslation："
    )

    sentiment_prompt = PromptTemplate(
        input_variables=["text"],
        template="分析以下文本的情感倾向（积极/消极/中性）：\n{text}\n\n情感："
    )

    # 创建并行链
    # RunnableParallel 是LangChain中的并行任务调度器
    #  - 输入共享：所有处理链接收相同的输入数据
    #  - 并行执行：同时运行多个独立的处理任务
    #  - 结果聚合：将结果统一组织到一个字典中
    parallel_chain = RunnableParallel({
        "summary": summary_prompt | llm | StrOutputParser(),
        "translation": translation_prompt | llm | StrOutputParser(),
        "sentiment": sentiment_prompt | llm | StrOutputParser()
    })

    test_text = "今天天气真好，我心情很愉快！"

    try:
        results = parallel_chain.invoke({"text": test_text})
        print(f"原始文本: {test_text}\n")
        print("并行处理结果:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        print()
    except Exception as e:
        print(f"并行处理时出错: {e}")

def conditional_chain_example():
    """条件链示例 - 根据条件选择不同的处理路径"""
    print("=== 条件链示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4.6",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 定义不同类型的提示模板
    technical_prompt = PromptTemplate(
        input_variables=["question"],
        template="请用技术性的语言回答这个问题：{question}"
    )

    simple_prompt = PromptTemplate(
        input_variables=["question"],
        template="请用简单易懂的语言回答这个问题：{question}"
    )

    # 条件函数 - 判断问题是否是技术性的
    def is_technical_question(question):
        tech_keywords = ["算法", "编程", "代码", "数据结构", "API", "数据库", "网络"]
        return any(keyword in question for keyword in tech_keywords)
#   1. tech_keywords 列表

#   tech_keywords = ["算法", "编程", "代码", "数据结构",
#   "API", "数据库", "网络"]
#   - 定义了技术领域的关键词
#   - 包含常见的编程和技术概念

#   2. any() 函数

#   - 作用：检查可迭代对象中是否有任何一个元素为True
#   - 返回值：True（如果至少有一个匹配）或
#   False（如果都没有匹配）

#   3. 列表推导式 + any()

#   any(keyword in question for keyword in tech_keywords)        

#   这相当于：
#   for keyword in tech_keywords:           # 
#   遍历每个技术关键词
#       if keyword in question:             # 
#   检查是否出现在问题中
#           return True                     # 
#   找到任何一个就返回True
#   return False                            # 
#   都没找到返回False

    # 创建条件链
    conditional_chain = (
        RunnablePassthrough.assign(
            is_technical=lambda x: is_technical_question(x["question"])
        )
        | RunnableBranch(
            (lambda x: x["is_technical"], technical_prompt | llm | StrOutputParser()),
            (lambda x: not x["is_technical"], simple_prompt | llm | StrOutputParser()),
        )
    )

    questions = [
        {"question": "什么是神经网络？"},
        {"question": "今天天气怎么样？"},
        {"question": "如何实现快速排序算法？"},
        {"question": "你最喜欢的颜色是什么？"}
    ]

    for item in questions:
        try:
            question = item["question"]
            print(f"问题: {question}")

            # 判断是否为技术问题
            is_tech = is_technical_question(question)
            print(f"类型: {'技术性问题' if is_tech else '一般性问题'}")

            # 获取回答
            answer = conditional_chain.invoke({"question": question})
            print(f"回答: {answer}\n")

        except Exception as e:
            print(f"处理问题时出错: {e}")

def sequential_chain_example():
    """顺序链示例 - 前一个步骤的输出作为后一个步骤的输入"""
    print("=== 顺序链示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 第一步：故事概要生成
    story_prompt = PromptTemplate(
        input_variables=["character"],
        template="创建一个关于{character}的故事概要，大约100字。"
    )

    # 第二步：基于概要生成故事标题
    title_prompt = PromptTemplate(
        input_variables=["story_summary"],
        template="基于以下故事概要，创作一个吸引人的标题：\n{story_summary}\n\n标题："
    )

    # 第三步：生成故事结局
    ending_prompt = PromptTemplate(
        input_variables=["title", "story_summary"],
        template="故事标题：{title}\n故事概要：{story_summary}\n\n请为这个故事写一个精彩的结尾，大约50字："
    )

    # 创建顺序链
    story_chain = (
        {"character": RunnablePassthrough()}
        | story_prompt
        | llm
        | StrOutputParser()
        | (lambda story: {"story_summary": story})
    )

    full_chain = (
        story_chain
        | RunnablePassthrough.assign(
            title=lambda x: (title_prompt | llm | StrOutputParser()).invoke({"story_summary": x["story_summary"]})
        )
        | RunnablePassthrough.assign(
            ending=lambda x: (ending_prompt | llm | StrOutputParser()).invoke({
                "title": x["title"],
                "story_summary": x["story_summary"]
            })
        )
    )

    character = "一个会说话的猫咪"

    try:
        result = full_chain.invoke(character)

        print(f"角色: {character}\n")
        print(f"故事概要: {result['story_summary']}\n")
        print(f"故事标题: {result['title']}\n")
        print(f"故事结尾: {result['ending']}\n")

        print("完整故事:")
        print(f"标题: {result['title']}")
        print(f"概要: {result['story_summary']}")
        print(f"结尾: {result['ending']}\n")

    except Exception as e:
        print(f"生成故事时出错: {e}")

def json_output_chain_example():
    """JSON输出链示例 - 结构化输出"""
    print("=== JSON输出链示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.3,  # 降低温度以获得更稳定的JSON输出
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建JSON输出解析器
    json_parser = JsonOutputParser()

    # 创建提示模板，要求JSON格式输出
    json_prompt = PromptTemplate(
        input_variables=["question"],
        template="""请回答以下问题，并以JSON格式返回结果。
要求：
1. answer: 直接回答问题
2. confidence: 置信度（0-1之间的数字）
3. sources: 相关信息来源（列表）

问题：{question}

请只返回JSON格式的结果，不要包含其他文字："""
    )

    # 创建JSON输出链
    json_chain = json_prompt | llm | json_parser

    questions = [
        "什么是机器学习？",
        "Python有哪些优势？",
        "如何提高编程技能？"
    ]

    for question in questions:
        try:
            result = json_chain.invoke({"question": question})
            print(f"问题: {question}")
            print("JSON回答:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            print()
        except Exception as e:
            print(f"处理问题时出错: {e}")

def custom_function_chain_example():
    """自定义函数链示例 - 在链中使用自定义函数"""
    print("=== 自定义函数链示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 自定义函数：文本预处理
    def preprocess_text(text):
        """预处理文本：清理格式、转换大小写等"""
        # 移除多余的空白字符
        cleaned = ' '.join(text.split())
        # 转换为小写（某些任务可能需要）
        # cleaned = cleaned.lower()
        return cleaned

    # 自定义函数：后处理结果
    def format_output(output):
        """格式化输出结果"""
        return f"📝 **处理结果**：\n{output}\n\n✨ 处理完成！"

    # 自定义函数：文本分析
    def analyze_text(text):
        """分析文本特征"""
        word_count = len(text.split())
        char_count = len(text)
        return {
            "word_count": word_count,
            "char_count": char_count,
            "complexity": "高" if word_count > 50 else "中" if word_count > 20 else "低"
        }

    # 创建提示模板
    analysis_prompt = PromptTemplate(
        input_variables=["text", "analysis"],
        template="""文本分析结果：
- 字数：{word_count}
- 字符数：{char_count}
- 复杂度：{complexity}

请基于以上分析，对这个文本提供详细的见解：
{text}

见解："""
    )

    # 创建包含自定义函数的链
    custom_chain = (
        RunnablePassthrough.assign(
            cleaned_text=lambda x: preprocess_text(x["text"]),
            analysis=lambda x: analyze_text(x["text"])
        )
        | RunnablePassthrough.assign(
            insights=lambda x: (analysis_prompt | llm | StrOutputParser()).invoke({
                "text": x["cleaned_text"],
                "word_count": x["analysis"]["word_count"],
                "char_count": x["analysis"]["char_count"],
                "complexity": x["analysis"]["complexity"]
            })
        )
        | RunnableLambda(format_output)
    )

    test_text = """
    LangChain 是一个强大的框架，它可以帮助开发人员构建基于大语言模型的应用程序。
    通过链式调用，可以组合多个步骤来完成复杂的任务。
    """

    try:
        result = custom_chain.invoke({"text": test_text})
        print(f"原始文本: {test_text}")
        print("\n处理结果:")
        print(result)
    except Exception as e:
        print(f"自定义函数链处理时出错: {e}")

def chat_template_chain_example():
    """聊天模板链示例 - 使用ChatPromptTemplate"""
    print("=== 聊天模板链示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建聊天提示模板
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{role}助手。回答要专业、准确、简洁。"),
        ("human", "{question}")
    ])

    # 创建聊天链
    chat_chain = chat_prompt | llm | StrOutputParser()

    # 测试不同角色的助手
    scenarios = [
        {"role": "Python编程", "question": "什么是装饰器？"},
        {"role": "数据科学", "question": "机器学习的主要类型有哪些？"},
        {"role": "心理咨询", "question": "如何管理工作压力？"}
    ]

    for scenario in scenarios:
        try:
            result = chat_chain.invoke(scenario)
            print(f"角色: {scenario['role']} 助手")
            print(f"问题: {scenario['question']}")
            print(f"回答: {result}\n")
        except Exception as e:
            print(f"处理聊天模板时出错: {e}")

if __name__ == "__main__":
    print("🔗 欢迎来到LangChain现代链式调用学习世界！\n")

    # 检查环境变量
    if not os.getenv("GLM_API_KEY") or not os.getenv("GLM_BASE_URL"):
        print("⚠️ 警告: 未找到GLM_API_KEY或GLM_BASE_URL环境变量")
        print("请确保在.env文件中设置了正确的智谱AI API配置")
        print()
        

    # 运行基础LCEL链示例
    print("🎯 1. 基础LCEL链")
    #basic_lcel_chain()

    print("\n" + "="*60 + "\n")

    # 运行并行链示例
    print("🎯 2. 并行链")
    #parallel_chain_example()

    print("\n" + "="*60 + "\n")

    # 运行条件链示例
    print("🎯 3. 条件链")
    #conditional_chain_example()

    print("\n" + "="*60 + "\n")

    # 运行顺序链示例
    print("🎯 4. 顺序链")
    #sequential_chain_example()

    print("\n" + "="*60 + "\n")

    # 运行JSON输出链示例
    print("🎯 5. JSON输出链")
    #json_output_chain_example()

    print("\n" + "="*60 + "\n")

    # 运行自定义函数链示例
    print("🎯 6. 自定义函数链")
    #custom_function_chain_example()

    print("\n" + "="*60 + "\n")

    # 运行聊天模板链示例
    print("🎯 7. 聊天模板链")
    chat_template_chain_example()

    print("\n✨ 现代链式调用示例完成！")
    print("您已经学会了如何使用最新的LCEL语法构建各种类型的链。")
    print()
    print("📚 LCEL主要特性:")
    print("  • 统一接口: 所有组件都使用相同的 invoke 方法")
    print("  • 原生流式: 支持流式输出")
    print("  • 异步支持: 原生支持异步操作")
    print("  • 批处理: 支持批量处理")
    print("  • 组合性: 使用管道操作符 | 组合组件")
    print("  • 并行化: 使用 RunnableParallel 实现并行执行")
    print("  • 条件逻辑: 使用 RunnableBranch 实现条件分支")
    print("  • 回退机制: 支持回退和错误处理")