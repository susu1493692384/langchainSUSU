#!/usr/bin/env python3
"""
LangChain 模板和记忆管理示例（修复版）
展示如何使用PromptTemplate和Memory来管理对话上下文
使用新的LangChain API
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 设置编码
if sys.platform == "win32":
    import locale
    locale.setlocale(locale.LC_ALL, 'Chinese (Simplified)_China.utf8')

# 加载环境变量
load_dotenv()

def basic_prompt_template_example():
    """基础提示模板示例"""
    print("=== 基础提示模板示例 ===")

    llm = ChatOpenAI(
        model="glm-4.6",
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),
        temperature=0.7
    )

    # 简单的提示模板
    simple_template = PromptTemplate(
        input_variables=["product", "feature"],
        template="请为{product}的{feature}功能写一段宣传语，要求简洁有力，不超过50个字。"
    )

    chain = simple_template | llm | StrOutputParser()

    # 测试不同的产品
    products = [
        {"product": "智能手表", "feature": "心率监测"},
        {"product": "智能手机", "feature": "拍照"},
        {"product": "智能音箱", "feature": "语音助手"}
    ]

    for item in products:
        try:
            result = chain.invoke(item)
            print(f"产品: {item['product']} | 功能: {item['feature']}")
            print(f"宣传语: {result}\n")
        except Exception as e:
            print(f"处理产品 {item['product']} 时出错: {e}")

def chat_prompt_template_example():
    """聊天提示模板示例"""
    print("=== 聊天提示模板示例 ===")

    llm = ChatOpenAI(
        model="glm-4.6",
        temperature=0.1,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建聊天模板
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的中文AI助手，专门帮助用户解决技术问题。请用友好、专业的方式回答。"),
        ("human", "{question}")
    ])

    chain = chat_template | llm | StrOutputParser()

    # 技术问题列表
    technical_questions = [
        "什么是Python装饰器？",
        "如何优化网站性能？",
        "Git和SVN有什么区别？"
    ]

    for question in technical_questions:
        try:
            result = chain.invoke({"question": question})
            print(f"问题: {question}")
            print(f"回答: {result}\n")
        except Exception as e:
            print(f"处理问题时出错: {e}")

def advanced_prompt_template_example():
    """高级提示模板示例"""
    print("=== 高级提示模板示例 ===")

    llm = ChatOpenAI(
        model="glm-4.6",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 复杂的多角色提示模板
    role_playing_template = ChatPromptTemplate.from_messages([
        ("system", """你是一个经验丰富的{role}。你的专业领域是{specialty}。
请以{tone}的语气回答用户的问题，并包含{detail_level}级别的细节。"""),
        ("human", "{question}")
    ])

    chain = role_playing_template | llm | StrOutputParser()

    # 不同角色配置
    role_configs = [
        {
            "role": "软件工程师",
            "specialty": "Python开发",
            "tone": "友好专业",
            "detail_level": "详细",
            "question": "如何写好的代码注释？"
        },
        {
            "role": "产品经理",
            "specialty": "用户体验设计",
            "tone": "务实建议",
            "detail_level": "全面",
            "question": "如何提高用户留存率？"
        },
        {
            "role": "数据科学家",
            "specialty": "机器学习",
            "tone": "学术严谨",
            "detail_level": "深度",
            "question": "如何评估模型的性能？"
        }
    ]

    for config in role_configs:
        try:
            result = chain.invoke(config)
            print(f"角色: {config['role']} ({config['specialty']})")
            print(f"问题: {config['question']}")
            print(f"回答: {result}\n")
        except Exception as e:
            print(f"处理角色 {config['role']} 时出错: {e}")

def conversation_buffer_memory_example():
    """对话缓冲记忆示例 - 使用新API"""
    print("=== 对话缓冲记忆示例 ===")

    llm = ChatOpenAI(
        model="glm-4.6",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建消息历史
    message_history = InMemoryChatMessageHistory()

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手。请根据对话历史回答用户的问题。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 创建链
    chain = prompt | llm | StrOutputParser()

    # 创建带消息历史的可运行对象
    runnable_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 模拟对话
    dialogues = [
        "你好，我想学习编程",
        "Python和Java哪个更适合初学者？",
        "学习Python需要什么基础知识？",
        "你刚才推荐了什么？",
        "能详细说说那个建议吗？"
    ]

    print("开始对话...")
    for i, user_input in enumerate(dialogues, 1):
        try:
            response = runnable_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "test_session"}}
            )
            print(f"轮次 {i}:")
            print(f"用户: {user_input}")
            print(f"AI助手: {response}\n")
        except Exception as e:
            print(f"处理对话轮次 {i} 时出错: {e}")

def window_memory_example():
    """窗口记忆示例 - 简化版本"""
    print("=== 窗口记忆示例 ===")

    llm = ChatOpenAI(
        model="glm-4.6",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建消息历史（手动管理窗口）
    message_history = InMemoryChatMessageHistory()
    window_size = 3  # 保留最近3轮对话

    def get_recent_messages():
        """获取最近的消息"""
        messages = message_history.messages
        # 保留最近6条消息（用户+AI各3条）
        return messages[-window_size*2:] if len(messages) > window_size*2 else messages

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是一个友好的AI助手。你只能记住最近{window_size}轮对话。请根据最近的对话历史回答用户的问题。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 创建链
    chain = prompt | llm | StrOutputParser()

    # 长对话测试
    long_dialogues = [
        "我的名字是张三",
        "我喜欢学习编程",
        "我正在学习Python",
        "Python是一门很好的语言",
        "我也喜欢机器学习",
        "机器学习很有趣",
        "我刚才说我叫什么名字？",  # 这个问题AI可能记不住，因为超过了3轮
        "我刚才说我喜欢什么？",    # 这个可能也记不住
        "机器学习怎么样？"         # 这个应该能回答，因为最近提到了
    ]

    print("开始长对话测试...")
    for i, user_input in enumerate(long_dialogues, 1):
        try:
            # 添加用户消息
            message_history.add_user_message(user_input)

            # 获取最近消息
            recent_messages = get_recent_messages()

            # 调用链
            response = chain.invoke({
                "input": user_input,
                "history": recent_messages
            })

            # 添加AI回复
            message_history.add_ai_message(response)

            print(f"轮次 {i}:")
            print(f"用户: {user_input}")
            print(f"AI助手: {response}\n")

        except Exception as e:
            print(f"处理对话轮次 {i} 时出错: {e}")

def summary_memory_example():
    """摘要记忆示例 - 简化版本"""
    print("=== 摘要记忆示例 ===")

    llm = ChatOpenAI(
        model="glm-4.6",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 手动管理摘要
    conversation_summary = ""
    message_count = 0
    summary_frequency = 3  # 每3轮对话更新一次摘要

    def update_summary(user_input, ai_response):
        nonlocal conversation_summary, message_count
        message_count += 1

        # 添加到摘要
        if conversation_summary:
            conversation_summary += f"\n用户: {user_input}\nAI: {ai_response}"
        else:
            conversation_summary = f"用户: {user_input}\nAI: {ai_response}"

        # 每3轮对话生成新摘要
        if message_count % summary_frequency == 0:
            try:
                summary_prompt = f"""请将以下对话内容总结为简洁的摘要：

{conversation_summary}

摘要："""

                new_summary = llm.invoke(summary_prompt)
                conversation_summary = f"对话摘要: {new_summary.content}"

                print(f"更新摘要: {conversation_summary}")
                print("-" * 50)

            except Exception as e:
                print(f"生成摘要时出错: {e}")

    # 复杂的多话题对话
    multi_topic_dialogues = [
        "我想了解人工智能的发展历史",
        "人工智能的里程碑事件有哪些？",
        "现在让我聊聊机器学习",
        "机器学习的主要类型有哪些？",
        "我还想了解深度学习",
        "深度学习和机器学习的关系是什么？",
        "根据我们之前的所有对话，请总结一下AI、ML、DL的关系"
    ]

    print("开始多话题对话测试...")
    for i, user_input in enumerate(multi_topic_dialogues, 1):
        try:
            # 创建包含摘要的提示
            if conversation_summary:
                system_prompt = f"""你是一个AI助手。以下是之前的对话摘要：

{conversation_summary}

请基于这个摘要和当前问题回答用户。"""
            else:
                system_prompt = "你是一个AI助手。请回答用户的问题。"

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            chain = prompt | llm | StrOutputParser()

            response = chain.invoke({"input": user_input})

            print(f"轮次 {i}:")
            print(f"用户: {user_input}")
            print(f"AI助手: {response}\n")

            # 更新摘要
            update_summary(user_input, response)

        except Exception as e:
            print(f"处理对话轮次 {i} 时出错: {e}")

def custom_memory_example():
    """自定义记忆示例"""
    print("=== 自定义记忆示例 ===")

    llm = ChatOpenAI(
        model="glm-4.6",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建自定义消息历史
    message_history = InMemoryChatMessageHistory()

    # 存储重要信息
    important_info = {}

    # 创建聊天模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个有记忆的AI助手。请记住对话的重要信息，并在适当时候引用之前的内容。

重要信息记录：
{important_info}

对话历史：
{history}"""),
        ("human", "{input}")
    ])

    # 创建链
    chain = prompt | llm | StrOutputParser()

    # 自定义对话管理函数
    def chat_with_memory(user_input: str):
        """带记忆的聊天函数"""
        # 提取重要信息
        if "我叫" in user_input or "我的名字是" in user_input:
            # 提取名字
            for word in user_input.split():
                if len(word) > 1 and word not in ["我叫", "名字", "是", "我的"]:
                    important_info["用户姓名"] = word
                    break

        if "使用" in user_input or "懂" in user_input or "会" in user_input:
            # 提取技能
            important_info["用户技能"] = user_input

        # 调用链获取响应
        response = chain.invoke({
            "input": user_input,
            "history": message_history.messages,
            "important_info": str(important_info) if important_info else "暂无重要信息"
        })

        return response

    # 测试记忆功能
    memory_test_dialogues = [
        "我叫李四，是一名软件工程师",
        "我主要使用React进行前端开发",
        "我还懂一些Node.js",
        "根据我们之前的对话，你能介绍一下我吗？",
        "我刚才提到我懂什么后端技术？",
        "我想提升我的前端技能，有什么建议吗？"
    ]

    print("开始自定义记忆对话测试...")
    for i, user_input in enumerate(memory_test_dialogues, 1):
        try:
            response = chat_with_memory(user_input)

            # 添加消息到历史
            message_history.add_user_message(user_input)
            message_history.add_ai_message(response)

            print(f"轮次 {i}:")
            print(f"用户: {user_input}")
            print(f"AI助手: {response}\n")

            # 显示重要信息
            if i == 3:  # 第三轮后显示记录的信息
                print("📝 记录的重要信息:")
                for key, value in important_info.items():
                    print(f"  {key}: {value}")
                print("-" * 50 + "\n")

        except Exception as e:
            print(f"处理对话轮次 {i} 时出错: {e}")

if __name__ == "__main__":
    print("欢迎来到LangChain模板和记忆管理学习世界！")

    # 运行基础提示模板示例
    basic_prompt_template_example()

    print("\n" + "="*50 + "\n")

    # 运行聊天提示模板示例
    chat_prompt_template_example()

    print("\n" + "="*50 + "\n")

    # 运行高级提示模板示例
    advanced_prompt_template_example()

    print("\n" + "="*50 + "\n")

    # 运行对话缓冲记忆示例
    conversation_buffer_memory_example()

    print("\n" + "="*50 + "\n")

    # 运行窗口记忆示例
    window_memory_example()

    print("\n" + "="*50 + "\n")

    # 运行摘要记忆示例
    summary_memory_example()

    print("\n" + "="*50 + "\n")

    # 运行自定义记忆示例
    custom_memory_example()

    print("\n模板和记忆管理示例完成！您已经学会了如何在LangChain中创建和管理各种模板及记忆系统。")