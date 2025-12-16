#!/usr/bin/env python3
"""
LangChain 消息类型完整参考指南
展示 ChatOpenAI.invoke() 方法中 message 参数可以使用的所有消息类型

作者: Claude
版本: 1.0
更新时间: 2025-11-28
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,           # 人类消息
    AIMessage,             # AI助手消息
    SystemMessage,         # 系统消息
    FunctionMessage,       # 函数调用结果消息(已弃用)
    ToolMessage,           # 工具调用结果消息
    ChatMessage,           # 通用聊天消息
)
from langchain_core.messages.message import Message
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# ================================
# 1. 基础消息类型
# ================================

def basic_message_types():
    """展示基础消息类型：HumanMessage, AIMessage, SystemMessage"""
    print("📝 === 基础消息类型示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=150,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 1. 人类消息 - 用户输入
    human_message = HumanMessage(content="你好！请用中文介绍一下你自己。")

    # 2. 系统消息 - 定义AI行为
    system_message = SystemMessage(content="你是一个专业的Python编程助手，回答要简洁明了。")

    # 3. AI消息 - 之前的AI回复
    ai_message = AIMessage(content="我是一个AI助手，可以帮助你解决编程问题。")

    # 单独使用人类消息
    try:
        print("👤 仅有HumanMessage:")
        response = llm.invoke([human_message])
        print(f"回答: {response.content}\n")
    except Exception as e:
        print(f"错误: {e}\n")

    # 组合使用消息
    try:
        print("🤖 组合消息 (System + Human):")
        messages = [system_message, human_message]
        response = llm.invoke(messages)
        print(f"系统指令: {system_message.content}")
        print(f"用户问题: {human_message.content}")
        print(f"AI回答: {response.content}\n")
    except Exception as e:
        print(f"错误: {e}\n")

    # 多轮对话
    try:
        print("💬 多轮对话:")
        conversation = [
            system_message,
            HumanMessage(content="什么是Python？"),
            AIMessage(content="Python是一种高级编程语言。"),
            HumanMessage(content="Python有哪些优点？")
        ]
        response = llm.invoke(conversation)
        print(f"对话历史:")
        for i, msg in enumerate(conversation):
            msg_type = type(msg).__name__
            print(f"  {i+1}. {msg_type}: {msg.content}")
        print(f"AI最新回答: {response.content}\n")
    except Exception as e:
        print(f"错误: {e}\n")

# ================================
# 2. 带元数据的消息
# ================================

def messages_with_metadata():
    """展示带元数据的消息"""
    print("🏷️ === 带元数据的消息示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=100,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 带名称的人类消息
    human_with_name = HumanMessage(
        content="帮我分析一下这段代码的性能",
        name="user_001",  # 用户标识
        additional_kwargs={"priority": "high"}  # 额外元数据
    )

    # 带工具调用信息的AI消息
    ai_with_tools = AIMessage(
        content="我来帮你分析代码性能。",
        tool_calls=[
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "analyze_code",
                    "arguments": '{"code": "sample_code"}'
                }
            }
        ]
    )

    try:
        print("🔧 带元数据的消息:")
        messages = [human_with_name]
        response = llm.invoke(messages)

        print(f"用户: {human_with_name.content}")
        print(f"用户ID: {human_with_name.name}")
        print(f"元数据: {human_with_name.additional_kwargs}")
        print(f"AI回答: {response.content}\n")

        print("🛠️ 带工具调用的AI消息:")
        print(f"AI: {ai_with_tools.content}")
        print(f"工具调用: {ai_with_tools.tool_calls}\n")
    except Exception as e:
        print(f"错误: {e}\n")

# ================================
# 3. 工具消息类型
# ================================

def tool_message_types():
    """展示工具相关的消息类型"""
    print("🔧 === 工具消息类型示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=100,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 工具调用消息
    ai_tool_call = AIMessage(
        content="我来为你计算一些数据。",
        tool_calls=[
            {
                "id": "calc_001",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression": "2+2"}'
                }
            }
        ]
    )

    # 工具执行结果消息
    tool_result = ToolMessage(
        content="4",
        tool_call_id="calc_001",
        name="calculate"
    )

    try:
        print("⚡ 工具调用流程:")
        messages = [
            HumanMessage(content="计算2+2等于多少？"),
            ai_tool_call,
            tool_result,
            HumanMessage(content="请解释计算结果。")
        ]

        response = llm.invoke(messages)

        print("对话流程:")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            if hasattr(msg, 'tool_call_id'):
                print(f"  {i+1}. {msg_type}: {msg.content} (工具ID: {msg.tool_call_id})")
            else:
                print(f"  {i+1}. {msg_type}: {msg.content}")

        print(f"最终回答: {response.content}\n")
    except Exception as e:
        print(f"错误: {e}\n")

# ================================
# 4. 通用聊天消息
# ================================

def chat_message_types():
    """展示通用ChatMessage类型"""
    print("💬 === 通用聊天消息示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=100,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 使用ChatMessage创建不同角色的消息
    developer_msg = ChatMessage(
        role="developer",
        content="请确保代码符合最佳实践。"
    )

    reviewer_msg = ChatMessage(
        role="reviewer",
        content="代码看起来不错，但可以进一步优化。"
    )

    assistant_msg = ChatMessage(
        role="assistant",
        content="我理解了，会按照最佳实践来优化代码。"
    )

    try:
        print("🎭 通用聊天消息:")
        messages = [
            developer_msg,
            reviewer_msg,
            HumanMessage(content="请重写这段代码。")
        ]

        response = llm.invoke(messages)

        for msg in messages:
            print(f"{msg.role}: {msg.content}")

        print(f"AI回答: {response.content}\n")
    except Exception as e:
        print(f"错误: {e}\n")

# ================================
# 5. 消息列表的不同组织方式
# ================================

def message_organization_examples():
    """展示消息列表的不同组织方式"""
    print("📋 === 消息组织方式示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=150,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 方式1: 简单单轮对话
    simple_chat = [HumanMessage(content="什么是机器学习？")]

    # 方式2: 带系统指令的对话
    system_guided = [
        SystemMessage(content="你是一个AI专家，用通俗易懂的语言解释技术概念。"),
        HumanMessage(content="解释什么是神经网络。")
    ]

    # 方式3: 多轮对话历史
    conversation_history = [
        SystemMessage(content="你是一个编程老师。"),
        HumanMessage(content="什么是变量？"),
        AIMessage(content="变量是用来存储数据的容器。"),
        HumanMessage(content="如何在Python中定义变量？"),
        AIMessage(content="在Python中，可以使用赋值语句定义变量，如：x = 10"),
        HumanMessage(content="可以给我更多例子吗？")
    ]

    # 方式4: 角色扮演对话
    role_playing = [
        SystemMessage(content="你是一个医生，用户是病人。"),
        HumanMessage(content="医生，我最近总是头痛，该怎么办？")
    ]

    # 方式5: 多专家讨论
    expert_discussion = [
        SystemMessage(content="现在有三个专家讨论一个问题。"),
        ChatMessage(role="frontend_developer", content="我们需要优化页面加载速度。"),
        ChatMessage(role="backend_developer", content="后端API响应时间需要优化。"),
        ChatMessage(role="devops", content="服务器配置也需要调整。"),
        HumanMessage(content="综合来看，我们应该从哪方面开始优化？")
    ]

    examples = [
        ("简单单轮对话", simple_chat),
        ("带系统指令", system_guided),
        ("多轮对话", conversation_history),
        ("角色扮演", role_playing),
        ("多专家讨论", expert_discussion)
    ]

    for name, messages in examples:
        try:
            print(f"🎯 {name}:")
            print("对话历史:")
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                if hasattr(msg, 'role'):
                    role = msg.role
                else:
                    role = msg_type.replace('Message', '')
                print(f"  {i+1}. {role}: {msg.content}")

            response = llm.invoke(messages)
            print(f"AI回答: {response.content}")
            print("-" * 50)
        except Exception as e:
            print(f"错误: {e}")
        print()

# ================================
# 6. 特殊用法示例
# ================================

def special_usage_examples():
    """展示消息的特殊用法"""
    print("✨ === 特殊用法示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=150,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 1. 包含代码和说明
    code_message = HumanMessage(content="""
请帮我分析这段Python代码：

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

这段代码有什么问题？如何优化？
""")

    # 2. 包含多语言内容
    multilingual_message = HumanMessage(content="""
Hello! Can you help me with programming?
你好！你能帮我编程吗？
こんにちは！プログラミングを手伝ってくれますか？

请用中文回答我关于编程的问题。
""")

    # 3. 包含结构化数据
    structured_message = HumanMessage(content="""
我有一个JSON数据：
```json
{
    "name": "张三",
    "age": 25,
    "skills": ["Python", "JavaScript", "SQL"],
    "experience": 3
}
```

请根据这个数据生成一个个人简介。
""")

    # 4. 包含指令和上下文
    instruction_message = [
        SystemMessage(content="你是一个专业的技术面试官。"),
        HumanMessage(content="""
候选人信息：
- 应聘岗位：Python开发工程师
- 工作经验：2年
- 技术栈：Python, Django, MySQL, Redis

请设计3个合适的技术面试问题。
""")
    ]

    special_messages = [
        ("代码分析", [code_message]),
        ("多语言支持", [multilingual_message]),
        ("结构化数据处理", [structured_message]),
        ("面试官角色", instruction_message)
    ]

    for name, messages in special_messages:
        try:
            print(f"🚀 {name}:")
            print("输入:")
            for msg in messages:
                print(f"  {msg.content}")

            response = llm.invoke(messages)
            print(f"输出:\n{response.content}")
            print("-" * 50)
        except Exception as e:
            print(f"错误: {e}")
        print()

# ================================
# 7. 错误处理和最佳实践
# ================================

def best_practices():
    """展示消息使用的最佳实践"""
    print("✅ === 最佳实践示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=100,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # ✅ 好的做法1: 清晰的系统消息
    good_system = [
        SystemMessage(content="你是一个Python编程专家。回答要准确、简洁，并包含代码示例。"),
        HumanMessage(content="如何在Python中读取文件？")
    ]

    # ✅ 好的做法2: 合理的对话长度
    good_length = [
        SystemMessage(content="简洁回答问题。"),
        HumanMessage(content="什么是列表推导式？")
    ]

    # ❌ 坏的做法1: 缺少系统消息
    bad_no_system = [
        HumanMessage(content="请告诉我如何开发一个完整的电商系统，包括前端、后端、数据库设计、部署等所有细节。")
    ]

    # ❌ 坏的做法2: 消息过长
    bad_too_long = [
        HumanMessage(content="请详细解释" + "详细" * 1000 + "的概念。")
    ]

    practices = [
        ("✅ 好的做法：清晰的系统消息", good_system),
        ("✅ 好的做法：合理的对话长度", good_length),
        ("❌ 坏的做法：缺少系统消息", bad_no_system),
        ("❌ 坏的做法：消息过长", bad_too_long)
    ]

    for name, messages in practices:
        try:
            print(f"{name}:")
            print(f"输入: {messages[0].content[:50]}...")

            response = llm.invoke(messages)
            print(f"输出: {response.content[:100]}...")
            print()
        except Exception as e:
            print(f"执行失败: {e}")
        print()

    # 最佳实践建议
    print("💡 最佳实践建议:")
    print("1. 始终使用SystemMessage定义AI的角色和行为")
    print("2. 保持消息内容简洁明了，避免冗长")
    print("3. 使用适当的角色扮演来获得更好的回答")
    print("4. 对于复杂任务，可以分步骤提问")
    print("5. 合理使用工具调用来扩展AI的能力")
    print("6. 注意消息的上下文连贯性")
    print("7. 避免在一个消息中包含过多不相关的内容")

# ================================
# 主函数
# ================================

def main():
    """主函数 - 运行所有消息类型示例"""
    import sys
    import io

    # 设置UTF-8编码输出
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("💬 LangChain 消息类型完整参考指南")
    print("=" * 60)
    print()

    # 检查环境变量
    if not os.getenv("GLM_API_KEY") or not os.getenv("GLM_BASE_URL"):
        print("⚠️ 警告: 未找到GLM_API_KEY或GLM_BASE_URL环境变量")
        print("请确保在.env文件中设置了正确的智谱AI API配置")
        print()
        return

    # 运行所有示例
    examples = [
        ("基础消息类型", basic_message_types),
        ("带元数据的消息", messages_with_metadata),
        ("工具消息类型", tool_message_types),
        ("通用聊天消息", chat_message_types),
        ("消息组织方式", message_organization_examples),
        ("特殊用法示例", special_usage_examples),
        ("最佳实践", best_practices)
    ]

    for name, func in examples:
        print(f"\n{'='*60}")
        print(f"📋 运行示例: {name}")
        print('='*60)
        print()

        try:
            func()
        except KeyboardInterrupt:
            print(f"\n⏹️ 用户中断了示例: {name}")
            break
        except Exception as e:
            print(f"❌ 示例 {name} 执行出错: {e}")

        # 询问是否继续
        print("\n" + "="*60)
        try:
            user_input = input("按Enter继续下一个示例，或输入'q'退出: ")
            if user_input.lower() == 'q':
                break
        except (EOFError, KeyboardInterrupt):
            print("\n👋 用户退出程序")
            break

    print("\n" + "="*60)
    print("✨ 消息类型参考指南结束！")
    print("="*60)
    print()

    # 消息类型总结
    print("📚 消息类型总结:")
    print()
    print("🏷️ 核心消息类型:")
    print("  • HumanMessage      - 人类用户消息")
    print("  • AIMessage         - AI助手回复消息")
    print("  • SystemMessage     - 系统指令消息")
    print("  • ToolMessage       - 工具执行结果消息")
    print("  • ChatMessage       - 通用角色消息")
    print()
    print("⚙️ 消息属性:")
    print("  • content           - 消息内容(必需)")
    print("  • name              - 消息名称/标识符")
    print("  • additional_kwargs - 额外元数据")
    print("  • response_metadata - 响应元数据(AIMessage)")
    print("  • tool_calls        - 工具调用信息(AIMessage)")
    print("  • tool_call_id      - 工具调用ID(ToolMessage)")
    print("  • role              - 消息角色(ChatMessage)")
    print()
    print("🎯 使用建议:")
    print("  • 每次对话都以SystemMessage开始，定义AI角色")
    print("  • 使用HumanMessage表示用户输入")
    print("  • AIMessage用于保存AI的回复历史")
    print("  • 对于工具调用，使用ToolMessage返回结果")
    print("  • 保持消息结构清晰，内容简洁")
    print("  • 合理组织对话历史，维持上下文连贯性")
    print()
    print("📖 更多信息:")
    print("  • LangChain消息文档: https://python.langchain.com/docs/modules/messages/")
    print("  • 消息类型API参考: https://api.python.langchain.com/en/latest/messages.html")

if __name__ == "__main__":
    main()