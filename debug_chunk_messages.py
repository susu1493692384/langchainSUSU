#!/usr/bin/env python3
"""
调试chunk结构和消息收集
"""

import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

@tool
def calculate(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

def debug_chunk_messages():
    """调试chunk消息结构"""
    print("调试chunk消息结构")
    print("=" * 50)

    # 创建智能体
    tools = [calculate]
    llm = ChatOpenAI(
        model="glm-4.5",
        temperature=0.1,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    system_message = SystemMessage(content="你是一个智能助手，可以使用计算工具。")
    agent = create_agent(llm, tools, system_prompt=system_message)

    question = "计算 15 * 6"
    print(f"用户问题: {question}")

    inputs = {"messages": [{"role": "user", "content": question}]}

    step_count = 0
    all_messages = []
    chunk_details = []

    for chunk in agent.stream(inputs):
        step_count += 1
        print(f"\n=== 步骤 {step_count} ===")
        print(f"Chunk类型: {type(chunk)}")
        print(f"Chunk键: {list(chunk.keys()) if isinstance(chunk, dict) else 'N/A'}")

        chunk_info = {'step': step_count, 'messages': []}

        if isinstance(chunk, dict):
            for key, value in chunk.items():
                print(f"\n  {key}:")
                if isinstance(value, dict) and 'messages' in value:
                    messages = value['messages']
                    print(f"    消息数量: {len(messages)}")

                    for i, msg in enumerate(messages):
                        msg_type = type(msg).__name__
                        msg_id = getattr(msg, 'id', f"no_id_{step_count}_{i}")
                        print(f"      消息 {i+1}: {msg_type} (ID: {msg_id})")

                        # 保存消息详情
                        chunk_info['messages'].append({
                            'type': msg_type,
                            'id': msg_id,
                            'content': getattr(msg, 'content', ''),
                            'tool_calls': getattr(msg, 'tool_calls', []),
                            'name': getattr(msg, 'name', None)
                        })

                        # 添加到全局消息列表（去重）
                        if not any(getattr(m, 'id', None) == msg_id for m in all_messages):
                            all_messages.append(msg)

                        if hasattr(msg, 'content') and msg.content and len(msg.content) < 100:
                            print(f"        内容: {msg.content}")
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"        工具调用: {[tc.get('name') for tc in msg.tool_calls]}")
                else:
                    print(f"    值类型: {type(value)}")
                    if isinstance(value, str) and len(value) < 100:
                        print(f"    值: {value}")

        chunk_details.append(chunk_info)

    print("\n" + "=" * 50)
    print("消息统计:")
    print(f"总步数: {step_count}")
    print(f"收集到的消息总数: {len(all_messages)}")

    print(f"\n所有消息详情:")
    for i, msg in enumerate(all_messages):
        msg_type = type(msg).__name__
        msg_id = getattr(msg, 'id', 'no_id')
        print(f"  {i+1}. {msg_type} (ID: {msg_id})")
        if hasattr(msg, 'content') and msg.content:
            content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"     内容: {content_preview}")
        if hasattr(msg, 'name') and msg.name:
            print(f"     工具名: {msg.name}")

if __name__ == "__main__":
    debug_chunk_messages()