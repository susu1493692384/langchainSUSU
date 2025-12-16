#!/usr/bin/env python3
"""
调试chunk结构，理解如何正确提取最终结果
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

def create_simple_agent():
    """创建简单的测试智能体"""
    tools = [calculate]

    llm = ChatOpenAI(
        model="glm-4.5",
        temperature=0.1,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    system_message = SystemMessage(content="你是一个智能助手，可以使用计算工具。")
    agent_executor = create_agent(llm, tools, system_prompt=system_message)
    return agent_executor

def debug_chunk_structure():
    """调试chunk结构"""
    print("调试chunk结构")
    print("=" * 50)

    agent = create_simple_agent()
    question = "计算 15 * 6"

    inputs = {"messages": [{"role": "user", "content": question}]}

    step_count = 0
    all_chunks = []

    for chunk in agent.stream(inputs):
        step_count += 1
        all_chunks.append(chunk)

        print(f"\n=== 步骤 {step_count} ===")
        print(f"Chunk类型: {type(chunk)}")
        print(f"Chunk键: {list(chunk.keys()) if isinstance(chunk, dict) else 'N/A'}")

        if isinstance(chunk, dict):
            for key, value in chunk.items():
                print(f"\n  {key}:")
                if isinstance(value, dict) and 'messages' in value:
                    print(f"    消息数量: {len(value['messages'])}")
                    for i, msg in enumerate(value['messages']):
                        msg_type = type(msg).__name__
                        print(f"      消息 {i+1}: {msg_type}")
                        if hasattr(msg, 'content') and msg.content and len(msg.content) < 100:
                            print(f"        内容: {msg.content}")
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"        工具调用: {[tc.get('name') for tc in msg.tool_calls]}")
                else:
                    print(f"    值类型: {type(value)}")

    print("\n" + "=" * 50)
    print("分析最终chunk...")

    if all_chunks:
        final_chunk = all_chunks[-1]
        print(f"最终chunk结构:")
        print(f"键: {list(final_chunk.keys())}")

        # 尝试找到最终的AI消息
        final_ai_message = None

        for key, value in final_chunk.items():
            if isinstance(value, dict) and 'messages' in value:
                for msg in value['messages']:
                    if type(msg).__name__ == 'AIMessage' and hasattr(msg, 'content') and msg.content.strip():
                        if not final_ai_message or len(msg.content) > 20:  # 优先选择更长的回答
                            final_ai_message = msg

        if final_ai_message:
            print(f"\n找到最终回答: {final_ai_message.content}")
        else:
            print("\n未找到最终AI回答")

if __name__ == "__main__":
    debug_chunk_structure()