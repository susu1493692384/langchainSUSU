#!/usr/bin/env python3
"""
简单的RAG工具使用示例
演示如何在AGENT+RAGFLOW目录中使用RAG检索工具
"""

from langchain_openai import ChatOpenAI
# 兼容新旧版本的LangChain导入
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

# 兼容不同版本的Agent导入
try:
    from langchain.agents import create_agent
    AGENT_AVAILABLE = True
except ImportError:
    try:
        from langgraph.prebuilt import create_agent
        AGENT_AVAILABLE = True
    except ImportError:
        AGENT_AVAILABLE = False

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def create_my_agent():
    
        from ragflow_retrieval_tool import get_rag_tools
        tools = get_rag_tools()
        print(f"加载了 {len(tools)} 个工具:")
    # 定义可用工具列表

    # 创建LLM实例
        llm = ChatOpenAI(
            model="glm-4.5",
            temperature=0.1,
            openai_api_key=os.getenv("GLM_API_KEY"),
            openai_api_base=os.getenv("GLM_BASE_URL")
        )

    # 创建提示模板
        system_prompt = SystemMessage(content="""你是一个智能助手，可以使用以下工具来帮助用户：
 
可用工具：
-list_knowledge_bases:获取RAGFlow中所有可用的知识库列表,
-search_documents:在RAGFlow知识库中搜索相关文档,
-ask_knowledge_base:基于RAGFlow知识库内容回答问题,
-get_document_summary:获取指定知识库或所有知识库的文档摘要信息

请根据用户的需求，选择合适的工具来完成任务,并说明使用了什么工具。如果需要使用多个工具，可以按步骤执行。
要使用工具时，请使用工具的准确名称和参数。""")
    # 创建智能体 (使用LangGraph的方式)
        try:
            
            agent_executor = create_agent(llm, tools, system_prompt=system_prompt )
            return agent_executor
        except Exception as e:
            print(f"创建智能体失败: {e}")
            # 如果create_agent失败，尝试直接使用LLM
            return llm

def basic_agent_example():
    """基础智能体示例"""
    print("=== 基础智能体示例 ===\n")

    agent = create_my_agent()

    # 测试问题
    questions = [
        "请列出所有知识库",
        "在知识库中搜索关于'人工智能'的文档",
        "知识库中关于'王书友'的内容是什么？",
        "请提供知识库的文档摘要信息"
    ]

    for question in questions:
        print(f"用户：{question}")
        try:
            # 检查agent类型并调用
            if hasattr(agent, 'invoke'):
                agent_type = str(type(agent))
                if 'CompiledStateGraph' in agent_type:
                    # 如果是LangGraph agent (CompiledStateGraph)
                    inputs = {"messages": [{"role": "user", "content": question}]}
                    result = agent.invoke(inputs)
                    if 'messages' in result and len(result['messages']) > 0:
                        print(f"助手：{result['messages'][-1].content}\n")
                    else:
                        print(f"助手：{result}\n")
                else:
                    # 如果是简单智能体或普通LLM
                    result = agent(question)
                    if hasattr(result, 'content'):
                        # 如果返回的是Message对象
                        print(f"助手：{result.content}\n")
                    else:
                        # 如果返回的是字符串
                        print(f"助手：{result}\n")
            else:
                print("Agent不可调用")
        except Exception as e:
            print(f"执行出错：{e}\n")
        print("-" * 50)
def main():
    """主函数"""
    basic_agent_example()

if __name__ == "__main__":
    main()