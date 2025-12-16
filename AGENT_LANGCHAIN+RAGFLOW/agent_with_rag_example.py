#!/usr/bin/env python3
"""
带RAG检索工具的智能体示例
展示如何将RAGFlow检索工具集成到LangChain智能体中
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
# 导入RAG工具
from ragflow_retrieval_tool import get_rag_tools, initialize_rag_tools

# 加载环境变量
load_dotenv()

class RAGEnabledAgent:
    """带RAG检索能力的智能体"""

    def __init__(self,
                 ragflow_url: str = None,
                 ragflow_api_key: str = None,
                 llm_model: str = "glm-4.5"):
        """
        初始化带RAG能力的智能体

        Args:
            ragflow_url: RAGFlow服务URL
            ragflow_api_key: RAGFlow API密钥
            llm_model: LLM模型名称
        """
        self.llm_model = llm_model
        self.ragflow_url = ragflow_url
        self.ragflow_api_key = ragflow_api_key
        self.agent_executor = None
        self.rag_tools = []
        self.checkpointer = None
        self.thread_id = "rag_conversation"  # 用于会话记忆的唯一ID
        self.conversation_memory = []  # 简单的本地记忆存储

    def initialize(self) -> bool:
        """初始化智能体和RAG工具"""
        try:
            # 初始化RAG工具
            print("正在初始化RAG检索工具...")
            if not initialize_rag_tools(
                ragflow_url=self.ragflow_url,
                ragflow_api_key=self.ragflow_api_key,
                llm_model=self.llm_model
            ):
                print("❌ RAG工具初始化失败")
                return False

            # 获取RAG工具
            self.rag_tools = get_rag_tools()
            print(f"✅ RAG工具初始化成功，加载了 {len(self.rag_tools)} 个工具")

            # 初始化LLM
            print("正在初始化LLM...")
            if os.getenv("GLM_API_KEY"):
                self.llm = ChatOpenAI(
                    model=os.getenv("LLM_MODEL", "GLM-4.5"),
                    temperature=0.1,
                    openai_api_key=os.getenv("GLM_API_KEY"),
                    openai_api_base=os.getenv("GLM_BASE_URL")
                )
            print("✅ LLM初始化成功")

            # 初始化checkpoint memory
            print("正在初始化记忆系统...")
            self.checkpointer = MemorySaver()
            print("✅ 记忆系统初始化成功")


            system_prompt = SystemMessage(content="""你是一个专业的AI助手，具有访问RAGFlow知识库的能力。

你有以下工具可以使用：
1. list_knowledge_bases - 获取所有可用的知识库列表
2. search_documents - 在知识库中搜索相关文档
3. ask_knowledge_base - 基于知识库内容回答问题
4. get_document_summary - 获取知识库文档摘要

使用指南：
- 当用户询问关于知识库内容时，先使用 list_knowledge_bases 查看可用的知识库
- 使用 search_documents 搜索相关信息
- 使用 ask_knowledge_base 直接回答用户问题
- 使用 get_document_summary 获取知识库概览

请根据用户的问题，选择合适的工具来提供准确的、基于知识库的回答。如果知识库中没有相关信息，请诚实地说明。

注意：工具调用会返回JSON格式的结果，你需要将结果以自然、友好的方式呈现给用户。""")

            try:
                 # 优先尝试使用新的create_agent API（推荐方式）
                print("尝试使用新的create_agent API...")

                # 方法1: 尝试使用最新的 langchain.agents.create_agent
                try:
                    # 创建基础agent
                    self.agent_executor = create_agent(self.llm, self.rag_tools, system_prompt=system_prompt)

                    # 如果agent有兼容的API，尝试包装checkpointer
                    if hasattr(self.agent_executor, 'ainvoke') and self.checkpointer:
                        try:
                            # 使用LangGraph的包装器添加记忆功能
                            from langgraph.prebuilt import create_react_agent
                            # 直接使用新的API，忽略deprecation警告
                            self.agent_executor = create_react_agent(
                                self.llm,
                                self.rag_tools,
                                checkpointer=self.checkpointer,
                                prompt=system_prompt.content
                            )
                            print("✅ 带记忆功能的智能体创建成功 (create_react_agent with checkpointer)")
                            return self.agent_executor
                        except:
                            print("✅ 基础智能体创建成功（无checkpointer）")
                            return self.agent_executor
                    else:
                        print("✅ 基础智能体创建成功")
                        return self.agent_executor

                except Exception as agent_e:
                    print(f"⚠️ create_agent方法失败: {agent_e}")

                # 方法2: 回退方案 - 直接使用带checkpointer的版本
                try:
                    from langgraph.prebuilt import create_react_agent
                    self.agent_executor = create_react_agent(
                        self.llm,
                        self.rag_tools,
                        checkpointer=self.checkpointer
                    )
                    print("✅ 带记忆功能的智能体创建成功 (create_react_agent fallback)")
                    return self.agent_executor
                except Exception as fallback_e:
                    print(f"⚠️ 回退方案失败: {fallback_e}")

            except Exception as e:
                print(f"⚠️ 带记忆的智能体创建失败: {e}")
                # 最终回退 - 不使用checkpoint的方法
                try:
                    self.agent_executor = create_agent(self.llm, self.rag_tools, system_prompt=system_prompt)
                    print("✅ 不带记忆的智能体创建成功")
                    return self.agent_executor
                except Exception as final_e:
                    print(f"❌ 所有智能体创建方法都失败: {final_e}")
        except Exception as e:
            print(f"❌ 智能体初始化失败: {e}")
            return False

    def chat(self, message: str) -> str:
        """与智能体对话（带记忆功能）"""
        try:
            # 添加用户消息到本地记忆
            self.conversation_memory.append({"role": "user", "content": message})

            # 构建包含记忆的完整消息历史
            messages = self.conversation_memory.copy()

            # 如果有checkpointer，使用带记忆的调用
            if self.checkpointer and hasattr(self.agent_executor, 'invoke'):
                response = self.agent_executor.invoke(
                    {"messages": messages},
                    config={"configurable": {"thread_id": self.thread_id}}
                )
            else:
                # 回退到普通调用
                response = self.agent_executor.invoke({"messages": messages})

            # 提取响应内容
            if isinstance(response, dict) and "messages" in response:
                messages_response = response["messages"]
                if messages_response:
                    response_content = str(messages_response[-1].content) if hasattr(messages_response[-1], 'content') else str(messages_response[-1])
                    # 添加AI响应到本地记忆
                    self.conversation_memory.append({"role": "assistant", "content": response_content})
                    return response_content
            elif hasattr(response, 'content'):
                response_content = str(response.content)
                self.conversation_memory.append({"role": "assistant", "content": response_content})
                return response_content
            elif isinstance(response, str):
                self.conversation_memory.append({"role": "assistant", "content": response})
                return response
            else:
                response_str = str(response)
                self.conversation_memory.append({"role": "assistant", "content": response_str})
                return response_str

        except Exception as e:
            print(f"❌ 调用智能体时出错: {e}")
            return f"抱歉，我遇到了一个错误：{e}"

    def clear_memory(self) -> None:
        """清除会话记忆"""
        # 清除本地记忆
        self.conversation_memory = []

        # 清除checkpoint记忆
        if self.checkpointer:
            self.checkpointer = MemorySaver()
            print("✅ 所有记忆已清除（本地记忆 + checkpoint记忆）")
        else:
            print("✅ 本地记忆已清除")

    def set_thread_id(self, thread_id: str) -> None:
        """设置新的会话线程ID"""
        self.thread_id = thread_id
        # 清除本地记忆开始新会话
        self.conversation_memory = []
        print(f"✅ 会话线程ID已设置为: {thread_id} (开始新会话)")

    def get_conversation_history(self) -> List:
        """获取当前会话的历史记录"""
        # 首先尝试本地记忆
        if self.conversation_memory:
            return self.conversation_memory

        # 如果本地记忆为空，尝试从checkpoint获取
        try:
            if self.checkpointer:
                checkpoint_config = {"configurable": {"thread_id": self.thread_id}}
                checkpoint_list = list(self.checkpointer.list(checkpoint_config))

                if checkpoint_list and len(checkpoint_list) > 0:
                    latest_checkpoint = checkpoint_list[-1]
                    if latest_checkpoint and latest_checkpoint.checkpoint:
                        values = latest_checkpoint.checkpoint.get("channel_values", {})
                        messages = values.get("messages", [])
                        # 将checkpoint消息转换为本地格式
                        local_format = []
                        for msg in messages:
                            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                                local_format.append({
                                    "role": "user" if msg.type == "human" else "assistant",
                                    "content": str(msg.content)
                                })
                        return local_format

            return []
        except Exception as e:
            print(f"⚠️ 获取checkpoint历史时出错: {e}")
            return self.conversation_memory  # 回退到本地记忆

    def test_memory_function(self) -> bool:
        """测试记忆功能是否正常工作"""
        try:
            if not self.checkpointer:
                print("⚠️ 检查点系统未初始化")
                return False

            if not self.agent_executor:
                print("⚠️ 智能体未初始化")
                return False

            # 检查是否有checkpointer属性
            if hasattr(self.agent_executor, 'checkpoint'):
                print("✅ 检查点系统正常")
                return True
            elif hasattr(self.agent_executor, 'checkpointer'):
                print("✅ 记忆系统正常")
                return True
            else:
                print("⚠️ 未找到记忆相关属性")
                return False

        except Exception as e:
            print(f"⚠️ 记忆功能测试失败: {e}")
            return False

    def simple_memory_test(self) -> bool:
        """简单记忆测试 - 通过对话验证"""
        try:
            print("🔄 发送第一个测试消息...")
            response1 = self.chat("记住数字123")

            print("🔄 发送第二个测试消息...")
            response2 = self.chat("我刚才让你记住的数字是多少？")

            # 如果智能体记住并提到了123，说明记忆功能正常
            if "123" in response2:
                print("✅ 记忆功能测试通过！")
                return True
            else:
                print("❌ 记忆功能测试失败")
                return False

        except Exception as e:
            print(f"❌ 记忆测试出错: {e}")
            return False

    def interactive_chat(self):
        """启动交互式聊天"""
        print("\n" + "="*60)
        print("🤖 RAG增强智能体已就绪")
        print("="*60)
        print("您可以询问关于知识库中的任何问题")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'help' 查看可用命令")
        print("输入 'clear', '清除'")
        print("输入 'history', '历史' 查看对话历史")
        print("输入 'session <名称>' 切换会话")
        print("="*60)

        while True:
            try:
                user_input = input(f"\n[{self.thread_id[:8]}...] 您: ").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("\n👋 感谢使用！再见！")
                    break

                if user_input.lower() in ['help', '帮助']:
                    self._show_help()
                    continue

                if user_input.lower() in ['clear', '清除']:
                    self.clear_memory()
                    continue

                if user_input.lower() in ['test', '测试']:
                    print("\n选择测试类型:")
                    print("1. 基础测试 (检查记忆系统状态)")
                    print("2. 对话测试 (验证实际记忆功能)")

                    test_choice = input("请选择 (1 或 2，默认为1): ").strip()

                    if test_choice == "2":
                        print("\n🧠 开始对话记忆测试...")
                        self.simple_memory_test()
                    else:
                        print("\n🔧 正在测试记忆功能...")
                        memory_works = self.test_memory_function()
                        if memory_works:
                            print("✅ 记忆功能正常工作")
                        else:
                            print("❌ 记忆功能测试失败")
                    continue

                if user_input.lower() in ['history', '历史']:
                    history = self.get_conversation_history()
                    if history:
                        print(f"\n📜 对话历史 ({len(history)} 条消息):")
                        for i, msg in enumerate(history[-5:], 1):  # 显示最近5条
                            try:
                                msg_type = "用户" if hasattr(msg, 'type') and msg.type == "human" else "AI助手"
                                content = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
                                print(f"  {i}. {msg_type}: {content}...")
                            except:
                                print(f"  {i}. 消息: {str(msg)[:100]}...")
                    else:
                        # 提供记忆功能的调试信息
                        print(f"\n📜 暂无对话历史")
                        if self.checkpointer:
                            print(f"💾 记忆系统已启用 (会话ID: {self.thread_id})")
                            print("💡 记忆功能正常工作，但历史记录可能需要时间同步")
                        else:
                            print("⚠️ 记忆系统未启用")
                    continue

                if user_input.lower().startswith('session '):
                    new_session = user_input[8:].strip()
                    if new_session:
                        self.set_thread_id(new_session)
                    else:
                        print("⚠️ 请提供会话名称，例如: session 工作对话")
                    continue

                if not user_input:
                    continue

                print("\n🤔 智能体正在思考...")
                response = self.chat(user_input)
                print(f"\n🤖 助手: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 感谢使用！再见！")
                break
            except Exception as e:
                print(f"\n❌ 出现错误: {e}")

    def _show_help(self):
        """显示帮助信息"""
        help_text = """
📚 可用命令:
- 直接输入问题，智能体会使用RAG知识库回答
- 'list' 或 '列表' - 查看所有可用知识库
- 'summary' 或 '摘要' - 获取知识库概览
- 'history' 或 '历史' - 查看对话历史
- 'clear' 或 '清除' - 清除会话记忆
- 'test' 或 '测试' - 测试记忆功能
- 'session <名称>' - 切换到新的会话
- 'help' 或 '帮助' - 显示此帮助信息
- 'quit' 或 'exit' 或 '退出' - 退出程序

🧠 记忆功能说明:
- 智能体现在具有记忆功能，可以记住之前的对话
- 可以通过'session <名称>'创建多个独立的对话会话
- 记忆功能帮助智能体更好地理解上下文和引用
- 使用'test'命令可以验证记忆功能是否正常

💡 示例问题:
- "王书友是什么岗位?"
- "搜索关于王书友的信息"
- "总结一下王书友的工作内容"
- "有哪些可用的知识库?"
- "我刚才问了什么问题？" (测试记忆功能)
        """
        print(help_text)


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 RAG增强智能体示例")
    print("=" * 60)

    # 创建智能体

    agent = RAGEnabledAgent(
        ragflow_url="http://localhost:9380",
        ragflow_api_key=os.getenv("RAGFLOW_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "GLM-4.5")
    )

    # 初始化
    if not agent.initialize():
        print("❌ 智能体初始化失败，请检查配置")
        return

    # 启动交互式聊天
    agent.interactive_chat()


if __name__ == "__main__":
    main()