#!/usr/bin/env python3
"""
RAGFlow检索工具 - 为智能体提供RAG知识库检索功能
这个工具可以让智能体轻松访问RAGFlow知识库中的信息
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
try:
    from langchain_core.tools import tool
except ImportError:
    # 兼容旧版本
    from langchain.tools import tool
try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        # 如果都不可用，创建一个简单的基类
        class BaseModel:
            pass
        def Field(**kwargs):
            return None

# type: ignore  # 忽略动态导入的类型检查警告
import logging

# 导入现有的RAGFlow集成
# type: ignore  # 忽略以下动态导入的类型检查警告
try:
    # 尝试多种导入路径
    from RAGFLOW_PLUS_LANGCHAIN.ragflow_langchain_integration import RAGFlowAPIConnector, RAGFlowLangChainApp
except ImportError:
    try:
        # 尝试从RAGFLOW+LANGCHAIN目录导入
        import sys
        import os
        # 添加上级目录中的RAGFLOW+LANGCHAIN到路径
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ragflow_path = os.path.join(parent_dir, 'RAGFLOW+LANGCHAIN')
        if ragflow_path not in sys.path:
            sys.path.append(ragflow_path)
        from ragflow_langchain_integration import RAGFlowAPIConnector, RAGFlowLangChainApp
    except ImportError:
        try:
            # 尝试其他可能的路径
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RAGFLOW+LANGCHAIN'))
            from ragflow_langchain_integration import RAGFlowAPIConnector, RAGFlowLangChainApp
        except ImportError:
            # 如果都不行，提供详细的错误信息
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)

            print(f"当前目录: {current_dir}")
            print(f"上级目录: {parent_dir}")
            print(f"尝试的RAGFlow路径: {os.path.join(parent_dir, 'RAGFLOW+LANGCHAIN')}")

            # 检查文件是否存在
            expected_file = os.path.join(parent_dir, 'RAGFLOW+LANGCHAIN', 'ragflow_langchain_integration.py')
            if os.path.exists(expected_file):
                print(f"文件存在: {expected_file}")
                sys.path.append(os.path.join(parent_dir, 'RAGFLOW+LANGCHAIN'))
                from ragflow_langchain_integration import RAGFlowAPIConnector, RAGFlowLangChainApp
            else:
                print(f"文件不存在: {expected_file}")
                raise ImportError(f"无法找到ragflow_langchain_integration模块。请检查路径: {expected_file}")

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetrievalTool:
    """RAG检索工具类 - 为智能体提供知识库检索功能"""

    def __init__(self,
                 ragflow_url: str = None,
                 ragflow_api_key: str = None,
                 llm_model: str = "glm-4.5"):
        """
        初始化RAG检索工具

        Args:
            ragflow_url: RAGFlow API服务地址 (如果为None，将从环境变量RAGFLOW_API_URL获取)
            ragflow_api_key: RAGFlow API密钥 (如果为None，将从环境变量RAGFLOW_API_KEY获取)
            llm_model: 使用的LLM模型
        """
        # 使用环境变量作为默认值
        self.ragflow_url = ragflow_url or os.getenv("RAGFLOW_API_URL", "http://localhost:9380")
        self.ragflow_api_key = ragflow_api_key or os.getenv("RAGFLOW_API_KEY")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "GLM-4.5")  # 保持与RAGFlow集成一致的默认值

        self.app = None
        self.available_kbs = []
        self._initialized = False

    def _ensure_initialized(self):
        """确保工具已初始化"""
        if not self._initialized:
            self.app = RAGFlowLangChainApp(
                ragflow_url=self.ragflow_url,
                ragflow_api_key=self.ragflow_api_key,
                llm_model=self.llm_model
            )

            if self.app.initialize():
                self.available_kbs = self.app.available_kbs
                self._initialized = True
                logger.info(f"RAG检索工具初始化成功，发现 {len(self.available_kbs)} 个知识库")
            else:
                logger.error("RAG检索工具初始化失败")
                raise RuntimeError("无法连接到RAGFlow服务")

    def get_available_knowledge_bases(self) -> List[Dict[str, Any]]:
        """
        获取所有可用的知识库列表

        Returns:
            知识库信息列表，每个知识库包含id、name、description等信息
        """
        self._ensure_initialized()

        kb_list = []
        for kb in self.available_kbs:
            if isinstance(kb, dict):
                kb_info = {
                    "id": kb.get('id', ''),
                    "name": kb.get('name', ''),
                    "description": kb.get('description', ''),
                    "document_count": kb.get('document_count', 0),
                    "chunk_count": kb.get('chunk_count', 0)
                }
            else:
                kb_info = {
                    "id": str(kb),
                    "name": str(kb),
                    "description": "",
                    "document_count": 0,
                    "chunk_count": 0
                }
            kb_list.append(kb_info)

        return kb_list

    def search_knowledge_base(self,
                            query: str,
                            knowledge_base: str = None,
                            top_k: int = 5,
                            include_content: bool = True) -> Dict[str, Any]:
        """
        在知识库中搜索相关信息

        Args:
            query: 搜索查询
            knowledge_base: 指定的知识库名称或ID，如果为None则搜索所有知识库
            top_k: 返回的最大结果数
            include_content: 是否包含文档内容

        Returns:
            搜索结果字典，包含results列表和统计信息
        """
        self._ensure_initialized()

        try:
            if knowledge_base:
                # 搜索指定知识库
                retriever = self.app.create_retriever(knowledge_base, top_k=top_k)
                docs = retriever.get_relevant_documents(query)
                searched_kbs = [knowledge_base]
            else:
                # 搜索所有知识库
                multi_retriever = self.app.create_multi_kb_retriever()
                docs = multi_retriever.get_relevant_documents(query)
                searched_kbs = list(multi_retriever.retrievers.keys())

            results = []
            for doc in docs:
                result = {
                    "content": doc.page_content if include_content else "",
                    "score": doc.metadata.get("score", 0.0),
                    "source": doc.metadata.get("source", ""),
                    "knowledge_base": doc.metadata.get("knowledge_base", knowledge_base or "未知"),
                    "title": doc.metadata.get("title", ""),
                    "doc_id": doc.metadata.get("doc_id", ""),
                    "url": doc.metadata.get("url", "")
                }
                results.append(result)

            return {
                "success": True,
                "query": query,
                "knowledge_bases": searched_kbs,
                "total_results": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"搜索知识库时出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }

    def ask_question(self,
                    question: str,
                    knowledge_base: str = None,
                    chain_type: str = "with_sources") -> Dict[str, Any]:
        """
        基于知识库内容回答问题

        Args:
            question: 用户问题
            knowledge_base: 指定的知识库名称或ID，如果为None则使用所有知识库
            chain_type: QA链类型 ("basic", "contextual", "with_sources")

        Returns:
            回答结果字典，包含answer和sources信息
        """
        self._ensure_initialized()

        try:
            if knowledge_base:
                # 使用指定知识库
                answer = self.app.chat(knowledge_base, question, chain_type)
                return {
                    "success": True,
                    "question": question,
                    "answer": answer,
                    "knowledge_base": knowledge_base,
                    "chain_type": chain_type
                }
            else:
                # 使用所有知识库
                multi_retriever = self.app.create_multi_kb_retriever()
                qa_chain = self.app.create_multi_kb_qa_chain(multi_retriever, chain_type)
                answer = qa_chain.invoke(question)

                return {
                    "success": True,
                    "question": question,
                    "answer": answer,
                    "knowledge_bases": list(multi_retriever.retrievers.keys()),
                    "chain_type": chain_type
                }

        except Exception as e:
            logger.error(f"回答问题时出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "answer": ""
            }

# 创建全局工具实例
rag_tool = RAGRetrievalTool()

# ============ LangChain Tools ============

@tool
def list_knowledge_bases() -> str:
    """
    获取RAGFlow中所有可用的知识库列表

    Returns:
        格式化的知识库列表字符串，包含知识库名称、ID和描述
    """
    try:
        kbs = rag_tool.get_available_knowledge_bases()

        if not kbs:
            return "当前没有可用的知识库。请先在RAGFlow中创建知识库并添加文档。"

        result = f"发现 {len(kbs)} 个可用知识库：\n\n"

        for i, kb in enumerate(kbs, 1):
            result += f"{i}. **{kb['name']}** (ID: {kb['id']})\n"
            if kb['description']:
                result += f"   描述: {kb['description']}\n"
            result += f"   文档数: {kb['document_count']}, Chunk数: {kb['chunk_count']}\n\n"

        return result.strip()

    except Exception as e:
        return f"获取知识库列表时出错: {str(e)}"

@tool
def search_documents(query: str,
                    knowledge_base: Optional[str] = None,
                    max_results: int = 5) -> str:
    """
    在RAGFlow知识库中搜索相关文档

    Args:
        query: 搜索查询词或问题
        knowledge_base: 可选，指定搜索的知识库名称或ID。如果不指定，将搜索所有知识库
        max_results: 最大返回结果数，默认为5

    Returns:
        格式化的搜索结果字符串，包含文档内容和元数据
    """
    try:
        # 构建搜索参数描述
        kb_desc = f"知识库 '{knowledge_base}'" if knowledge_base else "所有知识库"

        # 执行搜索
        search_result = rag_tool.search_knowledge_base(
            query=query,
            knowledge_base=knowledge_base,
            top_k=max_results,
            include_content=True
        )

        if not search_result["success"]:
            return f"搜索失败: {search_result['error']}"

        results = search_result["results"]

        if not results:
            return f"在{kb_desc}中没有找到与查询 '{query}' 相关的文档。"

        # 格式化结果
        response = f"在{kb_desc}中找到 {len(results)} 个相关文档：\n\n"

        for i, doc in enumerate(results, 1):
            response += f"**文档 {i}**\n"
            response += f"相关度: {doc['score']:.3f}\n"
            response += f"知识库: {doc['knowledge_base']}\n"

            if doc['title']:
                response += f"标题: {doc['title']}\n"

            if doc['source']:
                response += f"来源: {doc['source']}\n"

            response += f"内容: {doc['content']}\n\n"

        return response.strip()

    except Exception as e:
        return f"搜索文档时出错: {str(e)}"

@tool
def ask_knowledge_base(question: str,
                      knowledge_base: Optional[str] = None,
                      include_sources: bool = True) -> str:
    """
    基于RAGFlow知识库内容回答问题

    Args:
        question: 需要回答的问题
        knowledge_base: 可选，指定使用的知识库名称或ID。如果不指定，将使用所有知识库
        include_sources: 是否在回答中包含信息来源

    Returns:
        基于知识库内容的回答
    """
    try:
        # 选择链类型
        chain_type = "with_sources" if include_sources else "basic"

        # 构建参数描述
        kb_desc = f"知识库 '{knowledge_base}'" if knowledge_base else "所有知识库"

        # 执行问答
        qa_result = rag_tool.ask_question(
            question=question,
            knowledge_base=knowledge_base,
            chain_type=chain_type
        )

        if not qa_result["success"]:
            return f"回答问题失败: {qa_result['error']}"

        answer = qa_result["answer"]

        # 格式化响应
        response = f"基于{kb_desc}的回答：\n\n{answer}"

        if include_sources:
            response += f"\n\n*回答使用了{kb_desc}中的相关信息*"

        return response

    except Exception as e:
        return f"回答问题时出错: {str(e)}"

@tool
def get_document_summary(knowledge_base: Optional[str] = None) -> str:
    """
    获取指定知识库或所有知识库的文档摘要信息

    Args:
        knowledge_base: 可选，指定知识库名称或ID。如果不指定，返回所有知识库的摘要

    Returns:
        知识库文档摘要信息
    """
    try:
        kbs = rag_tool.get_available_knowledge_bases()

        if knowledge_base:
            # 返回指定知识库的信息
            target_kb = None
            for kb in kbs:
                if kb['id'] == knowledge_base or kb['name'] == knowledge_base:
                    target_kb = kb
                    break

            if not target_kb:
                return f"未找到知识库 '{knowledge_base}'"

            return f"""
知识库: **{target_kb['name']}** (ID: {target_kb['id']})
描述: {target_kb['description'] or '无描述'}
文档数量: {target_kb['document_count']}
知识块数量: {target_kb['chunk_count']}
            """.strip()
        else:
            # 返回所有知识库的摘要
            if not kbs:
                return "当前没有可用的知识库"

            total_docs = sum(kb['document_count'] for kb in kbs)
            total_chunks = sum(kb['chunk_count'] for kb in kbs)

            summary = f"知识库总览：\n"
            summary += f"知识库数量: {len(kbs)}\n"
            summary += f"总文档数量: {total_docs}\n"
            summary += f"总知识块数量: {total_chunks}\n\n"

            summary += "详细信息：\n"
            for kb in kbs:
                summary += f"- **{kb['name']}**: {kb['document_count']} 个文档, {kb['chunk_count']} 个知识块\n"

            return summary

    except Exception as e:
        return f"获取文档摘要时出错: {str(e)}"

# ============ 工具集合 ============

# 所有可用的RAG工具
RAG_TOOLS = [
    list_knowledge_bases,
    search_documents,
    ask_knowledge_base,
    get_document_summary
]

def get_rag_tools():
    """获取所有RAG检索工具"""
    return RAG_TOOLS

def initialize_rag_tools(ragflow_url: str = None,
                        ragflow_api_key: str = None,
                        llm_model: str = "glm-4.5") -> bool:
    """
    初始化RAG工具

    Args:
        ragflow_url: RAGFlow服务URL (如果为None，将从环境变量RAGFLOW_API_URL获取)
        ragflow_api_key: RAGFlow API密钥 (如果为None，将从环境变量RAGFLOW_API_KEY获取)
        llm_model: LLM模型名称 (如果为None，将从环境变量LLM_MODEL获取)

    Returns:
        初始化是否成功
    """
    global rag_tool

    try:
        # 创建工具实例，将自动使用环境变量作为默认值
        rag_tool = RAGRetrievalTool(
            ragflow_url=ragflow_url,
            ragflow_api_key=ragflow_api_key,
            llm_model=llm_model
        )

        # 测试初始化
        rag_tool._ensure_initialized()
        return True

    except Exception as e:
        logger.error(f"初始化RAG工具失败: {e}")
        return False

