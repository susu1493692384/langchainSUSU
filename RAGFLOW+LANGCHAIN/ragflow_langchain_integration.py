#!/usr/bin/env python3
"""
RAGFlow + LangChain 集成示例
展示如何在LangChain中使用RAGFlow的知识库
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, ConfigDict
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_openai.chat_models.base import ChatOpenAI as OpenAIChatBase
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 加载环境变量
load_dotenv()

def create_embeddings() -> Embeddings:
    """创建 embeddings 实例，支持多种配置"""
    if os.getenv("GLM_API_KEY"):
        # 使用 GLM 的 embeddings (兼容 OpenAI 格式的 API)
        try:
            return OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "embedding-2"),
                openai_api_key=os.getenv("GLM_API_KEY"),
                openai_api_base=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
            )
        except Exception as e:
            print(f"GLM embeddings 初始化失败: {e}")
            print("回退到 OpenAI embeddings")

    # 默认使用 OpenAI embeddings
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# ========================
# RAGFlow API 连接器
# ========================

class RAGFlowAPIConnector:
    """RAGFlow API连接器 - 通过API访问RAGFlow知识库"""

    def __init__(self,
                 base_url: str = None,
                 api_key: str = None,
                 timeout: int = 60):
        """
        初始化RAGFlow API连接器

        Args:
            base_url: RAGFlow API服务地址 (默认从环境变量 RAGFLOW_API_URL 获取，如果没有则使用 http://localhost:9380)
                       注意：不是 Web UI 端口(9000)，而是 API 服务端口(9380)
            api_key: API密钥 (从环境变量 RAGFLOW_API_KEY 获取)
            timeout: 请求超时时间
        """
        self.base_url = (os.getenv("RAGFLOW_API_URL") if base_url is None else base_url).rstrip('/')
        self.api_key = api_key or os.getenv("RAGFLOW_API_KEY")
        self.timeout = timeout
        self.session = requests.Session()

        # 设置请求头
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}' if self.api_key else None
        })

    def test_connection(self) -> bool:
        """测试RAGFlow连接"""
        try:
            # 尝试多个可能的健康检查端点
            endpoints = ["/api/health", "/health", "/", "/api/v1/datasets"]

            for endpoint in endpoints:
                try:
                    response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                    if response.status_code in [200, 401, 403]:  # 200成功，401/403说明服务可用但需要认证
                        return True
                except:
                    continue

            print("所有健康检查端点都无法访问")
            return False
        except Exception as e:
            print(f"连接RAGFlow失败: {e}")
            return False

    def get_knowledge_bases(self) -> List[Dict]:
        """获取所有知识库列表"""
        try:
            # 使用正确的 RAGFlow API 端点
            response = self.session.get(f"{self.base_url}/api/v1/datasets", timeout=self.timeout)
            if response.status_code == 200:
                result = response.json()

                # RAGFlow API 返回格式: {"code": 0, "data": [...], "message": "success"}
                if result.get("code") == 0 and isinstance(result.get("data"), list):
                    return result.get("data", [])
                else:
                    # API 返回错误
                    print(f"API 错误: {result.get('message', '未知错误')}")
                    return []
            else:
                print(f"获取知识库失败: HTTP {response.status_code}")
                return []
        except Exception as e:
            print(f"获取知识库异常: {e}")
            return []

    def search_knowledge_base(self,
                            kb_name: str,
                            query: str,
                            top_k: int = 5,
                            similarity_threshold: float = 0.7) -> List[Dict]:
        """
        在指定知识库中搜索

        Args:
            kb_name: 知识库名称
            query: 查询内容
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值

        Returns:
            搜索结果列表
        """
        try:
            # RAGFlow 搜索 API 参数格式
            data = {
                "question": query,
                "dataset_ids": [kb_name],  # RAGFlow 使用 dataset_ids 数组
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            }

            # 使用 RAGFlow 的搜索端点
            response = self.session.post(
                f"{self.base_url}/api/v1/retrieval",
                json=data,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()

                # RAGFlow API 返回格式检查
                if result.get("code") == 0:
                    # 检查是否有数据
                    data = result.get("data", {})
                    if isinstance(data, dict):
                        # 检查不同的可能的数据结构
                        chunks = data.get("chunks", [])
                        if not chunks:
                            # 如果没有 chunks，可能数据在 "data" 下面
                            if isinstance(data, list):
                                chunks = data
                            else:
                                # 尝试其他字段
                                chunks = data.get("documents", data.get("results", []))

                        if isinstance(chunks, list) and chunks:
                            # 转换为统一的格式
                            formatted_results = []
                            for chunk in chunks:
                                # 处理不同的数据结构
                                if isinstance(chunk, dict):
                                    content = chunk.get("content", chunk.get("text", str(chunk)))
                                    score = chunk.get("similarity", chunk.get("score", 0.0))
                                    doc_id = chunk.get("document_id", chunk.get("id", ""))
                                    title = chunk.get("document_name", chunk.get("title", chunk.get("document_keyword", "")))
                                    source = chunk.get("document_source", chunk.get("source", "ragflow"))

                                    formatted_results.append({
                                        "content": content,
                                        "source": source,
                                        "score": score,
                                        "doc_id": doc_id,
                                        "title": title,
                                        "url": chunk.get("document_source", ""),
                                        "raw_data": chunk
                                    })
                                else:
                                    # 如果 chunk 不是字典，转换为字符串
                                    formatted_results.append({
                                        "content": str(chunk),
                                        "source": "ragflow",
                                        "score": 0.0,
                                        "doc_id": "",
                                        "title": "",
                                        "url": "",
                                        "raw_data": chunk
                                    })
                            return formatted_results

                    # 如果没有找到 chunks，返回空列表
                    return []
                else:
                    print(f"搜索 API 错误: {result.get('message', '未知错误')}")
                    return []
            else:
                print(f"搜索失败: HTTP {response.status_code}")
                return []

        except Exception as e:
            print(f"搜索异常: {e}")
            return []

    def get_document_content(self, doc_id: str) -> str:
        """获取文档内容"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/documents/{doc_id}",
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json().get("content", "")
            else:
                print(f"获取文档内容失败: {response.status_code}")
                return ""

        except Exception as e:
            print(f"获取文档内容异常: {e}")
            return ""

# ========================
# RAGFlow 检索器
# ========================

class RAGFlowRetriever(BaseRetriever):
    """RAGFlow检索器 - 将RAGFlow集成到LangChain中"""

    connector: RAGFlowAPIConnector = Field(description="RAGFlow连接器")
    kb_name: str = Field(description="知识库名称")
    top_k: int = Field(default=5, description="返回结果数量")
    similarity_threshold: float = Field(default=0.7, description="相似度阈值")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """检索相关文档"""
        # 调用RAGFlow API搜索
        search_results = self.connector.search_knowledge_base(
            kb_name=self.kb_name,  # 使用知识库ID或名称
            query=query,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )

        # 转换为LangChain Document格式
        documents = []
        for result in search_results:
            doc = Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": result.get("source", "ragflow"),
                    "score": result.get("score", 0.0),
                    "doc_id": result.get("doc_id", ""),
                    "kb_name": self.kb_name,
                    "title": result.get("title", ""),
                    "url": result.get("url", "")
                }
            )
            documents.append(doc)

        return documents

    def get_relevant_documents(self, query: str) -> List[Document]:
        """公共方法：检索相关文档"""
        return self._get_relevant_documents(query)

# ========================
# 多知识库检索器
# ========================

class MultiKBRetriever(BaseRetriever):
    """多知识库检索器 - 能够同时搜索多个知识库"""

    retrievers: Dict[str, Any] = Field(default_factory=dict, description="知识库检索器字典")
    app: Any = Field(description="RAGFlow应用实例")

    def __init__(self, app, kb_names):
        super().__init__(
            app=app,
            retrievers={}
        )
        for kb_name in kb_names:
            self.retrievers[kb_name] = app.create_retriever(kb_name)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """从所有知识库中检索相关文档"""
        all_docs = []
        for kb_name, retriever in self.retrievers.items():
            docs = retriever.get_relevant_documents(query)
            # 添加知识库标识
            for doc in docs:
                doc.metadata["knowledge_base"] = kb_name
            all_docs.extend(docs)

        # 按相似度排序
        all_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
        return all_docs[:10]  # 返回前10个结果

    def get_relevant_documents(self, query: str) -> List[Document]:
        """公共方法：检索相关文档"""
        return self._get_relevant_documents(query)

    model_config = ConfigDict(arbitrary_types_allowed=True)

# ========================
# RAGFlow 数据导出导入工具
# ========================

class RAGFlowDataMigrator:
    """RAGFlow数据迁移工具 - 从RAGFlow导出数据到LangChain"""

    def __init__(self, connector: RAGFlowAPIConnector):
        self.connector = connector

    def export_knowledge_base(self, kb_name: str, output_file: str) -> bool:
        """导出知识库数据"""
        try:
            # 获取知识库中的所有文档（简化版本，实际可能需要分页）
            # 这里假设RAGFlow有导出API
            export_data = {
                "kb_name": kb_name,
                "documents": [],
                "metadata": {
                    "export_time": str(datetime.now()),
                    "source": "ragflow"
                }
            }

            # 实际实现需要调用RAGFlow的导出API
            # response = self.connector.export_kb(kb_name)
            # export_data["documents"] = response.json()

            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            print(f"知识库 '{kb_name}' 已导出到 {output_file}")
            return True

        except Exception as e:
            print(f"导出知识库失败: {e}")
            return False

    def import_to_langchain_vectorstore(self,
                                       export_file: str,
                                       embeddings: Embeddings,
                                       vectorstore_type: str = "faiss") -> Any:
        """导入到LangChain向量存储"""
        try:
            # 加载导出的数据
            with open(export_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 转换为LangChain Document格式
            documents = []
            for doc_data in data["documents"]:
                doc = Document(
                    page_content=doc_data.get("content", ""),
                    metadata={
                        "source": doc_data.get("source", "ragflow"),
                        "kb_name": data["kb_name"],
                        "doc_id": doc_data.get("doc_id", ""),
                        "title": doc_data.get("title", ""),
                        **doc_data.get("metadata", {})
                    }
                )
                documents.append(doc)

            # 创建向量存储
            if vectorstore_type.lower() == "faiss":
                vectorstore = FAISS.from_documents(documents, embeddings)
            elif vectorstore_type.lower() == "chroma":
                vectorstore = Chroma.from_documents(documents, embeddings)
            else:
                raise ValueError(f"不支持的向量存储类型: {vectorstore_type}")

            print(f"成功导入 {len(documents)} 个文档到 {vectorstore_type}")
            return vectorstore

        except Exception as e:
            print(f"导入到LangChain失败: {e}")
            return None

# ========================
# RAGFlow + LangChain 应用示例
# ========================

class RAGFlowLangChainApp:
    """RAGFlow + LangChain 应用类"""

    def __init__(self,
                 ragflow_url: str = None,
                 ragflow_api_key: str = None,
                 llm_model: str = "glm-4.5"):
        """
        初始化应用

        Args:
            ragflow_url: RAGFlow API服务地址 (默认从环境变量获取)
            ragflow_api_key: RAGFlow API密钥 (默认从环境变量获取)
            llm_model: 使用的LLM模型
        """
        # 初始化RAGFlow连接器
        self.connector = RAGFlowAPIConnector(
            base_url=ragflow_url,
            api_key=ragflow_api_key
        )

        # 初始化LLM - 支持 GLM 或 OpenAI
        if os.getenv("GLM_API_KEY"):
            # 使用 GLM (兼容 OpenAI 格式的 API)
            self.llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "GLM-4.5"),
                temperature=0.1,
                openai_api_key=os.getenv("GLM_API_KEY"),
                openai_api_base=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4")
            )
        else:
            # 使用 OpenAI (默认)
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=0.1,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

        # 可用的知识库
        self.available_kbs = []

        # 创建的检索器缓存
        self.retrievers = {}

    def initialize(self) -> bool:
        """初始化应用"""
        print("正在初始化RAGFlow + LangChain应用...")

        # 测试RAGFlow连接
        if not self.connector.test_connection():
            print("❌ RAGFlow连接失败，请检查服务是否运行")
            return False

        # 获取可用知识库
        self.available_kbs = self.connector.get_knowledge_bases()
        print(f"✅ 连接成功，发现 {len(self.available_kbs)} 个知识库")

        for kb in self.available_kbs:
            if isinstance(kb, str):
                print(f"  - {kb}")
            elif isinstance(kb, dict):
                print(f"  - {kb.get('name', '未知')}: {kb.get('description', '无描述')}")
            else:
                print(f"  - {str(kb)}")

        return True

    def create_retriever(self, kb_name: str, top_k: int = 5) -> Optional[RAGFlowRetriever]:
        """创建RAGFlow检索器"""
        # 处理知识库名称比较，支持字符串和字典格式
        available_kb_identifiers = []
        for kb in self.available_kbs:
            if isinstance(kb, str):
                available_kb_identifiers.append(kb)
            elif isinstance(kb, dict):
                # 同时支持 ID 和 名称
                available_kb_identifiers.append(kb.get('id'))
                available_kb_identifiers.append(kb.get('name'))

        if kb_name not in available_kb_identifiers:
            print(f"知识库 '{kb_name}' 不存在")
            print(f"可用知识库标识符: {[x for x in available_kb_identifiers if x]}")
            return None

        retriever = RAGFlowRetriever(
            connector=self.connector,
            kb_name=kb_name,
            top_k=top_k,
            similarity_threshold=0.1
        )

        self.retrievers[kb_name] = retriever
        print(f"✅ 为知识库 '{kb_name}' 创建检索器成功")

        return retriever

    def create_multi_kb_retriever(self, kb_names: List[str] = None):
        """创建多知识库检索器"""
        if kb_names is None:
            # 如果没有指定知识库名称，使用所有可用的知识库
            kb_names = []
            for kb in self.available_kbs:
                if isinstance(kb, str):
                    kb_names.append(kb)
                elif isinstance(kb, dict):
                    kb_names.append(kb.get('id'))

        return MultiKBRetriever(self, kb_names)

    def create_multi_kb_qa_chain(self, multi_retriever: MultiKBRetriever, chain_type: str = "with_sources") -> Any:
        """创建多知识库QA链"""
        if chain_type == "basic":
            return self._create_basic_multi_kb_qa_chain(multi_retriever)
        elif chain_type == "contextual":
            return self._create_contextual_multi_kb_qa_chain(multi_retriever)
        elif chain_type == "with_sources":
            return self._create_multi_kb_qa_chain_with_sources(multi_retriever)
        else:
            raise ValueError(f"不支持的链类型: {chain_type}")

    def _create_basic_multi_kb_qa_chain(self, multi_retriever: MultiKBRetriever):
        """创建基础多知识库QA链"""
        template = """你是一个专业的AI助手。请基于以下提供的上下文信息来回答用户的问题。
如果上下文中没有相关信息，请诚实地说明，不要编造信息。

上下文信息：
{context}

用户问题：{question}

请提供详细、准确的回答："""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context": multi_retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _create_multi_kb_qa_chain_with_sources(self, multi_retriever: MultiKBRetriever):
        """创建带来源的多知识库QA链"""
        template = """你是一个专业的AI助手。请基于以下提供的上下文信息来回答用户的问题，并在回答中引用信息来源。

上下文信息：
{context}

用户问题：{question}

请提供详细、准确的回答，并标注信息来源："""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        def format_docs_with_sources(docs):
            """格式化文档，包含来源信息"""
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知")
                title = doc.metadata.get("title", "")
                kb_name = doc.metadata.get("knowledge_base", "未知")
                score = doc.metadata.get("score", 0.0)

                doc_content = f"文档 {i} (知识库: {kb_name}, 来源: {source}, 相似度: {score:.3f})\n"
                if title:
                    doc_content += f"标题: {title}\n"
                doc_content += f"内容: {doc.page_content}"

                formatted_docs.append(doc_content)

            return "\n\n---\n\n".join(formatted_docs)

        chain = (
            {
                "context": multi_retriever | format_docs_with_sources,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def create_qa_chain(self, kb_name: str, chain_type: str = "with_sources") -> Any:
        """创建QA链"""
        retriever = self.retrievers.get(kb_name)
        if not retriever:
            retriever = self.create_retriever(kb_name)
            if not retriever:
                return None

        if chain_type == "basic":
            return self._create_basic_qa_chain(retriever)
        elif chain_type == "contextual":
            return self._create_contextual_qa_chain(retriever)
        elif chain_type == "with_sources":
            return self._create_qa_chain_with_sources(retriever)
        else:
            raise ValueError(f"不支持的链类型: {chain_type}")

    def _create_basic_qa_chain(self, retriever: RAGFlowRetriever):
        """创建基础QA链"""
        template = """你是一个专业的AI助手。请基于以下提供的上下文信息来回答用户的问题。
如果上下文中没有相关信息，请诚实地说明，不要编造信息。

上下文信息：
{context}

用户问题：{question}

请提供详细、准确的回答："""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _create_contextual_qa_chain(self, retriever: RAGFlowRetriever):
        """创建上下文增强QA链"""
        template = """你是一个专业的AI助手，专门帮助用户基于知识库内容回答问题。

知识库上下文：
{context}

用户问题：{question}

请基于上下文信息提供准确、详细的回答。如果上下文中信息不足，请明确说明。
回答时请保持专业性和友好性："""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs_with_scores(docs):
            """格式化文档，包含相似度分数"""
            formatted_docs = []
            for doc in docs:
                score = doc.metadata.get("score", 0.0)
                source = doc.metadata.get("source", "未知")
                formatted_docs.append(f"[相似度: {score:.3f}] 来源: {source}\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted_docs)

        chain = (
            {
                "context": retriever | format_docs_with_scores,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _create_qa_chain_with_sources(self, retriever: RAGFlowRetriever):
        """创建带来源引用的QA链"""
        template = """你是一个专业的AI助手。请基于以下提供的上下文信息来回答用户的问题，并在回答中引用信息来源。

上下文信息：
{context}

用户问题：{question}

请提供详细、准确的回答，并标注信息来源："""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        def format_docs_with_sources(docs):
            """格式化文档，包含来源信息"""
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知")
                title = doc.metadata.get("title", "")
                score = doc.metadata.get("score", 0.0)

                doc_content = f"文档 {i} (来源: {source}, 相似度: {score:.3f})\n"
                if title:
                    doc_content += f"标题: {title}\n"
                doc_content += f"内容: {doc.page_content}"

                formatted_docs.append(doc_content)

            return "\n\n---\n\n".join(formatted_docs)

        chain = (
            {
                "context": retriever | format_docs_with_sources,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def chat(self, kb_name: str, question: str, chain_type: str) -> str:
        """与知识库对话"""
        chain = self.create_qa_chain(kb_name, chain_type)
        if not chain:
            return "无法创建问答链，请检查知识库配置"

        try:
            return chain.invoke(question)
        except Exception as e:
            return f"回答问题时出错: {e}"

# ========================
# 演示函数
# ========================

def demo_ragflow_langchain_integration():
    """演示RAGFlow + LangChain集成"""
    print("=" * 60)
    print("🚀 RAGFlow + LangChain 集成演示")
    print("=" * 60)

    # 创建应用实例
    app = RAGFlowLangChainApp(
        ragflow_url="http://localhost:9380",  # RAGFlow服务地址
        ragflow_api_key=os.getenv("RAGFLOW_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "GLM-4.5")  # 从环境变量获取LLM模型
    )

    # 初始化应用
    if not app.initialize():
        print("❌ 应用初始化失败")
        return

    # 选择知识库
    if not app.available_kbs:
        print("❌ 没有可用的知识库")
        print("请先在RAGFlow中创建知识库并添加文档")
        return

    print(f"\n可用知识库数量: {len(app.available_kbs)}")
    print("所有可用知识库:")
    for i, kb in enumerate(app.available_kbs, 1):
        if isinstance(kb, str):
            print(f"{i}. {kb}")
        elif isinstance(kb, dict):
            kb_id = kb.get('id', 'unknown')
            kb_name = kb.get('name', 'unknown')
            kb_desc = kb.get('description', '无描述')
            doc_count = kb.get('document_count', 0)
            chunk_count = kb.get('chunk_count', 0)
            print(f"{i}. {kb_name} (ID: {kb_id})")
            print(f"   描述: {kb_desc}")
            print(f"   文档数: {doc_count}, Chunk数: {chunk_count}")
        else:
            print(f"{i}. {str(kb)}")

    print(f"\n🚀 使用所有知识库进行检索...")

    # 创建多知识库检索器
    multi_retriever = app.create_multi_kb_retriever()
    print(f"✅ 多知识库检索器创建成功，包含 {len(multi_retriever.retrievers)} 个知识库")

    # 测试检索
    test_queries = [
        "王书友是什么岗位?",
        "王书友上周做了什么",
        "总结近几周王书友的工作内容"
    ]

    print("\n📊 测试多知识库检索功能:")
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        docs = multi_retriever.get_relevant_documents(query)

        print(f"📋 找到 {len(docs)} 个相关文档:")
        for i, doc in enumerate(docs, 1):
            score = doc.metadata.get("score", 0.0)
            source = doc.metadata.get("source", "未知")
            kb_name = doc.metadata.get("knowledge_base", "未知")
            title = doc.metadata.get("title", "")
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content

            print(f"  {i}. [分数: {score:.3f}] 知识库: {kb_name}")
            if title:
                print(f"     标题: {title}")
            print(f"     来源: {source}")
            print(f"     内容: {content_preview}")

    # 测试多知识库问答
    print("\n💬 测试多知识库问答功能:")

    # 创建多知识库QA链
    multi_kb_qa_chain = app.create_multi_kb_qa_chain(multi_retriever, chain_type="with_sources")

    for query in test_queries[:2]:  # 只测试前两个问题
        print(f"\n❓ 问题: {query}")

        try:
            # 多知识库问答
            answer = multi_kb_qa_chain.invoke(query)
            print(f"🤖 多知识库回答:")
            print(answer)
        except Exception as e:
            print(f"❌ 多知识库问答出错: {e}")


def main():
    """主函数"""
    print("RAGFlow + LangChain 集成指南")
    print("=" * 60)
    print("本示例展示如何在LangChain中使用RAGFlow的知识库")

    while True:
        print("\n" + "=" * 60)
        print("选择演示功能：")
        print("=" * 60)
        print("1. RAGFlow + LangChain 集成演示")
        print("2. 数据迁移功能演示")
        print("3. 连接测试")
        print("0. 退出")

        choice = input("\n请选择 (0-3): ").strip()

        if choice == "0":
            print("\n感谢使用RAGFlow + LangChain集成指南！")
            break
        elif choice == "1":
            demo_ragflow_langchain_integration()
        elif choice == "2":
            # 连接测试
            connector = RAGFlowAPIConnector()
            if connector.test_connection():
                print("✅ RAGFlow连接成功")
                kbs = connector.get_knowledge_bases()
                print(f"发现 {len(kbs)} 个知识库")
                for kb in kbs:
                    # 处理知识库可能是字符串或字典的情况
                    if isinstance(kb, str):
                        print(f"  - {kb}")
                    elif isinstance(kb, dict):
                        print(f"  - {kb.get('name', '未知')}")
                    else:
                        print(f"  - {str(kb)}")
            else:
                print("❌ RAGFlow连接失败")
        else:
            print("无效选择，请输入0-3之间的数字")

if __name__ == "__main__":
    main()