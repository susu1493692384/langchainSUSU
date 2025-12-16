#!/usr/bin/env python3
"""
LangChain 专题教程 - RAG（检索增强生成）完全指南
Retrieval-Augmented Generation 完整开发教程
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.retrievers import BaseRetriever
# Note: Some retriever components may not be available in current LangChain version
# We'll implement basic functionality without these advanced retrievers
# Note: Some chain modules may not be available in current LangChain version
# from langchain.chains import RetrievalQA
# from langchain.chains.conversational_retrieval import ConversationalRetrievalChain
# Note: Memory module may not be available in current LangChain version
# from langchain_core.memory import ConversationBufferMemory
# Note: Callbacks module may not be available in current LangChain version
# from langchain_community.callbacks import get_openai_callback
# Note: Using OpenAI-compatible embeddings instead of ZhipuAiClient
# from zai import ZhipuAiClient
# 加载环境变量
load_dotenv()

# ========================
# RAG核心组件 (RAG Core Components)
# ========================

class RAGSystem:
    """RAG系统主类"""

    def __init__(self,
                 llm_model: str = "glm-4.6",
                 embedding_model: str = "embedding-3",
                 temperature: float = 0.1):
        """
        初始化RAG系统

        Args:
            llm_model: 使用的LLM模型
            embedding_model: 使用的嵌入模型
            temperature: LLM温度参数
        """
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=os.getenv("GLM_API_KEY"),
            openai_api_base=os.getenv("GLM_BASE_URL")
        )
        # 使用 OpenAI 兼容的 embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("GLM_API_KEY"),
            openai_api_base=os.getenv("GLM_BASE_URL")
        )

        self.vector_store = None
        self.retriever = None
        self.qa_chain = None

# ========================
# 文档处理组件 (Document Processing)
# ========================

class DocumentProcessor:
    """文档处理器"""

    def __init__(self):
        self.chunk_strategies = {
            'recursive': RecursiveCharacterTextSplitter,
            'token': TokenTextSplitter
        }

    def create_sample_documents(self) -> List[Document]:
        """创建示例文档集合"""
        docs = [
            Document(
                page_content="""
                人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
                AI包括机器学习、深度学习、自然语言处理、计算机视觉等多个子领域。
                现代AI技术在医疗诊断、自动驾驶、语音识别、图像识别等领域都有广泛应用。
                """,
                metadata={"source": "AI基础知识", "category": "技术", "difficulty": "入门"}
            ),
            Document(
                page_content="""
                机器学习是AI的核心技术之一，它使计算机能够从数据中学习而无需明确编程。
                主要类型包括监督学习、无监督学习和强化学习。
                监督学习使用标记数据训练模型，无监督学习发现数据模式，强化学习通过奖励机制学习。
                """,
                metadata={"source": "机器学习介绍", "category": "技术", "difficulty": "中级"}
            ),
            Document(
                page_content="""
                深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的工作方式。
                卷积神经网络（CNN）在图像识别中表现出色，循环神经网络（RNN）和Transformer在自然语言处理中效果显著。
                GPT、BERT等预训练模型是深度学习在NLP领域的重大突破。
                """,
                metadata={"source": "深度学习原理", "category": "技术", "difficulty": "高级"}
            ),
            Document(
                page_content="""
                LangChain是一个用于构建LLM应用的框架，提供了模块化的组件来简化开发过程。
                核心组件包括Models（模型）、Prompts（提示）、Chains（链）、Memory（记忆）、Retrievers（检索器）和Agents（智能体）。
                LangChain支持多种LLM提供商和向量数据库，使开发者能够快速构建复杂的AI应用。
                """,
                metadata={"source": "LangChain框架", "category": "框架", "difficulty": "中级"}
            ),
            Document(
                page_content="""
                RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。
                它首先从知识库中检索相关文档，然后将检索到的内容作为上下文提供给LLM生成回答。
                RAG能够减少幻觉，提高回答的准确性和时效性，在知识问答、文档分析等场景中表现优异。
                """,
                metadata={"source": "RAG技术", "category": "技术", "difficulty": "中级"}
            ),
            Document(
                page_content="""
                向量数据库是RAG系统的核心组件，用于存储和检索文档的向量表示。
                常见的向量数据库包括FAISS、Chroma、Pinecone、Weaviate等。
                向量数据库使用相似度搜索（如余弦相似度、欧几里得距离）来找到最相关的文档。
                """,
                metadata={"source": "向量数据库", "category": "数据库", "difficulty": "中级"}
            )
        ]
        return docs

    def load_documents_from_files(self, file_paths: List[str]) -> List[Document]:
        """从文件加载文档"""
        documents = []

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"警告: 文件 {file_path} 不存在")
                continue

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": path.name,
                        "file_type": path.suffix,
                        "file_size": len(content),
                        "modified_time": path.stat().st_mtime
                    }
                )
                documents.append(doc)
                print(f"成功加载文档: {path.name}")

            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")

        return documents

    def chunk_documents(self,
                        documents: List[Document],
                        strategy: str = 'recursive',
                        chunk_size: int = 1000,
                        chunk_overlap: int = 200) -> List[Document]:
        """分割文档为块"""

        if strategy not in self.chunk_strategies:
            raise ValueError(f"不支持的分割策略: {strategy}")

        splitter_class = self.chunk_strategies[strategy]

        if strategy == 'recursive':
            splitter = splitter_class(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", " ", ""]
            )
        elif strategy == 'token':
            splitter = splitter_class(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        chunks = splitter.split_documents(documents)

        # 为每个块添加唯一ID
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = f"{chunk.metadata['source']}_chunk_{i}"
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)

        print(f"使用 {strategy} 策略分割文档: {len(documents)} 个文档 → {len(chunks)} 个块")
        return chunks

# ========================
# 向量存储管理 (Vector Storage Management)
# ========================

class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.stores = {}

    def create_faiss_store(self, documents: List[Document]) -> FAISS:
        """创建FAISS向量存储"""
        print("创建FAISS向量存储...")

        # 创建向量存储
        vector_store = FAISS.from_documents(documents, self.embeddings)

        # 添加文档索引
        docstore = {f"doc_{i}": doc for i, doc in enumerate(documents)}
        index_to_docstore_id = {i: f"doc_{i}" for i in range(len(documents))}

        vector_store.docstore = docstore
        vector_store.index_to_docstore_id = index_to_docstore_id

        print(f"FAISS存储创建完成，包含 {len(documents)} 个文档")
        return vector_store

    def create_chroma_store(self,
                          documents: List[Document],
                          collection_name: str = "rag_collection") -> Chroma:
        """创建Chroma向量存储"""
        print(f"创建Chroma向量存储 (集合: {collection_name})...")

        # 创建持久化目录
        persist_directory = f"./chroma_db_{collection_name}"

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        print(f"Chroma存储创建完成，包含 {len(documents)} 个文档")
        return vector_store

    def load_chroma_store(self,
                         collection_name: str = "rag_collection") -> Optional[Chroma]:
        """加载已存在的Chroma存储"""
        persist_directory = f"./chroma_db_{collection_name}"

        if not os.path.exists(persist_directory):
            print(f"Chroma存储不存在: {persist_directory}")
            return None

        try:
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            print(f"成功加载Chroma存储: {collection_name}")
            return vector_store
        except Exception as e:
            print(f"加载Chroma存储失败: {e}")
            return None

    def save_faiss_store(self, vector_store: FAISS, file_path: str):
        """保存FAISS存储到文件"""
        try:
            vector_store.save_local(file_path)
            print(f"FAISS存储已保存到: {file_path}")
        except Exception as e:
            print(f"保存FAISS存储失败: {e}")

    def load_faiss_store(self, file_path: str, embeddings) -> Optional[FAISS]:
        """从文件加载FAISS存储"""
        try:
            vector_store = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
            print(f"成功加载FAISS存储: {file_path}")
            return vector_store
        except Exception as e:
            print(f"加载FAISS存储失败: {e}")
            return None

# ========================
# 高级检索器 (Advanced Retrievers)
# ========================

class AdvancedRetrievers:
    """高级检索器集合 - 简化版本"""

    @staticmethod
    def create_basic_retriever(vector_store, search_kwargs={"k": 3}):
        """创建基础检索器"""
        print("创建基础检索器...")
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    # 注意：以下检索器在当前LangChain版本中可能不可用
    # 我们使用基础检索器来替代

    @staticmethod
    def create_multi_query_retriever_alternative(llm, base_retriever):
        """多查询检索器的替代实现"""
        print("创建多查询检索器（简化版本）...")
        # 暂时返回基础检索器
        return base_retriever

    @staticmethod
    def create_contextual_compression_retriever_alternative(llm, base_retriever):
        """上下文压缩检索器的替代实现"""
        print("创建上下文压缩检索器（简化版本）...")
        # 暂时返回基础检索器
        return base_retriever

    @staticmethod
    def create_parent_document_retriever_alternative(child_splitter,
                                                  parent_splitter,
                                                  vector_store):
        """父子文档检索器的替代实现"""
        print("创建父子文档检索器（简化版本）...")
        # 暂时返回基础检索器
        return vector_store.as_retriever(search_kwargs={"k": 3})

# ========================
# RAG链构建 (RAG Chain Construction)
# ========================

class RAGChainBuilder:
    """RAG链构建器"""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def create_basic_rag_chain(self):
        """创建基础RAG链"""
        print("创建基础RAG链...")

        # RAG提示模板
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

        # 格式化检索到的文档
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 创建链
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def create_conversational_rag_chain(self):
        """创建对话式RAG链（简化版本）"""
        print("创建对话式RAG链（简化版本）...")

        # 对话提示模板
        template = """你是一个专业的对话AI助手。请基于以下提供的上下文信息来回答用户的问题。

        上下文信息：
        {context}

        用户问题：{question}

        请提供自然、连贯的回答："""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # 格式化检索到的文档
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 创建简化的对话链（使用基础RAG链结构）
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def create_rag_with_source_chain(self):
        """创建带来源引用的RAG链"""
        print("创建带来源引用的RAG链...")

        # 格式化文档的函数
        def format_docs(docs):
            return "\n\n".join([
                f"文档来源: {doc.metadata.get('source', '未知')}\n"
                f"内容: {doc.page_content}"
                for doc in docs
            ])

        # RAG提示模板
        template = """你是一个专业的AI助手。请基于以下提供的上下文信息来回答用户的问题。
        在回答中请引用信息的来源。

        上下文信息：
        {context}

        用户问题：{question}

        请提供详细、准确的回答，并标注信息来源："""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # 创建链
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

# ========================
# RAG系统评估 (RAG System Evaluation)
# ========================

class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self, llm):
        self.llm = llm

    def evaluate_retrieval_quality(self,
                                 query: str,
                                 retrieved_docs: List[Document]) -> Dict[str, Any]:
        """评估检索质量"""
        print(f"评估查询 '{query}' 的检索质量...")

        # 评估指标
        metrics = {
            "query": query,
            "retrieved_count": len(retrieved_docs),
            "avg_doc_length": np.mean([len(doc.page_content) for doc in retrieved_docs]) if retrieved_docs else 0,
            "sources": list(set([doc.metadata.get('source', '未知') for doc in retrieved_docs])),
            "categories": list(set([doc.metadata.get('category', '未知') for doc in retrieved_docs]))
        }

        # 相关性评分（简化版本）
        query_words = set(query.lower().split())
        relevance_scores = []

        for doc in retrieved_docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words & doc_words)
            similarity = overlap / len(query_words) if query_words else 0
            relevance_scores.append(similarity)

        metrics["avg_relevance_score"] = np.mean(relevance_scores) if relevance_scores else 0

        return metrics

    def evaluate_response_quality(self,
                                 question: str,
                                 context: str,
                                 response: str) -> Dict[str, Any]:
        """评估回答质量"""
        print("评估回答质量...")

        # 评估提示
        evaluation_prompt = PromptTemplate(
            input_variables=["question", "context", "response"],
            template="""请评估以下AI回答的质量，从多个维度进行评分（1-10分）：

        问题: {question}

        上下文信息: {context}

        AI回答: {response}

        请从以下维度评分：
        1. 准确性 - 回答是否基于上下文信息且准确无误
        2. 完整性 - 是否充分回答了问题
        3. 清晰性 - 表达是否清晰易懂
        4. 相关性 - 是否直接回应了用户问题
        5. 有用性 - 对用户是否有实际帮助

        请以JSON格式返回评分结果："""
        )

        try:
            result = self.llm.invoke(evaluation_prompt.format(
                question=question,
                context=context[:1000] + "..." if len(context) > 1000 else context,
                response=response
            ))

            # 这里应该解析JSON，简化处理直接返回
            return {
                "evaluation": result.content,
                "response_length": len(response),
                "uses_context": any(word in response.lower() for word in context.lower().split()[:10])
            }

        except Exception as e:
            return {
                "error": str(e),
                "evaluation": "评估失败"
            }

    def benchmark_retrieval(self,
                          queries: List[str],
                          retriever) -> Dict[str, Any]:
        """检索性能基准测试"""
        print(f"开始检索基准测试，共 {len(queries)} 个查询...")

        results = []
        total_start_time = time.time()

        for i, query in enumerate(queries):
            print(f"测试查询 {i+1}/{len(queries)}: {query}")

            start_time = time.time()
            try:
                docs = retriever.get_relevant_documents(query)
                end_time = time.time()

                metrics = self.evaluate_retrieval_quality(query, docs)
                metrics["retrieval_time"] = end_time - start_time
                metrics["success"] = True

                results.append(metrics)

            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False,
                    "retrieval_time": time.time() - start_time
                })

        total_time = time.time() - total_start_time

        # 汇总统计
        successful_results = [r for r in results if r.get("success", False)]

        summary = {
            "total_queries": len(queries),
            "successful_queries": len(successful_results),
            "success_rate": len(successful_results) / len(queries) * 100,
            "total_time": total_time,
            "avg_retrieval_time": np.mean([r.get("retrieval_time", 0) for r in successful_results]) if successful_results else 0,
            "avg_retrieved_docs": np.mean([r.get("retrieved_count", 0) for r in successful_results]) if successful_results else 0,
            "avg_relevance_score": np.mean([r.get("avg_relevance_score", 0) for r in successful_results]) if successful_results else 0,
            "detailed_results": results
        }

        return summary

# ========================
# 演示函数 (Demo Functions)
# ========================

def basic_rag_demo():
    """基础RAG演示"""
    print("=" * 60)
    print("🚀 基础RAG系统演示")
    print("=" * 60)

    # 初始化组件
    rag_system = RAGSystem()
    processor = DocumentProcessor()
    vector_manager = VectorStoreManager(rag_system.embeddings)

    # 1. 加载和处理文档
    print("\n📚 步骤1: 加载和处理文档")
    documents = processor.create_sample_documents()
    chunks = processor.chunk_documents(documents, strategy='recursive')

    # 2. 创建向量存储
    print("\n💾 步骤2: 创建向量存储")
    vector_store = vector_manager.create_faiss_store(chunks)

    # 3. 创建检索器
    print("\n🔍 步骤3: 创建检索器")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 4. 构建RAG链
    print("\n⛓️ 步骤4: 构建RAG链")
    chain_builder = RAGChainBuilder(rag_system.llm, retriever)
    rag_chain = chain_builder.create_basic_rag_chain()

    # 5. 测试问答
    print("\n💬 步骤5: 测试问答")
    test_questions = [
        "什么是人工智能？",
        "机器学习有哪些类型？",
        "LangChain是什么？",
        "RAG技术的优势是什么？"
    ]

    for question in test_questions:
        print(f"\n❓ 问题: {question}")
        try:
            start_time = time.time()
            answer = rag_chain.invoke(question)
            end_time = time.time()

            print(f"🤖 回答: {answer}")
            print(f"⏱️ 响应时间: {end_time - start_time:.2f}秒")

        except Exception as e:
            print(f"❌ 处理问题失败: {e}")

        print("-" * 40)

def advanced_rag_demo():
    """高级RAG演示"""
    print("=" * 60)
    print("🚀 高级RAG系统演示（简化版本）")
    print("=" * 60)

    # 初始化组件
    rag_system = RAGSystem()
    processor = DocumentProcessor()
    vector_manager = VectorStoreManager(rag_system.embeddings)
    advanced_retrievers = AdvancedRetrievers()

    # 1. 准备文档和向量存储
    print("\n📚 准备文档和向量存储")
    documents = processor.create_sample_documents()
    chunks = processor.chunk_documents(documents, strategy='recursive')
    vector_store = vector_manager.create_faiss_store(chunks)

    # 2. 创建基础检索器
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. 创建高级检索器（简化版本）
    print("\n🔍 创建高级检索器（简化版本）")

    # 多查询检索器（简化版本）
    multi_query_retriever = advanced_retrievers.create_multi_query_retriever_alternative(
        rag_system.llm, base_retriever
    )

    # 上下文压缩检索器（简化版本）
    compression_retriever = advanced_retrievers.create_contextual_compression_retriever_alternative(
        rag_system.llm, base_retriever
    )

    # 4. 创建不同的RAG链
    print("\n⛓️ 创建不同的RAG链")

    chains = {
        "基础RAG": RAGChainBuilder(rag_system.llm, base_retriever).create_basic_rag_chain(),
        "多查询RAG（简化版）": RAGChainBuilder(rag_system.llm, multi_query_retriever).create_basic_rag_chain(),
        "压缩RAG（简化版）": RAGChainBuilder(rag_system.llm, compression_retriever).create_basic_rag_chain(),
        "带来源RAG": RAGChainBuilder(rag_system.llm, base_retriever).create_rag_with_source_chain()
    }

    # 5. 比较不同检索器的效果
    print("\n📊 比较不同RAG方法的效果")
    test_question = "深度学习和机器学习有什么关系？"

    for method_name, chain in chains.items():
        print(f"\n--- {method_name} ---")
        try:
            start_time = time.time()
            answer = chain.invoke(test_question)
            end_time = time.time()

            print(f"回答: {answer[:200]}...")
            print(f"响应时间: {end_time - start_time:.2f}秒")

        except Exception as e:
            print(f"处理失败: {e}")

def conversational_rag_demo():
    """对话式RAG演示"""
    print("=" * 60)
    print("🚀 对话式RAG系统演示")
    print("=" * 60)

    # 初始化组件
    rag_system = RAGSystem()
    processor = DocumentProcessor()
    vector_manager = VectorStoreManager(rag_system.embeddings)

    # 1. 准备文档和向量存储
    documents = processor.create_sample_documents()
    chunks = processor.chunk_documents(documents)
    vector_store = vector_manager.create_faiss_store(chunks)

    # 2. 创建检索器和对话链
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    chain_builder = RAGChainBuilder(rag_system.llm, retriever)
    conversational_chain = chain_builder.create_conversational_rag_chain()

    # 3. 模拟对话
    print("\n💬 开始对话（输入 'quit' 退出）")

    while True:
        question = input("\n❓ 您的问题: ").strip()
        if question.lower() in ['quit', 'exit', '退出']:
            break

        if not question:
            continue

        try:
            answer = conversational_chain.invoke(question)
            print(f"\n🤖 助手: {answer}")

        except Exception as e:
            print(f"❌ 处理失败: {e}")

def rag_evaluation_demo():
    """RAG系统评估演示"""
    print("=" * 60)
    print("🚀 RAG系统评估演示")
    print("=" * 60)

    # 初始化组件
    rag_system = RAGSystem()
    processor = DocumentProcessor()
    vector_manager = VectorStoreManager(rag_system.embeddings)
    evaluator = RAGEvaluator(rag_system.llm)

    # 1. 准备RAG系统
    documents = processor.create_sample_documents()
    chunks = processor.chunk_documents(documents)
    vector_store = vector_manager.create_faiss_store(chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. 创建RAG链
    chain_builder = RAGChainBuilder(rag_system.llm, retriever)
    rag_chain = chain_builder.create_basic_rag_chain()

    # 3. 检索质量评估
    print("\n📊 检索质量评估")
    test_queries = [
        "什么是AI？",
        "深度学习的应用",
        "LangChain的特点"
    ]

    for query in test_queries:
        print(f"\n🔍 评估查询: {query}")
        docs = retriever.get_relevant_documents(query)
        metrics = evaluator.evaluate_retrieval_quality(query, docs)

        print(f"检索到 {metrics['retrieved_count']} 个文档")
        print(f"平均相关性分数: {metrics['avg_relevance_score']:.3f}")
        print(f"文档来源: {', '.join(metrics['sources'])}")

    # 4. 基准测试
    print("\n🏃 基准性能测试")
    benchmark_results = evaluator.benchmark_retrieval(test_queries, retriever)

    print(f"成功率: {benchmark_results['success_rate']:.1f}%")
    print(f"平均检索时间: {benchmark_results['avg_retrieval_time']:.3f}秒")
    print(f"平均检索文档数: {benchmark_results['avg_retrieved_docs']:.1f}")
    print(f"平均相关性分数: {benchmark_results['avg_relevance_score']:.3f}")

    # 5. 回答质量评估
    print("\n📝 回答质量评估")
    test_question = "什么是RAG技术？"

    # 获取上下文和回答
    docs = retriever.get_relevant_documents(test_question)
    context = "\n".join([doc.page_content for doc in docs])
    response = rag_chain.invoke(test_question)

    # 评估回答质量
    quality_metrics = evaluator.evaluate_response_quality(
        test_question, context, response
    )

    print(f"回答长度: {quality_metrics.get('response_length', 0)} 字符")
    print(f"使用上下文: {'是' if quality_metrics.get('uses_context', False) else '否'}")
    print(f"评估结果: {quality_metrics.get('evaluation', '评估失败')}")

def file_based_rag_demo():
    """基于文件的RAG演示"""
    print("=" * 60)
    print("🚀 基于文件的RAG系统演示")
    print("=" * 60)

    # 创建示例文件
    sample_files = {
        "AI_tutorial.txt": """
人工智能教程
============

1. 人工智能概述
人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。
AI系统可以学习、推理、感知和理解自然语言。

2. 机器学习基础
机器学习是AI的核心技术，使计算机能够从数据中学习。
主要类型：
- 监督学习：使用标记数据训练
- 无监督学习：发现数据模式
- 强化学习：通过奖励学习

3. 深度学习进阶
深度学习使用多层神经网络处理复杂问题。
应用领域包括：
- 图像识别
- 自然语言处理
- 语音识别

4. 实践应用
AI技术在各个领域都有广泛应用：
- 医疗诊断
- 自动驾驶
- 智能客服
- 推荐系统
        """,

        "LangChain_guide.txt": """
LangChain开发指南
=================

1. 框架介绍
LangChain是构建LLM应用的开源框架，提供模块化组件。

2. 核心组件
- Models: 语言模型接口
- Prompts: 提示工程
- Chains: 链式调用
- Memory: 对话记忆
- Retrievers: 文档检索
- Agents: 智能代理

3. 快速开始
安装: pip install langchain
基础使用: from langchain import OpenAI, LLMChain

4. 高级功能
- 自定义组件
- 工具集成
- 多模型支持
        """
    }

    # 保存示例文件
    for filename, content in sample_files.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"创建示例文件: {filename}")

    # 初始化组件
    rag_system = RAGSystem()
    processor = DocumentProcessor()
    vector_manager = VectorStoreManager(rag_system.embeddings)

    # 1. 从文件加载文档
    print("\n📚 从文件加载文档")
    file_paths = list(sample_files.keys())
    documents = processor.load_documents_from_files(file_paths)

    # 2. 分割文档
    chunks = processor.chunk_documents(documents, strategy='recursive')

    # 3. 创建向量存储并保存
    vector_store = vector_manager.create_faiss_store(chunks)
    vector_manager.save_faiss_store(vector_store, "./file_rag_index")

    # 4. 创建RAG链
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    chain_builder = RAGChainBuilder(rag_system.llm, retriever)
    rag_chain = chain_builder.create_rag_with_source_chain()

    # 5. 测试问答
    print("\n💬 测试基于文件的问答")
    test_questions = [
        "如何开始学习机器学习？",
        "LangChain有哪些核心组件？",
        "深度学习有哪些应用？"
    ]

    for question in test_questions:
        print(f"\n❓ {question}")
        try:
            answer = rag_chain.invoke(question)
            print(f"🤖 {answer}")
        except Exception as e:
            print(f"❌ 回答失败: {e}")
        print("-" * 40)

    # 清理文件
    for filename in file_paths:
        try:
            os.remove(filename)
            print(f"删除临时文件: {filename}")
        except:
            pass

# ========================
# 主函数 (Main Function)
# ========================

def main():
    """主函数"""
    print("LangChain RAG（检索增强生成）完全指南")
    print("=" * 60)
    print("本教程将带您深入掌握RAG技术的各个方面")

    demo_options = {
        "1": ("基础RAG系统", basic_rag_demo),
        "2": ("高级RAG技术", advanced_rag_demo),
        "3": ("对话式RAG", conversational_rag_demo),
        "4": ("RAG系统评估", rag_evaluation_demo),
        "5": ("基于文件的RAG", file_based_rag_demo),
        "6": ("运行所有演示", run_all_demos)
    }

    while True:
        print("\n" + "=" * 60)
        print("选择要演示的RAG技术：")
        print("=" * 60)

        for key, (name, _) in demo_options.items():
            print(f"{key}. {name}")
        print("0. 退出")

        choice = input("\n请选择 (0-6): ").strip()

        if choice == "0":
            print("\n感谢使用RAG完全指南！")
            break
        elif choice in demo_options:
            name, demo_func = demo_options[choice]
            print(f"\n开始演示: {name}")
            try:
                demo_func()
            except Exception as e:
                print(f"演示出错: {e}")
                print("请检查环境配置和依赖安装")
        else:
            print("无效选择，请输入0-6之间的数字")

def run_all_demos():
    """运行所有演示"""
    print("\n🚀 运行所有RAG演示...")

    demos = [
        ("基础RAG系统", basic_rag_demo),
        ("高级RAG技术", advanced_rag_demo),
        ("RAG系统评估", rag_evaluation_demo),
        ("基于文件的RAG", file_based_rag_demo)
    ]

    for name, demo_func in demos:
        print(f"\n{'='*80}")
        print(f"🚀 演示: {name}")
        print(f"{'='*80}")
        try:
            demo_func()
        except Exception as e:
            print(f"❌ 演示 '{name}' 出错: {e}")

        input("\n按回车键继续下一个演示...")

if __name__ == "__main__":
    main()