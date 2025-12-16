#!/usr/bin/env python3
"""
LangChain 向量存储和检索示例
展示如何使用向量数据库进行文档检索和问答
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
import json
from typing import List, Dict

# 加载环境变量
load_dotenv()

def create_sample_documents():
    """创建示例文档集合"""
    print("=== 创建示例文档 ===\n")

    documents = [
        Document(
            page_content="""
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
AI包括机器学习、深度学习、自然语言处理、计算机视觉等子领域。

机器学习是AI的核心部分，它使计算机能够从数据中学习，而无需明确编程。
深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。
            """,
            metadata={"source": "AI基础知识", "category": "AI概述"}
        ),
        Document(
            page_content="""
Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。
Python在数据科学、机器学习、Web开发、自动化脚本等领域广泛使用。

Python的主要特点包括：
- 简洁易读的语法
- 丰富的标准库
- 跨平台兼容性
- 强大的社区支持
- 大量第三方库和框架

在AI开发中，TensorFlow、PyTorch、Scikit-learn等流行的机器学习库都提供Python接口。
            """,
            metadata={"source": "Python编程", "category": "编程语言"}
        ),
        Document(
            page_content="""
数据科学是一个跨学科领域，结合了统计学、数学、编程和领域知识来从数据中提取见解。

数据科学工作流程包括：
1. 数据收集和清洗
2. 探索性数据分析
3. 特征工程
4. 模型构建和训练
5. 模型评估和优化
6. 部署和监控

常用的数据科学工具包括：
- 编程语言：Python、R
- 数据处理：Pandas、NumPy
- 可视化：Matplotlib、Seaborn
- 机器学习：Scikit-learn、TensorFlow
            """,
            metadata={"source": "数据科学", "category": "数据科学"}
        ),
        Document(
            page_content="""
深度学习是机器学习的一个分支，使用人工神经网络来学习数据的表示。

深度学习的主要架构包括：
- 卷积神经网络（CNN）：主要用于图像处理
- 循环神经网络（RNN）：主要用于序列数据处理
- Transformer：主要用于自然语言处理
- 生成对抗网络（GAN）：主要用于生成模型

深度学习在以下领域取得了重大突破：
- 图像识别和分类
- 自然语言处理
- 语音识别
- 自动驾驶
- 医疗诊断
            """,
            metadata={"source": "深度学习", "category": "机器学习"}
        ),
        Document(
            page_content="""
提示词工程是设计和优化AI模型输入提示词的艺术和科学。
好的提示词可以显著提高AI模型的输出质量和相关性。

有效的提示词设计原则：
1. 明确具体：清楚地说明你想要什么
2. 提供上下文：给模型足够的信息
3. 设定角色：告诉模型扮演什么角色
4. 指定格式：要求特定的输出格式
5. 使用示例：提供好的例子（Few-shot学习）

提示词工程对于构建可靠、一致的AI应用至关重要。
            """,
            metadata={"source": "提示词工程", "category": "AI应用"}
        )
    ]

    print(f"创建了 {len(documents)} 个示例文档")
    for doc in documents:
        print(f"- {doc.metadata['source']} ({doc.metadata['category']})")
    print()

    return documents

def text_splitting_example(documents):
    """文本分割示例"""
    print("=== 文本分割示例 ===\n")

    # 字符分割器
    char_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )

    char_chunks = char_splitter.split_documents(documents)
    print(f"字符分割器生成了 {len(char_chunks)} 个文档块")

    # 递归字符分割器
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    recursive_chunks = recursive_splitter.split_documents(documents)
    print(f"递归分割器生成了 {len(recursive_chunks)} 个文档块")

    # 显示分割结果
    print("\n分割结果示例:")
    for i, chunk in enumerate(recursive_chunks[:3]):
        print(f"块 {i+1} (来源: {chunk.metadata['source']}):")
        print(f"{chunk.page_content[:150]}...")
        print()

    return recursive_chunks

def vector_store_example(chunks):
    """向量存储示例"""
    print("=== 向量存储示例 ===\n")

    # 创建嵌入模型
    embeddings = OpenAIEmbeddings()

    # 使用FAISS创建向量存储
    print("创建FAISS向量存储...")
    faiss_vectorstore = FAISS.from_documents(chunks, embeddings)
    print("FAISS向量存储创建完成")

    # 使用Chroma创建向量存储
    print("创建Chroma向量存储...")
    chroma_vectorstore = Chroma.from_documents(chunks, embeddings)
    print("Chroma向量存储创建完成")

    return faiss_vectorstore, chroma_vectorstore, embeddings

def similarity_search_example(vectorstore):
    """相似性搜索示例"""
    print("=== 相似性搜索示例 ===\n")

    # 测试查询
    queries = [
        "什么是机器学习？",
        "Python有什么特点？",
        "如何学习数据科学？",
        "深度学习应用在哪里？",
        "怎样写好的提示词？"
    ]

    for query in queries:
        print(f"查询: {query}")

        # 相似性搜索
        docs = vectorstore.similarity_search(query, k=2)

        for i, doc in enumerate(docs, 1):
            print(f"  结果 {i} (来源: {doc.metadata['source']}):")
            print(f"    {doc.page_content[:100]}...")

        print()

def max_marginal_relevance_search_example(vectorstore):
    """最大边际相关性搜索示例"""
    print("=== 最大边际相关性搜索示例 ===\n")

    query = "人工智能和机器学习的关系"

    print(f"查询: {query}")

    # 普通相似性搜索
    print("\n普通相似性搜索结果:")
    sim_docs = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(sim_docs, 1):
        print(f"  {i}. {doc.metadata['source']}: {doc.page_content[:50]}...")

    # 最大边际相关性搜索（增加多样性）
    print("\n最大边际相关性搜索结果 (增加多样性):")
    mmr_docs = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)
    for i, doc in enumerate(mmr_docs, 1):
        print(f"  {i}. {doc.metadata['source']}: {doc.page_content[:50]}...")

    print()

def retrieval_qa_example(vectorstore):
    """检索问答链示例"""
    print("=== 检索问答链示例 ===\n")

    # 创建LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 创建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 将所有检索到的文档"塞"进上下文
        retriever=retriever,
        return_source_documents=True
    )

    # 测试问题
    questions = [
        "请解释人工智能和机器学习的关系",
        "Python在数据科学中有什么优势？",
        "深度学习的主要应用领域有哪些？",
        "如何提高AI模型的输出质量？"
    ]

    for question in questions:
        print(f"问题: {question}")
        try:
            result = qa_chain.invoke({"query": question})
            print(f"回答: {result['result']}")
            print("相关文档:")
            for doc in result['source_documents']:
                print(f"  - {doc.metadata['source']}")
        except Exception as e:
            print(f"处理问题时出错: {e}")
        print()

def custom_retrieval_chain_example(vectorstore, embeddings):
    """自定义检索链示例"""
    print("=== 自定义检索链示例 ===\n")

    # 创建LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 创建上下文压缩器
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    # 创建自定义提示模板
    prompt = PromptTemplate(
        template="""基于以下上下文信息回答问题。如果信息不足，请说明无法回答。

上下文:
{context}

问题: {question}

详细回答:""",
        input_variables=["context", "question"]
    )

    # 创建链
    def retrieve_and_answer(question: str):
        """检索并回答问题的自定义函数"""
        # 检索相关文档
        docs = retriever.get_relevant_documents(question)

        # 如果没有相关文档，使用压缩检索器
        if not docs or len(docs[0].page_content) < 50:
            docs = compression_retriever.get_relevant_documents(question)

        # 如果还是没有结果，返回无法回答
        if not docs:
            return "抱歉，我找不到相关信息来回答您的问题。"

        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in docs])

        # 生成回答
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"context": context, "question": question})

        return result, docs

    # 测试问题
    questions = [
        "深度学习和传统机器学习有什么区别？",
        "如何在项目中应用Python进行数据分析？",
        "什么是神经网络？",
        "AI模型的性能如何评估？"
    ]

    for question in questions:
        print(f"问题: {question}")
        try:
            answer, docs = retrieve_and_answer(question)
            print(f"回答: {answer}")
            print("相关文档:")
            for doc in docs:
                print(f"  - {doc.metadata['source']}")
        except Exception as e:
            print(f"处理问题时出错: {e}")
        print()

def conversational_retrieval_example(vectorstore):
    """对话式检索示例"""
    print("=== 对话式检索示例 ===\n")

    # 创建LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 创建对话式检索链
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )

    # 模拟对话
    dialogues = [
        {"question": "我想学习AI，应该从哪里开始？"},
        {"question": "Python在AI开发中重要吗？"},
        {"question": "能推荐一些学习资源吗？"},
        {"question": "刚才提到的主要内容有哪些？"}  # 这个问题需要记忆
    ]

    chat_history = []

    for dialogue in dialogues:
        question = dialogue["question"]
        print(f"用户: {question}")

        try:
            result = conversation_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })

            answer = result["answer"]
            print(f"AI助手: {answer}")

            # 更新对话历史
            chat_history.append((question, answer))

            print("相关文档:")
            for doc in result["source_documents"]:
                print(f"  - {doc.metadata['source']}")

        except Exception as e:
            print(f"处理对话时出错: {e}")

        print()

def save_and_load_vectorstore_example(chunks, embeddings):
    """保存和加载向量存储示例"""
    print("=== 保存和加载向量存储示例 ===\n")

    # 创建向量存储
    print("创建向量存储...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 保存向量存储
    save_path = "./faiss_index"
    print(f"保存向量存储到 {save_path}...")
    vectorstore.save_local(save_path)
    print("向量存储已保存")

    # 加载向量存储
    print(f"从 {save_path} 加载向量存储...")
    loaded_vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    print("向量存储已加载")

    # 测试加载的向量存储
    query = "什么是深度学习？"
    print(f"\n测试查询: {query}")
    docs = loaded_vectorstore.similarity_search(query, k=2)

    for i, doc in enumerate(docs, 1):
        print(f"  结果 {i}: {doc.page_content[:100]}...")

    print()

if __name__ == "__main__":
    print("📚 欢迎来到LangChain向量存储和检索学习世界！\n")

    # 创建示例文档
    documents = create_sample_documents()

    # 文本分割
    chunks = text_splitting_example(documents)

    # 向量存储
    faiss_vectorstore, chroma_vectorstore, embeddings = vector_store_example(chunks)

    print("\n" + "="*50 + "\n")

    # 相似性搜索
    similarity_search_example(faiss_vectorstore)

    print("\n" + "="*50 + "\n")

    # 最大边际相关性搜索
    max_marginal_relevance_search_example(faiss_vectorstore)

    print("\n" + "="*50 + "\n")

    # 检索问答链
    retrieval_qa_example(faiss_vectorstore)

    print("\n" + "="*50 + "\n")

    # 自定义检索链
    custom_retrieval_chain_example(faiss_vectorstore, embeddings)

    print("\n" + "="*50 + "\n")

    # 对话式检索
    conversational_retrieval_example(faiss_vectorstore)

    print("\n" + "="*50 + "\n")

    # 保存和加载向量存储
    save_and_load_vectorstore_example(chunks, embeddings)

    print("\n✨ 向量存储和检索示例完成！您已经学会了如何在LangChain中使用向量存储进行文档检索和问答。")