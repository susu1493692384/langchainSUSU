#!/usr/bin/env python3
"""
LangChain Embeddings 完全使用指南
"""

import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
from sklearn.metrics.pairwise import cosine_similarity

# 加载环境变量
load_dotenv()

# ========================
# 1. 基本Embeddings使用
# ========================

def basic_embeddings_demo():
    """基本Embeddings使用演示"""
    print("=" * 60)
    print("🔍 基本Embeddings使用演示")
    print("=" * 60)

    # 初始化OpenAI Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",  # 或 "text-embedding-3-small", "text-embedding-3-large"
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    # 示例文本
    texts = [
        "人工智能是计算机科学的一个分支",
        "AI致力于创建智能机器",
        "机器学习是AI的核心技术",
        "深度学习使用神经网络",
        "今天天气很好",
        "苹果是一种水果"
    ]

    print(f"\n📝 处理 {len(texts)} 个文本...")

    # 生成embeddings
    print("\n🔄 生成embeddings...")
    embeddings_list = embeddings.embed_documents(texts)

    print(f"✅ 生成了 {len(embeddings_list)} 个embedding向量")
    print(f"📏 每个向量的维度: {len(embeddings_list[0])}")

    # 查询单个文本的embedding
    query_text = "什么是深度学习？"
    query_embedding = embeddings.embed_query(query_text)
    print(f"\n❓ 查询文本: {query_text}")
    print(f"📏 查询向量维度: {len(query_embedding)}")

    return embeddings, embeddings_list, query_embedding

# ========================
# 2. 相似性计算
# ========================

def similarity_calculations_demo(embeddings, embeddings_list, query_embedding):
    """相似性计算演示"""
    print("\n" + "=" * 60)
    print("📊 文本相似性计算演示")
    print("=" * 60)

    # 示例文本
    texts = [
        "人工智能是计算机科学的一个分支",
        "AI致力于创建智能机器",
        "机器学习是AI的核心技术",
        "深度学习使用神经网络",
        "今天天气很好",
        "苹果是一种水果"
    ]

    # 计算余弦相似度
    similarities = []
    for i, doc_embedding in enumerate(embeddings_list):
        similarity = cosine_similarity(
            [query_embedding],
            [doc_embedding]
        )[0][0]
        similarities.append((i, texts[i], similarity))

    # 按相似度排序
    similarities.sort(key=lambda x: x[2], reverse=True)

    print(f"\n🎯 与查询 '{texts[3]}' 最相似的文本:")
    print("-" * 60)
    for i, (idx, text, similarity) in enumerate(similarities):
        print(f"{i+1}. {text}")
        print(f"   相似度: {similarity:.4f}")
        print()

    return similarities

# ========================
# 3. Hugging Face Embeddings
# ========================

def huggingface_embeddings_demo():
    """Hugging Face Embeddings演示"""
    print("\n" + "=" * 60)
    print("🤗 Hugging Face Embeddings演示")
    print("=" * 60)

    try:
        # 使用中文模型
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",  # 中文embedding模型
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        texts = [
            "机器学习是人工智能的重要分支",
            "深度学习属于机器学习的一个子集",
            "自然语言处理是AI的应用领域",
            "今天天气晴朗，适合外出"
        ]

        print(f"\n📝 使用模型: BAAI/bge-small-zh-v1.5")
        print(f"🔄 正在处理 {len(texts)} 个中文文本...")

        # 生成embeddings
        embeddings_list = embeddings.embed_documents(texts)
        print(f"✅ 生成完成，向量维度: {len(embeddings_list[0])}")

        # 计算相似度
        query = "人工智能技术"
        query_embedding = embeddings.embed_query(query)

        similarities = []
        for i, doc_embedding in enumerate(embeddings_list):
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append((texts[i], similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🎯 与查询 '{query}' 的相似度排序:")
        for text, similarity in similarities:
            print(f"   {text}: {similarity:.4f}")

        return embeddings

    except Exception as e:
        print(f"❌ Hugging Face Embeddings出错: {e}")
        print("💡 提示: 需要安装 transformers, sentence_transformers 库")
        return None

# ========================
# 4. 批处理优化
# ========================

def batch_processing_demo():
    """批处理优化演示"""
    print("\n" + "=" * 60)
    print("⚡ 批处理优化演示")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    # 生成大量文本
    texts = [f"这是第{i}个示例文本，用于演示批处理效果" for i in range(100)]

    import time

    # 逐个处理
    print("\n🐌 逐个处理模式:")
    start_time = time.time()
    individual_embeddings = []
    for text in texts[:10]:  # 只处理前10个作为示例
        embedding = embeddings.embed_query(text)
        individual_embeddings.append(embedding)
    individual_time = time.time() - start_time
    print(f"   处理10个文本耗时: {individual_time:.2f}秒")

    # 批量处理
    print("\n🚀 批量处理模式:")
    start_time = time.time()
    batch_embeddings = embeddings.embed_documents(texts[:10])
    batch_time = time.time() - start_time
    print(f"   处理10个文本耗时: {batch_time:.2f}秒")

    print(f"\n💡 批量处理比逐个处理快 {individual_time/batch_time:.1f} 倍")

# ========================
# 5. 向量数据库集成
# ========================

def vector_store_demo():
    """向量数据库集成演示"""
    print("\n" + "=" * 60)
    print("🗄️ 向量数据库集成演示")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    # 创建示例文档
    documents = [
        Document(page_content="Python是一种流行的编程语言", metadata={"source": "编程指南"}),
        Document(page_content="机器学习算法可以从数据中学习模式", metadata={"source": "AI教程"}),
        Document(page_content="深度学习使用多层神经网络", metadata={"source": "AI教程"}),
        Document(page_content="JavaScript主要用于网页开发", metadata={"source": "Web开发"}),
        Document(page_content="自然语言处理帮助计算机理解人类语言", metadata={"source": "AI教程"})
    ]

    print(f"📝 准备了 {len(documents)} 个文档")

    # 创建FAISS向量存储
    print("\n🔧 创建FAISS向量存储...")
    faiss_store = FAISS.from_documents(documents, embeddings)
    print("✅ FAISS存储创建完成")

    # 相似性搜索
    query = "人工智能相关技术"
    print(f"\n🔍 搜索查询: {query}")

    # 检索相似文档
    similar_docs = faiss_store.similarity_search(query, k=3)

    print(f"\n📋 找到 {len(similar_docs)} 个相似文档:")
    for i, doc in enumerate(similar_docs, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   来源: {doc.metadata.get('source', '未知')}")
        print()

    # 带分数的相似性搜索
    print("📊 带相似度分数的搜索:")
    docs_with_scores = faiss_store.similarity_search_with_score(query, k=3)

    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   相似度分数: {score:.4f}")
        print()

    return faiss_store

# ========================
# 6. 实际应用场景
# ========================

def real_world_applications():
    """实际应用场景演示"""
    print("\n" + "=" * 60)
    print("🌍 实际应用场景演示")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    # 场景1: 文档分类
    print("\n📂 场景1: 文档分类")
    documents = [
        "苹果公司发布了新的iPhone手机",
        "研究表明苹果含有丰富的维生素",
        "谷歌推出新的AI搜索功能",
        "健康饮食建议每天吃水果",
        "微软收购了OpenAI"
    ]

    # 预定义类别
    categories = {
        "科技": ["苹果公司", "iPhone", "谷歌", "AI", "微软", "OpenAI"],
        "健康": ["苹果", "维生素", "健康饮食", "水果"]
    }

    # 为每个文档分类
    doc_embeddings = embeddings.embed_documents(documents)

    for i, (doc, embedding) in enumerate(zip(documents, doc_embeddings)):
        # 简单的关键词匹配分类（实际应用中会更复杂）
        if any(tech_word in doc for tech_word in categories["科技"]):
            category = "科技"
        else:
            category = "健康"

        print(f"文档 {i+1}: {doc[:30]}...")
        print(f"分类: {category}")
        print()

    # 场景2: 语义搜索
    print("🔍 场景2: 语义搜索")
    knowledge_base = [
        "机器学习是AI的一个子领域，专注于算法和统计模型",
        "深度学习是机器学习的一个分支，使用神经网络",
        "自然语言处理帮助计算机理解和生成人类语言",
        "计算机视觉使机器能够理解和分析图像",
        "强化学习通过试错来训练智能体"
    ]

    # 创建向量存储
    vector_store = FAISS.from_texts(knowledge_base, embeddings)

    # 语义搜索查询
    queries = [
        "如何让计算机看懂图片",
        "AI如何学习",
        "智能聊天机器人原理"
    ]

    for query in queries:
        print(f"\n❓ 查询: {query}")
        results = vector_store.similarity_search(query, k=2)
        for j, result in enumerate(results, 1):
            print(f"   {j}. {result.page_content}")

# ========================
# 7. 性能优化技巧
# ========================

def performance_optimization():
    """性能优化技巧演示"""
    print("\n" + "=" * 60)
    print("⚡ 性能优化技巧演示")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # 使用更小更快的模型
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    # 技巧1: 缓存embeddings
    print("\n💾 技巧1: 缓存embeddings")

    embedding_cache = {}

    def get_cached_embedding(text):
        if text not in embedding_cache:
            embedding_cache[text] = embeddings.embed_query(text)
            print(f"   计算并缓存: {text[:20]}...")
        else:
            print(f"   使用缓存: {text[:20]}...")
        return embedding_cache[text]

    # 重复文本演示缓存效果
    texts = ["人工智能技术", "人工智能技术", "机器学习", "人工智能技术"]

    for text in texts:
        _ = get_cached_embedding(text)

    print(f"缓存大小: {len(embedding_cache)}")

    # 技巧2: 选择合适的模型
    print("\n🎯 技巧2: 模型选择对比")

    models_info = {
        "text-embedding-3-small": {"dimensions": 1536, "cost": "低", "speed": "快"},
        "text-embedding-3-large": {"dimensions": 3072, "cost": "高", "speed": "中"},
        "text-embedding-ada-002": {"dimensions": 1536, "cost": "中", "speed": "中"}
    }

    print("模型对比:")
    for model, info in models_info.items():
        print(f"  {model}:")
        print(f"    维度: {info['dimensions']}")
        print(f"    成本: {info['cost']}")
        print(f"    速度: {info['speed']}")
        print()

    # 技巧3: 批处理大小优化
    print("📦 技巧3: 批处理大小优化")
    print("   建议批处理大小:")
    print("   - OpenAI: 100-1000个文本")
    print("   - 本地模型: 根据GPU内存调整")
    print("   - 过大会导致超时，过小会影响效率")

# ========================
# 8. 错误处理和最佳实践
# ========================

def error_handling_best_practices():
    """错误处理和最佳实践演示"""
    print("\n" + "=" * 60)
    print("🛡️ 错误处理和最佳实践演示")
    print("=" * 60)

    # 最佳实践1: 异常处理
    print("\n🔧 最佳实践1: 异常处理")

    def safe_embed_text(text, embeddings, max_retries=3):
        """安全的文本embedding函数"""
        for attempt in range(max_retries):
            try:
                return embeddings.embed_query(text)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"   ❌ 嵌入失败: {e}")
                    return None
                print(f"   ⚠️ 第{attempt + 1}次尝试失败，重试中...")
                time.sleep(2 ** attempt)  # 指数退避

    # 测试异常处理
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    result = safe_embed_text("测试文本", embeddings)
    print(f"   ✅ 嵌入成功: {result is not None}")

    # 最佳实践2: 文本预处理
    print("\n🧹 最佳实践2: 文本预处理")

    def preprocess_text(text):
        """文本预处理"""
        # 移除多余空白
        text = ' '.join(text.split())
        # 截断过长文本（根据模型限制）
        max_length = 8191  # OpenAI的token限制
        if len(text) > max_length * 4:  # 粗略估算token数
            text = text[:max_length * 4] + "..."
        return text

    long_text = "这是一个很长的文本，" * 1000
    processed_text = preprocess_text(long_text)
    print(f"   原始长度: {len(long_text)} 字符")
    print(f"   处理后长度: {len(processed_text)} 字符")

    # 最佳实践3: 监控和日志
    print("\n📊 最佳实践3: 监控和日志")

    class EmbeddingMonitor:
        def __init__(self):
            self.request_count = 0
            self.total_tokens = 0

        def log_request(self, text_length):
            self.request_count += 1
            # 粗略估算token数（1 token ≈ 4字符）
            estimated_tokens = text_length // 4
            self.total_tokens += estimated_tokens
            print(f"   请求 #{self.request_count}: ~{estimated_tokens} tokens")

    monitor = EmbeddingMonitor()
    test_texts = ["短文本", "这是一个中等长度的测试文本", "这是一个" * 100]

    for text in test_texts:
        monitor.log_request(len(text))

    print(f"   总请求数: {monitor.request_count}")
    print(f"   总token数: ~{monitor.total_tokens}")

# ========================
# 主函数
# ========================

def main():
    """主函数"""
    print("🚀 LangChain Embeddings 完全使用指南")
    print("=" * 80)

    try:
        # 1. 基本用法
        embeddings, embeddings_list, query_embedding = basic_embeddings_demo()

        # 2. 相似性计算
        similarities = similarity_calculations_demo(embeddings, embeddings_list, query_embedding)

        # 3. Hugging Face Embeddings
        hf_embeddings = huggingface_embeddings_demo()

        # 4. 批处理优化
        batch_processing_demo()

        # 5. 向量数据库集成
        vector_store = vector_store_demo()

        # 6. 实际应用场景
        real_world_applications()

        # 7. 性能优化
        performance_optimization()

        # 8. 错误处理和最佳实践
        error_handling_best_practices()

        print("\n" + "=" * 80)
        print("🎉 Embeddings教程完成！")
        print("=" * 80)

        print("\n💡 主要要点总结:")
        print("1. ✅ Embeddings将文本转换为语义向量")
        print("2. 🔍 可用于相似性搜索和文本分析")
        print("3. ⚡ 批处理大幅提升处理效率")
        print("4. 🗄️ 与向量数据库集成构建搜索系统")
        print("5. 🛡️ 需要适当的错误处理和优化")

    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        print("\n💡 请检查:")
        print("1. 环境变量是否正确设置")
        print("2. 网络连接是否正常")
        print("3. API密钥是否有效")

if __name__ == "__main__":
    main()