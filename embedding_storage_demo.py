#!/usr/bin/env python3
"""
Embeddings存储原理和数据结构详解
"""

import os
import pickle
import json
import sqlite3
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

# 加载环境变量
load_dotenv()

# ========================
# 1. Embeddings本质演示
# ========================

def embeddings_basic_demo():
    """演示Embeddings的本质"""
    print("=" * 60)
    print("🔢 Embeddings本质演示")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    texts = [
        "人工智能",
        "机器学习",
        "深度学习",
        "神经网络",
        "自然语言处理"
    ]

    print(f"\n📝 处理文本: {texts}")

    # 生成embeddings
    print("\n🔄 生成embeddings...")
    embedding_vectors = embeddings.embed_documents(texts)

    # 展示embedding向量的特点
    print(f"\n📊 Embedding向量信息:")
    print(f"✅ 生成了 {len(embedding_vectors)} 个向量")
    print(f"📏 每个向量的维度: {len(embedding_vectors[0])}")
    print(f"🔢 数据类型: {type(embedding_vectors[0])}")
    print(f"📐 向量示例(前10维): {embedding_vectors[0][:10]}")

    # 保存原始向量数据到文件
    print(f"\n💾 保存原始向量数据...")

    # 保存为numpy格式
    vectors_array = np.array(embedding_vectors)
    np.save("text_vectors.npy", vectors_array)
    print(f"✅ 保存为npy格式: text_vectors.npy")

    # 保存为pickle格式
    with open("text_vectors.pkl", "wb") as f:
        pickle.dump({
            "texts": texts,
            "vectors": embedding_vectors
        }, f)
    print(f"✅ 保存为pickle格式: text_vectors.pkl")

    return texts, embedding_vectors

# ========================
# 2. 向量数据库结构演示
# ========================

def vector_database_structure_demo():
    """演示向量数据库的内部结构"""
    print("\n" + "=" * 60)
    print("🗄️ 向量数据库结构演示")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    # 创建文档
    documents = [
        Document(
            page_content="人工智能是计算机科学的一个分支",
            metadata={"id": 1, "category": "AI基础", "source": "textbook"}
        ),
        Document(
            page_content="机器学习使计算机能够从数据中学习",
            metadata={"id": 2, "category": "ML", "source": "paper"}
        ),
        Document(
            page_content="深度学习使用多层神经网络",
            metadata={"id": 3, "category": "DL", "source": "article"}
        )
    ]

    print(f"\n📝 准备文档: {len(documents)} 个")

    # 生成向量
    print("\n🔄 生成文档向量...")
    vectors = embeddings.embed_documents([doc.page_content for doc in documents])

    # 展示向量数据库的三个核心组件
    print(f"\n🏗️ 向量数据库的核心组件:")

    # 1. 向量索引
    print(f"\n1️⃣ 向量索引 (Vector Index):")
    print(f"   📊 存储内容: {len(vectors)} 个 {len(vectors[0])} 维向量")
    print(f"   🎯 用途: 快速相似性搜索")
    print(f"   📁 文件: FAISS创建 .index 文件")

    # 2. 文档存储
    print(f"\n2️⃣ 文档存储 (Document Store):")
    print(f"   📄 存储内容:")
    for i, doc in enumerate(documents):
        print(f"      ID {doc.metadata['id']}: {doc.page_content[:30]}...")
    print(f"   🎯 用途: 存储原始文本内容")
    print(f"   📁 文件: 通常存储为 pickle 或 JSON")

    # 3. ID映射
    print(f"\n3️⃣ ID映射 (ID Mapping):")
    print(f"   🔗 映射关系:")
    for i, doc in enumerate(documents):
        print(f"      向量索引 {i} → 文档ID {doc.metadata['id']}")
    print(f"   🎯 用途: 连接向量和文档")
    print(f"   📁 文件: 内部索引文件")

    return documents, vectors

# ========================
# 3. 不同存储格式演示
# ========================

def storage_formats_demo(documents, vectors):
    """演示不同的存储格式"""
    print("\n" + "=" * 60)
    print("💾 不同存储格式演示")
    print("=" * 60)

    # 格式1: FAISS (内存 + 文件)
    print(f"\n1️⃣ FAISS 存储格式:")
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    faiss_store = FAISS.from_documents(documents, embeddings)
    faiss_store.save_local("faiss_demo")

    print(f"   📁 生成的文件:")
    for file in os.listdir("."):
        if file.startswith("faiss_demo"):
            print(f"      - {file}")

    print(f"   🎯 特点:")
    print(f"      - 内存数据库，支持持久化")
    print(f"      - 高性能向量搜索")
    print(f"      - 适合大规模数据")

    # 格式2: Chroma (数据库)
    print(f"\n2️⃣ Chroma 存储格式:")
    chroma_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="demo_collection",
        persist_directory="./chroma_demo"
    )

    print(f"   📁 生成的目录结构:")
    if os.path.exists("./chroma_demo"):
        for root, dirs, files in os.walk("./chroma_demo"):
            level = root.replace("./chroma_demo", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

    print(f"   🎯 特点:")
    print(f"      - 真正的数据库系统")
    print(f"      - 支持元数据过滤")
    print(f"      - 持久化存储")

    # 格式3: SQLite (传统数据库)
    print(f"\n3️⃣ SQLite 自定义存储:")

    # 创建自定义数据库
    conn = sqlite3.connect("vectors.db")
    cursor = conn.cursor()

    # 创建表结构
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        content TEXT,
        metadata TEXT,
        vector BLOB
    )
    ''')

    # 插入数据
    for i, (doc, vector) in enumerate(zip(documents, vectors)):
        cursor.execute('''
        INSERT INTO documents (id, content, metadata, vector)
        VALUES (?, ?, ?, ?)
        ''', (
            doc.metadata['id'],
            doc.page_content,
            json.dumps(doc.metadata),
            pickle.dumps(vector)
        ))

    conn.commit()
    print(f"   📁 生成文件: vectors.db")
    print(f"   📊 存储了 {len(documents)} 条记录")

    # 查询演示
    cursor.execute("SELECT id, content FROM documents")
    records = cursor.fetchall()
    print(f"   🔍 查询结果示例:")
    for record in records:
        print(f"      ID {record[0]}: {record[1][:30]}...")

    conn.close()

    return faiss_store, chroma_store

# ========================
# 4. 搜索原理演示
# ========================

def search_principle_demo(faiss_store, chroma_store):
    """演示向量搜索的原理"""
    print("\n" + "=" * 60)
    print("🔍 向量搜索原理演示")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    # 搜索查询
    query = "什么是神经网络？"
    print(f"❓ 搜索查询: {query}")

    # 生成查询向量
    print(f"\n🔄 生成查询向量...")
    query_vector = embeddings.embed_query(query)
    print(f"📏 查询向量维度: {len(query_vector)}")

    # FAISS搜索
    print(f"\n🚀 FAISS相似性搜索:")
    faiss_results = faiss_store.similarity_search_with_score(query, k=2)
    print(f"   找到 {len(faiss_results)} 个相似文档:")
    for i, (doc, score) in enumerate(faiss_results, 1):
        print(f"      {i}. {doc.page_content}")
        print(f"         相似度分数: {score:.4f}")
        print(f"         元数据: {doc.metadata}")

    # Chroma搜索
    print(f"\n🗄️ Chroma相似性搜索:")
    chroma_results = chroma_store.similarity_search_with_score(query, k=2)
    print(f"   找到 {len(chroma_results)} 个相似文档:")
    for i, (doc, score) in enumerate(chroma_results, 1):
        print(f"      {i}. {doc.page_content}")
        print(f"         距离分数: {score:.4f}")
        print(f"         元数据: {doc.metadata}")

    # 手动计算相似度（演示原理）
    print(f"\n🧮 手动计算相似度原理:")

    # 获取所有文档向量
    conn = sqlite3.connect("vectors.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, vector FROM documents")
    db_records = cursor.fetchall()
    conn.close()

    # 计算余弦相似度
    def cosine_similarity(vec1, vec2):
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = []
    for doc_id, content, vector_blob in db_records:
        stored_vector = pickle.loads(vector_blob)
        similarity = cosine_similarity(query_vector, stored_vector)
        similarities.append((doc_id, content, similarity))

    # 按相似度排序
    similarities.sort(key=lambda x: x[2], reverse=True)

    print(f"   相似度计算结果:")
    for i, (doc_id, content, similarity) in enumerate(similarities[:2], 1):
        print(f"      {i}. ID {doc_id}: {content[:30]}...")
        print(f"         余弦相似度: {similarity:.4f}")

# ========================
# 5. 性能对比演示
# ========================

def performance_comparison_demo():
    """演示不同存储方式的性能对比"""
    print("\n" + "=" * 60)
    print("⚡ 存储方式性能对比")
    print("=" * 60)

    import time

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    # 创建大量测试数据
    print(f"\n📊 创建测试数据...")
    test_docs = [
        Document(
            page_content=f"这是第{i}个测试文档，内容是关于AI技术的第{i}个方面",
            metadata={"id": i, "category": f"category_{i % 5}"}
        )
        for i in range(1000)
    ]

    print(f"   生成了 {len(test_docs)} 个测试文档")

    # 测试不同存储方式的性能
    storage_methods = {
        "FAISS": lambda: FAISS.from_documents(test_docs[:100], embeddings),  # 限制数量以节省时间
        "Chroma": lambda: Chroma.from_documents(
            test_docs[:100], embeddings,
            collection_name="perf_test",
            persist_directory="./perf_chroma"
        ),
        "SQLite": lambda: create_sqlite_store(test_docs[:100], embeddings)
    }

    results = {}

    for method_name, create_func in storage_methods.items():
        print(f"\n🔧 测试 {method_name} 存储...")

        # 存储性能测试
        start_time = time.time()
        try:
            store = create_func()
            storage_time = time.time() - start_time

            # 搜索性能测试
            start_time = time.time()
            if hasattr(store, 'similarity_search'):
                _ = store.similarity_search("AI技术", k=5)
            search_time = time.time() - start_time

            results[method_name] = {
                "storage_time": storage_time,
                "search_time": search_time,
                "success": True
            }
            print(f"   ✅ 存储时间: {storage_time:.2f}秒")
            print(f"   ✅ 搜索时间: {search_time:.2f}秒")

        except Exception as e:
            results[method_name] = {
                "error": str(e),
                "success": False
            }
            print(f"   ❌ 测试失败: {e}")

    # 性能总结
    print(f"\n📈 性能对比总结:")
    print(f"{'方法':<10} {'存储时间':<12} {'搜索时间':<12} {'状态':<8}")
    print("-" * 50)
    for method, result in results.items():
        if result["success"]:
            print(f"{method:<10} {result['storage_time']:<12.2f} {result['search_time']:<12.4f} {'成功':<8}")
        else:
            print(f"{method:<10} {'N/A':<12} {'N/A':<12} {'失败':<8}")

def create_sqlite_store(documents, embeddings):
    """创建SQLite向量存储"""
    conn = sqlite3.connect(":memory:")  # 内存数据库
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        content TEXT,
        metadata TEXT,
        vector BLOB
    )
    ''')

    vectors = embeddings.embed_documents([doc.page_content for doc in documents])

    for i, (doc, vector) in enumerate(zip(documents, vectors)):
        cursor.execute('''
        INSERT INTO documents (id, content, metadata, vector)
        VALUES (?, ?, ?, ?)
        ''', (
            doc.metadata['id'],
            doc.page_content,
            json.dumps(doc.metadata),
            pickle.dumps(vector)
        ))

    conn.commit()
    return conn

# ========================
# 6. 实际应用场景演示
# ========================

def real_world_scenarios_demo():
    """演示实际应用场景"""
    print("\n" + "=" * 60)
    print("🌍 实际应用场景演示")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    # 场景1: 文档知识库
    print(f"\n📚 场景1: 文档知识库")
    knowledge_docs = [
        Document(page_content="Python是一种高级编程语言，语法简洁易学",
                metadata={"type": "编程语言", "difficulty": "初级"}),
        Document(page_content="机器学习算法可以从数据中自动学习模式",
                metadata={"type": "AI技术", "difficulty": "高级"}),
        Document(page_content="深度学习使用多层神经网络进行特征学习",
                metadata={"type": "AI技术", "difficulty": "高级"}),
        Document(page_content="SQL是用于管理关系数据库的标准语言",
                metadata={"type": "数据库", "difficulty": "中级"}),
    ]

    knowledge_store = FAISS.from_documents(knowledge_docs, embeddings)

    # 模拟知识库查询
    questions = [
        "如何学习编程？",
        "什么是人工智能？",
        "数据库操作语言"
    ]

    for question in questions:
        print(f"\n❓ 问题: {question}")
        results = knowledge_store.similarity_search(question, k=2)
        for i, doc in enumerate(results, 1):
            print(f"   📄 {i}. {doc.page_content}")
            print(f"      🏷️ 标签: {doc.metadata}")

    # 场景2: 产品推荐系统
    print(f"\n🛒 场景2: 产品推荐系统")
    products = [
        Document(page_content="iPhone 15 Pro - 最新款苹果手机，钛金属设计",
                metadata={"category": "手机", "brand": "Apple", "price": "高端"}),
        Document(page_content="MacBook Pro M3 - 专业笔记本电脑，性能强劲",
                metadata={"category": "笔记本", "brand": "Apple", "price": "高端"}),
        Document(page_content="小米14 - 性价比高的国产旗舰手机",
                metadata={"category": "手机", "brand": "Xiaomi", "price": "中端"}),
        Document(page_content="ThinkPad X1 - 商务办公笔记本，键盘手感好",
                metadata={"category": "笔记本", "brand": "Lenovo", "price": "高端"}),
    ]

    product_store = FAISS.from_documents(products, embeddings)

    # 模拟用户查询
    user_queries = [
        "想要一部拍照好的手机",
        "办公用的笔记本电脑",
        "苹果公司的产品"
    ]

    for query in user_queries:
        print(f"\n🔍 用户查询: {query}")
        results = product_store.similarity_search(query, k=2)
        for i, doc in enumerate(results, 1):
            print(f"   🛍️ {i}. {doc.page_content}")
            print(f"      💰 价格段: {doc.metadata['price']}")

# ========================
# 清理函数
# ========================

def cleanup_demo_files():
    """清理演示生成的文件"""
    print(f"\n🧹 清理演示文件...")

    files_to_remove = [
        "text_vectors.npy", "text_vectors.pkl",
        "faiss_demo.index", "vectors.db"
    ]

    import shutil
    dirs_to_remove = ["chroma_demo", "perf_chroma"]

    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"   🗑️ 删除文件: {file}")

    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   🗑️ 删除目录: {dir_name}")

    print("✅ 清理完成")

# ========================
# 主函数
# ========================

def main():
    """主函数"""
    print("🔍 Embeddings存储原理和数据结构详解")
    print("=" * 80)

    try:
        # 1. Embeddings本质
        texts, vectors = embeddings_basic_demo()

        # 2. 向量数据库结构
        documents, doc_vectors = vector_database_structure_demo()

        # 3. 不同存储格式
        faiss_store, chroma_store = storage_formats_demo(documents, doc_vectors)

        # 4. 搜索原理
        search_principle_demo(faiss_store, chroma_store)

        # 5. 性能对比
        performance_comparison_demo()

        # 6. 实际应用场景
        real_world_scenarios_demo()

        print("\n" + "=" * 80)
        print("🎉 Embeddings存储原理演示完成！")
        print("=" * 80)

        print("\n💡 关键要点总结:")
        print("1. 🔢 Embeddings生成的是数值向量，不是数据库文件")
        print("2. 🗄️ 向量数据库存储三个核心组件：向量索引、文档内容、ID映射")
        print("3. 🚀 不同存储方式有不同特点：FAISS(快)、Chroma(功能全)、SQLite(灵活)")
        print("4. 🔍 搜索原理：计算查询向量与存储向量的相似度")
        print("5. ⚡ 选择存储方式要考虑数据规模、查询频率、功能需求")

    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")

    finally:
        # 清理文件
        cleanup_demo_files()

if __name__ == "__main__":
    main()