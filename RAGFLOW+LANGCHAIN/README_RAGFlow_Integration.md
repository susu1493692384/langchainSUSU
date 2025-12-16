# RAGFlow + LangChain 集成指南

本指南展示如何在LangChain中使用RAGFlow的知识库，实现强大的RAG应用。

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install langchain langchain-openai langchain-community requests python-dotenv

# 或使用requirements.txt
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# RAGFlow 配置
RAGFLOW_BASE_URL=http://localhost:9380
RAGFLOW_API_KEY=your_ragflow_api_key_here

# OpenAI 配置
OPENAI_API_KEY=your_openai_api_key_here

# 模型配置
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002

# 检索配置
DEFAULT_TOP_K=5
DEFAULT_SIMILARITY_THRESHOLD=0.7
```

### 3. 基础使用

```python
from ragflow_langchain_integration import RAGFlowLangChainApp

# 创建应用
app = RAGFlowLangChainApp(
    ragflow_url="http://localhost:9380",
    ragflow_api_key="your_api_key",
    llm_model="gpt-3.5-turbo"
)

# 初始化
if app.initialize():
    # 选择知识库
    kb_name = "your_knowledge_base"

    # 创建QA链
    qa_chain = app.create_qa_chain(kb_name, chain_type="with_sources")

    # 问答
    answer = qa_chain.invoke("什么是人工智能？")
    print(answer)
```

## 📚 核心组件

### 1. RAGFlowAPIConnector

RAGFlow API连接器，负责与RAGFlow服务通信：

```python
from ragflow_langchain_integration import RAGFlowAPIConnector

connector = RAGFlowAPIConnector(
    base_url="http://localhost:9380",
    api_key="your_api_key"
)

# 测试连接
if connector.test_connection():
    print("RAGFlow连接成功")

    # 获取知识库列表
    kbs = connector.get_knowledge_bases()
    print(f"发现 {len(kbs)} 个知识库")

    # 搜索知识库
    results = connector.search_knowledge_base(
        kb_name="tech_docs",
        query="机器学习",
        top_k=5
    )
```

### 2. RAGFlowRetriever

LangChain检索器，将RAGFlow集成到LangChain生态：

```python
from ragflow_langchain_integration import RAGFlowRetriever, RAGFlowAPIConnector

# 创建连接器
connector = RAGFlowAPIConnector()

# 创建检索器
retriever = RAGFlowRetriever(
    connector=connector,
    kb_name="your_knowledge_base",
    top_k=5,
    similarity_threshold=0.1
)

# 检索文档
docs = retriever.get_relevant_documents("AI技术")
for doc in docs:
    print(f"来源: {doc.metadata['source']}")
    print(f"内容: {doc.page_content[:100]}...")
```

### 3. RAGFlowLangChainApp

完整的应用类，提供开箱即用的RAG功能：

```python
from ragflow_langchain_integration import RAGFlowLangChainApp

app = RAGFlowLangChainApp()

if app.initialize():
    # 基础问答
    answer = app.chat("tech_docs", "什么是深度学习？")

    # 带来源的问答
    answer_with_sources = app.chat(
        "tech_docs",
        "机器学习的应用领域",
        chain_type="with_sources"
    )
```

## 🔧 高级功能

### 1. 数据迁移

从RAGFlow导出数据到LangChain向量存储：

```python
from ragflow_langchain_integration import RAGFlowDataMigrator, RAGFlowAPIConnector
from langchain_openai import OpenAIEmbeddings

# 创建连接器和迁移工具
connector = RAGFlowAPIConnector()
migrator = RAGFlowDataMigrator(connector)

# 导出知识库
migrator.export_knowledge_base("my_kb", "export.json")

# 导入到LangChain
embeddings = OpenAIEmbeddings()
vectorstore = migrator.import_to_langchain_vectorstore(
    "export.json",
    embeddings,
    vectorstore_type="faiss"
)
```

### 2. 自定义检索器

创建自定义的RAGFlow检索器：

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class CustomRAGFlowRetriever(BaseRetriever):
    def __init__(self, connector, kb_name, **kwargs):
        super().__init__(**kwargs)
        self.connector = connector
        self.kb_name = kb_name

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 自定义检索逻辑
        results = self.connector.search_knowledge_base(
            self.kb_name,
            query,
            top_k=10,
            similarity_threshold=0.6
        )

        # 自定义文档处理
        documents = []
        for result in results:
            # 添加自定义处理逻辑
            doc = Document(
                page_content=result["content"],
                metadata={
                    "score": result["score"],
                    "custom_field": "custom_value"
                }
            )
            documents.append(doc)

        return documents
```

### 3. 多知识库融合

同时使用多个RAGFlow知识库：

```python
class MultiKBRetriever:
    def __init__(self, app, kb_names):
        self.retrievers = {}
        for kb_name in kb_names:
            self.retrievers[kb_name] = app.create_retriever(kb_name)

    def get_relevant_documents(self, query: str):
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
```

## 🎯 使用场景

### 1. 企业知识问答

```python
# 企业内部知识库问答
app = RAGFlowLangChainApp()
app.initialize()

# HR政策问答
hr_answer = app.chat("hr_policy", "公司的年假政策是什么？")

# 技术文档问答
tech_answer = app.chat("tech_docs", "如何配置微服务架构？")
```

### 2. 多轮对话

```python
# 对话式RAG
class ConversationalRAG:
    def __init__(self, app, kb_name):
        self.app = app
        self.kb_name = kb_name
        self.history = []

    def chat(self, question):
        # 将历史对话加入上下文
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.history[-5:]])

        # 构建增强查询
        enhanced_query = f"对话历史:\n{context}\n\n当前问题: {question}"

        # 获取回答
        answer = self.app.chat(self.kb_name, enhanced_query)

        # 更新历史
        self.history.append((question, answer))
        return answer
```

### 3. 文档分析

```python
# 批量文档分析
def analyze_documents(app, kb_name, questions):
    results = {}
    for question in questions:
        answer = app.chat(kb_name, question, chain_type="with_sources")
        results[question] = answer
    return results

# 使用示例
questions = [
    "这个项目的主要技术栈是什么？",
    "项目的架构设计有什么特点？",
    "存在哪些技术风险？"
]

analysis = analyze_documents(app, "project_docs", questions)
```

## 🛠️ 故障排除

### 常见问题

1. **连接失败**
   ```
   检查RAGFlow服务是否启动
   确认API地址和端口是否正确
   验证API密钥是否有效
   ```

2. **检索结果为空**
   ```
   检查知识库是否存在文档
   调整相似度阈值
   尝试不同的查询关键词
   ```

3. **回答质量不佳**
   ```
   增加返回文档数量（top_k）
   降低相似度阈值
   检查知识库文档质量
   优化问题表述
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查检索结果
retriever = app.create_retriever("kb_name")
docs = retriever.get_relevant_documents("测试查询")

for i, doc in enumerate(docs):
    print(f"文档 {i+1}:")
    print(f"  相似度: {doc.metadata.get('score', 0)}")
    print(f"  来源: {doc.metadata.get('source', '未知')}")
    print(f"  内容: {doc.page_content[:100]}...")
```

## 📋 最佳实践

### 1. 知识库管理

- 定期更新知识库内容
- 合理设置文档分块大小
- 维护清晰的文档元数据
- 定期备份知识库数据

### 2. 性能优化

- 使用缓存减少API调用
- 批量处理相似查询
- 监控响应时间
- 合理设置top_k参数

### 3. 安全考虑

- 保护API密钥安全
- 实施访问权限控制
- 记录查询日志
- 定期审查数据使用

## 📖 相关文档

- [RAGFlow官方文档](https://ragflow.io/docs)
- [LangChain文档](https://python.langchain.com/docs/)
- [向量数据库指南](https://python.langchain.com/docs/modules/data_connection/vectorstores/)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个集成方案！