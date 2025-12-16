# RAGFlow检索工具使用指南

这个工具包为LangChain智能体提供了强大的RAGFlow知识库检索能力，让智能体能够轻松访问和查询知识库中的信息。

## 📋 功能特性

### 🔧 核心工具
1. **list_knowledge_bases** - 获取所有可用的知识库列表
2. **search_documents** - 在知识库中搜索相关文档
3. **ask_knowledge_base** - 基于知识库内容回答问题
4. **get_document_summary** - 获取知识库文档摘要

### 🚀 主要功能
- 支持单知识库和多知识库检索
- 智能相似度匹配和排序
- 自动格式化和结果处理
- 完整的错误处理和日志记录
- 易于集成的LangChain工具接口

## 🛠 安装和配置

### 环境要求
```bash
pip install langchain langchain-openai requests python-dotenv
```

### 环境变量配置
创建 `.env` 文件并配置以下变量：

```env
# RAGFlow配置
RAGFLOW_API_URL=http://localhost:9380
RAGFLOW_API_KEY=your_ragflow_api_key

# LLM配置 (二选一)
# GLM配置
GLM_API_KEY=your_glm_api_key
GLM_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4
LLM_MODEL=GLM-4.5

# 或者OpenAI配置
OPENAI_API_KEY=your_openai_api_key

# Embedding配置 (可选)
EMBEDDING_MODEL=embedding-2
```

## 📖 使用方法

### 基础使用

#### 1. 直接使用工具函数

```python
from ragflow_retrieval_tool import list_knowledge_bases, search_documents, ask_knowledge_base

# 获取知识库列表
kbs = list_knowledge_bases.invoke({})
print(kbs)

# 搜索文档
results = search_documents.invoke({
    "query": "王书友",
    "knowledge_base": None,  # 搜索所有知识库
    "max_results": 5
})
print(results)

# 回答问题
answer = ask_knowledge_base.invoke({
    "question": "王书友是什么岗位?",
    "knowledge_base": None,
    "include_sources": True
})
print(answer)
```

#### 2. 使用工具类

```python
from ragflow_retrieval_tool import RAGRetrievalTool

# 创建工具实例
tool = RAGRetrievalTool(
    ragflow_url="http://localhost:9380",
    ragflow_api_key="your_api_key"
)

# 搜索文档
results = tool.search_knowledge_base(
    query="王书友",
    knowledge_base="your_kb_name",
    top_k=5
)

# 回答问题
qa_result = tool.ask_question(
    question="王书友是什么岗位?",
    knowledge_base="your_kb_name"
)
```

### 集成到LangChain智能体

#### 方法1：使用预构建的智能体类

```python
from agent_with_rag_example import RAGEnabledAgent

# 创建智能体
agent = RAGEnabledAgent(
    ragflow_url="http://localhost:9380",
    ragflow_api_key="your_api_key",
    llm_model="GLM-4.5"
)

# 初始化
if agent.initialize():
    # 对话
    response = agent.chat("王书友是什么岗位?")
    print(response)

    # 或启动交互式聊天
    agent.interactive_chat()
```

#### 方法2：手动集成到自定义智能体

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from ragflow_retrieval_tool import get_rag_tools, initialize_rag_tools

# 初始化RAG工具
initialize_rag_tools()

# 获取工具
tools = get_rag_tools()

# 创建LLM
llm = ChatOpenAI(model="GLM-4.5", temperature=0.1)

# 创建提示词
prompt = ChatPromptTemplate.from_template("""
你是一个AI助手，可以使用以下工具来帮助用户：
{tools}

请根据用户问题选择合适的工具来回答。
""")

# 创建智能体
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 使用
response = agent_executor.invoke({"input": "搜索关于王书友的信息"})
```

## 🔍 工具详细说明

### 1. list_knowledge_bases
获取所有可用的知识库列表，包括名称、ID、文档数量等信息。

**参数：** 无

**返回：** 格式化的知识库列表字符串

```python
# 使用示例
result = list_knowledge_bases.invoke({})
print(result)
# 输出：
# 发现 3 个可用知识库：
#
# 1. **weekly_reports** (ID: weekly_reports)
#    描述: 周报知识库
#    文档数: 10, Chunk数: 156
#
# 2. **hr_docs** (ID: hr_docs)
#    描述: 人力资源文档
#    文档数: 25, Chunk数: 432
```

### 2. search_documents
在知识库中搜索相关文档。

**参数：**
- `query` (必需): 搜索查询词
- `knowledge_base` (可选): 指定知识库名称或ID
- `max_results` (可选): 最大结果数，默认5

**返回：** 格式化的搜索结果

```python
# 使用示例
result = search_documents.invoke({
    "query": "王书友",
    "knowledge_base": "weekly_reports",
    "max_results": 3
})
print(result)
```

### 3. ask_knowledge_base
基于知识库内容回答问题，提供自然的回答格式。

**参数：**
- `question` (必需): 要回答的问题
- `knowledge_base` (可选): 指定知识库名称或ID
- `include_sources` (可选): 是否包含来源信息，默认True

**返回：** 基于知识库的回答

```python
# 使用示例
result = ask_knowledge_base.invoke({
    "question": "王书友是什么岗位?",
    "knowledge_base": None,
    "include_sources": True
})
print(result)
```

### 4. get_document_summary
获取知识库的文档摘要信息。

**参数：**
- `knowledge_base` (可选): 指定知识库名称或ID

**返回：** 知识库摘要信息

```python
# 获取所有知识库摘要
summary = get_document_summary.invoke({"knowledge_base": None})
print(summary)

# 获取特定知识库摘要
kb_summary = get_document_summary.invoke({"knowledge_base": "weekly_reports"})
print(kb_summary)
```

## 🚀 高级用法

### 批量搜索
```python
queries = ["王书友", "项目进度", "周报内容"]
for query in queries:
    result = search_documents.invoke({"query": query})
    print(f"\n查询: {query}")
    print(result)
```

### 多知识库对比
```python
# 获取所有知识库
kbs = list_knowledge_bases.invoke({})
print(kbs)

# 在不同知识库中搜索同一问题
question = "工作总结"
for kb_name in ["weekly_reports", "hr_docs"]:
    result = ask_knowledge_base.invoke({
        "question": question,
        "knowledge_base": kb_name
    })
    print(f"\n知识库 {kb_name} 的回答:")
    print(result)
```

### 结果处理
```python
# 解析搜索结果
def parse_search_results(result_str):
    """解析搜索结果字符串"""
    lines = result_str.split('\n')
    documents = []
    current_doc = {}

    for line in lines:
        if line.startswith("**文档"):
            if current_doc:
                documents.append(current_doc)
            current_doc = {"title": line}
        elif line.startswith("相关度:"):
            current_doc["score"] = line.split(":")[1].strip()
        elif line.startswith("内容:"):
            current_doc["content"] = line.split(":", 1)[1].strip()

    if current_doc:
        documents.append(current_doc)

    return documents

# 使用示例
search_result = search_documents.invoke({"query": "王书友"})
documents = parse_search_results(search_result)
for doc in documents:
    print(f"文档: {doc['title']}")
    print(f"相关度: {doc['score']}")
    print(f"内容: {doc['content'][:100]}...")
```

## 🔧 故障排除

### 常见问题

1. **连接RAGFlow失败**
   - 检查RAGFlow服务是否运行在正确端口 (默认9380)
   - 验证API密钥是否正确
   - 确认网络连接正常

2. **没有找到知识库**
   - 确保在RAGFlow中创建了知识库
   - 检查知识库是否有文档内容
   - 验证知识库ID或名称拼写正确

3. **搜索结果为空**
   - 尝试降低相似度阈值
   - 使用更通用的搜索关键词
   - 检查文档是否正确分块和索引

4. **LLM调用失败**
   - 检查API密钥是否有效
   - 确认模型名称正确
   - 验证API服务端点可访问

### 调试模式

启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 然后使用工具，会看到详细的API调用信息
```

### 性能优化

1. **缓存结果**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query, kb_name, max_results):
    return search_documents.invoke({
        "query": query,
        "knowledge_base": kb_name,
        "max_results": max_results
    })
```

2. **批量处理**
```python
def batch_search(queries, **kwargs):
    results = []
    for query in queries:
        result = search_documents.invoke({"query": query, **kwargs})
        results.append(result)
    return results
```

## 📝 示例场景

### 人力资源助手
```python
# 创建HR助手
class HRAssistant:
    def __init__(self):
        self.agent = RAGEnabledAgent()
        self.agent.initialize()

    def answer_employee_question(self, question):
        """回答员工问题"""
        response = self.agent.chat(question)
        return response

    def search_policy(self, policy_name):
        """搜索公司政策"""
        result = search_documents.invoke({
            "query": policy_name,
            "knowledge_base": "hr_policies"
        })
        return result

# 使用
hr_assistant = HRAssistant()
answer = hr_assistant.answer_employee_question("年假政策是什么？")
```

### 知识管理助手
```python
def knowledge_qa():
    """知识问答助手"""
    print("欢迎使用知识问答助手！")

    while True:
        question = input("\n请输入您的问题: ").strip()
        if question.lower() in ['quit', 'exit']:
            break

        # 首先搜索相关文档
        search_result = search_documents.invoke({"query": question})
        print("\n📄 相关文档:")
        print(search_result)

        # 然后提供直接回答
        answer = ask_knowledge_base.invoke({"question": question})
        print("\n💬 智能回答:")
        print(answer)

# 启动
knowledge_qa()
```

## 📚 API参考

### RAGRetrievalTool类

```python
class RAGRetrievalTool:
    def __init__(self, ragflow_url=None, ragflow_api_key=None, llm_model="glm-4.5")
    def get_available_knowledge_bases(self) -> List[Dict[str, Any]]
    def search_knowledge_base(self, query, knowledge_base=None, top_k=5, include_content=True) -> Dict[str, Any]
    def ask_question(self, question, knowledge_base=None, chain_type="with_sources") -> Dict[str, Any]
```

### 工具函数

```python
def list_knowledge_bases() -> str
def search_documents(query: str, knowledge_base: str = None, max_results: int = 5) -> str
def ask_knowledge_base(question: str, knowledge_base: str = None, include_sources: bool = True) -> str
def get_document_summary(knowledge_base: str = None) -> str
def get_rag_tools() -> List
def initialize_rag_tools(ragflow_url=None, ragflow_api_key=None, llm_model="glm-4.5") -> bool
```

## 🤝 贡献

欢迎提交问题报告和改进建议！