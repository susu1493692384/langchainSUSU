# AGENT + RAGFLOW - 智能体RAG检索工具

这个目录包含了修复后的LangChain智能体与RAGFlow集成工具，现在可以在AGENT_LANGCHAIN+RAGFLOW目录中正常工作。

## 🔧 已修复的问题

### 1. 导入路径问题
- ✅ 修复了`RAGFLOW_PLUS_LANGCHAIN`模块导入错误
- ✅ 添加了多路径导入逻辑，支持不同的目录结构
- ✅ 动态添加正确的Python路径

### 2. LangChain兼容性
- ✅ 修复了AgentExecutor导入问题
- ✅ 提供了简单工具调用模式作为后备方案
- ✅ 兼容不同版本的LangChain

### 3. 编码问题
- ✅ 修复了Windows环境下的emoji编码问题
- ✅ 确保在Windows环境下正常显示

## 📁 文件结构

```
AGENT_LANGCHAIN+RAGFLOW/
├── ragflow_retrieval_tool.py      # 核心RAG检索工具
├── agent_with_rag_example.py     # 智能体集成示例
├── test_agent_working.py         # 功能测试
├── simple_usage_example.py       # 使用示例
└── README.md                     # 本文件

../RAGFLOW+LANGCHAIN/              # RAGFlow集成模块
├── ragflow_langchain_integration.py
└── ...
```

## 🚀 快速开始

### 1. 环境配置

确保在项目根目录的`.env`文件中配置：

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
```

### 2. 直接使用工具

```python
from ragflow_retrieval_tool import (
    list_knowledge_bases,
    search_documents,
    ask_knowledge_base
)

# 获取知识库列表
result = list_knowledge_bases.invoke({})
print(result)

# 搜索文档
search_result = search_documents.invoke({
    "query": "王书友",
    "knowledge_base": None,
    "max_results": 5
})
print(search_result)

# 回答问题
answer = ask_knowledge_base.invoke({
    "question": "王书友是什么岗位?",
    "knowledge_base": None,
    "include_sources": True
})
print(answer)
```

### 3. 使用智能体

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

## 🛠 运行测试

```bash
# 在AGENT_LANGCHAIN+RAGFLOW目录中运行
cd F:\SOFE\langchain\AGENT_LANGCHAIN+RAGFLOW

# 运行功能测试
python test_agent_working.py

# 运行使用示例
python simple_usage_example.py
```

## 📋 可用的工具

### 1. list_knowledge_bases
- **功能**: 获取所有可用的知识库列表
- **参数**: 无
- **返回**: 格式化的知识库列表

### 2. search_documents
- **功能**: 在知识库中搜索相关文档
- **参数**:
  - `query` (必需): 搜索查询
  - `knowledge_base` (可选): 指定知识库
  - `max_results` (可选): 最大结果数
- **返回**: 格式化的搜索结果

### 3. ask_knowledge_base
- **功能**: 基于知识库内容回答问题
- **参数**:
  - `question` (必需): 要回答的问题
  - `knowledge_base` (可选): 指定知识库
  - `include_sources` (可选): 是否包含来源
- **返回**: 基于知识库的回答

### 4. get_document_summary
- **功能**: 获取知识库文档摘要
- **参数**:
  - `knowledge_base` (可选): 指定知识库
- **返回**: 知识库摘要信息

## 🔧 故障排除

### 如果遇到导入错误

1. **检查RAGFlow集成模块**:
   ```bash
   # 确保RAGFLOW+LANGCHAIN目录存在
   ls ../RAGFLOW+LANGCHAIN/ragflow_langchain_integration.py
   ```

2. **检查Python路径**:
   ```python
   import sys
   print(sys.path)  # 确认包含正确的路径
   ```

### 如果RAGFlow连接失败

1. **检查服务状态**:
   - 确保RAGFlow运行在 http://localhost:9380
   - 检查API密钥是否正确

2. **检查网络连接**:
   ```bash
   curl http://localhost:9380/api/health
   ```

### 如果工具调用失败

1. **检查环境变量**:
   ```bash
   echo $RAGFLOW_API_KEY
   echo $GLM_API_KEY
   ```

2. **检查知识库**:
   - 确保在RAGFlow中创建了知识库
   - 确保知识库中有文档内容

## 🎯 使用场景

### 1. 人力资源助手
```python
# 回答员工关于政策的问题
answer = ask_knowledge_base.invoke({
    "question": "公司的年假政策是什么？",
    "knowledge_base": "hr_policies"
})
```

### 2. 文档检索助手
```python
# 搜索相关文档
results = search_documents.invoke({
    "query": "项目管理流程",
    "max_results": 10
})
```

### 3. 智能客服
```python
# 集成到客服智能体中
agent = RAGEnabledAgent()
agent.initialize()
response = agent.chat("如何申请报销？")
```

## 📚 更多示例

- `simple_usage_example.py`: 基本使用示例
- `agent_with_rag_example.py`: 完整智能体示例
- `test_agent_working.py`: 功能测试

## ✨ 特性

- ✅ **多知识库支持**: 可同时搜索多个知识库
- ✅ **智能工具选择**: 自动选择最合适的工具
- ✅ **错误处理**: 完善的异常处理和错误提示
- ✅ **兼容性**: 支持不同版本的LangChain
- ✅ **易用性**: 简单直观的API设计

## 🤝 贡献

如果遇到问题或需要改进，请：
1. 检查现有的测试文件
2. 运行测试确保基本功能正常
3. 查看错误日志定位问题
4. 根据文档进行配置调整

---

**现在RAG检索工具已经完全修复并可以在智能体中正常使用了！** 🎉