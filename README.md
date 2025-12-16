# LangChain 学习教程 🤖

欢迎使用这个全面的LangChain学习教程！本教程通过实际示例带您从入门到进阶，掌握LangChain的核心功能。

## 📁 教程结构

### 1. 基础LLM调用 (`01_basic_llm.py`)
- 简单的LLM调用
- 多个问题处理
- 自定义参数设置（temperature、max_tokens等）
- 错误处理

**学习重点：**
- 如何创建ChatOpenAI实例
- 如何发送消息并获取响应
- 不同参数对输出的影响

### 2. 链式调用 (`02_chains.py`)
- 简单链（PromptTemplate + LLM）
- 顺序链（多步骤处理）
- 条件链（基于条件执行不同逻辑）
- 复杂工作流（模拟真实AI助手场景）

**学习重点：**
- LangChain Expression Language (LCEL) 语法
- 如何组合多个处理步骤
- 如何创建复杂的AI工作流

### 3. 模板和记忆管理 (`03_templates_memory.py`)
- 基础提示模板（PromptTemplate）
- 聊天提示模板（ChatPromptTemplate）
- 高级提示模板（多角色、多参数）
- 对话缓冲记忆（ConversationBufferMemory）
- 窗口记忆（ConversationBufferWindowMemory）
- 摘要记忆（ConversationSummaryMemory）
- 自定义记忆系统

**学习重点：**
- 如何设计有效的提示词
- 如何管理对话上下文
- 不同记忆类型的适用场景

### 4. 向量存储和检索 (`04_vector_storage.py`)
- 文档创建和加载
- 文本分割策略
- 向量存储（FAISS、Chroma）
- 相似性搜索
- 最大边际相关性搜索
- 检索问答链
- 对话式检索系统
- 向量存储的保存和加载

**学习重点：**
- 如何构建文档检索系统
- 向量数据库的使用
- RAG（Retrieval-Augmented Generation）的基础实现

### 4.1 🆕 RAG完全指南 (`04_1_rag_comprehensive.py` + `RAG_GUIDE.md`)
- **基础RAG系统**：文档检索和生成的完整流程
- **高级RAG技术**：多查询检索、上下文压缩、父子文档
- **对话式RAG**：支持多轮对话的智能检索
- **RAG评估**：性能监控和质量评估
- **实战项目**：企业问答、学术助手、法律分析

**学习重点：**
- RAG技术的核心原理和最佳实践
- 不同场景下的RAG优化策略
- 生产级RAG系统的构建方法
- 性能评估和持续改进

**🚀 RAG快速启动：**
```bash
# 运行RAG快速启动器
python quickstart_rag.py

# 或直接运行完整教程
python 04_1_rag_comprehensive.py

# 查看详细指南
cat RAG_GUIDE.md
```

## 🚀 快速开始

### 1. 环境设置

首先确保您已经安装了必要的包：

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

### 2. API密钥配置

复制 `.env.example` 文件为 `.env` 并填入您的API密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件，添加您的OpenAI API密钥：

```
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. 运行示例

按照顺序运行示例：

```bash
# 基础LLM调用
python 01_basic_llm.py

# 链式调用
python 02_chains_modern.py

# 模板和记忆管理
python 03_templates_memory_fixed.py

# 向量存储和检索
python 04_vector_storage.py

# 🆕 RAG完全指南
python 04_1_rag_comprehensive.py
# 或使用快速启动器
python quickstart_rag.py
```

## 📚 学习路径

### 初学者路径
1. 先运行 `01_basic_llm.py`，理解基本的LLM调用
2. 学习 `02_chains_modern.py` 中的简单链示例
3. 掌握 `03_templates_memory_fixed.py` 中的基础模板

### 进阶学习路径
1. 深入学习链式调用中的复杂工作流
2. 掌握各种记忆类型的使用
3. 学习向量存储和检索系统

### 🆕 RAG专题学习路径
1. **RAG基础**：运行 `04_1_rag_comprehensive.py` 的基础演示
2. **高级技术**：掌握多查询检索、上下文压缩等优化方法
3. **实践应用**：构建企业问答、学术助手等实际项目

**推荐学习顺序：**
```bash
# 1. RAG快速入门
python quickstart_rag.py

# 2. 完整RAG教程
python 04_1_rag_comprehensive.py

# 3. 查看理论指南
cat RAG_GUIDE.md
```

### 高级应用
1. 构建完整的RAG系统
2. 开发对话式AI应用
3. 实现复杂的AI工作流

## 🛠️ 核心概念

### LangChain Expression Language (LCEL)
现代LangChain的核心语法，使用 `|` 操作符连接组件：

```python
chain = prompt | llm | output_parser
result = chain.invoke({"input": "your question"})
```

### 组件类型
- **Models**: LLM和ChatModel
- **Prompts**: 提示模板系统
- **Chains**: 组合多个步骤的工作流
- **Memory**: 对话上下文管理
- **Retrievers**: 文档检索系统
- **Agents**: 能够使用工具的智能体

## 💡 最佳实践

### 1. 提示词设计
- 明确指定AI的角色和任务
- 提供足够的上下文信息
- 使用具体的输出格式要求
- 提供示例（Few-shot learning）

### 2. 链的设计
- 每个链组件职责单一
- 合理使用错误处理
- 考虑性能和成本优化
- 适当的参数调优

### 3. 记忆选择
- 简短对话：使用BufferMemory
- 长对话：使用WindowMemory或SummaryMemory
- 需要精确记忆：使用自定义记忆系统

### 4. 向量存储
- 选择合适的文本分割策略
- 根据需求选择向量数据库
- 优化检索参数（k值、搜索类型）
- 定期更新和优化向量存储

## 🔧 常见问题

### Q: 如何选择合适的temperature？
A:
- 0.0-0.3: 用于需要确定性的任务（代码生成、事实查询）
- 0.4-0.7: 用于一般对话和内容创作
- 0.8-1.0: 用于需要创意的任务（创意写作、头脑风暴）

### Q: 向量存储应该选择什么？
A:
- FAISS: 快速、内存中，适合中小型数据集
- Chroma: 持久化、功能丰富，适合生产环境
- Pinecone: 云服务、高性能，适合大规模应用

### Q: 什么时候使用不同类型的记忆？
A:
- BufferMemory: 简短对话，需要完整记忆
- WindowMemory: 长对话，只需要最近的记忆
- SummaryMemory: 很长的对话，需要摘要记忆

## 📖 进阶主题

当您完成本教程后，可以进一步学习：
1. **Agents**: 创建能够使用工具的AI智能体
2. **Tools**: 集成外部API和工具
3. **Evaluation**: 评估AI应用的性能
4. **LangSmith**: 监控和调试AI应用
5. **Deployment**: 部署生产级AI应用

## 🤝 贡献

欢迎贡献您的示例和改进建议！您可以：
- 添加新的示例代码
- 改进现有代码
- 完善文档说明
- 报告问题和bug

## 📄 许可证

本教程采用MIT许可证，您可以自由使用和分享。

---

**开始您的LangChain学习之旅吧！** 🎉

记住：最好的学习方式是实践。修改示例代码，尝试不同的参数，构建您自己的AI应用！