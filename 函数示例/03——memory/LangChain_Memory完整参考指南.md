# LangChain Memory 完整参数参考指南

## 目录
1. [Memory类型总览](#memory类型总览)
2. [基础参数详解](#基础参数详解)
3. [各类Memory详细参数](#各类memory详细参数)
4. [使用示例](#使用示例)
5. [快速查表](#快速查表)

---

## Memory类型总览

### 1. ConversationBufferMemory（对话缓冲记忆）
- **用途**: 保存完整对话历史
- **特点**: 记录所有对话，适合短期对话
- **适用场景**: 聊天机器人、问答系统

### 2. ConversationBufferWindowMemory（窗口记忆）
- **用途**: 只保留最近K轮对话
- **特点**: 内存效率高，适合长期对话
- **适用场景**: 长期运行的对话系统

### 3. ConversationSummaryMemory（摘要记忆）
- **用途**: 对话历史摘要
- **特点**: 自动总结，节省token
- **适用场景**: 长文档对话、会议记录

### 4. ConversationKGMemory（知识图谱记忆）
- **用途**: 提取和存储知识图谱
- **特点**: 结构化记忆，便于查询
- **适用场景**: 知识密集型应用

### 5. VectorStoreRetrieverMemory（向量存储记忆）
- **用途**: 基于向量搜索的相关记忆检索
- **特点**: 语义搜索，高效检索
- **适用场景**: 大规模对话历史检索

---

## 基础参数详解

### 通用基础参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `memory_key` | str | `"history"` | 在prompt中引用记忆的变量名 |
| `return_messages` | bool | `False` | 是否返回消息对象格式 |
| `input_key` | str | `"input"` | 输入文本的键名 |
| `output_key` | str | `"output"` | 输出文本的键名 |
| `chat_memory` | BaseChatMessageHistory | `None` | 自定义聊天记忆存储 |
| `human_prefix` | str | `"Human"` | 用户消息前缀 |
| `ai_prefix` | str | `"AI"` | AI回复前缀 |

---

## 各类Memory详细参数

### 1. ConversationBufferMemory

```python
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    # 基础参数
    memory_key="chat_history",           # 在prompt中使用的变量名
    return_messages=True,                # 返回消息格式
    input_key="input",                   # 输入键名
    output_key="output",                 # 输出键名

    # 自定义存储
    chat_memory=None,                    # 自定义消息历史存储

    # 消息格式
    human_prefix="Human",                # 用户消息前缀
    ai_prefix="AI",                      # AI消息前缀

    # 其他
    verbose=False                        # 是否显示详细日志
)
```

**完整示例**:
```python
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 创建自定义存储
custom_history = InMemoryChatMessageHistory()

memory = ConversationBufferMemory(
    memory_key="conversation_history",
    return_messages=True,
    chat_memory=custom_history,
    human_prefix="用户",
    ai_prefix="助手"
)
```

### 2. ConversationBufferWindowMemory

```python
from langchain_classic.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    # 基础参数
    memory_key="chat_history",
    return_messages=True,
    k=5,                                 # 保留最近5轮对话
    input_key="input",
    output_key="output",

    # 消息格式
    human_prefix="Human",
    ai_prefix="AI",

    # 其他
    verbose=False
)
```

**关键参数说明**:
- `k`: 保留的对话轮数（用户+AI算一轮）
- 当k=3时，只保留最近3轮对话

### 3. ConversationSummaryMemory

```python
from langchain_classic.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

memory = ConversationSummaryMemory(
    # 核心参数
    llm=llm,                            # 用于生成摘要的LLM
    memory_key="chat_history",
    return_messages=True,

    # 摘要控制
    prompt=None,                        # 自定义摘要提示
    max_token_limit=1000,              # 最大token限制
    summary_prefix="Summary:",         # 摘要前缀

    # 基础参数
    input_key="input",
    output_key="output",
    human_prefix="Human",
    ai_prefix="AI",

    # 其他
    verbose=False
)
```

**自定义摘要提示示例**:
```python
from langchain_core.prompts import PromptTemplate

summary_prompt = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template="现有对话摘要：{summary}\n\n新的对话内容：{new_lines}\n\n请生成新的摘要："
)

memory = ConversationSummaryMemory(
    llm=llm,
    prompt=summary_prompt,
    max_token_limit=800
)
```

### 4. VectorStoreRetrieverMemory

```python
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 创建向量存储
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embedding_model)

memory = VectorStoreRetrieverMemory(
    # 核心参数
    retriever=vectorstore.as_retriever(), # 向量检索器

    # 检索参数
    search_kwargs={"k": 4},              # 检索返回的文档数量

    # 基础参数
    memory_key="relevant_history",
    input_key="input",

    # 其他
    verbose=False
)
```

### 5. MotorheadMemory

```python
from langchain_community.memory import MotorheadMemory

memory = MotorheadMemory(
    # 核心参数
    session_id="user_session_123",       # 会话ID
    url="mongodb://localhost:27017",     # MongoDB连接URL
    memory_key="chat_history",

    # 基础参数
    return_messages=True,
    input_key="input",
    output_key="output",

    # 其他
    verbose=False
)
```

### 6. ZepMemory

```python
from langchain_community.memory import ZepMemory

memory = ZepMemory(
    # 核心参数
    session_id="user_session_456",       # 会话ID
    url="http://localhost:8000",         # Zep服务URL
    api_key="your_zep_api_key",          # Zep API密钥

    # 基础参数
    memory_key="chat_history",
    return_messages=True,

    # 其他
    verbose=False
)
```

---

## 使用示例

### 示例1: 基础对话缓冲记忆

```python
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    human_prefix="用户",
    ai_prefix="AI助手"
)

# 创建对话链
llm = ChatOpenAI(model="gpt-3.5-turbo")
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 使用
response1 = conversation.predict(input="你好，我想学习Python")
response2 = conversation.predict(input="Python有什么优势？")
```

### 示例2: 窗口记忆配置

```python
from langchain_classic.memory import ConversationBufferWindowMemory

# 只保留最近3轮对话
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="recent_history",
    return_messages=True
)

# 测试记忆限制
for i in range(6):
    print(f"轮次 {i+1}")
    response = conversation.predict(input=f"测试消息 {i+1}")
    print(f"当前记忆中的消息数: {len(memory.chat_memory.messages)}")
```

### 示例3: 自定义摘要记忆

```python
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 自定义摘要提示
custom_prompt = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template="""
之前的对话摘要：{summary}

新的对话内容：{new_lines}

请生成包含新内容的完整摘要：
"""
)

memory = ConversationSummaryMemory(
    llm=llm,
    prompt=custom_prompt,
    max_token_limit=500,
    return_messages=True
)
```

### 示例4: 向量存储记忆

```python
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 创建FAISS向量存储
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    ["之前的对话内容1", "之前的对话内容2"],
    embeddings
)

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    memory_key="relevant_context"
)

# 在prompt中使用
template = """
相关历史上下文：
{relevant_context}

当前问题：{input}

请根据上下文回答：
"""
```

---

## 高级配置技巧

### 1. 自定义消息历史存储

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

class CustomChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def clear(self) -> None:
        self.messages = []

# 使用自定义存储
custom_history = CustomChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=custom_history,
    return_messages=True
)
```

### 2. 混合记忆策略

```python
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory
)

# 同时使用缓冲记忆和摘要记忆
buffer_memory = ConversationBufferMemory(
    memory_key="recent_chat",
    return_messages=True
)

summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_summary",
    return_messages=True
)

# 在prompt中组合使用
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    对话摘要：{chat_summary}
    最近对话：{recent_chat}
    """),
    ("human", "{input}")
])
```

### 3. 条件记忆保存

```python
class ConditionalMemory(ConversationBufferMemory):
    def save_context(self, inputs: dict, outputs: dict) -> None:
        # 只有当消息包含重要信息时才保存
        user_input = inputs.get(self.input_key, "")

        if self._is_important(user_input):
            super().save_context(inputs, outputs)

    def _is_important(self, text: str) -> bool:
        # 自定义重要性判断逻辑
        important_keywords = ["重要", "记住", "关键", "总结"]
        return any(keyword in text for keyword in important_keywords)

# 使用条件记忆
memory = ConditionalMemory(
    memory_key="important_history",
    return_messages=True
)
```

---

## 性能优化建议

### 1. Token使用优化

```python
# 使用窗口记忆而不是缓冲记忆
memory = ConversationBufferWindowMemory(k=5)  # 限制记忆长度

# 或使用摘要记忆
memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=500  # 限制摘要长度
)
```

### 2. 异步操作

```python
import asyncio
from langchain_classic.memory import ConversationBufferMemory

class AsyncMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory()

    async def add_conversation(self, user_input: str, ai_response: str):
        # 异步添加对话
        await asyncio.sleep(0.01)  # 模拟异步操作
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(ai_response)

# 使用异步记忆
async_memory = AsyncMemory()
await async_memory.add_conversation("你好", "你好！有什么可以帮您的吗？")
```

### 3. 持久化存储

```python
import json
from langchain_classic.memory import ConversationBufferMemory

class PersistentMemory:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.memory = ConversationBufferMemory()
        self.load()

    def save(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory.chat_memory.messages, f, default=str)

    def load(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
                # 恢复消息到记忆中
                for msg in messages:
                    if msg['type'] == 'human':
                        self.memory.chat_memory.add_user_message(msg['content'])
                    else:
                        self.memory.chat_memory.add_ai_message(msg['content'])
        except FileNotFoundError:
            pass

# 使用持久化记忆
persistent_memory = PersistentMemory("chat_history.json")
```

---

## 错误处理和调试

### 1. 常见错误处理

```python
try:
    response = conversation.predict(input="用户输入")
except Exception as e:
    print(f"对话生成失败: {e}")
    # 重置记忆或使用备用方案
    memory.clear()
    response = "抱歉，出现了一些问题，让我们重新开始。"
```

### 2. 记忆调试

```python
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True, verbose=True)

# 调试记忆内容
def debug_memory(memory):
    print("=== 记忆调试信息 ===")
    print(f"消息总数: {len(memory.chat_memory.messages)}")

    for i, msg in enumerate(memory.chat_memory.messages):
        msg_type = "用户" if msg.type == "human" else "AI"
        print(f"{msg_type} {i+1}: {msg.content}")

    print(f"记忆缓冲: {memory.buffer}")
    print("==================")

# 使用调试
conversation.predict(input="测试消息")
debug_memory(memory)
```

---

## 最佳实践

### 1. 记忆类型选择指南

| 场景 | 推荐Memory | 配置建议 |
|------|-----------|----------|
| 短期聊天 | ConversationBufferMemory | 无特殊配置 |
| 长期对话 | ConversationBufferWindowMemory | k=5-10 |
| 大量历史 | ConversationSummaryMemory | 设置合适的token限制 |
| 语义检索 | VectorStoreRetrieverMemory | 使用合适的embedding模型 |
| 生产环境 | ZepMemory/MotorheadMemory | 配置持久化存储 |

### 2. 安全性考虑

```python
# 敏感信息过滤
class SecureMemory(ConversationBufferMemory):
    def save_context(self, inputs: dict, outputs: dict) -> None:
        # 过滤敏感信息
        filtered_inputs = self._filter_sensitive_data(inputs)
        filtered_outputs = self._filter_sensitive_data(outputs)

        super().save_context(filtered_inputs, filtered_outputs)

    def _filter_sensitive_data(self, data: dict) -> dict:
        # 实现敏感数据过滤逻辑
        sensitive_patterns = ["密码", "身份证", "银行卡"]
        filtered_data = data.copy()

        for key, value in filtered_data.items():
            if isinstance(value, str):
                for pattern in sensitive_patterns:
                    if pattern in value:
                        filtered_data[key] = "[敏感信息已过滤]"

        return filtered_data
```

这个完整的参考指南涵盖了LangChain Memory的所有主要参数和用法，可以作为开发时的完整参考文档。