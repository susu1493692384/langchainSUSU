# LangChain Memory 快速查表

## 🚀 快速开始

### 基础导入
```python
from langchain_classic.memory import (
    ConversationBufferMemory,           # 缓冲记忆
    ConversationBufferWindowMemory,     # 窗口记忆
    ConversationSummaryMemory,          # 摘要记忆
    ConversationKGMemory,               # 知识图谱记忆
    VectorStoreRetrieverMemory          # 向量记忆
)
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI
```

## 📋 Memory类型速查

| 类型 | 用途 | 记忆长度 | 适用场景 | 配置复杂度 |
|------|------|----------|----------|-----------|
| `ConversationBufferMemory` | 完整对话 | 无限制 | 短期聊天 | ⭐ |
| `ConversationBufferWindowMemory` | 最近N轮 | N轮 | 长期对话 | ⭐⭐ |
| `ConversationSummaryMemory` | 摘要 | 智能摘要 | 大量历史 | ⭐⭐⭐ |
| `ConversationKGMemory` | 知识图谱 | 结构化 | 知识密集 | ⭐⭐⭐⭐ |
| `VectorStoreRetrieverMemory` | 向量检索 | 无限制 | 语义搜索 | ⭐⭐⭐⭐ |

## ⚡ 常用配置模板

### 1. 基础缓冲记忆 ⭐
```python
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

### 2. 窗口记忆 ⭐⭐
```python
memory = ConversationBufferWindowMemory(
    k=5,                          # 保留最近5轮
    memory_key="recent_history",
    return_messages=True
)
```

### 3. 摘要记忆 ⭐⭐⭐
```python
memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    max_token_limit=800,
    memory_key="summary_history",
    return_messages=True
)
```

### 4. 向量检索记忆 ⭐⭐⭐⭐
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory_key="relevant_history"
)
```

## 🔧 通用参数速查

### 基础参数（所有Memory通用）
```python
memory = AnyMemoryClass(
    memory_key="chat_history",        # 在prompt中引用的变量名
    return_messages=True,             # 返回消息对象格式
    input_key="input",                # 输入键名
    output_key="output",              # 输出键名
    human_prefix="Human",             # 用户消息前缀
    ai_prefix="AI",                   # AI消息前缀
    verbose=False                     # 是否显示详细日志
)
```

## 📊 性能对比

| 特性 | 缓冲记忆 | 窗口记忆 | 摘要记忆 | 向量记忆 |
|------|----------|----------|----------|----------|
| **内存使用** | ❌ 高 | ✅ 低 | ✅ 低 | ✅ 可控 |
| **上下文完整性** | ✅ 完整 | ❌ 有限 | ⚠️ 摘要 | ✅ 相关 |
| **检索速度** | ✅ 快 | ✅ 快 | ✅ 快 | ❌ 慢 |
| **Token消耗** | ❌ 高 | ✅ 低 | ✅ 低 | ✅ 低 |
| **实现复杂度** | ⭐ 简单 | ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐⭐⭐ 复杂 |

## 🎯 场景选择指南

### 短期对话 (< 10轮)
```python
# 推荐：缓冲记忆
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)
```

### 长期聊天 (> 10轮)
```python
# 推荐：窗口记忆
memory = ConversationBufferWindowMemory(
    k=5,                              # 只保留最近5轮
    return_messages=True
)
```

### 知识问答/文档对话
```python
# 推荐：摘要记忆
memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    max_token_limit=1000,
    return_messages=True
)
```

### 大规模对话历史
```python
# 推荐：向量记忆
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory_key="relevant_history"
)
```

## 🔥 快速配置代码

### 最简单配置（复制即用）
```python
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI

# 1. 创建记忆
memory = ConversationBufferMemory()

# 2. 创建对话链
conversation = ConversationChain(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    memory=memory,
    verbose=False
)

# 3. 开始对话
response = conversation.predict(input="你好！")
print(response)

# 4. 继续对话（自动记住上下文）
response = conversation.predict(input="刚才我说了什么？")
print(response)  # AI会记住之前的对话
```

### 常用配置模板
```python
# 配置1：窗口记忆（推荐用于长期对话）
def create_window_memory(k=5):
    return ConversationBufferWindowMemory(
        k=k,
        return_messages=True,
        memory_key="chat_history"
    )

# 配置2：摘要记忆（推荐用于文档对话）
def create_summary_memory(llm_model="gpt-3.5-turbo"):
    return ConversationSummaryMemory(
        llm=ChatOpenAI(model=llm_model),
        max_token_limit=800,
        return_messages=True,
        memory_key="summary"
    )

# 配置3：带自定义前缀的缓冲记忆
def create_custom_memory():
    return ConversationBufferMemory(
        return_messages=True,
        memory_key="conversation",
        human_prefix="用户",
        ai_prefix="助手"
    )
```

## 🛠️ 常用操作方法

### 查看记忆内容
```python
# 查看所有消息
for message in memory.chat_memory.messages:
    print(f"{message.type}: {message.content}")

# 查看缓冲区内容（仅缓冲记忆）
print(memory.buffer)

# 查看消息数量
print(f"消息总数: {len(memory.chat_memory.messages)}")
```

### 清空记忆
```python
# 清空所有记忆
memory.clear()

# 手动移除最后一条消息
memory.chat_memory.messages.pop()
```

### 手动添加消息
```python
from langchain_core.messages import HumanMessage, AIMessage

# 添加用户消息
memory.chat_memory.add_user_message("这是一条用户消息")

# 添加AI回复
memory.chat_memory.add_ai_message("这是一条AI回复")
```

## 🚨 常见错误解决

### 错误1：ModuleNotFoundError
```python
# 错误写法
from langchain.memory import ConversationBufferMemory

# 正确写法
from langchain_classic.memory import ConversationBufferMemory
```

### 错误2：记忆长度过长
```python
# 解决方案：使用窗口记忆
memory = ConversationBufferWindowMemory(k=10)  # 限制为10轮

# 或使用摘要记忆
memory = ConversationSummaryMemory(max_token_limit=500)
```

### 错误3：Token超限
```python
# 解决方案：设置摘要记忆的token限制
memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=1000  # 限制摘要长度
)
```

## 📈 性能优化技巧

### 1. 选择合适的Memory类型
- **< 5轮对话**: 使用 `ConversationBufferMemory`
- **5-20轮对话**: 使用 `ConversationBufferWindowMemory(k=10)`
- **> 20轮对话**: 使用 `ConversationSummaryMemory`

### 2. 优化Token使用
```python
# 设置合理的窗口大小
memory = ConversationBufferWindowMemory(k=5)  # 5轮通常够用

# 或限制摘要长度
memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=500  # 根据模型限制调整
)
```

### 3. 异步操作（高级）
```python
import asyncio

async def async_conversation():
    memory = ConversationBufferMemory()

    # 异步处理对话
    tasks = []
    for i in range(5):
        task = asyncio.create_task(
            process_async_input(memory, f"消息 {i}")
        )
        tasks.append(task)

    await asyncio.gather(*tasks)
    return memory

async def process_async_input(memory, user_input):
    # 模拟异步处理
    await asyncio.sleep(0.1)
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(f"回复: {user_input}")
```

## 🔍 调试技巧

### 启用详细日志
```python
# 创建记忆时启用verbose
memory = ConversationBufferMemory(
    return_messages=True,
    verbose=True  # 显示详细操作日志
)

# 或在对话链中启用
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # 显示对话详细过程
)
```

### 检查记忆状态
```python
def debug_memory(memory):
    print(f"=== {memory.__class__.__name__} 调试信息 ===")
    print(f"消息数量: {len(memory.chat_memory.messages)}")

    # 显示最近5条消息
    recent_messages = memory.chat_memory.messages[-5:]
    for i, msg in enumerate(recent_messages, 1):
        print(f"{i}. [{msg.type}]: {msg.content[:50]}...")

    print("=" * 50)

# 使用调试
debug_memory(memory)
```

## 📚 完整示例模板

### 完整的聊天机器人
```python
def create_chatbot(memory_type="buffer", **kwargs):
    """
    创建聊天机器人

    Args:
        memory_type: "buffer", "window", "summary", "vector"
        **kwargs: 记忆特定参数

    Returns:
        ConversationChain: 配置好的对话链
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # 选择记忆类型
    if memory_type == "buffer":
        memory = ConversationBufferMemory(
            return_messages=True,
            **kwargs
        )
    elif memory_type == "window":
        memory = ConversationBufferWindowMemory(
            k=kwargs.get('k', 5),
            return_messages=True
        )
    elif memory_type == "summary":
        memory = ConversationSummaryMemory(
            llm=llm,
            max_token_limit=kwargs.get('max_token_limit', 800),
            return_messages=True
        )
    else:
        raise ValueError(f"不支持的memory_type: {memory_type}")

    # 创建对话链
    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

# 使用示例
chatbot = create_chatbot("window", k=3)
response = chatbot.predict(input="你好，请记住我的名字是小明")
print(response)

response = chatbot.predict(input="我叫什么名字？")
print(response)  # 应该回答"小明"
```

## 🎯 总结

### 快速选择指南
- **新手入门** → `ConversationBufferMemory`
- **长期对话** → `ConversationBufferWindowMemory(k=5)`
- **文档对话** → `ConversationSummaryMemory`
- **语义搜索** → `VectorStoreRetrieverMemory`

### 最佳实践
1. 优先考虑使用窗口记忆（平衡性能和功能）
2. 设置合适的token限制
3. 启用verbose模式进行调试
4. 定期清理不必要的记忆
5. 选择适合应用场景的记忆类型

### 性能警告
- ⚠️ 缓冲记忆会无限增长
- ⚠️ 大量历史对话考虑使用向量记忆
- ⚠️ 注意API调用成本和token限制