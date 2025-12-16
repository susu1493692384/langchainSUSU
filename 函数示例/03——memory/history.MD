history有哪些 
  ⎿ ⧉ Selected 1 lines from 03_templates_memory.py in Visual Studio 
    Code


● 很好的问题！在 LangChain
  中有多种历史消息管理方式，让我详细列出：

  1. 内存消息历史 (InMemoryChatMessageHistory)

  from langchain_core.chat_history import
  InMemoryChatMessageHistory

  # 就像第261行这样使用
  message_history = InMemoryChatMessageHistory()

  特点：
  - 对话历史存储在内存中
  - 程序结束后历史会丢失
  - 适合简单对话和测试

  2. Redis 消息历史 (RedisChatMessageHistory)

  from langchain_redis import RedisChatMessageHistory

  message_history = RedisChatMessageHistory(
      session_id="user_123",
      url="redis://localhost:6379"
  )

  特点：
  - 历史存储在 Redis 中
  - 可以跨会话持久化
  - 支持分布式部署

  3. PostgreSQL 消息历史 (PostgresChatMessageHistory)

  from langchain_postgres import PostgresChatMessageHistory

  message_history = PostgresChatMessageHistory(
      connection_string="postgresql://...",
      session_id="user_123"
  )

  4. 文件系统消息历史 (FileChatMessageHistory)

  from langchain_community.chat_message_histories import
  FileChatMessageHistory

  message_history = FileChatMessageHistory(
      file_path="chat_history.json"
  )

  5. MongoDB 消息历史

  from langchain_mongodb.chat_message_histories import
  MongoDBChatMessageHistory

  message_history = MongoDBChatMessageHistory(
      connection_string="mongodb://...",
      session_id="user_123"
  )

  6. SQLite 消息历史

  from langchain_community.chat_message_histories import
  SQLChatMessageHistory

  message_history = SQLChatMessageHistory(
      session_id="user_123",
      connection_string="sqlite:///chat.db"
  )

  常用接口方法

  所有这些历史类都遵循相同接口：

  # 添加消息
  message_history.add_user_message("你好")
  message_history.add_ai_message("你好！有什么可以帮助你的？")      

  # 获取所有消息
  all_messages = message_history.messages

  # 清空历史
  message_history.clear()

  # 获取特定格式的历史
  messages = message_history.get_messages()

  选择建议

  - 开发/测试: InMemoryChatMessageHistory
  - 生产环境: Redis 或 PostgreSQL
  - 简单部署: FileChatMessageHistory 或 SQLite
  - 云原生: Redis 或 MongoDB

  第261行使用 InMemoryChatMessageHistory
  是因为它最简单，适合学习示例！