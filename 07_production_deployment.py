#!/usr/bin/env python3
"""
LangChain 进阶示例 - 生产部署 (Production Deployment)
展示如何将LangChain应用部署到生产环境
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from langchain.schema import BaseOutputParser
from langchain.callbacks import AsyncCallbackHandler
import redis
import aioredis
from contextlib import asynccontextmanager

# 加载环境变量
load_dotenv()

# ========================
# 配置和日志 (Configuration & Logging)
# ========================

class Config:
    """应用配置类"""

    # API配置
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # LLM配置
    LLM_MODEL = os.getenv("LLM_MODEL", "glm-4")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

    # Redis配置
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

    # 安全配置
    API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
    API_KEY = os.getenv("API_KEY", "your-api-key-here")

    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")

# 设置日志
def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ========================
# 数据模型 (Data Models)
# ========================

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息", min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, description="会话ID")
    temperature: Optional[float] = Field(None, description="温度参数", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="最大Token数", ge=1, le=4000)

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="AI回复")
    session_id: str = Field(..., description="会话ID")
    timestamp: str = Field(..., description="时间戳")
    tokens_used: int = Field(..., description="使用的Token数")
    response_time: float = Field(..., description="响应时间(秒)")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="检查时间")
    version: str = Field("1.0.0", description="版本号")
    dependencies: Dict[str, str] = Field(..., description="依赖状态")

class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    message_count: int
    created_at: str
    last_activity: str

# ========================
# 异步回调处理器 (Async Callback Handler)
# ========================

class StreamingCallbackHandler(AsyncCallbackHandler):
    """流式回调处理器"""

    def __init__(self):
        self.tokens = []
        self.start_time = None

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """处理新的Token"""
        self.tokens.append(token)
        # 这里可以实现实时流式输出
        if len(self.tokens) % 10 == 0:  # 每10个token记录一次
            logger.info(f"Generated {len(self.tokens)} tokens so far")

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """LLM开始时调用"""
        self.start_time = datetime.now()
        self.tokens = []
        logger.info(f"LLM started with {len(prompts)} prompts")

    async def on_llm_end(self, response, **kwargs: Any) -> None:
        """LLM结束时调用"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        logger.info(f"LLM completed in {duration:.2f}s, generated {len(self.tokens)} tokens")

# ========================
# LangChain服务类 (LangChain Service)
# ========================

class LangChainService:
    """LangChain服务类"""

    def __init__(self):
        self.llm = None
        self.memory_store = {}
        self.callback_handler = StreamingCallbackHandler()
        self._initialize_llm()

    def _initialize_llm(self):
        """初始化LLM"""
        try:
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
                openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
                openai_api_base=os.getenv("ANTHROPIC_BASE_URL"),
                streaming=True,
                callbacks=[self.callback_handler]
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def get_memory(self, session_id: str) -> ConversationBufferMemory:
        """获取或创建会话记忆"""
        if session_id not in self.memory_store:
            self.memory_store[session_id] = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        return self.memory_store[session_id]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求"""
        start_time = datetime.now()

        try:
            # 获取或生成会话ID
            session_id = request.session_id or f"session_{int(start_time.timestamp())}"

            # 获取记忆
            memory = self.get_memory(session_id)

            # 创建提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个专业的AI助手，请提供准确、有帮助的回答。"),
                ("human", "{input}")
            ])

            # 创建链
            chain = prompt | self.llm

            # 准备输入
            inputs = {"input": request.message}

            # 如果有历史记录，添加到输入中
            if memory.chat_memory.messages:
                inputs["chat_history"] = memory.chat_memory.messages

            # 调用链
            response = await chain.ainvoke(inputs)

            # 更新记忆
            memory.chat_memory.add_user_message(request.message)
            memory.chat_memory.add_ai_message(response.content)

            # 计算响应时间
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            # 估算Token数（简化版本）
            tokens_used = len(response.content.split()) + len(request.message.split())

            logger.info(f"Chat completed for session {session_id}, took {response_time:.2f}s")

            return ChatResponse(
                response=response.content,
                session_id=session_id,
                timestamp=end_time.isoformat(),
                tokens_used=tokens_used,
                response_time=response_time
            )

        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话信息"""
        if session_id not in self.memory_store:
            return None

        memory = self.memory_store[session_id]
        message_count = len(memory.chat_memory.messages)

        return SessionInfo(
            session_id=session_id,
            message_count=message_count,
            created_at=session_id.split("_")[1] if "_" in session_id else "unknown",
            last_activity=datetime.now().isoformat()
        )

    def clear_session(self, session_id: str) -> bool:
        """清除会话"""
        if session_id in self.memory_store:
            del self.memory_store[session_id]
            logger.info(f"Cleared session {session_id}")
            return True
        return False

# ========================
# Redis服务类 (Redis Service)
# ========================

class RedisService:
    """Redis服务类，用于持久化存储"""

    def __init__(self):
        self.redis_client = None
        self._initialize_redis()

    async def _initialize_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_client = await aioredis.from_url(
                Config.REDIS_URL,
                password=Config.REDIS_PASSWORD,
                decode_responses=True
            )
            # 测试连接
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            self.redis_client = None

    async def save_conversation(self, session_id: str, messages: List[Dict]):
        """保存对话到Redis"""
        if not self.redis_client:
            return False

        try:
            key = f"conversation:{session_id}"
            await self.redis_client.set(
                key,
                json.dumps(messages),
                ex=86400  # 24小时过期
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save conversation to Redis: {e}")
            return False

    async def load_conversation(self, session_id: str) -> Optional[List[Dict]]:
        """从Redis加载对话"""
        if not self.redis_client:
            return None

        try:
            key = f"conversation:{session_id}"
            data = await self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Failed to load conversation from Redis: {e}")
            return None

# ========================
# FastAPI应用 (FastAPI Application)
# ========================

# 全局服务实例
langchain_service = LangChainService()
redis_service = RedisService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("Application startup")
    yield
    # 关闭时执行
    logger.info("Application shutdown")

# 创建FastAPI应用
app = FastAPI(
    title="LangChain Production API",
    description="生产级LangChain聊天API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# API路由 (API Routes)
# ========================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """聊天API端点"""
    try:
        response = await langchain_service.chat(request)

        # 后台任务：保存对话到Redis
        background_tasks.add_task(
            redis_service.save_conversation,
            response.session_id,
            [{"role": "user", "content": request.message},
             {"role": "assistant", "content": response.response}]
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    dependencies = {}

    # 检查LLM
    try:
        if langchain_service.llm:
            dependencies["llm"] = "healthy"
        else:
            dependencies["llm"] = "unhealthy"
    except Exception as e:
        dependencies["llm"] = f"error: {str(e)}"

    # 检查Redis
    try:
        if redis_service.redis_client:
            await redis_service.redis_client.ping()
            dependencies["redis"] = "healthy"
        else:
            dependencies["redis"] = "disabled"
    except Exception as e:
        dependencies["redis"] = f"error: {str(e)}"

    all_healthy = all(
        status in ["healthy", "disabled"]
        for status in dependencies.values()
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now().isoformat(),
        dependencies=dependencies
    )

@app.get("/session/{session_id}", response_model=Optional[SessionInfo])
async def get_session_info(session_id: str):
    """获取会话信息"""
    session_info = langchain_service.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_info

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    success = langchain_service.clear_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": f"Session {session_id} deleted successfully"}

@app.get("/stats")
async def get_stats():
    """获取应用统计信息"""
    return {
        "active_sessions": len(langchain_service.memory_store),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": Config.LLM_MODEL,
            "temperature": Config.LLM_TEMPERATURE,
            "max_tokens": Config.LLM_MAX_TOKENS
        }
    }

# ========================
# 中间件 (Middleware)
# ========================

@app.middleware("http")
async def logging_middleware(request, call_next):
    """请求日志中间件"""
    start_time = datetime.now()

    response = await call_next(request)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )

    response.headers["X-Response-Time"] = str(duration)
    return response

# API Key验证中间件（如果启用）
if Config.API_KEY_REQUIRED:
    @app.middleware("http")
    async def api_key_middleware(request, call_next):
        """API Key验证中间件"""
        if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi"):
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if api_key != Config.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API Key")

        return await call_next(request)

# ========================
# 主函数 (Main Function)
# ========================

def main():
    """主函数 - 演示如何运行生产服务器"""
    print("LangChain 生产部署示例")
    print("=" * 50)
    print("本示例展示如何将LangChain应用部署到生产环境")
    print("\n功能包括：")
    print("- FastAPI异步API服务")
    print("- 会话管理和持久化")
    print("- Redis集成")
    print("- 日志和监控")
    print("- 健康检查")
    print("- 性能统计")
    print("\n启动命令：")
    print("python 07_production_deployment.py")
    print("\nAPI文档地址：")
    print("http://localhost:8000/docs")
    print("\n健康检查：")
    print("http://localhost:8000/health")

    # 启动服务器
    try:
        uvicorn.run(
            "07_production_deployment:app",
            host=Config.API_HOST,
            port=Config.API_PORT,
            reload=True,  # 开发模式，生产环境设为False
            log_level=Config.LOG_LEVEL.lower()
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")

# 简化的演示函数
def demo_client():
    """演示客户端如何使用API"""
    print("\n=== API客户端演示 ===")
    print("以下是如何使用API的示例代码：")

    demo_code = '''
import requests

# 聊天请求
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "你好，介绍一下LangChain",
        "session_id": "test_session"
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"AI回复: {result['response']}")
    print(f"会话ID: {result['session_id']}")
    print(f"响应时间: {result['response_time']:.2f}秒")

# 健康检查
health = requests.get("http://localhost:8000/health")
print(f"服务状态: {health.json()['status']}")
'''

    print(demo_code)

if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "demo":
        demo_client()
    else:
        main()