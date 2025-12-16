#!/usr/bin/env python3
"""
ChatOpenAI 快速查表
基于 chatopenai_complete_reference.py 的精华版本
所有常用参数的快速参考和示例

用法：复制粘贴即可使用
"""

import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# ================================
# 🔥 最常用配置 (90%情况下使用这个)
# ================================

# 基础配置 - 适用于智谱AI/GLM
def basic_config():
    return ChatOpenAI(
        model="glm-4",
        temperature=0.7,        # 0-2之间，控制随机性
        max_tokens=150,         # 限制输出长度
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

# OpenAI官方API配置
def openai_config():
    return ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        max_tokens=150,
        api_key=os.getenv("OPENAI_API_KEY")  # 或直接传入密钥
    )

# ================================
# ⚙️ 参数快速参考
# ================================

# 🎯 核心参数
def core_parameters():
    return ChatOpenAI(
        # === 基础参数 ===
        model="glm-4",           # 模型名称
        temperature=0.7,         # 随机性: 0=确定性, 1=标准, 2=最大随机性
        max_tokens=150,          # 最大输出token数
        top_p=0.9,              # 核采样: 0.1=只考虑前10%概率的tokens

        # === 惩罚参数 ===
        frequency_penalty=0.0,   # 频率惩罚: -2到2，正值减少重复
        presence_penalty=0.0,    # 存在惩罚: -2到2，正值鼓励新话题

        # === API配置 ===
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),
        timeout=30.0,           # 请求超时时间(秒)
        max_retries=3,           # 最大重试次数

        # === 输出控制 ===
        stop=["\n", "END"],     # 遇到这些序列时停止生成

        verbose=True             # 显示详细日志
    )

# 🚀 生产环境配置
def production_config():
    return ChatOpenAI(
        model="glm-4",
        temperature=0.3,         # 生产环境建议较低的随机性
        max_tokens=500,

        # 生产环境重要参数
        timeout=60.0,
        max_retries=5,
        streaming=False,         # 生产环境通常关闭流式

        # API配置
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),

        # 性能优化
        default_headers={
            "User-Agent": "ProductionApp/1.0",
            "X-Request-ID": "production"
        }
    )

# 🎨 创意写作配置
def creative_config():
    return ChatOpenAI(
        model="glm-4",
        temperature=1.2,         # 高随机性，适合创意内容
        max_tokens=300,
        top_p=0.95,             # 更多样性

        # 创意优化
        frequency_penalty=0.3,   # 减少重复
        presence_penalty=0.5,   # 鼓励新思路

        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

# 💻 代码助手配置
def code_assistant_config():
    return ChatOpenAI(
        model="glm-4",
        temperature=0.1,         # 代码需要确定性
        max_tokens=800,

        # 代码相关设置
        model_kwargs={
            "response_format": {"type": "text"}  # 确保纯文本输出
        },

        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),

        # 停止符，适合代码输出
        stop=["```", "END", "完"]
    )

# ================================
# 🔧 特殊场景配置
# ================================

# 📊 JSON输出配置
def json_output_config():
    return ChatOpenAI(
        model="glm-4",
        temperature=0.3,
        max_tokens=200,

        # 强制JSON输出
        model_kwargs={
            "response_format": {"type": "json_object"}
        },

        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

# 🌐 高并发配置
def high_concurrency_config():
    # 自定义HTTP客户端
    custom_client = httpx.Client(
        limits=httpx.Limits(
            max_keepalive_connections=20,  # 保持连接数
            max_connections=50              # 最大连接数
        ),
        timeout=httpx.Timeout(10.0, connect=5.0)  # 连接和读取超时
    )

    return ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=150,

        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),

        # 高并发优化
        http_client=custom_client,
        timeout=10.0,
        max_retries=2,

        verbose=False  # 高并发时关闭详细日志
    )

# 🔒 可重现结果配置
def reproducible_config():
    return ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=150,
        seed=42,  # 随机种子，确保相同输入得到相同输出

        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

# ⚡ 快速响应配置
def fast_response_config():
    return ChatOpenAI(
        model="glm-4",
        temperature=0.5,
        max_tokens=100,          # 限制输出长度以加快响应

        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),

        # 快速响应设置
        timeout=15.0,
        max_retries=1,

        # 新版API参数 (如果支持)
        # service_tier="flex",     # 延迟优化层
        # reasoning_effort="low"   # 减少推理努力
    )

# ================================
# 📋 参数说明速查
# ================================

def parameter_reference():
    """
    ChatOpenAI 参数速查表

    🔥 核心参数 (必须了解):
    - model: str              # 模型名称 (glm-4, gpt-4, 等)
    - temperature: float       # 随机性 0-2 (0.7标准, 0.1精确, 1.2创意)
    - max_tokens: int          # 最大输出长度
    - openai_api_key: str      # API密钥

    ⚙️ 性能参数 (推荐设置):
    - timeout: float          # 超时时间 (30s标准, 60s生产)
    - max_retries: int         # 重试次数 (3标准, 5生产)
    - streaming: bool         # 流式输出 (False标准, True实时)

    🎛️ 输出控制:
    - top_p: float           # 核采样 0-1 (0.9标准)
    - stop: list[str]        # 停止序列
    - frequency_penalty: float # 频率惩罚 -2到2
    - presence_penalty: float  # 存在惩罚 -2到2

    🔧 高级参数 (特殊需求):
    - seed: int              # 随机种子 (用于重现结果)
    - model_kwargs: dict     # 自定义API参数
    - http_client: httpx.Client # 自定义HTTP客户端

    📊 智谱AI特有:
    - openai_api_base: str    # "https://open.bigmodel.cn/api/paas/v4/"
    """
    pass

# ================================
# 🎯 使用示例
# ================================

def usage_examples():
    """实际使用示例"""

    # 示例1: 基础问答
    def basic_qa():
        llm = basic_config()
        messages = [{"role": "user", "content": "你好！请介绍一下自己。"}]
        # response = llm.invoke(messages)
        # print(response.content)

    # 示例2: 代码生成
    def code_generation():
        llm = code_assistant_config()
        messages = [{"role": "user", "content": "写一个Python函数计算斐波那契数列"}]
        # response = llm.invoke(messages)
        # print(response.content)

    # 示例3: 创意写作
    def creative_writing():
        llm = creative_config()
        messages = [{"role": "user", "content": "写一个关于AI的未来故事"}]
        # response = llm.invoke(messages)
        # print(response.content)

    # 示例4: JSON数据提取
    def json_extraction():
        llm = json_output_config()
        messages = [{"role": "user", "content": "从这段文本提取关键信息并返回JSON格式"}]
        # response = llm.invoke(messages)
        # print(response.content)

    # 示例5: 高并发应用
    def concurrent_requests():
        llm = high_concurrency_config()
        messages = [{"role": "user", "content": "快速回答这个问题"}]
        # response = llm.invoke(messages)
        # print(response.content)

# ================================
# 🚨 错误处理最佳实践
# ================================

def error_handling_template():
    """推荐的错误处理模板"""

    def safe_llm_call(llm, messages, max_attempts=3):
        """
        安全的LLM调用，包含重试和错误处理
        """
        for attempt in range(max_attempts):
            try:
                response = llm.invoke(messages)
                return response
            except Exception as e:
                print(f"尝试 {attempt + 1}/{max_attempts} 失败: {e}")
                if attempt == max_attempts - 1:
                    print("所有尝试都失败了，请检查配置")
                    return None
                import time
                time.sleep(2 ** attempt)  # 指数退避

    # 使用示例
    llm = production_config()
    messages = [{"role": "user", "content": "测试消息"}]
    # response = safe_llm_call(llm, messages)
    # if response:
    #     print(f"成功获得回复: {response.content}")

# ================================
# 🏆 推荐配置
# ================================

def recommended_configurations():
    """针对不同场景的推荐配置"""

    recommendations = {
        "🏢 生产环境": """
        model="glm-4",
        temperature=0.3,
        max_tokens=500,
        timeout=60.0,
        max_retries=5,
        streaming=False
        """,

        "🎨 创意写作": """
        model="glm-4",
        temperature=1.2,
        max_tokens=300,
        frequency_penalty=0.3,
        presence_penalty=0.5
        """,

        "💻 代码生成": """
        model="glm-4",
        temperature=0.1,
        max_tokens=800,
        stop=["```", "END"]
        """,

        "⚡ 快速响应": """
        model="glm-4",
        temperature=0.5,
        max_tokens=100,
        timeout=15.0,
        max_retries=1
        """,

        "🔄 可重现结果": """
        model="glm-4",
        temperature=0.7,
        max_tokens=150,
        seed=42
        """
    }

    return recommendations

if __name__ == "__main__":
    print("🚀 ChatOpenAI 快速查表")
    print("=" * 50)
    print()

    print("🔥 最常用配置 (复制即用):")
    print("""
# 智谱AI/GLM基础配置
llm = ChatOpenAI(
    model="glm-4",
    temperature=0.7,
    max_tokens=150,
    openai_api_key=os.getenv("GLM_API_KEY"),
    openai_api_base=os.getenv("GLM_BASE_URL")
)
""")

    print("🎯 不同场景推荐:")
    configs = recommended_configurations()
    for scene, config in configs.items():
        print(f"\n{scene}:")
        print(config)

    print(f"\n📋 完整参考请查看: chatopenai_complete_reference.py")
    print(f"💬 消息类型参考: message_types_reference.py")