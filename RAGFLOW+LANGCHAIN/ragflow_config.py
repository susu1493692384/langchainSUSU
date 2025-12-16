#!/usr/bin/env python3
"""
RAGFlow + LangChain 配置文件
包含RAGFlow API配置和使用示例
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class RAGFlowConfig:
    """RAGFlow配置类"""

    # RAGFlow 服务配置
    RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", "http://localhost:9380")
    RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY", "ragflow-om0edpurycQmm8HFyO73hJtp5qTbhdewc9nnrVsb-lw")

    # LangChain 配置
    OPENAI_API_KEY = os.getenv("GLM_API_KEY", "c687408340974d5993fcc2a8c04fcd4e.15vKrjujYsrT6ZBi")
    OPENAI_API_BASE = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4")
    LLM_MODEL = os.getenv("LLM_MODEL", "glm-4.5")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-2")

    # 检索配置
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.7"))

    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """验证配置"""
        validation_result = {
            "ragflow_url": bool(cls.RAGFLOW_BASE_URL),
            "openai_api_key": bool(cls.OPENAI_API_KEY),
            "ragflow_api_key": bool(cls.RAGFLOW_API_KEY)
        }
        return validation_result

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("当前RAGFlow + LangChain配置:")
        print(f"RAGFlow服务地址: {cls.RAGFLOW_BASE_URL}")
        print(f"RAGFlow API密钥: {'已设置' if cls.RAGFLOW_API_KEY else '未设置'}")
        print(f"OpenAI API密钥: {'已设置' if cls.OPENAI_API_KEY else '未设置'}")
        print(f"LLM模型: {cls.LLM_MODEL}")
        print(f"嵌入模型: {cls.EMBEDDING_MODEL}")
        print(f"默认返回数量: {cls.DEFAULT_TOP_K}")
        print(f"默认相似度阈值: {cls.DEFAULT_SIMILARITY_THRESHOLD}")

class KnowledgeBaseConfig:
    """知识库配置"""

    # 示例知识库配置
    SAMPLE_KNOWLEDGE_BASES = {
        "tech_docs": {
            "name": "技术文档",
            "description": "包含AI、机器学习等技术文档",
            "top_k": 3,
            "similarity_threshold": 0.75
        },
        "company_policy": {
            "name": "公司政策",
            "description": "公司规章制度和政策文档",
            "top_k": 5,
            "similarity_threshold": 0.8
        },
        "product_manual": {
            "name": "产品手册",
            "description": "产品使用说明和技术手册",
            "top_k": 4,
            "similarity_threshold": 0.7
        }
    }

# 环境变量示例
ENVIRONMENT_EXAMPLE = """
# RAGFlow + LangChain 环境变量配置示例
# 复制此内容到 .env 文件

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
"""

# 快速使用示例
QUICK_START_EXAMPLE = '''
# 快速开始示例

from ragflow_langchain_integration import RAGFlowLangChainApp

# 1. 创建应用实例
app = RAGFlowLangChainApp(
    ragflow_url="http://localhost:9380",
    ragflow_api_key="your_api_key",
    llm_model="gpt-3.5-turbo"
)

# 2. 初始化应用
if app.initialize():
    # 3. 创建检索器
    retriever = app.create_retriever("your_knowledge_base_name")

    # 4. 创建QA链
    qa_chain = app.create_qa_chain("your_knowledge_base_name", chain_type="with_sources")

    # 5. 开始问答
    answer = qa_chain.invoke("你的问题")
    print(answer)
'''

# 命令行使用示例
CLI_USAGE_EXAMPLE = '''
# 命令行使用

# 运行集成演示
python ragflow_langchain_integration.py

# 测试连接
python -c "
from ragflow_langchain_integration import RAGFlowAPIConnector
connector = RAGFlowAPIConnector()
print('连接状态:', connector.test_connection())
"
'''

if __name__ == "__main__":
    print("RAGFlow + LangChain 配置检查")
    print("=" * 50)

    # 打印配置
    RAGFlowConfig.print_config()

    # 验证配置
    print("\n配置验证:")
    validation = RAGFlowConfig.validate_config()
    for key, status in validation.items():
        status_str = "✅" if status else "❌"
        print(f"{status_str} {key}: {'正确' if status else '需要配置'}")

    # 输出环境变量示例
    print(f"\n环境变量配置示例:")
    print(ENVIRONMENT_EXAMPLE)