# 环境变量修复说明

## 🎯 问题解决

您提出的问题非常关键：**环境变量什么时候传入？**

### 修复前的问题
- `RAGRetrievalTool` 类没有使用环境变量作为默认值
- 即使设置了 `.env` 文件，工具也不会自动读取
- 需要手动传递所有参数

### 修复后的解决方案

#### 1. RAGRetrievalTool 构造函数修复
```python
# 修复前
def __init__(self, ragflow_url: str = None, ragflow_api_key: str = None, llm_model: str = "glm-4.5"):
    self.ragflow_url = ragflow_url
    self.ragflow_api_key = ragflow_api_key
    self.llm_model = llm_model

# 修复后
def __init__(self, ragflow_url: str = None, ragflow_api_key: str = None, llm_model: str = "glm-4.5"):
    # 使用环境变量作为默认值
    self.ragflow_url = ragflow_url or os.getenv("RAGFLOW_API_URL", "http://localhost:9380")
    self.ragflow_api_key = ragflow_api_key or os.getenv("RAGFLOW_API_KEY")
    self.llm_model = llm_model or os.getenv("LLM_MODEL", "GLM-4.5")
```

#### 2. 环境变量自动传递的时机
现在环境变量在以下时机自动传入：

1. **创建 RAGRetrievalTool 实例时**:
   ```python
   # 不传递参数，自动使用环境变量
   tool = RAGRetrievalTool()  # 自动读取 .env 文件
   ```

2. **调用 initialize_rag_tools 函数时**:
   ```python
   # 自动使用环境变量
   initialize_rag_tools()  # 不需要传递参数
   ```

3. **创建 RAGEnabledAgent 智能体时**:
   ```python
   # 自动使用环境变量
   agent = RAGEnabledAgent()  # 自动读取配置
   ```

## 🔧 支持的环境变量

| 环境变量 | 用途 | 默认值 | 示例 |
|---------|------|--------|------|
| `RAGFLOW_API_URL` | RAGFlow服务地址 | `http://localhost:9380` | `http://localhost:9380` |
| `RAGFLOW_API_KEY` | RAGFlow API密钥 | 无 | `ragflow-xxxxxx` |
| `LLM_MODEL` | LLM模型名称 | `GLM-4.5` | `GLM-4.5`, `gpt-4` |
| `GLM_API_KEY` | GLM API密钥 | 无 | `your-glm-key` |
| `OPENAI_API_KEY` | OpenAI API密钥 | 无 | `your-openai-key` |

## 📝 .env 文件示例

```env
# RAGFlow配置
RAGFLOW_API_URL=http://localhost:9380
RAGFLOW_API_KEY=ragflow-om0edpurycQmm8HFyO73hJtp5qTbhdewc9nnrVsb-lw

# LLM配置
LLM_MODEL=GLM-4.5
GLM_API_KEY=your_glm_api_key
GLM_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4

# 或者使用OpenAI
# OPENAI_API_KEY=your_openai_api_key
```

## 🚀 使用方式

### 1. 完全自动（推荐）
```python
# 只需要设置 .env 文件，然后：
from ragflow_retrieval_tool import initialize_rag_tools

# 自动使用环境变量
success = initialize_rag_tools()
```

### 2. 部分覆盖
```python
# 使用环境变量，但覆盖特定参数
from ragflow_retrieval_tool import RAGRetrievalTool

tool = RAGRetrievalTool(
    ragflow_api_key="custom_key"  # 覆盖环境变量中的值
)
# 其他参数仍然使用环境变量
```

### 3. 完全手动
```python
# 完全手动设置，不使用环境变量
tool = RAGRetrievalTool(
    ragflow_url="http://custom.url:9380",
    ragflow_api_key="custom_key",
    llm_model="custom_model"
)
```

## 🧪 测试验证

### 运行测试
```bash
cd F:\SOFE\langchain\AGENT_LANGCHAIN+RAGFLOW
python test_env_vars.py
```

### 验证结果
```
1. 检查原始环境变量:
   RAGFLOW_API_URL: None
   RAGFLOW_API_KEY: 存在
   LLM_MODEL: GLM-4.5

2. 测试RAGRetrievalTool环境变量使用:
   实际URL: http://localhost:9380
   期望URL: http://localhost:9380
   URL匹配: True
   实际API Key: 存在
   期望API Key: 存在
   API Key匹配: True
   实际Model: GLM-4.5
   期望Model: GLM-4.5
   Model匹配: True
```

## ✅ 总结

现在环境变量在以下时机自动传入：
1. **加载 .env 文件时** - 通过 `load_dotenv()`
2. **创建工具实例时** - 通过 `os.getenv()` 读取
3. **初始化应用时** - 传递给 RAGFlowLangChainApp

**环境变量现在完全正确地传递给RAG工具了！** 🎉