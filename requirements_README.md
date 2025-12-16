# 依赖文件说明

本项目包含了3个不同版本的依赖文件，以满足不同的使用场景：

## 📁 依赖文件

| 文件名 | 用途 | 说明 |
|--------|------|------|
| `requirements.txt` | 完整依赖 | 包含当前环境的所有包（63个） |
| `requirements_categorized.txt` | 分类依赖 | 按功能分类的完整依赖（推荐） |
| `requirements_minimal.txt` | 最小依赖 | 仅核心必需包（8个） |

## 🚀 快速开始

### 生产环境推荐
```bash
# 安装分类依赖（推荐）
pip install -r requirements_categorized.txt

# 或者安装最小依赖+常用包
pip install -r requirements_minimal.txt
pip install tqdm langchain-community
```

### 开发环境
```bash
# 安装完整依赖
pip install -r requirements.txt

# 或者安装分类依赖
pip install -r requirements_categorized.txt
```

### 最小化环境
```bash
# 仅安装核心包
pip install -r requirements_minimal.txt
```

## 📦 依赖分类

### 🔥 核心依赖（必需）
- `langchain` - LangChain主框架
- `langchain-core` - LangChain核心组件
- `langchain-openai` - OpenAI模型支持
- `openai` - OpenAI官方SDK
- `tiktoken` - Token计算工具
- `python-dotenv` - 环境变量管理
- `pydantic` - 数据验证
- `requests` - HTTP请求库
- `numpy` - 数值计算

### 📊 LangGraph生态
- `langgraph` - 状态机和工作流
- `langgraph-checkpoint` - 检查点支持
- `langgraph-prebuilt` - 预构建组件
- `langgraph-sdk` - SDK工具

### 🛠️ 常用工具
- `SQLAlchemy` - 数据库ORM
- `tqdm` - 进度条
- `PyYAML` - YAML解析
- `tenacity` - 重试机制

## 🔧 环境配置

### 1. 创建虚拟环境
```bash
python -m venv langchain_env
source langchain_env/bin/activate  # Linux/Mac
# 或
langchain_env\Scripts\activate     # Windows
```

### 2. 安装依赖
```bash
# 选择合适的依赖文件
pip install -r requirements_minimal.txt
```

### 3. 配置环境变量
创建 `.env` 文件：
```env
# 智谱AI配置
GLM_API_KEY=your_glm_api_key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/

# OpenAI配置（可选）
OPENAI_API_KEY=your_openai_api_key

# LangSmith配置（可选）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```

## 📋 版本兼容性

| 包名 | 当前版本 | 兼容性说明 |
|------|----------|------------|
| langchain | 1.1.0 | 最新稳定版 |
| langchain-core | 1.1.0 | 与langchain匹配 |
| langchain-openai | 1.1.0 | 最新版本 |
| openai | 2.8.1 | 支持最新API |
| python | 3.8+ | 最低Python版本 |

## 🚨 常见问题

### 1. 版本冲突
如果遇到版本冲突，建议使用虚拟环境：
```bash
pip install --upgrade pip
pip install -r requirements_minimal.txt
```

### 2. 网络问题
如果安装缓慢，可以使用国内镜像：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements_minimal.txt
```

### 3. 权限问题
如果遇到权限问题，使用用户安装：
```bash
pip install --user -r requirements_minimal.txt
```

## 📚 推荐安装顺序

1. **基础环境**
   ```bash
   pip install python-dotenv pydantic requests numpy
   ```

2. **LangChain核心**
   ```bash
   pip install langchain langchain-core langchain-openai
   ```

3. **AI支持**
   ```bash
   pip install openai tiktoken
   ```

4. **可选组件**
   ```bash
   pip install langchain-community langchainhub
   pip install tqdm SQLAlchemy PyYAML
   ```

## 🔄 更新依赖

定期更新依赖以获得最新功能和安全补丁：
```bash
# 更新单个包
pip install --upgrade langchain

# 更新所有依赖
pip list --outdated
pip install --upgrade -r requirements_categorized.txt
```

## 📝 备注

- 本项目基于Python 3.8+开发
- 推荐使用虚拟环境进行依赖管理
- 生产环境建议使用固定版本号
- 开发环境可以使用最新版本