# RAGFlow 现代化GUI使用指南

## 🚀 快速开始

### 方法1：使用启动器（推荐）
```bash
python start_gui.py
```
启动器会自动检查依赖、配置并启动GUI。

### 方法2：直接启动
```bash
python ragflow_modern_gui.py
```

## 📋 环境配置

### 1. 环境变量配置
复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：
```bash
# RAGFlow API 配置
RAGFLOW_API_URL=http://localhost:9380
RAGFLOW_API_KEY=your_ragflow_api_key

# LLM 配置
GLM_API_KEY=your_glm_api_key
LLM_MODEL=glm-4.5

# OpenAI 配置（可选）
OPENAI_API_KEY=your_openai_api_key
```

### 2. 依赖安装
```bash
pip install -r requirements_gui.txt
```

或安装主要依赖：
```bash
pip install tkinter langchain requests python-dotenv Pillow
```

## 🎨 界面功能

### 主界面布局

#### 💬 智能问答标签页
- **连接控制**：连接/断开RAGFlow服务
- **知识库选择**：选择要查询的知识库
- **对话窗口**：实时问答交互
- **输入区域**：输入问题并支持快捷键

#### ⚙️ 系统配置标签页
- **RAGFlow配置**：服务地址和API密钥
- **LLM配置**：选择语言模型
- **检索参数**：调整检索数量和相似度阈值

## 🎯 核心功能

### 1. 连接RAGFlow
1. 在配置页面设置RAGFlow服务地址和API密钥
2. 点击"🔌 连接RAGFlow"按钮
3. 等待连接状态变为"🟢 已连接"

### 2. 选择知识库
1. 连接成功后，左侧会显示可用知识库列表
2. 使用搜索框快速查找知识库
3. 点击知识库名称进行选择

### 3. 智能问答
1. 在底部输入框输入您的问题
2. 使用 **Ctrl+Enter** 发送消息
3. 使用 **Shift+Enter** 换行
4. 查看AI助手的回答

### 4. 对话管理
- **清空对话**：清除所有聊天记录
- **导出对话**：保存对话为文本或Markdown文件
- **字数统计**：实时显示输入字数（最多2000字符）

## ⌨️ 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+Enter` | 发送消息 |
| `Shift+Enter` | 输入换行 |
| `Ctrl+F` | 搜索知识库（在搜索框获得焦点时） |

## 🔧 配置说明

### RAGFlow配置
- **服务地址**：默认 `http://localhost:9380`
- **API密钥**：您的RAGFlow API密钥

### LLM配置
- **模型选择**：支持 glm-4.5、gpt-3.5-turbo、gpt-4
- **API密钥**：对应模型的API密钥

### 检索参数
- **结果数量**：返回相关文档数量（1-20）
- **相似度阈值**：文档相似度最低要求（0.1-1.0）

## 🚨 常见问题

### 1. 连接失败
**问题**：点击连接按钮后显示"连接失败"
**解决方案**：
- 检查RAGFlow服务是否运行（端口9380）
- 验证API密钥是否正确
- 确认网络连接正常

### 2. 知识库列表为空
**问题**：连接成功但没有显示知识库
**解决方案**：
- 确认RAGFlow中已创建知识库
- 检查API权限设置
- 点击"刷新"按钮更新列表

### 3. 问答无响应
**问题**：发送问题后没有回复
**解决方案**：
- 确保已选择知识库
- 检查LLM配置和API密钥
- 验证知识库中是否有相关文档

### 4. 依赖包问题
**问题**：启动时提示缺少依赖包
**解决方案**：
- 使用 `pip install -r requirements_gui.txt` 安装
- 或运行 `python start_gui.py` 自动安装

### 5. 界面显示异常
**问题**：界面元素显示不正常
**解决方案**：
- 确认系统支持tkinter
- 检查屏幕分辨率是否满足最低要求（900x600）

## 🎨 界面特性

### 现代化设计
- 现代化UI配色和布局
- 响应式界面设计
- 智能状态指示器

### 用户体验
- 实时输入反馈
- 智能占位符提示
- 多格式对话导出

### 交互优化
- 键盘快捷键支持
- 自动滚动到最新消息
- 一键操作按钮

## 📝 使用建议

### 1. 首次使用
1. 先配置环境变量文件 `.env`
2. 启动GUI并连接RAGFlow
3. 选择合适知识库
4. 尝试简单的问答

### 2. 高效问答
- 问题要具体明确
- 使用关键词搜索知识库
- 合理设置相似度阈值

### 3. 对话管理
- 定期导出重要对话
- 适时清空对话历史
- 按主题分类保存对话

## 🔗 相关文件

- `ragflow_modern_gui.py` - GUI主程序
- `start_gui.py` - 启动脚本
- `ragflow_langchain_integration.py` - RAGFlow集成模块
- `.env.example` - 环境变量示例
- `requirements_gui.txt` - 依赖包列表

## 📞 技术支持

如遇到问题，请检查：
1. Python版本（建议3.8+）
2. 依赖包是否正确安装
3. RAGFlow服务是否正常运行
4. 网络连接是否稳定

---

💡 **提示**：使用 `python start_gui.py` 可以自动检查和解决大部分启动问题！