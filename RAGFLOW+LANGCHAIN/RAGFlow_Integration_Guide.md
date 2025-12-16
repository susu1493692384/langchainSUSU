# RAGFlow + LangChain 完整代码示例指南

## 📋 目录

1. [环境准备](#环境准备)
2. [RAGFlow 连接配置](#ragflow-连接配置)
3. [基本检索功能](#基本检索功能)
4. [提示词与上下文构建](#提示词与上下文构建)
5. [完整应用示例](#完整应用示例)
6. [高级功能](#高级功能)

---

## 🔧 环境准备

### 1. 安装依赖包

```bash
pip install langchain langchain-core langchain-community
pip install langchain-openai
pip install requests python-dotenv pydantic
pip install faiss-cpu chromadb  # 可选向量数据库
```

### 2. 环境变量配置

创建 `.env` 文件：

```bash
# RAGFlow 配置
RAGFLOW_API_URL=http://localhost:9380
RAGFLOW_API_KEY=your_ragflow_api_key_here

# LLM 配置 (GLM 或 OpenAI)
GLM_API_KEY=your_glm_api_key_here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-4.5

# 或者使用 OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# 嵌入模型配置
EMBEDDING_MODEL=embedding-2
```

---

## 🔗 RAGFlow 连接配置

### 1. 基本连接器

```python
import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class RAGFlowAPIConnector:
    """RAGFlow API连接器"""

    def __init__(self,
                 base_url: str = None,
                 api_key: str = None,
                 timeout: int = 60):
        """
        初始化RAGFlow连接器

        Args:
            base_url: RAGFlow服务地址 (默认: http://localhost:9380)
            api_key: RAGFlow API密钥
            timeout: 请求超时时间
        """
        self.base_url = (os.getenv("RAGFLOW_API_URL") if base_url is None else base_url).rstrip('/')
        self.api_key = api_key or os.getenv("RAGFLOW_API_KEY")
        self.timeout = timeout
        self.session = requests.Session()

        # 设置请求头
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}' if self.api_key else None
        })

    def test_connection(self) -> bool:
        """测试RAGFlow连接"""
        try:
            # 尝试多个健康检查端点
            endpoints = ["/api/health", "/health", "/", "/api/v1/datasets"]

            for endpoint in endpoints:
                try:
                    response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                    if response.status_code in [200, 401, 403]:
                        return True
                except:
                    continue

            print("所有健康检查端点都无法访问")
            return False
        except Exception as e:
            print(f"连接RAGFlow失败: {e}")
            return False

    def get_knowledge_bases(self) -> List[Dict]:
        """获取所有知识库列表"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/datasets", timeout=self.timeout)
            if response.status_code == 200:
                result = response.json()

                if result.get("code") == 0 and isinstance(result.get("data"), list):
                    return result.get("data", [])
                else:
                    print(f"API 错误: {result.get('message', '未知错误')}")
                    return []
            else:
                print(f"获取知识库失败: HTTP {response.status_code}")
                return []
        except Exception as e:
            print(f"获取知识库异常: {e}")
            return []

# 使用示例
def connection_example():
    """连接示例"""
    # 创建连接器
    connector = RAGFlowAPIConnector()

    # 测试连接
    if connector.test_connection():
        print("✅ RAGFlow连接成功!")

        # 获取知识库列表
        knowledge_bases = connector.get_knowledge_bases()
        print(f"📚 发现 {len(knowledge_bases)} 个知识库:")

        for kb in knowledge_bases:
            if isinstance(kb, str):
                print(f"  - {kb}")
            elif isinstance(kb, dict):
                kb_name = kb.get('name', '未知')
                kb_desc = kb.get('description', '无描述')
                print(f"  - {kb_name}: {kb_desc}")
    else:
        print("❌ RAGFlow连接失败")

if __name__ == "__main__":
    connection_example()
```

### 2. 检索功能实现

```python
def search_knowledge_base(self,
                        kb_name: str,
                        query: str,
                        top_k: int = 5,
                        similarity_threshold: float = 0.7) -> List[Dict]:
    """
    在指定知识库中搜索文档

    Args:
        kb_name: 知识库名称或ID
        query: 查询内容
        top_k: 返回结果数量
        similarity_threshold: 相似度阈值

    Returns:
        搜索结果列表
    """
    try:
        # 构建搜索请求
        data = {
            "question": query,
            "dataset_ids": [kb_name],  # RAGFlow使用dataset_ids数组
            "top_k": top_k,
            "similarity_threshold": similarity_threshold
        }

        # 发送搜索请求
        response = self.session.post(
            f"{self.base_url}/api/v1/retrieval",
            json=data,
            timeout=self.timeout
        )

        if response.status_code == 200:
            result = response.json()

            if result.get("code") == 0:
                # 提取搜索结果
                data = result.get("data", {})
                chunks = data.get("chunks", [])

                if not chunks and isinstance(data, list):
                    chunks = data

                # 格式化结果
                formatted_results = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        formatted_results.append({
                            "content": chunk.get("content", chunk.get("text", str(chunk))),
                            "source": chunk.get("document_source", "ragflow"),
                            "score": chunk.get("similarity", chunk.get("score", 0.0)),
                            "doc_id": chunk.get("document_id", chunk.get("id", "")),
                            "title": chunk.get("document_name", chunk.get("title", "")),
                            "raw_data": chunk
                        })

                return formatted_results
            else:
                print(f"搜索 API 错误: {result.get('message', '未知错误')}")
                return []
        else:
            print(f"搜索失败: HTTP {response.status_code}")
            return []

    except Exception as e:
        print(f"搜索异常: {e}")
        return []

# 使用示例
def search_example():
    """搜索示例"""
    connector = RAGFlowAPIConnector()

    if connector.test_connection():
        # 获取知识库
        knowledge_bases = connector.get_knowledge_bases()

        if knowledge_bases:
            kb_name = knowledge_bases[0].get('id') if isinstance(knowledge_bases[0], dict) else knowledge_bases[0]

            # 执行搜索
            query = "什么是人工智能？"
            results = connector.search_knowledge_base(
                kb_name=kb_name,
                query=query,
                top_k=5,
                similarity_threshold=0.7
            )

            print(f"\n🔍 搜索查询: {query}")
            print(f"📋 找到 {len(results)} 个相关文档:")

            for i, result in enumerate(results, 1):
                print(f"\n{i}. [相似度: {result['score']:.3f}] {result['title']}")
                print(f"   来源: {result['source']}")
                content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                print(f"   内容: {content_preview}")
```

---

## 🔍 基本检索功能

### 1. LangChain 集成检索器

```python
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, ConfigDict

class RAGFlowRetriever(BaseRetriever):
    """RAGFlow检索器 - 将RAGFlow集成到LangChain中"""

    connector: RAGFlowAPIConnector = Field(description="RAGFlow连接器")
    kb_name: str = Field(description="知识库名称")
    top_k: int = Field(default=5, description="返回结果数量")
    similarity_threshold: float = Field(default=0.7, description="相似度阈值")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """检索相关文档"""
        # 调用RAGFlow API搜索
        search_results = self.connector.search_knowledge_base(
            kb_name=self.kb_name,
            query=query,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )

        # 转换为LangChain Document格式
        documents = []
        for result in search_results:
            doc = Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": result.get("source", "ragflow"),
                    "score": result.get("score", 0.0),
                    "doc_id": result.get("doc_id", ""),
                    "kb_name": self.kb_name,
                    "title": result.get("title", ""),
                    "url": result.get("url", "")
                }
            )
            documents.append(doc)

        return documents

    def get_relevant_documents(self, query: str) -> List[Document]:
        """公共方法：检索相关文档"""
        return self._get_relevant_documents(query)

# 使用示例
def retriever_example():
    """检索器使用示例"""
    # 创建连接器和检索器
    connector = RAGFlowAPIConnector()

    if connector.test_connection():
        # 获取知识库
        knowledge_bases = connector.get_knowledge_bases()
        if knowledge_bases:
            kb_name = knowledge_bases[0].get('id') if isinstance(knowledge_bases[0], dict) else knowledge_bases[0]

            # 创建检索器
            retriever = RAGFlowRetriever(
                connector=connector,
                kb_name=kb_name,
                top_k=5,
                similarity_threshold=0.7
            )

            # 执行检索
            query = "机器学习的基本概念"
            documents = retriever.get_relevant_documents(query)

            print(f"🔍 查询: {query}")
            print(f"📋 检索到 {len(documents)} 个文档:")

            for i, doc in enumerate(documents, 1):
                metadata = doc.metadata
                print(f"\n{i}. [分数: {metadata['score']:.3f}] {metadata['title']}")
                print(f"   来源: {metadata['source']}")
                print(f"   内容: {doc.page_content[:100]}...")
```

### 2. 多知识库检索器

```python
class MultiKBRetriever(BaseRetriever):
    """多知识库检索器 - 同时搜索多个知识库"""

    retrievers: Dict[str, RAGFlowRetriever] = Field(default_factory=dict, description="知识库检索器字典")

    def __init__(self, connector: RAGFlowAPIConnector, kb_names: List[str]):
        super().__init__(retrievers={})
        for kb_name in kb_names:
            self.retrievers[kb_name] = RAGFlowRetriever(
                connector=connector,
                kb_name=kb_name,
                top_k=3,  # 每个知识库返回3个结果
                similarity_threshold=0.7
            )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """从所有知识库中检索相关文档"""
        all_docs = []
        for kb_name, retriever in self.retrievers.items():
            docs = retriever.get_relevant_documents(query)
            # 添加知识库标识
            for doc in docs:
                doc.metadata["knowledge_base"] = kb_name
            all_docs.extend(docs)

        # 按相似度排序
        all_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
        return all_docs[:10]  # 返回前10个结果

    def get_relevant_documents(self, query: str) -> List[Document]:
        """公共方法：检索相关文档"""
        return self._get_relevant_documents(query)

    model_config = ConfigDict(arbitrary_types_allowed=True)

# 使用示例
def multi_kb_example():
    """多知识库检索示例"""
    connector = RAGFlowAPIConnector()

    if connector.test_connection():
        knowledge_bases = connector.get_knowledge_bases()
        if len(knowledge_bases) >= 2:
            # 选择前两个知识库
            kb_names = []
            for kb in knowledge_bases[:2]:
                kb_name = kb.get('id') if isinstance(kb, dict) else kb
                kb_names.append(kb_name)

            # 创建多知识库检索器
            multi_retriever = MultiKBRetriever(connector, kb_names)

            # 执行检索
            query = "深度学习的应用领域"
            documents = multi_retriever.get_relevant_documents(query)

            print(f"🔍 多知识库查询: {query}")
            print(f"📋 检索到 {len(documents)} 个文档:")

            for i, doc in enumerate(documents, 1):
                metadata = doc.metadata
                kb_name = metadata.get('knowledge_base', '未知')
                print(f"\n{i}. [知识库: {kb_name}] [分数: {metadata['score']:.3f}] {metadata['title']}")
                print(f"   来源: {metadata['source']}")
                print(f"   内容: {doc.page_content[:100]}...")
```

---

## 💬 提示词与上下文构建

### 1. 基础提示词模板

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

class RAGFlowQAChain:
    """RAGFlow问答链"""

    def __init__(self, retriever: BaseRetriever, llm: any):
        self.retriever = retriever
        self.llm = llm

    def create_basic_chain(self):
        """创建基础问答链"""
        template = """你是一个专业的AI助手。请基于以下提供的上下文信息来回答用户的问题。
如果上下文中没有相关信息，请诚实地说明，不要编造信息。

上下文信息：
{context}

用户问题：{question}

请提供详细、准确的回答："""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            """格式化检索到的文档"""
            return "\n\n".join(doc.page_content for doc in docs)

        # 构建链
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def create_contextual_chain(self):
        """创建上下文增强问答链"""
        template = """你是一个专业的AI助手，专门帮助用户基于知识库内容回答问题。

知识库上下文：
{context}

用户问题：{question}

请基于上下文信息提供准确、详细的回答。如果上下文中信息不足，请明确说明。
回答时请保持专业性和友好性："""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs_with_scores(docs):
            """格式化文档，包含相似度分数"""
            formatted_docs = []
            for doc in docs:
                score = doc.metadata.get("score", 0.0)
                source = doc.metadata.get("source", "未知")
                title = doc.metadata.get("title", "")

                doc_content = f"[相似度: {score:.3f}] 来源: {source}"
                if title:
                    doc_content += f" | 标题: {title}"
                doc_content += f"\n{doc.page_content}"

                formatted_docs.append(doc_content)

            return "\n\n---\n\n".join(formatted_docs)

        chain = (
            {
                "context": self.retriever | format_docs_with_scores,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def create_chain_with_sources(self):
        """创建带来源引用的问答链"""
        template = """你是一个专业的AI助手。请基于以下提供的上下文信息来回答用户的问题，并在回答中引用信息来源。

上下文信息：
{context}

用户问题：{question}

请提供详细、准确的回答，并标注信息来源："""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        def format_docs_with_sources(docs):
            """格式化文档，包含详细来源信息"""
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知")
                title = doc.metadata.get("title", "")
                score = doc.metadata.get("score", 0.0)
                kb_name = doc.metadata.get("kb_name", "")
                doc_id = doc.metadata.get("doc_id", "")

                doc_content = f"文档 {i}:"
                doc_content += f"\n- 标题: {title}"
                doc_content += f"\n- 来源: {source}"
                doc_content += f"\n- 知识库: {kb_name}"
                doc_content += f"\n- 相似度: {score:.3f}"
                doc_content += f"\n- 文档ID: {doc_id}"
                doc_content += f"\n- 内容: {doc.page_content}"

                formatted_docs.append(doc_content)

            return "\n\n---\n\n".join(formatted_docs)

        chain = (
            {
                "context": self.retriever | format_docs_with_sources,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

# 使用示例
def qa_chain_example():
    """问答链示例"""
    # 初始化组件
    connector = RAGFlowAPIConnector()

    if connector.test_connection():
        # 获取知识库
        knowledge_bases = connector.get_knowledge_bases()
        if knowledge_bases:
            kb_name = knowledge_bases[0].get('id') if isinstance(knowledge_bases[0], dict) else knowledge_bases[0]

            # 创建检索器和LLM
            retriever = RAGFlowRetriever(connector, kb_name)

            # 使用GLM或OpenAI
            if os.getenv("GLM_API_KEY"):
                llm = ChatOpenAI(
                    model=os.getenv("LLM_MODEL", "glm-4"),
                    temperature=0.1,
                    openai_api_key=os.getenv("GLM_API_KEY"),
                    openai_api_base=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
                )
            else:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )

            # 创建问答链
            qa_chain = RAGFlowQAChain(retriever, llm)

            # 测试不同类型的链
            test_question = "什么是深度学习？"

            print(f"❓ 问题: {test_question}")
            print("\n" + "="*50)

            # 基础链
            basic_chain = qa_chain.create_basic_chain()
            print("🔗 基础问答链:")
            answer = basic_chain.invoke(test_question)
            print(answer)

            print("\n" + "="*50)

            # 上下文增强链
            contextual_chain = qa_chain.create_contextual_chain()
            print("🔗 上下文增强问答链:")
            answer = contextual_chain.invoke(test_question)
            print(answer)

            print("\n" + "="*50)

            # 带来源链
            sources_chain = qa_chain.create_chain_with_sources()
            print("🔗 带来源引用问答链:")
            answer = sources_chain.invoke(test_question)
            print(answer)
```

### 2. 高级提示词技巧

```python
class AdvancedPromptTemplate:
    """高级提示词模板"""

    @staticmethod
    def create_conditional_template():
        """创建条件响应模板"""
        template = """你是一个专业的AI助手。请基于提供的上下文回答用户问题。

{context}

回答指南：
1. 如果上下文中包含相关信息，请基于上下文详细回答
2. 如果上下文中信息不足，请诚实地说明
3. 如果完全无法找到相关信息，请礼貌地解释无法回答

用户问题：{question}

请提供专业的回答："""

        return PromptTemplate(template=template, input_variables=["context", "question"])

    @staticmethod
    def create_step_by_step_template():
        """创建分步推理模板"""
        template = """你是一个专业的分析师。请按照以下步骤回答问题：

步骤1: 分析用户问题的核心需求
步骤2: 从提供的上下文中提取相关信息
步骤3: 综合信息得出结论
步骤4: 提供清晰、准确的回答

上下文信息：
{context}

用户问题：{question}

请按步骤进行分析和回答："""

        return PromptTemplate(template=template, input_variables=["context", "question"])

    @staticmethod
    def create_role_based_template(role: str):
        """创建角色化模板"""
        role_templates = {
            "专家": """你是一位资深专家，拥有深厚的专业知识。请基于以下信息提供权威、专业的回答。

{context}

作为领域专家，请对以下问题进行专业分析：
{question}

请提供专业的见解和建议：""",

            "教师": """你是一位耐心的教师，擅长用简单易懂的方式解释复杂概念。

{context}

请用教学的方式回答以下问题：
{question}

请提供清晰、易懂的解释：""",

            "分析师": """你是一位数据分析师，擅长从信息中提取关键洞察。

{context}

请从数据分析的角度回答：
{question}

请提供基于数据的分析："""
        }

        template = role_templates.get(role, role_templates["专家"])
        return PromptTemplate(template=template, input_variables=["context", "question"])

# 使用示例
def advanced_prompt_example():
    """高级提示词示例"""
    connector = RAGFlowAPIConnector()

    if connector.test_connection():
        knowledge_bases = connector.get_knowledge_bases()
        if knowledge_bases:
            kb_name = knowledge_bases[0].get('id') if isinstance(knowledge_bases[0], dict) else knowledge_bases[0]

            retriever = RAGFlowRetriever(connector, kb_name)
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

            # 创建高级提示词链
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            test_question = "深度学习的优势是什么？"

            # 条件响应模板
            conditional_prompt = AdvancedPromptTemplate.create_conditional_template()
            conditional_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | conditional_prompt
                | llm
                | StrOutputParser()
            )

            print(f"❓ 问题: {test_question}")
            print("\n🎯 条件响应模板:")
            print(conditional_chain.invoke(test_question))

            # 分步推理模板
            step_prompt = AdvancedPromptTemplate.create_step_by_step_template()
            step_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | step_prompt
                | llm
                | StrOutputParser()
            )

            print("\n🔍 分步推理模板:")
            print(step_chain.invoke(test_question))

            # 角色化模板
            expert_prompt = AdvancedPromptTemplate.create_role_based_template("专家")
            expert_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | expert_prompt
                | llm
                | StrOutputParser()
            )

            print("\n👨‍🏫 专家角色模板:")
            print(expert_chain.invoke(test_question))
```

---

## 🚀 完整应用示例

### 1. 完整的RAGFlow应用类

```python
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 加载环境变量
load_dotenv()

class RAGFlowApp:
    """完整的RAGFlow应用"""

    def __init__(self, ragflow_url: str = None, ragflow_api_key: str = None):
        """初始化应用"""
        # RAGFlow连接器
        self.connector = RAGFlowAPIConnector(ragflow_url, ragflow_api_key)

        # LLM配置
        if os.getenv("GLM_API_KEY"):
            self.llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "glm-4.5"),
                temperature=0.1,
                openai_api_key=os.getenv("GLM_API_KEY"),
                openai_api_base=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
            )
        else:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

        self.available_kbs = []
        self.retrievers = {}

    def initialize(self) -> bool:
        """初始化应用"""
        print("🚀 初始化RAGFlow应用...")

        # 测试连接
        if not self.connector.test_connection():
            print("❌ RAGFlow连接失败")
            return False

        # 获取知识库
        self.available_kbs = self.connector.get_knowledge_bases()
        print(f"✅ 连接成功，发现 {len(self.available_kbs)} 个知识库")

        return True

    def create_retriever(self, kb_name: str, top_k: int = 5) -> Optional[RAGFlowRetriever]:
        """创建检索器"""
        retriever = RAGFlowRetriever(
            connector=self.connector,
            kb_name=kb_name,
            top_k=top_k,
            similarity_threshold=0.7
        )

        self.retrievers[kb_name] = retriever
        return retriever

    def create_qa_chain(self, kb_name: str, chain_type: str = "basic"):
        """创建问答链"""
        if kb_name not in self.retrievers:
            self.create_retriever(kb_name)

        retriever = self.retrievers[kb_name]

        if chain_type == "basic":
            return self._create_basic_chain(retriever)
        elif chain_type == "with_sources":
            return self._create_chain_with_sources(retriever)
        elif chain_type == "contextual":
            return self._create_contextual_chain(retriever)
        else:
            raise ValueError(f"不支持的链类型: {chain_type}")

    def _create_basic_chain(self, retriever):
        """创建基础链"""
        template = """你是一个专业的AI助手。请基于以下提供的上下文信息来回答用户的问题。

{context}

用户问题：{question}

请提供详细、准确的回答："""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _create_chain_with_sources(self, retriever):
        """创建带来源的链"""
        template = """你是一个专业的AI助手。请基于以下提供的上下文信息来回答用户的问题，并在回答中引用信息来源。

{context}

用户问题：{question}

请提供详细、准确的回答，并标注信息来源："""

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        def format_docs_with_sources(docs):
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知")
                title = doc.metadata.get("title", "")
                score = doc.metadata.get("score", 0.0)

                doc_content = f"文档 {i} (来源: {source}, 相似度: {score:.3f})\n"
                if title:
                    doc_content += f"标题: {title}\n"
                doc_content += f"内容: {doc.page_content}"

                formatted_docs.append(doc_content)

            return "\n\n---\n\n".join(formatted_docs)

        chain = (
            {
                "context": retriever | format_docs_with_sources,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def chat(self, kb_name: str, question: str, chain_type: str = "basic") -> str:
        """聊天功能"""
        try:
            chain = self.create_qa_chain(kb_name, chain_type)
            return chain.invoke(question)
        except Exception as e:
            return f"聊天出错: {e}"

# 完整使用示例
def complete_app_example():
    """完整应用示例"""
    print("🎯 RAGFlow完整应用示例")
    print("=" * 60)

    # 创建应用实例
    app = RAGFlowApp()

    # 初始化
    if not app.initialize():
        return

    # 显示可用知识库
    print("\n📚 可用知识库:")
    for i, kb in enumerate(app.available_kbs, 1):
        if isinstance(kb, str):
            print(f"{i}. {kb}")
        elif isinstance(kb, dict):
            kb_name = kb.get('name', '未知')
            kb_desc = kb.get('description', '无描述')
            print(f"{i}. {kb_name}: {kb_desc}")

    # 选择第一个知识库进行演示
    if app.available_kbs:
        first_kb = app.available_kbs[0]
        kb_name = first_kb.get('id') if isinstance(first_kb, dict) else first_kb

        print(f"\n🎯 选择知识库: {kb_name}")

        # 测试问题
        test_questions = [
            "什么是机器学习？",
            "深度学习有哪些应用？",
            "自然语言处理的挑战是什么？"
        ]

        for question in test_questions:
            print(f"\n❓ 问题: {question}")
            print("-" * 40)

            try:
                # 基础问答
                print("🔗 基础问答:")
                basic_answer = app.chat(kb_name, question, "basic")
                print(basic_answer)

                print("\n🔗 带来源问答:")
                sources_answer = app.chat(kb_name, question, "with_sources")
                print(sources_answer)

            except Exception as e:
                print(f"❌ 问答出错: {e}")

            print("=" * 60)

if __name__ == "__main__":
    complete_app_example()
```

### 2. 交互式命令行应用

```python
def interactive_qa_app():
    """交互式问答应用"""
    print("🤖 RAGFlow 交互式问答应用")
    print("=" * 50)

    # 初始化应用
    app = RAGFlowApp()
    if not app.initialize():
        print("❌ 应用初始化失败")
        return

    # 选择知识库
    if not app.available_kbs:
        print("❌ 没有可用的知识库")
        return

    print("\n📚 请选择知识库:")
    for i, kb in enumerate(app.available_kbs, 1):
        if isinstance(kb, str):
            print(f"{i}. {kb}")
        elif isinstance(kb, dict):
            kb_name = kb.get('name', '未知')
            kb_desc = kb.get('description', '无描述')[:50]
            print(f"{i}. {kb_name} - {kb_desc}")

    try:
        choice = int(input("\n请选择知识库 (输入数字): ")) - 1
        if 0 <= choice < len(app.available_kbs):
            selected_kb = app.available_kbs[choice]
            kb_name = selected_kb.get('id') if isinstance(selected_kb, dict) else selected_kb

            print(f"\n✅ 已选择知识库: {kb_name}")
            print("💬 开始问答 (输入 'quit' 退出)")
            print("-" * 50)

            while True:
                question = input("\n❓ 请输入您的问题: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 感谢使用！")
                    break

                if not question:
                    continue

                print("🤔 正在思考...")

                try:
                    # 获取回答
                    answer = app.chat(kb_name, question, "with_sources")
                    print(f"\n🤖 回答:")
                    print(answer)
                    print("-" * 50)
                except Exception as e:
                    print(f"❌ 回答出错: {e}")

        else:
            print("❌ 无效选择")

    except ValueError:
        print("❌ 请输入有效数字")

if __name__ == "__main__":
    interactive_qa_app()
```

---

## 🔧 高级功能

### 1. 批量处理功能

```python
def batch_search_example():
    """批量搜索示例"""
    connector = RAGFlowAPIConnector()

    if connector.test_connection():
        knowledge_bases = connector.get_knowledge_bases()
        if knowledge_bases:
            kb_name = knowledge_bases[0].get('id') if isinstance(knowledge_bases[0], dict) else knowledge_bases[0]

            # 批量查询
            questions = [
                "机器学习的定义",
                "深度学习的发展历史",
                "神经网络的基本原理",
                "自然语言处理的应用"
            ]

            print("🔄 批量搜索中...")

            for i, question in enumerate(questions, 1):
                print(f"\n{i}. 问题: {question}")

                results = connector.search_knowledge_base(
                    kb_name=kb_name,
                    query=question,
                    top_k=3,
                    similarity_threshold=0.6
                )

                if results:
                    print(f"   找到 {len(results)} 个相关结果")
                    for j, result in enumerate(results, 1):
                        print(f"     {j}. [相似度: {result['score']:.3f}] {result['title']}")
                else:
                    print("   未找到相关结果")

if __name__ == "__main__":
    batch_search_example()
```

### 2. 性能监控

```python
import time
from typing import Dict, List

class PerformanceMonitor:
    """性能监控类"""

    def __init__(self):
        self.search_times: List[float] = []
        self.llm_times: List[float] = []
        self.total_times: List[float] = []

    def monitor_search(self, connector, kb_name: str, query: str):
        """监控搜索性能"""
        start_time = time.time()

        results = connector.search_knowledge_base(
            kb_name=kb_name,
            query=query,
            top_k=5,
            similarity_threshold=0.7
        )

        search_time = time.time() - start_time
        self.search_times.append(search_time)

        return results, search_time

    def monitor_qa_chain(self, chain, question: str):
        """监控问答链性能"""
        start_time = time.time()

        answer = chain.invoke(question)

        total_time = time.time() - start_time
        self.total_times.append(total_time)

        return answer, total_time

    def get_stats(self) -> Dict:
        """获取性能统计"""
        stats = {}

        if self.search_times:
            stats['search'] = {
                'count': len(self.search_times),
                'avg_time': sum(self.search_times) / len(self.search_times),
                'min_time': min(self.search_times),
                'max_time': max(self.search_times)
            }

        if self.total_times:
            stats['total'] = {
                'count': len(self.total_times),
                'avg_time': sum(self.total_times) / len(self.total_times),
                'min_time': min(self.total_times),
                'max_time': max(self.total_times)
            }

        return stats

    def print_stats(self):
        """打印性能统计"""
        stats = self.get_stats()

        print("📊 性能统计报告:")
        print("=" * 40)

        if 'search' in stats:
            search_stats = stats['search']
            print(f"🔍 搜索性能:")
            print(f"   次数: {search_stats['count']}")
            print(f"   平均时间: {search_stats['avg_time']:.3f}s")
            print(f"   最快时间: {search_stats['min_time']:.3f}s")
            print(f"   最慢时间: {search_stats['max_time']:.3f}s")

        if 'total' in stats:
            total_stats = stats['total']
            print(f"\n💬 问答性能:")
            print(f"   次数: {total_stats['count']}")
            print(f"   平均时间: {total_stats['avg_time']:.3f}s")
            print(f"   最快时间: {total_stats['min_time']:.3f}s")
            print(f"   最慢时间: {total_stats['max_time']:.3f}s")

def performance_test_example():
    """性能测试示例"""
    connector = RAGFlowAPIConnector()
    monitor = PerformanceMonitor()

    if connector.test_connection():
        knowledge_bases = connector.get_knowledge_bases()
        if knowledge_bases:
            kb_name = knowledge_bases[0].get('id') if isinstance(knowledge_bases[0], dict) else knowledge_bases[0]

            # 创建应用和问答链
            app = RAGFlowApp()
            app.initialize()
            chain = app.create_qa_chain(kb_name, "with_sources")

            # 测试问题
            test_questions = [
                "什么是人工智能",
                "机器学习的应用",
                "深度学习的发展",
                "自然语言处理技术",
                "计算机视觉的应用领域"
            ]

            print("🚀 开始性能测试...")

            for question in test_questions:
                print(f"\n🔍 测试问题: {question}")

                # 监控搜索
                results, search_time = monitor.monitor_search(connector, kb_name, question)
                print(f"   搜索时间: {search_time:.3f}s, 结果数: {len(results)}")

                # 监控问答
                answer, total_time = monitor.monitor_qa_chain(chain, question)
                print(f"   总时间: {total_time:.3f}s")

            # 打印性能统计
            print("\n")
            monitor.print_stats()

if __name__ == "__main__":
    performance_test_example()
```

### 3. 缓存机制

```python
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

class RAGFlowCache:
    """RAGFlow缓存系统"""

    def __init__(self, cache_dir: str = "ragflow_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.search_cache = {}
        self.retriever_cache = {}

    def _get_cache_key(self, kb_name: str, query: str, **kwargs) -> str:
        """生成缓存键"""
        # 创建包含所有参数的字符串
        params_str = f"{kb_name}:{query}:{sorted(kwargs.items())}"
        # 生成MD5哈希作为缓存键
        return hashlib.md5(params_str.encode()).hexdigest()

    def cache_search_result(self, kb_name: str, query: str, results: List[Dict], **kwargs):
        """缓存搜索结果"""
        cache_key = self._get_cache_key(kb_name, query, **kwargs)
        cache_data = {
            'results': results,
            'timestamp': time.time(),
            'kb_name': kb_name,
            'query': query,
            'params': kwargs
        }

        # 保存到内存
        self.search_cache[cache_key] = cache_data

        # 保存到文件
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"缓存保存失败: {e}")

    def get_cached_search_result(self, kb_name: str, query: str, **kwargs) -> Optional[List[Dict]]:
        """获取缓存的搜索结果"""
        cache_key = self._get_cache_key(kb_name, query, **kwargs)

        # 先检查内存缓存
        if cache_key in self.search_cache:
            cached_data = self.search_cache[cache_key]
            # 检查缓存是否过期（1小时）
            if time.time() - cached_data['timestamp'] < 3600:
                return cached_data['results']
            else:
                # 删除过期缓存
                del self.search_cache[cache_key]

        # 检查文件缓存
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # 检查缓存是否过期
                if time.time() - cached_data['timestamp'] < 3600:
                    # 加载到内存缓存
                    self.search_cache[cache_key] = cached_data
                    return cached_data['results']
                else:
                    # 删除过期缓存文件
                    cache_file.unlink()
            except Exception as e:
                print(f"缓存读取失败: {e}")

        return None

    def clear_cache(self):
        """清空缓存"""
        self.search_cache.clear()
        self.retriever_cache.clear()

        # 删除缓存文件
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"删除缓存文件失败: {e}")

# 缓存版本的连接器
class CachedRAGFlowConnector(RAGFlowAPIConnector):
    """带缓存的RAGFlow连接器"""

    def __init__(self, base_url: str = None, api_key: str = None, timeout: int = 60, cache: RAGFlowCache = None):
        super().__init__(base_url, api_key, timeout)
        self.cache = cache or RAGFlowCache()

    def search_knowledge_base(self, kb_name: str, query: str, top_k: int = 5, similarity_threshold: float = 0.7):
        """带缓存的搜索"""
        # 先检查缓存
        cached_results = self.cache.get_cached_search_result(
            kb_name=kb_name,
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        if cached_results is not None:
            print("💾 使用缓存结果")
            return cached_results

        # 缓存未命中，执行搜索
        print("🔍 执行新搜索")
        results = super().search_knowledge_base(kb_name, query, top_k, similarity_threshold)

        # 缓存结果
        self.cache.cache_search_result(
            kb_name=kb_name,
            query=query,
            results=results,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        return results

def cache_example():
    """缓存使用示例"""
    print("📦 RAGFlow缓存示例")
    print("=" * 40)

    # 创建缓存和连接器
    cache = RAGFlowCache()
    connector = CachedRAGFlowConnector(cache=cache)

    if connector.test_connection():
        knowledge_bases = connector.get_knowledge_bases()
        if knowledge_bases:
            kb_name = knowledge_bases[0].get('id') if isinstance(knowledge_bases[0], dict) else knowledge_bases[0]

            test_queries = [
                "什么是人工智能",
                "什么是人工智能",  # 相同查询，应该使用缓存
                "机器学习的发展历史"
            ]

            print(f"🎯 知识库: {kb_name}")
            print("\n🔍 执行测试查询:")

            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. 查询: {query}")

                start_time = time.time()
                results = connector.search_knowledge_base(
                    kb_name=kb_name,
                    query=query,
                    top_k=3,
                    similarity_threshold=0.7
                )
                search_time = time.time() - start_time

                print(f"   搜索时间: {search_time:.3f}s")
                print(f"   结果数: {len(results)}")

                if results:
                    for j, result in enumerate(results[:2], 1):
                        print(f"     {j}. {result['title'][:50]}... [分数: {result['score']:.3f}]")

if __name__ == "__main__":
    cache_example()
```

---

## 📚 总结

这个完整的代码示例展示了如何：

1. **连接RAGFlow服务**：通过API连接器实现稳定连接
2. **实现检索功能**：单知识库和多知识库检索
3. **构建提示词**：多种高级提示词模板和技巧
4. **创建完整应用**：包括交互式应用和性能监控
5. **高级功能**：缓存机制、批量处理等

### 🔑 关键要点

- **配置管理**：使用环境变量管理敏感信息
- **错误处理**：完善的异常处理和降级机制
- **性能优化**：缓存和性能监控
- **可扩展性**：模块化设计，易于扩展
- **用户体验**：丰富的反馈和进度提示

### 🚀 快速开始

```bash
# 1. 安装依赖
pip install langchain langchain-core langchain-community langchain-openai
pip install requests python-dotenv pydantic

# 2. 配置环境变量
# 创建 .env 文件并配置必要的API密钥

# 3. 运行示例
python your_script.py
```

这个示例提供了完整的RAGFlow + LangChain集成解决方案，您可以根据具体需求进行调整和扩展。