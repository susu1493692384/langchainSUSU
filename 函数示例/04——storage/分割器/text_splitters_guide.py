"""
LangChain 文本分割器完整指南
包含所有分割器类型、参数说明、示例代码和快速查表功能
"""

# 导入所需的所有分割器
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    TokenTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
    SemanticChunker
)
from langchain.embeddings.openai import OpenAIEmbeddings
import pandas as pd
from typing import Dict, Any, List, Optional


class SplitterGuide:
    """LangChain文本分割器指南类"""

    def __init__(self):
        self.splitter_info = self._create_splitter_info()

    def _create_splitter_info(self) -> Dict[str, Dict[str, Any]]:
        """创建分割器信息字典"""
        return {
            "CharacterTextSplitter": {
                "description": "基于指定字符分割文本，是最基础的分割器",
                "use_case": "简单文本、有明确分隔符的场景",
                "parameters": {
                    "separator": {"type": "str", "default": '"\\n\\n"', "description": "分割符，用于确定在哪里分割文本"},
                    "chunk_size": {"type": "int", "default": 1000, "description": "每个文本块的最大字符数"},
                    "chunk_overlap": {"type": "int", "default": 200, "description": "文本块之间的重叠字符数"},
                    "length_function": {"type": "callable", "default": "len", "description": "用于计算文本长度的函数"},
                    "keep_separator": {"type": "bool", "default": False, "description": "是否在分割后的文本块中保留分隔符"}
                },
                "example": self._character_splitter_example,
                "pros": ["简单快速", "可控性强"],
                "cons": ["可能破坏语义结构"],
                "performance": "最快"
            },

            "RecursiveCharacterTextSplitter": {
                "description": "递归字符文本分割器，按优先级顺序尝试不同的分隔符来保持文本的语义完整性",
                "use_case": "通用文本、RAG系统，最常用的分割器",
                "parameters": {
                    "chunk_size": {"type": "int", "default": 1000, "description": "每个文本块的最大字符数"},
                    "chunk_overlap": {"type": "int", "default": 200, "description": "文本块之间的重叠字符数"},
                    "length_function": {"type": "callable", "default": "len", "description": "用于计算文本长度的函数"},
                    "separators": {"type": "List[str]", "default": '["\\n\\n", "\\n", " ", ""]', "description": "分隔符优先级列表"},
                    "keep_separator": {"type": "bool", "default": False, "description": "是否在分割后的文本块中保留分隔符"}
                },
                "example": self._recursive_character_splitter_example,
                "pros": ["保持语义完整", "最常用", "平衡性好"],
                "cons": ["可能产生不均匀的文本块"],
                "performance": "平衡"
            },

            "HTMLHeaderTextSplitter": {
                "description": "专门用于HTML文档，基于标题标签(h1, h2, h3等)进行分割",
                "use_case": "HTML网页、富文本文档",
                "parameters": {
                    "headers_to_split_on": {"type": "List[Tuple[str, str]]", "default": None, "description": "要分割的标题标签列表"},
                    "return_each_element": {"type": "bool", "default": False, "description": "是否返回每个HTML元素"}
                },
                "example": self._html_splitter_example,
                "pros": ["保持HTML结构", "语义清晰"],
                "cons": ["仅适用于HTML"],
                "performance": "中等"
            },

            "MarkdownTextSplitter": {
                "description": "专门用于Markdown文档，基于标题级别进行分割",
                "use_case": "Markdown文档、技术文档、README文件",
                "parameters": {
                    "chunk_size": {"type": "int", "default": 1000, "description": "每个文本块的最大字符数"},
                    "chunk_overlap": {"type": "int", "default": 200, "description": "文本块之间的重叠字符数"},
                    "headers_to_split_on": {"type": "List[Tuple[str, str]]", "default": None, "description": "要分割的Markdown标题级别"}
                },
                "example": self._markdown_splitter_example,
                "pros": ["保持文档结构", "适合技术文档"],
                "cons": ["仅适用于Markdown"],
                "performance": "中等"
            },

            "PythonCodeTextSplitter": {
                "description": "专门用于Python代码，基于类、函数等逻辑结构进行分割",
                "use_case": "Python代码文档、源码分析",
                "parameters": {
                    "chunk_size": {"type": "int", "default": 1000, "description": "每个文本块的最大字符数"},
                    "chunk_overlap": {"type": "int", "default": 200, "description": "文本块之间的重叠字符数"},
                    "language": {"type": "str", "default": '"python"', "description": "编程语言"}
                },
                "example": self._python_code_splitter_example,
                "pros": ["保持代码逻辑结构", "理解性强"],
                "cons": ["仅适用于Python代码"],
                "performance": "中等"
            },

            "TokenTextSplitter": {
                "description": "基于令牌数量而非字符数进行分割，更适合LLM的上下文窗口限制",
                "use_case": "LLM上下文限制、API调用场景",
                "parameters": {
                    "chunk_size": {"type": "int", "default": 1000, "description": "每个文本块的最大令牌数"},
                    "chunk_overlap": {"type": "int", "default": 200, "description": "文本块之间的重叠令牌数"},
                    "model_name": {"type": "str", "default": None, "description": "用于计算令牌数量的模型名称"},
                    "encoding_name": {"type": "str", "default": None, "description": "使用的编码名称"}
                },
                "example": self._token_splitter_example,
                "pros": ["精确控制令牌数", "适合LLM"],
                "cons": ["需要额外依赖", "计算成本高"],
                "performance": "中等"
            },

            "NLTKTextSplitter": {
                "description": "使用NLTK库进行更智能的文本分割，基于句子边界",
                "use_case": "自然语言文本、学术文档",
                "parameters": {
                    "chunk_size": {"type": "int", "default": 1000, "description": "每个文本块的最大字符数"},
                    "chunk_overlap": {"type": "int", "default": 200, "description": "文本块之间的重叠字符数"}
                },
                "example": self._nltk_splitter_example,
                "pros": ["智能句子边界", "语言学准确"],
                "cons": ["需要NLTK依赖", "需要下载语言包"],
                "performance": "中等"
            },

            "SpacyTextSplitter": {
                "description": "使用SpaCy库进行基于语言学特征的文本分割",
                "use_case": "专业文档、多语言文本",
                "parameters": {
                    "chunk_size": {"type": "int", "default": 1000, "description": "每个文本块的最大字符数"},
                    "chunk_overlap": {"type": "int", "default": 200, "description": "文本块之间的重叠字符数"},
                    "pipeline": {"type": "str", "default": None, "description": "使用的SpaCy管道名称"}
                },
                "example": self._spacy_splitter_example,
                "pros": ["高质量语言学分析", "多语言支持"],
                "cons": ["需要SpaCy依赖", "资源占用大"],
                "performance": "较慢"
            },

            "SemanticChunker": {
                "description": "基于文本的语义相似度进行分割，保持语义相关的文本在一起",
                "use_case": "高质量RAG系统、语义检索",
                "parameters": {
                    "embeddings": {"type": "Embeddings", "default": None, "description": "用于计算语义相似度的嵌入模型"},
                    "buffer_size": {"type": "int", "default": None, "description": "缓冲区大小"},
                    "min_chunk_size": {"type": "int", "default": None, "description": "最小文本块大小"},
                    "max_chunk_size": {"type": "int", "default": None, "description": "最大文本块大小"},
                    "breakpoint_threshold_type": {"type": "str", "default": '"percentile"', "description": "断点阈值类型"}
                },
                "example": self._semantic_chunker_example,
                "pros": ["保持语义连贯性", "质量最高"],
                "cons": ["计算成本高", "需要嵌入模型"],
                "performance": "最慢"
            }
        }

    def _character_splitter_example(self):
        """CharacterTextSplitter 示例"""
        print("=== CharacterTextSplitter 示例 ===")

        # 基本用法
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=200,
            chunk_overlap=50,
            length_function=len
        )

        text = """这是一段很长的文本...
包含多行内容...
需要被分割成合适的块。
每个块都应该有合适的大小。
这样可以更好地处理长文本。"""

        chunks = splitter.split_text(text)
        print(f"原始文本长度: {len(text)}")
        print(f"分割后得到 {len(chunks)} 个文本块")
        for i, chunk in enumerate(chunks):
            print(f"块 {i+1}: {chunk[:50]}...")

        return chunks

    def _recursive_character_splitter_example(self):
        """RecursiveCharacterTextSplitter 示例"""
        print("\n=== RecursiveCharacterTextSplitter 示例 ===")

        # 自定义分隔符优先级（适合中文）
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", "！", "？", " ", ""]
        )

        text = """这是一段包含多个段落的中文文本。每个段落都有不同的内容！
我们希望保持语义完整性；同时确保文本块大小合适。

这是第二段内容，包含更多要展示的信息。
我们需要测试分割器如何处理不同类型的内容。"""

        chunks = splitter.split_text(text)
        print(f"原始文本长度: {len(text)}")
        print(f"分割后得到 {len(chunks)} 个文本块")
        for i, chunk in enumerate(chunks):
            print(f"块 {i+1}: {chunk[:50]}...")

        return chunks

    def _html_splitter_example(self):
        """HTMLHeaderTextSplitter 示例"""
        print("\n=== HTMLHeaderTextSplitter 示例 ===")

        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ]

        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        html_content = """
        <h1>主标题</h1>
        <p>这是主标题下的第一段内容，介绍主要内容。</p>
        <h2>二级标题</h2>
        <p>这是二级标题下的内容，包含详细信息。</p>
        <h3>三级标题</h3>
        <p>这是三级标题下的具体内容。</p>
        <h2>另一个二级标题</h2>
        <p>这是另一个二级标题下的内容。</p>
        """

        chunks = html_splitter.split_text(html_content)
        print(f"HTML内容分割后得到 {len(chunks)} 个块")
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, 'metadata'):
                print(f"块 {i+1} 标题: {chunk.metadata.get('Header 1', 'N/A')}")
            print(f"内容: {chunk.page_content[:50]}...")

        return chunks

    def _markdown_splitter_example(self):
        """MarkdownTextSplitter 示例"""
        print("\n=== MarkdownTextSplitter 示例 ===")

        markdown_splitter = MarkdownTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        markdown_content = """# 主标题

这是主标题下的内容。主标题通常用于整个文档的主题。

## 二级标题

这是二级标题下的内容。二级标题用于组织文档的主要章节。

### 三级标题

这是三级标题下的内容。三级标题用于细分章节内容。

#### 四级标题

这是四级标题下的内容，提供更详细的分类。"""

        chunks = markdown_splitter.split_text(markdown_content)
        print(f"Markdown内容分割后得到 {len(chunks)} 个块")
        for i, chunk in enumerate(chunks):
            print(f"块 {i+1}: {chunk[:50]}...")

        return chunks

    def _python_code_splitter_example(self):
        """PythonCodeTextSplitter 示例"""
        print("\n=== PythonCodeTextSplitter 示例 ===")

        python_splitter = PythonCodeTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        python_code = '''
def hello_world():
    """这是一个简单的函数"""
    print("Hello, World!")
    return "Hello, World!"

class MyClass:
    """这是一个示例类"""
    def __init__(self):
        self.value = 0
        self.name = "示例"

    def increment(self):
        """增加数值"""
        self.value += 1
        return self.value

    def get_info(self):
        """获取信息"""
        return f"名称: {self.name}, 值: {self.value}"

def main():
    """主函数"""
    obj = MyClass()
    print(obj.increment())
    print(obj.get_info())

if __name__ == "__main__":
    main()
        '''

        chunks = python_splitter.split_text(python_code)
        print(f"Python代码分割后得到 {len(chunks)} 个块")
        for i, chunk in enumerate(chunks):
            lines = chunk.strip().split('\n')
            print(f"块 {i+1}: {len(lines)} 行代码")

        return chunks

    def _token_splitter_example(self):
        """TokenTextSplitter 示例"""
        print("\n=== TokenTextSplitter 示例 ===")

        try:
            token_splitter = TokenTextSplitter(
                chunk_size=100,  # 100个令牌
                chunk_overlap=10,
                model_name="gpt-3.5-turbo"
            )

            text = """这是一段需要基于令牌数量进行分割的文本。
            令牌分割器对于处理大语言模型的上下文窗口限制非常有用。
            它可以确保每个文本块都不超过模型的令牌限制。
            这样可以避免API调用时出现令牌超限的错误。"""

            chunks = token_splitter.split_text(text)
            print(f"文本分割后得到 {len(chunks)} 个块")
            for i, chunk in enumerate(chunks):
                print(f"块 {i+1}: {chunk[:50]}...")

            return chunks
        except Exception as e:
            print(f"TokenTextSplitter需要tiktoken库或适当的依赖: {e}")
            return []

    def _nltk_splitter_example(self):
        """NLTKTextSplitter 示例"""
        print("\n=== NLTKTextSplitter 示例 ===")

        try:
            nltk_splitter = NLTKTextSplitter(
                chunk_size=200,
                chunk_overlap=30
            )

            text = "这是第一句话。这是第二句话！这是第三句话？NLTK分割器能够智能地识别句子边界。它使用自然语言处理技术来确保分割的准确性。这样可以保持句子的完整性。"

            chunks = nltk_splitter.split_text(text)
            print(f"文本分割后得到 {len(chunks)} 个块")
            for i, chunk in enumerate(chunks):
                print(f"块 {i+1}: {chunk[:50]}...")

            return chunks
        except Exception as e:
            print(f"NLTKTextSplitter需要nltk库和数据包: {e}")
            print("请运行: pip install nltk && python -c 'import nltk; nltk.download(\"punkt\")'")
            return []

    def _spacy_splitter_example(self):
        """SpacyTextSplitter 示例"""
        print("\n=== SpaCyTextSplitter 示例 ===")

        try:
            spacy_splitter = SpacyTextSplitter(
                chunk_size=200,
                chunk_overlap=30,
                pipeline="zh_core_web_sm"  # 中文模型
            )

            text = "这是一段需要使用spaCy进行智能分割的中文文本。spaCy是一个强大的自然语言处理库，它提供了高质量的语言分析功能。"

            chunks = spacy_splitter.split_text(text)
            print(f"文本分割后得到 {len(chunks)} 个块")
            for i, chunk in enumerate(chunks):
                print(f"块 {i+1}: {chunk[:50]}...")

            return chunks
        except Exception as e:
            print(f"SpacyTextSplitter需要spacy库和语言模型: {e}")
            print("请运行: pip install spacy && python -m spacy download zh_core_web_sm")
            return []

    def _semantic_chunker_example(self):
        """SemanticChunker 示例"""
        print("\n=== SemanticChunker 示例 ===")

        try:
            # 需要OpenAI API密钥
            embeddings = OpenAIEmbeddings()
            semantic_splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile"
            )

            text = """这是一段需要基于语义进行分割的长文本。
            语义分割器能够识别文本中语义相似的部分，
            并将它们组织在一起，形成一个有意义的文本块。
            这种方法特别适合构建高质量的检索增强生成(RAG)系统，
            因为它能够保持上下文的连贯性和语义的完整性。
            相比传统的基于字符或令牌的分割方法，
            语义分割能够产生更高质量的文本块，
            从而提高检索和生成的质量。"""

            chunks = semantic_splitter.split_text(text)
            print(f"文本分割后得到 {len(chunks)} 个块")
            for i, chunk in enumerate(chunks):
                print(f"块 {i+1}: {chunk[:50]}...")

            return chunks
        except Exception as e:
            print(f"SemanticChunker需要OpenAI API密钥和依赖: {e}")
            return []

    def print_quick_reference_table(self):
        """打印快速查表"""
        print("\n" + "="*80)
        print("📊 LangChain 文本分割器快速查表")
        print("="*80)

        # 创建表格数据
        table_data = []
        for name, info in self.splitter_info.items():
            table_data.append({
                "分割器类型": name,
                "主要用途": info["use_case"],
                "性能": info["performance"],
                "优点": ", ".join(info["pros"]),
                "缺点": ", ".join(info["cons"])
            })

        # 转换为DataFrame并打印
        df = pd.DataFrame(table_data)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)

        print(df.to_string(index=False))
        print("="*80)

    def print_detailed_info(self, splitter_name: Optional[str] = None):
        """打印详细的分割器信息"""
        if splitter_name:
            if splitter_name in self.splitter_info:
                self._print_single_splitter_info(splitter_name, self.splitter_info[splitter_name])
            else:
                print(f"未找到分割器: {splitter_name}")
                print(f"可用的分割器: {list(self.splitter_info.keys())}")
        else:
            for name, info in self.splitter_info.items():
                self._print_single_splitter_info(name, info)
                print("\n" + "-"*60)

    def _print_single_splitter_info(self, name: str, info: Dict[str, Any]):
        """打印单个分割器的详细信息"""
        print(f"\n🔧 {name}")
        print(f"📝 描述: {info['description']}")
        print(f"🎯 用途: {info['use_case']}")
        print(f"⚡ 性能: {info['performance']}")
        print(f"✅ 优点: {', '.join(info['pros'])}")
        print(f"❌ 缺点: {', '.join(info['cons'])}")

        print("\n📋 参数说明:")
        for param, details in info["parameters"].items():
            print(f"  • {param} ({details['type']}): {details['description']}")
            if details['default'] is not None:
                print(f"    默认值: {details['default']}")

    def run_all_examples(self):
        """运行所有示例"""
        print("🚀 运行所有分割器示例...")

        for name, info in self.splitter_info.items():
            try:
                info["example"]()
                print(f"✅ {name} 示例运行成功")
            except Exception as e:
                print(f"❌ {name} 示例运行失败: {e}")

            print("\n" + "="*60)

    def get_recommended_splitter(self, use_case: str) -> str:
        """根据用例推荐分割器"""
        recommendations = {
            "通用": "RecursiveCharacterTextSplitter",
            "html": "HTMLHeaderTextSplitter",
            "markdown": "MarkdownTextSplitter",
            "python": "PythonCodeTextSplitter",
            "代码": "PythonCodeTextSplitter",
            "令牌": "TokenTextSplitter",
            "语义": "SemanticChunker",
            "快速": "CharacterTextSplitter",
            "nlp": "NLTKTextSplitter",
            "语言学": "SpacyTextSplitter"
        }

        use_case_lower = use_case.lower()
        for key, splitter in recommendations.items():
            if key in use_case_lower:
                return splitter

        return "RecursiveCharacterTextSplitter"  # 默认推荐


def create_splitter_config_template() -> str:
    """创建分割器配置模板"""
    return '''
# 分割器配置模板
def create_splitter(splitter_type="recursive", **kwargs):
    """根据类型创建分割器

    Args:
        splitter_type: 分割器类型
        **kwargs: 额外参数

    Returns:
        对应的分割器实例
    """
    defaults = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "length_function": len
    }
    defaults.update(kwargs)

    if splitter_type == "character":
        return CharacterTextSplitter(**defaults)
    elif splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(**defaults)
    elif splitter_type == "html":
        return HTMLHeaderTextSplitter(
            headers_to_split_on=defaults.get("headers_to_split_on", [
                ("h1", "Header 1"), ("h2", "Header 2")
            ])
        )
    elif splitter_type == "markdown":
        return MarkdownTextSplitter(**defaults)
    elif splitter_type == "python":
        return PythonCodeTextSplitter(**defaults)
    elif splitter_type == "token":
        return TokenTextSplitter(
            chunk_size=defaults["chunk_size"],
            chunk_overlap=defaults["chunk_overlap"],
            model_name=defaults.get("model_name", "gpt-3.5-turbo")
        )
    # 添加其他分割器...
    else:
        raise ValueError(f"不支持的分割器类型: {splitter_type}")


# 使用示例
if __name__ == "__main__":
    # 创建分割器
    splitter = create_splitter(
        splitter_type="recursive",
        chunk_size=1500,
        chunk_overlap=300
    )

    # 使用分割器
    text = "你的长文本内容..."
    chunks = splitter.split_text(text)
    print(f"分割得到 {len(chunks)} 个块")
'''


if __name__ == "__main__":
    # 创建指南实例
    guide = SplitterGuide()

    print("🎯 LangChain 文本分割器完整指南")
    print("="*50)

    # 显示快速查表
    guide.print_quick_reference_table()

    # 询问用户是否要查看详细信息
    print("\n📚 可用操作:")
    print("1. 查看所有分割器详细信息")
    print("2. 查看特定分割器信息")
    print("3. 运行所有示例")
    print("4. 获取分割器推荐")
    print("5. 生成配置模板")

    # 这里可以添加用户交互逻辑
    # 为了演示，我们直接显示一些信息

    print("\n💡 推荐分割器选择:")
    print("• 通用场景: RecursiveCharacterTextSplitter")
    print("• HTML文档: HTMLHeaderTextSplitter")
    print("• Markdown: MarkdownTextSplitter")
    print("• Python代码: PythonCodeTextSplitter")
    print("• 令牌控制: TokenTextSplitter")
    print("• 语义分割: SemanticChunker")

    # 生成配置模板
    print("\n📝 配置模板:")
    print(create_splitter_config_template())