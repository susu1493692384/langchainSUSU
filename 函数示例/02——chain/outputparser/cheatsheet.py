#!/usr/bin/env python3
"""
LangChain 输出解析器 (Output Parser) 速查手册
快速参考指南 - 包含常用解析器的快速使用方法和示例
"""

from typing import List, Dict, Any, Optional, Type
import json
from langchain_core.output_parsers import (
    BaseOutputParser,
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    StructuredOutputParser,
    OutputFixingParser,
    RetryOutputParser,
    XMLOutputParser
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os

# ============================================================================
# 🚀 快速速查表
# ============================================================================

# 1. 基础解析器选择指南
PARSER_QUICK_GUIDE = {
    "字符串输出": "StrOutputParser()",
    "JSON数据": "JsonOutputParser()",
    "结构化对象": "PydanticOutputParser(pydantic_object=MyModel)",
    "逗号分隔列表": "CommaSeparatedListOutputParser()",
    "多个字段": "StructuredOutputParser.from_response_schemas(schemas)",
    "XML格式": "XMLOutputParser()",
    "错误修复": "OutputFixingParser(parser=base_parser)",
    "重试机制": "RetryOutputParser(parser=base_parser)"
}

# 2. 常用解析器快速创建
QUICK_CREATORS = {
    "字符串": lambda: StrOutputParser(),
    "JSON": lambda: JsonOutputParser(),
    "列表": lambda: CommaSeparatedListOutputParser(),
    "结构化": lambda schemas: StructuredOutputParser.from_response_schemas(schemas)
}

# 3. 常用Pydantic模型模板
COMMON_PYDANTIC_MODELS = {
    "用户信息": {
        "name": "str",
        "age": "int",
        "email": "Optional[str]"
    },
    "分析结果": {
        "sentiment": "str",
        "confidence": "float",
        "topics": "List[str]"
    },
    "产品信息": {
        "name": "str",
        "price": "float",
        "features": "List[str]",
        "rating": "Optional[float]"
    },
    "任务清单": {
        "title": "str",
        "priority": "str",
        "status": "str",
        "deadline": "Optional[str]"
    }
}

# ============================================================================
# 📝 快速使用示例
# ============================================================================

class QuickExamples:
    """快速使用示例集合"""

    @staticmethod
    def string_parser_example():
        """字符串解析器 - 最简单直接"""
        print("🔤 字符串解析器 (StrOutputParser)")

        # 代码示例
        code = '''
# 创建解析器
parser = StrOutputParser()

# 在链中使用
chain = prompt | llm | parser
result = chain.invoke({"topic": "人工智能"})
# 输出: "人工智能是..."
        '''
        print(code)
        return code

    @staticmethod
    def json_parser_example():
        """JSON解析器 - 结构化数据"""
        print("📊 JSON解析器 (JsonOutputParser)")

        code = '''
# 创建解析器
parser = JsonOutputParser()

# 获取格式说明 (添加到提示词中)
format_instructions = parser.get_format_instructions()

# 提示模板
prompt = PromptTemplate(
    template="回答问题并以JSON格式返回: {question}\\n{format_instructions}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

# 链式调用
chain = prompt | llm | parser
result = chain.invoke({"question": "什么是机器学习?"})
# 输出: {"answer": "...", "confidence": 0.9}
        '''
        print(code)
        return code

    @staticmethod
    def pydantic_parser_example():
        """Pydantic解析器 - 类型安全"""
        print("🏗️ Pydantic解析器 (PydanticOutputParser)")

        code = '''
# 1. 定义数据模型
class UserInfo(BaseModel):
    name: str = Field(description="用户姓名")
    age: int = Field(description="年龄", ge=0, le=150)
    skills: List[str] = Field(description="技能列表")

# 2. 创建解析器
parser = PydanticOutputParser(pydantic_object=UserInfo)

# 3. 获取格式说明
format_instructions = parser.get_format_instructions()

# 4. 创建链
chain = prompt | llm | parser
result = chain.invoke({"text": "张三，25岁，精通Python和Java"})
# 输出: UserInfo(name="张三", age=25, skills=["Python", "Java"])
        '''
        print(code)
        return code

    @staticmethod
    def list_parser_example():
        """列表解析器 - 处理列表数据"""
        print("📋 列表解析器 (CommaSeparatedListOutputParser)")

        code = '''
# 创建解析器
parser = CommaSeparatedListOutputParser()

# 获取格式说明
format_instructions = parser.get_format_instructions()

# 使用
chain = prompt | llm | parser
result = chain.invoke({"topic": "编程语言"})
# 输出: ["Python", "Java", "JavaScript"]
        '''
        print(code)
        return code

    @staticmethod
    def structured_parser_example():
        """结构化解析器 - 多字段输出"""
        print("🏗️ 结构化解析器 (StructuredOutputParser)")

        code = '''
# 定义响应模式
response_schemas = [
    {"name": "summary", "description": "内容摘要", "type": "string"},
    {"name": "sentiment", "description": "情感倾向", "type": "string"},
    {"name": "confidence", "description": "置信度", "type": "number"}
]

# 创建解析器
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 使用
chain = prompt | llm | parser
result = chain.invoke({"text": "今天天气真好"})
# 输出: {"summary": "描述好天气", "sentiment": "积极", "confidence": 0.95}
        '''
        print(code)
        return code

# ============================================================================
# 🛠️ 实用工具函数
# ============================================================================

class ParserUtils:
    """解析器实用工具"""

    @staticmethod
    def quick_parser(parsers: List[str]):
        """快速创建多个解析器"""
        return [QUICK_CREATORS.get(parser)() for parser in parsers]

    @staticmethod
    def create_pydantic_model(model_name: str, fields: Dict[str, str]):
        """动态创建Pydantic模型"""
        field_definitions = {}
        for field_name, field_type in fields.items():
            field_definitions[field_name] = (eval(field_type), Field(description=field_name))

        return type(model_name, (BaseModel,), field_definitions)

    @staticmethod
    def format_parser_results(results: Dict[str, Any], parser_type: str):
        """格式化解析结果用于显示"""
        if parser_type == "json":
            return json.dumps(results, indent=2, ensure_ascii=False)
        elif parser_type == "pydantic":
            if hasattr(results, 'dict'):
                return json.dumps(results.dict(), indent=2, ensure_ascii=False)
        return str(results)

    @staticmethod
    def validate_parser_output(output: Any, expected_type: str):
        """验证解析器输出"""
        if expected_type == "json":
            try:
                json.dumps(output)
                return True
            except:
                return False
        elif expected_type == "list":
            return isinstance(output, list)
        elif expected_type == "string":
            return isinstance(output, str)
        return True

# ============================================================================
# 🔧 解析器工厂
# ============================================================================

class ParserFactory:
    """解析器工厂 - 根据需求快速创建合适的解析器"""

    @staticmethod
    def create_for_simple_text():
        """简单文本解析器"""
        return StrOutputParser()

    @staticmethod
    def create_for_structured_data(schema: Dict[str, str]):
        """结构化数据解析器"""
        if "pydantic_model" in schema:
            return PydanticOutputParser(pydantic_object=schema["pydantic_model"])
        else:
            response_schemas = [
                {"name": key, "description": desc, "type": "string"}
                for key, desc in schema.items()
            ]
            return StructuredOutputParser.from_response_schemas(response_schemas)

    @staticmethod
    def create_for_list_data():
        """列表数据解析器"""
        return CommaSeparatedListOutputParser()

    @staticmethod
    def create_for_json_data():
        """JSON数据解析器"""
        return JsonOutputParser()

    @staticmethod
    def create_with_error_handling(base_parser: BaseOutputParser):
        """带错误处理的解析器"""
        return OutputFixingParser(parser=base_parser)

    @staticmethod
    def create_with_retry(base_parser: BaseOutputParser, max_retries: int = 3):
        """带重试机制的解析器"""
        return RetryOutputParser(parser=base_parser, max_retries=max_retries)

# ============================================================================
# 📋 常见使用场景
# ============================================================================

class CommonUseCases:
    """常见使用场景的解析器配置"""

    @staticmethod
    def user_profile_extraction():
        """用户信息提取"""
        code = '''
# 场景: 从文本中提取用户信息
class UserProfile(BaseModel):
    name: str = Field(description="用户姓名")
    age: int = Field(description="年龄")
    interests: List[str] = Field(description="兴趣爱好")
    location: Optional[str] = Field(description="所在地")

parser = PydanticOutputParser(pydantic_object=UserProfile)
        '''
        print("👤 用户信息提取")
        print(code)
        return code

    @staticmethod
    def sentiment_analysis():
        """情感分析"""
        code = '''
# 场景: 文本情感分析
response_schemas = [
    {"name": "sentiment", "description": "情感倾向 (正面/负面/中性)", "type": "string"},
    {"name": "confidence", "description": "置信度 (0-1)", "type": "number"},
    {"name": "keywords", "description": "关键词列表", "type": "array"}
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
        '''
        print("😊 情感分析")
        print(code)
        return code

    @staticmethod
    def product_review():
        """产品评价提取"""
        code = '''
# 场景: 产品评价信息提取
class ProductReview(BaseModel):
    product_name: str = Field(description="产品名称")
    rating: float = Field(description="评分 (1-5)", ge=1, le=5)
    pros: List[str] = Field(description="优点")
    cons: List[str] = Field(description="缺点")
    recommendation: str = Field(description="推荐意见")

parser = PydanticOutputParser(pydantic_object=ProductReview)
        '''
        print("⭐ 产品评价")
        print(code)
        return code

    @staticmethod
    def task_management():
        """任务管理"""
        code = '''
# 场景: 任务列表解析
parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template="将以下任务分解为具体步骤，用逗号分隔: {task}",
    input_variables=["task"]
)

chain = prompt | llm | parser
result = chain.invoke({"task": "学习LangChain"})
# 输出: ["学习基础概念", "安装环境", "练习示例", "构建项目"]
        '''
        print("✅ 任务管理")
        print(code)
        return code

    @staticmethod
    def data_extraction():
        """数据提取"""
        code = '''
# 场景: 从非结构化文本提取结构化数据
response_schemas = [
    {"name": "companies", "description": "提到的公司", "type": "array"},
    {"name": "technologies", "description": "提到的技术", "type": "array"},
    {"name": "dates", "description": "提到的日期", "type": "array"}
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
        '''
        print("📊 数据提取")
        print(code)
        return code

# ============================================================================
# ⚡ 性能优化技巧
# ============================================================================

class PerformanceTips:
    """性能优化技巧"""

    @staticmethod
    def caching_parsers():
        """解析器缓存"""
        code = '''
# 缓存解析器实例
_parsers = {}

def get_parser(parser_type: str, **kwargs):
    key = f"{parser_type}_{hash(tuple(sorted(kwargs.items())))}"
    if key not in _parsers:
        if parser_type == "pydantic":
            _parsers[key] = PydanticOutputParser(**kwargs)
        elif parser_type == "json":
            _parsers[key] = JsonOutputParser(**kwargs)
        # ... 其他解析器
    return _parsers[key]
        '''
        print("💾 解析器缓存")
        print(code)
        return code

    @staticmethod
    def batch_processing():
        """批量处理"""
        code = '''
# 批量处理减少LLM调用
def batch_parse(texts: List[str], parser: BaseOutputParser):
    # 合并为单个提示
    combined_prompt = f"分别解析以下文本:\\n" + "\\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])

    # 一次性处理
    results = parser.parse(combined_prompt)

    # 分离结果
    return [result.get(str(i+1)) for i in range(len(texts))]
        '''
        print("📦 批量处理")
        print(code)
        return code

    @staticmethod
    def async_parsing():
        """异步解析"""
        code = '''
# 异步解析提高并发性能
import asyncio

async def async_parse_multiple(texts: List[str], parser: BaseOutputParser):
    tasks = []
    for text in texts:
        # 创建异步任务
        task = asyncio.create_task(async_parse(text, parser))
        tasks.append(task)

    # 并发执行
    results = await asyncio.gather(*tasks)
    return results
        '''
        print("⚡ 异步解析")
        print(code)
        return code

# ============================================================================
# 🐛 错误处理和调试
# ============================================================================

class ErrorHandling:
    """错误处理和调试技巧"""

    @staticmethod
    def safe_parsing():
        """安全解析"""
        code = '''
def safe_parse(text: str, parser: BaseOutputParser, fallback="解析失败"):
    try:
        return parser.parse(text)
    except Exception as e:
        print(f"解析错误: {e}")
        return fallback
        '''
        print("🛡️ 安全解析")
        print(code)
        return code

    @staticmethod
    def debug_parsing():
        """调试解析"""
        code = '''
def debug_parse(text: str, parser: BaseOutputParser):
    print(f"输入文本: {text}")
    print(f"解析器类型: {type(parser).__name__}")

    if hasattr(parser, 'get_format_instructions'):
        print(f"格式要求: {parser.get_format_instructions()}")

    try:
        result = parser.parse(text)
        print(f"解析结果: {result}")
        print(f"结果类型: {type(result)}")
        return result
    except Exception as e:
        print(f"解析失败: {e}")
        return None
        '''
        print("🔍 调试解析")
        print(code)
        return code

    @staticmethod
    def validation_parser():
        """验证解析器"""
        code = '''
class ValidatingParser(BaseOutputParser):
    def __init__(self, base_parser, validator_func):
        self.base_parser = base_parser
        self.validator = validator_func

    def parse(self, text: str):
        result = self.base_parser.parse(text)
        if not self.validator(result):
            raise ValueError("解析结果验证失败")
        return result
        '''
        print("✅ 验证解析器")
        print(code)
        return code

# ============================================================================
# 📚 主函数 - 完整速查手册
# ============================================================================

def main():
    """主函数 - 显示完整的速查手册"""
    print("🚀 LangChain 输出解析器速查手册")
    print("=" * 60)

    # 快速指南
    print("\n📖 快速选择指南:")
    for use_case, parser in PARSER_QUICK_GUIDE.items():
        print(f"  {use_case}: {parser}")

    # 常用示例
    print("\n💡 常用示例:")
    examples = QuickExamples()
    examples.string_parser_example()
    examples.json_parser_example()
    examples.pydantic_parser_example()
    examples.list_parser_example()
    examples.structured_parser_example()

    # 常见场景
    print("\n🎯 常见使用场景:")
    use_cases = CommonUseCases()
    use_cases.user_profile_extraction()
    use_cases.sentiment_analysis()
    use_cases.product_review()
    use_cases.task_management()
    use_cases.data_extraction()

    # 性能优化
    print("\n⚡ 性能优化:")
    tips = PerformanceTips()
    tips.caching_parsers()
    tips.batch_processing()
    tips.async_parsing()

    # 错误处理
    print("\n🛡️ 错误处理:")
    error_handling = ErrorHandling()
    error_handling.safe_parsing()
    error_handling.debug_parsing()
    error_handling.validation_parser()

    print("\n" + "=" * 60)
    print("✅ 速查手册完成！")

    print("\n📋 常用Pydantic模型模板:")
    for model_name, fields in COMMON_PYDANTIC_MODELS.items():
        print(f"  {model_name}: {fields}")

    print("\n🔗 更多资源:")
    print("  • LangChain文档: https://python.langchain.com/")
    print("  • Pydantic文档: https://pydantic-docs.helpmanual.io/")
    print("  • JSON Schema: https://json-schema.org/")

if __name__ == "__main__":
    main()