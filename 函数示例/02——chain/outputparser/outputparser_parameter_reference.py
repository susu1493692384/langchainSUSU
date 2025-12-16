#!/usr/bin/env python3
"""
LangChain 输出解析器 (Output Parser) 参数参考手册
提供各种输出解析器的详细参数说明和使用示例
"""

import json
import re
from typing import List, Dict, Any, Optional, Type, Union
from datetime import datetime
from langchain_core.output_parsers import (
    BaseOutputParser,
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    OutputFixingParser,
    RetryOutputParser,
    StructuredOutputParser,
    XMLOutputParser
)
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, validator
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

# ============================================================================
# 1. BaseOutputParser - 基础输出解析器
# ============================================================================

class CustomBaseParser(BaseOutputParser[str]):
    """
    自定义基础解析器示例
    继承BaseOutputParser需要实现parse()方法
    """

    def parse(self, text: str) -> str:
        """必须实现的解析方法"""
        return text.strip().upper()

    @property
    def _type(self) -> str:
        """必须实现的类型标识"""
        return "custom_upper_case"

# 基础解析器参数说明
base_parser_params = {
    "parse": {
        "类型": "方法",
        "参数": "text: str",
        "返回值": "Any",
        "说明": "解析输入文本的核心方法，必须实现",
        "示例": "def parse(self, text: str) -> str: return text.strip()"
    },
    "_type": {
        "类型": "属性",
        "参数": "无",
        "返回值": "str",
        "说明": "返回解析器的类型标识，必须实现",
        "示例": "@property def _type(self) -> str: return 'my_parser'"
    },
    "get_format_instructions": {
        "类型": "方法",
        "参数": "无",
        "返回值": "str",
        "说明": "获取输出格式说明，可选实现",
        "示例": "def get_format_instructions(self) -> str: return '请输出JSON格式'"
    }
}

# ============================================================================
# 2. StrOutputParser - 字符串输出解析器
# ============================================================================

def demonstrate_str_output_parser():
    """StrOutputParser参数和使用示例"""
    print("=== StrOutputParser 参数参考 ===\n")

    # 基础用法
    str_parser = StrOutputParser()

    # StrOutputParser 支持的参数
    str_parser_params = {
        "初始化参数": {
            "参数": "无特殊参数",
            "说明": "StrOutputParser不接受任何初始化参数"
        },
        "parse方法": {
            "输入": "AIMessage或字符串",
            "输出": "str",
            "说明": "提取消息的content内容并转换为字符串"
        }
    }

    # 示例代码
    print("参数说明:")
    for category, params in str_parser_params.items():
        print(f"  {category}:")
        for param, desc in params.items():
            print(f"    {param}: {desc}")

    print(f"\n基础使用示例:")
    print(f"  parser = StrOutputParser()")
    print(f"  result = parser.parse('这是一个测试文本')")
    print(f"  # 输出: '这是一个测试文本'")

    return str_parser

# ============================================================================
# 3. JsonOutputParser - JSON输出解析器
# ============================================================================

def demonstrate_json_output_parser():
    """JsonOutputParser参数和使用示例"""
    print("\n=== JsonOutputParser 参数参考 ===\n")

    # JsonOutputParser 支持的参数
    json_parser_params = {
        "初始化参数": {
            "pydantic_model": "可选",
            "说明": "如果提供，将使用Pydantic模型验证和解析JSON"
        },
        "parse方法": {
            "输入": "字符串",
            "输出": "Dict[str, Any] 或 Pydantic模型实例",
            "说明": "解析JSON字符串为Python对象"
        },
        "get_format_instructions": {
            "返回值": "str",
            "说明": "返回JSON格式说明，可用于提示模板"
        }
    }

    # 示例代码
    print("参数说明:")
    for category, params in json_parser_params.items():
        print(f"  {category}:")
        for param, desc in params.items():
            print(f"    {param}: {desc}")

    print(f"\n使用示例:")
    json_parser = JsonOutputParser()

    # 获取格式说明
    format_instructions = json_parser.get_format_instructions()
    print(f"  格式说明: {format_instructions}")

    # 解析示例
    test_json = '{"name": "张三", "age": 25, "skills": ["Python", "LangChain"]}'
    try:
        result = json_parser.parse(test_json)
        print(f"  解析结果: {result}")
        print(f"  数据类型: {type(result)}")
    except Exception as e:
        print(f"  解析错误: {e}")

    return json_parser

# ============================================================================
# 4. PydanticOutputParser - 数据模型输出解析器
# ============================================================================

# 定义Pydantic模型
class PersonInfo(BaseModel):
    """人员信息数据模型"""
    name: str = Field(description="人员姓名")
    age: int = Field(description="年龄", ge=0, le=150)
    email: Optional[str] = Field(description="邮箱地址", default=None)
    skills: List[str] = Field(description="技能列表")
    is_active: bool = Field(description="是否活跃", default=True)

    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('邮箱格式不正确')
        return v

def demonstrate_pydantic_output_parser():
    """PydanticOutputParser参数和使用示例"""
    print("\n=== PydanticOutputParser 参数参考 ===\n")

    # PydanticOutputParser 支持的参数
    pydantic_parser_params = {
        "初始化参数": {
            "pydantic_object": "必需",
            "类型": "Type[BaseModel]",
            "说明": "要解析到的Pydantic模型类"
        },
        "parse方法": {
            "输入": "字符串",
            "输出": "BaseModel实例",
            "说明": "解析文本为Pydantic模型实例，包含数据验证"
        },
        "get_format_instructions": {
            "返回值": "str",
            "说明": "返回基于Pydantic模型的JSON格式说明"
        },
        "get_schema": {
            "返回值": "dict",
            "说明": "返回Pydantic模型的JSON Schema"
        }
    }

    print("参数说明:")
    for category, params in pydantic_parser_params.items():
        print(f"  {category}:")
        for param, desc in params.items():
            print(f"    {param}: {desc}")

    # 创建解析器
    pydantic_parser = PydanticOutputParser(pydantic_object=PersonInfo)

    print(f"\n使用示例:")
    print(f"  模型Schema: {pydantic_parser.get_schema()}")

    format_instructions = pydantic_parser.get_format_instructions()
    print(f"  格式说明: {format_instructions[:100]}...")

    # 测试解析
    test_data = '''
    {
        "name": "李四",
        "age": 30,
        "email": "lisi@example.com",
        "skills": ["Java", "Spring", "MySQL"],
        "is_active": true
    }
    '''

    try:
        result = pydantic_parser.parse(test_data)
        print(f"  解析结果:")
        print(f"    姓名: {result.name}")
        print(f"    年龄: {result.age}")
        print(f"    邮箱: {result.email}")
        print(f"    技能: {result.skills}")
        print(f"    活跃: {result.is_active}")
        print(f"    数据类型: {type(result)}")
    except Exception as e:
        print(f"  解析错误: {e}")

    return pydantic_parser

# ============================================================================
# 5. CommaSeparatedListOutputParser - 逗号分隔列表解析器
# ============================================================================

def demonstrate_comma_separated_parser():
    """CommaSeparatedListOutputParser参数和使用示例"""
    print("\n=== CommaSeparatedListOutputParser 参数参考 ===\n")

    # 解析器参数
    list_parser_params = {
        "初始化参数": {
            "参数": "无特殊参数",
            "说明": "不接受初始化参数"
        },
        "parse方法": {
            "输入": "字符串",
            "输出": "List[str]",
            "说明": "将逗号分隔的文本转换为字符串列表"
        },
        "get_format_instructions": {
            "返回值": "str",
            "说明": "返回逗号分隔格式的说明"
        }
    }

    print("参数说明:")
    for category, params in list_parser_params.items():
        print(f"  {category}:")
        for param, desc in params.items():
            print(f"    {param}: {desc}")

    # 使用示例
    list_parser = CommaSeparatedListOutputParser()

    print(f"\n使用示例:")
    test_lists = [
        "Python, JavaScript, Java",
        "机器学习, 深度学习, 自然语言处理",
        "北京, 上海, 广州, 深圳",
        "单项目",  # 单个元素
        "  包含空格  ,   项目2  ,  项目3  "  # 包含多余空格
    ]

    for i, test_list in enumerate(test_lists, 1):
        try:
            result = list_parser.parse(test_list)
            print(f"  测试{i}: '{test_list}'")
            print(f"    结果: {result}")
        except Exception as e:
            print(f"  测试{i}: 解析错误 - {e}")

    return list_parser

# ============================================================================
# 6. StructuredOutputParser - 结构化输出解析器
# ============================================================================

def demonstrate_structured_output_parser():
    """StructuredOutputParser参数和使用示例"""
    print("\n=== StructuredOutputParser 参数参考 ===\n")

    # 定义响应模式
    response_schemas = [
        {
            "name": "answer",
            "description": "回答用户的问题",
            "type": "string"
        },
        {
            "name": "confidence",
            "description": "回答的置信度 (0-1)",
            "type": "number"
        },
        {
            "name": "sources",
            "description": "信息来源列表",
            "type": "array"
        }
    ]

    # 解析器参数
    structured_parser_params = {
        "初始化参数": {
            "response_schemas": "必需",
            "类型": "List[Dict[str, str]]",
            "说明": "定义输出结构的响应模式列表"
        },
        "parse方法": {
            "输入": "字符串",
            "输出": "Dict[str, Any]",
            "说明": "根据响应模式解析结构化输出"
        },
        "get_format_instructions": {
            "返回值": "str",
            "说明": "返回基于响应模式的格式说明"
        }
    }

    print("参数说明:")
    for category, params in structured_parser_params.items():
        print(f"  {category}:")
        for param, desc in params.items():
            print(f"    {param}: {desc}")

    # 创建解析器
    structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    print(f"\n使用示例:")
    format_instructions = structured_parser.get_format_instructions()
    print(f"  格式说明长度: {len(format_instructions)} 字符")

    # 测试解析
    test_structured = '''
    ```json
    {
        "answer": "LangChain是一个用于构建基于大语言模型应用程序的框架",
        "confidence": 0.95,
        "sources": ["官方文档", "GitHub仓库", "技术博客"]
    }
    ```
    '''

    try:
        result = structured_parser.parse(test_structured)
        print(f"  解析结果:")
        for key, value in result.items():
            print(f"    {key}: {value} ({type(value).__name__})")
    except Exception as e:
        print(f"  解析错误: {e}")

    return structured_parser

# ============================================================================
# 7. 错误处理和修复解析器
# ============================================================================

class RetryWithErrorOutputParser(BaseOutputParser[str]):
    """自定义重试错误处理解析器示例"""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.base_parser = StrOutputParser()

    def parse(self, text: str) -> str:
        """带重试的解析方法"""
        for attempt in range(self.max_retries):
            try:
                result = self.base_parser.parse(text)
                if len(result.strip()) > 0:
                    return result
                else:
                    raise ValueError("输出为空")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"解析失败 (重试{self.max_retries}次): {str(e)}"
                continue

        return "未知错误"

    @property
    def _type(self) -> str:
        return "retry_with_error"

def demonstrate_error_handling_parsers():
    """错误处理解析器示例"""
    print("\n=== 错误处理解析器参数参考 ===\n")

    # OutputFixingParser 参数
    fixing_parser_params = {
        "初始化参数": {
            "parser": "必需",
            "类型": "BaseOutputParser",
            "说明": "基础解析器，用于解析成功的情况"
        },
        "retry_chain": "可选",
        "type": "LLMChain",
        "说明": "用于修复输出的LLM链"
    }

    # RetryOutputParser 参数
    retry_parser_params = {
        "初始化参数": {
            "parser": "必需",
            "类型": "BaseOutputParser",
            "说明": "基础解析器"
        },
        "max_retries": "可选",
        "类型": "int",
            "默认值": "3",
            "说明": "最大重试次数"
        }
    }

    print("OutputFixingParser参数:")
    for param, desc in fixing_parser_params.items():
        print(f"  {param}: {desc}")

    print("\nRetryOutputParser参数:")
    for param, desc in retry_parser_params.items():
        print(f"  {param}: {desc}")

    # 自定义错误处理示例
    print(f"\n自定义错误处理示例:")
    error_parser = RetryWithErrorOutputParser(max_retries=2)

    test_cases = [
        "正常内容",
        "",
        "   ",
        "还有一些内容"
    ]

    for i, test_case in enumerate(test_cases, 1):
        result = error_parser.parse(test_case)
        print(f"  测试{i}: '{test_case}' -> '{result}'")

# ============================================================================
# 8. 性能优化和最佳实践
# ============================================================================

def performance_optimization_tips():
    """性能优化和最佳实践"""
    print("\n=== 输出解析器性能优化指南 ===\n")

    optimization_tips = {
        "选择合适的解析器": {
            "简单文本": "使用StrOutputParser",
            "结构化数据": "使用JsonOutputParser或PydanticOutputParser",
            "列表数据": "使用CommaSeparatedListOutputParser"
        },
        "错误处理": {
            "预期错误": "使用OutputFixingParser自动修复",
            "重试机制": "使用RetryOutputParser处理临时错误",
            "自定义验证": "在Pydantic模型中添加验证规则"
        },
        "性能考虑": {
            "缓存解析器": "重用解析器实例而不是重复创建",
            "批量处理": "使用批处理减少LLM调用次数",
            "异步处理": "使用异步方法提高并发性能"
        },
        "提示词优化": {
            "明确格式要求": "在提示词中明确说明输出格式",
            "提供示例": "给出格式示例帮助理解",
            "使用格式说明": "利用get_format_instructions()生成说明"
        }
    }

    for category, tips in optimization_tips.items():
        print(f"{category}:")
        for tip, detail in tips.items():
            print(f"  • {tip}: {detail}")

# ============================================================================
# 9. 实际应用示例
# ============================================================================

def real_world_examples():
    """实际应用示例"""
    print("\n=== 实际应用示例 ===\n")

    # 示例1: 用户评论分析
    class CommentAnalysis(BaseModel):
        sentiment: str = Field(description="情感倾向 (正面/负面/中性)")
        topics: List[str] = Field(description="讨论话题")
        confidence: float = Field(description="分析置信度", ge=0, le=1)

    comment_parser = PydanticOutputParser(pydantic_object=CommentAnalysis)

    print("示例1: 用户评论分析")
    print(f"  Schema: {comment_parser.get_schema()}")

    # 示例2: 产品信息提取
    product_parser = StructuredOutputParser.from_response_schemas([
        {"name": "product_name", "description": "产品名称", "type": "string"},
        {"name": "price", "description": "价格", "type": "number"},
        {"name": "features", "description": "产品特性", "type": "array"}
    ])

    print("\n示例2: 产品信息提取")
    print(f"  格式说明: {len(product_parser.get_format_instructions())} 字符")

    # 示例3: 任务列表解析
    task_parser = CommaSeparatedListOutputParser()

    print("\n示例3: 任务列表解析")
    sample_tasks = "完成报告, 发送邮件, 准备会议, 整理文档"
    parsed_tasks = task_parser.parse(sample_tasks)
    print(f"  输入: '{sample_tasks}'")
    print(f"  输出: {parsed_tasks}")

# ============================================================================
# 主函数 - 运行所有示例
# ============================================================================

def main():
    """主函数 - 演示所有输出解析器参数"""
    print("🔧 LangChain 输出解析器 (Output Parser) 参数参考手册")
    print("=" * 60)

    # 演示各种解析器
    demonstrate_str_output_parser()
    demonstrate_json_output_parser()
    demonstrate_pydantic_output_parser()
    demonstrate_comma_separated_parser()
    demonstrate_structured_output_parser()
    demonstrate_error_handling_parsers()

    # 最佳实践和示例
    performance_optimization_tips()
    real_world_examples()

    print("\n" + "=" * 60)
    print("📚 参数参考手册完成！")
    print("\n🔗 相关资源:")
    print("  • LangChain官方文档: https://python.langchain.com/")
    print("  • 输出解析器API: https://api.python.langchain.com/")
    print("  • Pydantic文档: https://pydantic-docs.helpmanual.io/")

if __name__ == "__main__":
    main()