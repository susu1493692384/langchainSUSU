#!/usr/bin/env python3
"""
PromptTemplate 快速查表
所有常用参数和用法的快速参考

复制粘贴即可使用
"""

from langchain_core.prompts import (
    PromptTemplate,           # 🔥 基础提示模板
    ChatPromptTemplate,       # 💬 聊天提示模板
    FewShotPromptTemplate,    # 🎯 少样本提示模板
    PipelinePromptTemplate,   # 🔗 管道提示模板
    MessagesPlaceholder       # 📧 消息占位符
)

# ================================
# 🔥 基础PromptTemplate (95%情况下使用)
# ================================

# 最常用 - f-string格式
basic_prompt = PromptTemplate(
    input_variables=["topic", "style"],          # 必需：变量名
    template="请用{style}风格写一篇关于{topic}的文章。",  # 必需：模板
    validate_template=True,                      # 推荐：验证模板格式
    metadata={"purpose": "writing"}              # 可选：元数据
)

# 条件判断 - Jinja2格式
conditional_prompt = PromptTemplate(
    input_variables=["score"],
    template="""
{% if score >= 90 %}
评级：优秀
{% elif score >= 80 %}
评级：良好
{% elif score >= 60 %}
评级：及格
{% else %}
评级：不及格
{% endif %}""",
    template_format="jinja2"
)

# 简洁模板 - Mustache格式
simple_prompt = PromptTemplate(
    input_variables=["name", "greeting"],
    template="{{greeting}} {{name}}！",
    template_format="mustache"
)

# 多行模板
multiline_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
基于以下上下文回答问题：

上下文：{context}

问题：{question}

回答："""
)

# ================================
# 💬 ChatPromptTemplate (对话场景)
# ================================

# 基础聊天模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}助手。"),
    ("human", "请回答：{question}"),
    ("ai", "{ai_response}"),
    ("human", "{followup}")
])

# 带历史记录的聊天模板
chat_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}助手。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 聊天历史
    ("human", "{input}")
])

# 多角色对话
multi_role_chat = ChatPromptTemplate.from_messages([
    ("system", "你正在参与一个关于{topic}的讨论。"),
    ("human", "用户问题：{user_question}"),
    ("ai", "专家回答：{expert_answer}"),
    ("human", "追问：{followup}")
])

# ================================
# 🎯 FewShotPromptTemplate (少样本学习)
# ================================

# 数学计算示例
from langchain_core.prompts.few_shot import FewShotPromptTemplate

examples = [
    {
        "question": "2 + 2 = ?",
        "answer": "4"
    },
    {
        "question": "5 * 3 = ?",
        "answer": "15"
    }
]

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,                              # 示例列表
    example_prompt=PromptTemplate(                    # 示例格式
        input_variables=["question", "answer"],
        template="问题: {question}\n回答: {answer}"
    ),
    prefix="以下是一些数学计算的例子：",             # 前缀
    suffix="现在请计算：{question}",                 # 后缀
    input_variables=["question"],                    # 输入变量
    example_separator="\n\n"                         # 示例分隔符
)

# ================================
# 🔗 PipelinePromptTemplate (组合模板)
# ================================

from langchain_core.prompts.pipeline import PipelinePromptTemplate

# 组合多个模板
base_prompt = PromptTemplate(
    input_variables=["topic"],
    template="主题：{topic}"
)

analysis_prompt = PromptTemplate(
    input_variables=["base_output"],
    template="分析：{base_output}"
)

final_prompt = PromptTemplate(
    input_variables=["analysis_result"],
    template="最终报告：{analysis_result}"
)

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=final_prompt,
    pipeline_prompts=[
        ("base", base_prompt),
        ("analysis", analysis_prompt),
    ]
)

# ================================
# 📧 MessagesPlaceholder (消息占位符)
# ================================

# 聊天历史占位符
history_placeholder = MessagesPlaceholder(variable_name="chat_history")

# 上下文消息占位符
context_placeholder = MessagesPlaceholder(variable_name="context_messages")

# 示例消息占位符
examples_placeholder = MessagesPlaceholder(variable_name="examples")

# ================================
# 🎯 常用模板格式
# ================================

def template_formats():
    """不同模板格式的示例"""

    # 1. f-string格式（默认，最常用）
    fstring_template = PromptTemplate(
        input_variables=["name", "task", "deadline"],
        template="任务：{name}，负责：{task}，截止日期：{deadline}"
    )

    # 2. Jinja2格式（支持逻辑和循环）
    jinja2_template = PromptTemplate(
        input_variables=["tasks"],
        template="""
任务清单：
{% for task in tasks %}
- {{ task.name }} ({{ task.status }})
{% endfor %}
""",
        template_format="jinja2"
    )

    # 3. Mustache格式（简洁）
    mustache_template = PromptTemplate(
        input_variables=["greeting", "name"],
        template="{{#greeting}}{{greeting}} {{/greeting}}{{name}}！",
        template_format="mustache"
    )

    return {
        "f-string": fstring_template,
        "jinja2": jinja2_template,
        "mustache": mustache_template
    }

# ================================
# 🛠️ 实用模板示例
# ================================

def practical_templates():
    """实际应用中的常用模板"""

    # 1. 代码生成模板
    code_gen_template = PromptTemplate(
        input_variables=["language", "functionality", "requirements"],
        template="""
请用{language}编写一个{functionality}函数。

要求：
{requirements}

请提供完整的代码和注释：
"""
    )

    # 2. 文档分析模板
    doc_analysis_template = PromptTemplate(
        input_variables=["document", "analysis_type"],
        template="""
请对以下文档进行{analysis_type}分析：

文档内容：
{document}

分析结果：
"""
    )

    # 3. 翻译模板
    translation_template = PromptTemplate(
        input_variables=["source_text", "source_lang", "target_lang"],
        template="""
请将以下{source_lang}文本翻译成{target_lang}：

原文：
{source_text}

译文：
"""
    )

    # 4. 数据分析模板
    data_analysis_template = PromptTemplate(
        input_variables=["data_description", "analysis_goal"],
        template="""
数据描述：{data_description}

分析目标：{analysis_goal}

请提供详细的数据分析报告：
"""
    )

    return {
        "code_generation": code_gen_template,
        "document_analysis": doc_analysis_template,
        "translation": translation_template,
        "data_analysis": data_analysis_template
    }

# ================================
# ✅ 最佳实践模板
# ================================

def best_practice_templates():
    """最佳实践模板示例"""

    # 1. 带验证的模板
    validated_template = PromptTemplate(
        input_variables=["user_input"],
        template="用户说：{user_input}",
        validate_template=True,  # 启用模板验证
        input_types={"user_input": "str"}  # 定义输入类型
    )

    # 2. 部分填充模板
    partial_template = PromptTemplate(
        input_variables=["topic", "style", "length"],
        template="请用{style}风格写一篇{length}字的关于{topic}的文章。"
    ).partial(style="专业")  # 部分填充style变量

    # 3. 元数据模板
    metadata_template = PromptTemplate(
        input_variables=["question"],
        template="问题：{question}",
        metadata={
            "version": "1.0",
            "purpose": "question_answering",
            "author": "AI助手"
        }
    )

    # 4. 错误处理模板
    def safe_format_template(template, data):
        """安全的模板格式化"""
        try:
            # 检查必需变量
            missing = set(template.input_variables) - set(data.keys())
            if missing:
                raise ValueError(f"缺少变量: {missing}")

            return template.format(**data)
        except Exception as e:
            print(f"模板格式化错误: {e}")
            return None

    return {
        "validated": validated_template,
        "partial": partial_template,
        "metadata": metadata_template,
        "safe_formatter": safe_format_template
    }

# ================================
# 📋 参数速查表
# ================================

def parameter_reference():
    """
    PromptTemplate 参数速查表

    🔥 核心参数（必须）:
    - input_variables: list     # 输入变量名列表
    - template: str            # 模板字符串

    ⚙️ 格式参数（常用）:
    - template_format: str     # "f-string"(默认), "jinja2", "mustache"
    - validate_template: bool  # 是否验证模板格式（默认True）

    🏷️ 元数据参数（可选）:
    - metadata: dict          # 额外元数据
    - input_types: dict       # 输入变量类型定义

    🎯 使用场景选择:
    • 简单插值 → f-string格式
    • 需要循环/条件 → Jinja2格式
    • 前端模板 → Mustache格式
    • 对话场景 → ChatPromptTemplate
    • 少样本学习 → FewShotPromptTemplate
    • 复杂组合 → PipelinePromptTemplate

    ✅ 最佳实践:
    1. 总是启用模板验证
    2. 使用明确的变量命名
    3. 根据需要选择合适的模板格式
    4. 实现错误处理
    5. 使用部分模板减少重复
    """
    pass

# ================================
# 🚀 快速使用示例
# ================================

def quick_examples():
    """快速使用示例"""

    # 示例1：基础使用
    def basic_usage():
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="请介绍一下{topic}"
        )
        formatted = prompt.format(topic="人工智能")
        # result = llm.invoke(formatted)
        return formatted

    # 示例2：条件生成
    def conditional_usage():
        prompt = PromptTemplate(
            input_variables=["score"],
            template="""
{% if score >= 90 %}
优秀
{% else %}
需要改进
{% endif %}""",
            template_format="jinja2"
        )
        formatted = prompt.format(score=85)
        return formatted

    # 示例3：聊天对话
    def chat_usage():
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是{role}助手"),
            ("human", "{question}")
        ])
        formatted = prompt.format_messages(
            role="编程",
            question="什么是Python？"
        )
        return formatted

    # 示例4：少样本学习
    def few_shot_usage():
        examples = [
            {"input": "猫", "output": "meow"},
            {"input": "狗", "output": "woof"}
        ]

        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=PromptTemplate(
                input_variables=["input", "output"],
                template="输入: {input}\n输出: {output}"
            ),
            prefix="动物声音示例：",
            suffix="输入: {new_input}\n输出：",
            input_variables=["new_input"]
        )

        formatted = prompt.format(new_input="牛")
        return formatted

    return {
        "basic": basic_usage,
        "conditional": conditional_usage,
        "chat": chat_usage,
        "few_shot": few_shot_usage
    }

# ================================
# 📚 完整使用指南
# ================================

def usage_guide():
    """
    PromptTemplate 使用指南

    🔥 1. 基础使用（推荐90%情况）：
    ```python
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate(
        input_variables=["topic"],
        template="请介绍{topic}"
    )
    formatted = prompt.format(topic="AI")
    ```

    💬 2. 对话场景：
    ```python
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是{role}助手"),
        ("human", "{question}")
    ])
    ```

    🎯 3. 少样本学习：
    ```python
    from langchain_core.prompts.few_shot import FewShotPromptTemplate

    few_shot = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_format,
        suffix="现在回答：{question}"
    )
    ```

    🔧 4. 复杂逻辑（使用Jinja2）：
    ```python
    prompt = PromptTemplate(
        input_variables=["items"],
        template="""
    列表：
    {% for item in items %}
    - {{ item }}
    {% endfor %}
    """,
        template_format="jinja2"
    )
    ```

    ✅ 5. 安全使用：
    ```python
    def safe_format(prompt, data):
        try:
            return prompt.format(**data)
        except Exception as e:
            print(f"错误: {e}")
            return None
    ```

    🎯 选择建议：
    • 简单文本插值 → PromptTemplate + f-string
    • 对话交互 → ChatPromptTemplate
    • 需要示例 → FewShotPromptTemplate
    • 复杂逻辑 → PromptTemplate + Jinja2
    • 模板组合 → PipelinePromptTemplate
    """
    pass

if __name__ == "__main__":
    print("📋 PromptTemplate 快速查表")
    print("=" * 50)
    print()

    print("🔥 最常用模板:")
    print("basic_prompt = PromptTemplate(")
    print("    input_variables=['topic', 'style'],")
    print("    template='请用{style}风格写{topic}'")
    print(")")
    print()

    print("💬 对话模板:")
    print("chat_prompt = ChatPromptTemplate.from_messages([")
    print("    ('system', '你是{role}助手'),")
    print("    ('human', '{question}')")
    print("])")
    print()

    print("🎯 少样本模板:")
    print("few_shot = FewShotPromptTemplate(")
    print("    examples=examples,")
    print("    example_prompt=example_format,")
    print("    suffix='回答：{question}'")
    print(")")
    print()

    print("📊 模板格式对比:")
    formats = {
        "f-string": "'Hello {name}' (默认，推荐）",
        "Jinja2": "'Hello {{name}}' (支持循环、条件）",
        "Mustache": "'Hello {{name}}' (简洁）"
    }
    for format_type, description in formats.items():
        print(f"  • {format_type}: {description}")
    print()

    print("✅ 使用示例:")
    examples = quick_examples()
    for name, example_func in examples.items():
        print(f"  • {name}: {example_func.__doc__}")
    print()

    print("📖 完整参考: prompttemplate_complete_reference.py")
    print("📚 文档: https://python.langchain.com/docs/modules/prompts/")