#!/usr/bin/env python3
"""
PromptTemplate 完整参数参考指南
LangChain中PromptTemplate的所有参数、用法和最佳实践

作者: Claude
版本: 1.0
更新时间: 2025-11-28
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,           # 基础提示模板
    ChatPromptTemplate,       # 聊天提示模板
    FewShotPromptTemplate,    # 少样本提示模板
    PipelinePromptTemplate,   # 管道提示模板
    MessagesPlaceholder       # 消息占位符
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ChatMessage
)
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

# ================================
# 1. PromptTemplate 基础参数
# ================================

def basic_prompt_template_parameters():
    """展示PromptTemplate的所有基础参数"""
    print("📋 === PromptTemplate 基础参数参考 ===\n")

    # === 基础参数 ===
    template1 = PromptTemplate(
        # 核心参数
        input_variables=["topic", "style"],          # 必需：输入变量列表
        template="请用{style}的风格写一篇关于{topic}的文章。",  # 必需：模板字符串

        # 格式参数
        template_format="f-string",                 # 模板格式: "f-string"(默认), "jinja2", "mustache"
        validate_template=True,                      # 是否验证模板格式

        # 元数据
        metadata={"purpose": "article_writing", "version": "1.0"},  # 额外元数据

        # 示例数据（用于文档和测试）
        input_types={"topic": "str", "style": "str"}  # 输入变量类型
    )

    # === Jinja2模板示例 ===
    template2 = PromptTemplate(
        input_variables=["name", "items"],
        template="""
用户: {{ name }}
购物清单:
{% for item in items %}
- {{ item }}
{% endfor %}
""",
        template_format="jinja2"
    )

    # === Mustache模板示例 ===
    template3 = PromptTemplate(
        input_variables=["greeting", "name"],
        template="{{greeting}} {{name}}！欢迎来到我们的平台。",
        template_format="mustache"
    )

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 测试基础模板
    test_data1 = {"topic": "人工智能", "style": "科普"}
    print("📝 基础f-string模板:")
    print(f"模板: {template1.template}")
    print(f"输入变量: {template1.input_variables}")
    print(f"测试数据: {test_data1}")

    try:
        formatted = template1.format(**test_data1)
        print(f"格式化结果: {formatted}")
        response = llm.invoke(formatted)
        print(f"AI回答: {response.content}\n")
    except Exception as e:
        print(f"格式化错误: {e}\n")

    # 测试Jinja2模板
    test_data2 = {"name": "张三", "items": ["苹果", "香蕉", "橙子"]}
    print("🔧 Jinja2模板:")
    print(f"模板: {template2.template}")
    print(f"测试数据: {test_data2}")

    try:
        formatted = template2.format(**test_data2)
        print(f"格式化结果: {formatted}")
        response = llm.invoke(formatted)
        print(f"AI回答: {response.content}\n")
    except Exception as e:
        print(f"格式化错误: {e}\n")

    # 测试Mustache模板
    test_data3 = {"greeting": "你好", "name": "李四"}
    print("🎯 Mustache模板:")
    print(f"模板: {template3.template}")
    print(f"测试数据: {test_data3}")

    try:
        formatted = template3.format(**test_data3)
        print(f"格式化结果: {formatted}")
        response = llm.invoke(formatted)
        print(f"AI回答: {response.content}\n")
    except Exception as e:
        print(f"格式化错误: {e}\n")

def advanced_prompt_template_features():
    """展示PromptTemplate的高级功能"""
    print("🚀 === PromptTemplate 高级功能 ===\n")

    # === 验证模板 ===
    print("✅ 模板验证示例:")

    # 正确的模板
    valid_template = PromptTemplate(
        input_variables=["topic", "length"],
        template="写一篇关于{topic}的{length}字文章。",
        validate_template=True  # 启用验证
    )

    try:
        formatted = valid_template.format(topic="Python", length="100")
        print(f"验证通过: {formatted}")
    except Exception as e:
        print(f"验证失败: {e}")

    # === 部分变量模板 ===
    print("\n🔧 部分变量模板:")

    partial_template = PromptTemplate(
        input_variables=["topic", "style"],
        template="请用{style}风格写一篇关于{topic}的技术文章，大约500字。"
    )

    # 部分填充
    partial_filled = partial_template.partial(style="专业")
    print(f"部分填充后剩余变量: {partial_filled.input_variables}")

    try:
        final_format = partial_filled.format(topic="机器学习")
        print(f"最终格式化: {final_format}")
    except Exception as e:
        print(f"格式化错误: {e}")

    # === 组合模板 ===
    print("\n🔗 组合模板示例:")

    intro_template = PromptTemplate(
        input_variables=["subject"],
        template="关于{subject}的介绍："
    )

    content_template = PromptTemplate(
        input_variables=["content"],
        template="详细内容：{content}"
    )

    conclusion_template = PromptTemplate(
        input_variables=["summary"],
        template="总结：{summary}"
    )

    # 组合多个模板
    combined = PromptTemplate(
        input_variables=["subject", "content", "summary"],
        template="""
{intro}

{main_content}

{conclusion}
""".format(
            intro=intro_template.format(subject="{subject}"),
            main_content=content_template.format(content="{content}"),
            conclusion=conclusion_template.format(summary="{summary}")
        )
    )

    test_data = {
        "subject": "人工智能",
        "content": "人工智能是计算机科学的一个分支...",
        "summary": "AI正在改变我们的生活方式。"
    }

    try:
        formatted = combined.format(**test_data)
        print("组合模板结果:")
        print(formatted)
    except Exception as e:
        print(f"组合错误: {e}")

# ================================
# 2. ChatPromptTemplate 完整参数
# ================================

def chat_prompt_template_parameters():
    """展示ChatPromptTemplate的完整参数"""
    print("💬 === ChatPromptTemplate 参数参考 ===\n")

    # === 基础消息格式 ===
    chat_prompt1 = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{role}助手。"),
        ("human", "请回答：{question}"),
        ("ai", "我理解了，让我来回答这个问题。"),
        ("human", "{followup_question}")
    ])

    # === 使用消息对象 ===
    chat_prompt2 = ChatPromptTemplate(
        input_variables=["role", "question"],
        messages=[
            SystemMessage(content="你是一个专业的{role}助手。"),
            HumanMessage(content="请回答：{question}")
        ],
        validate_template=True
    )

    # === 复杂聊天模板 ===
    chat_prompt3 = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}专家，专门处理{domain}相关的问题。"),
        MessagesPlaceholder(variable_name="chat_history"),  # 聊天历史占位符
        ("human", "{input}"),
        ("ai", "{ai_response}"),
        ("human", "{followup}")
    ])

    # === 使用占位符 ===
    chat_prompt4 = ChatPromptTemplate.from_messages([
        ("system", "系统角色：{role}"),
        MessagesPlaceholder(variable_name="context_messages"),  # 上下文消息
        MessagesPlaceholder(variable_name="examples"),         # 示例消息
        ("human", "问题：{question}"),
        ("ai", "{answer}")
    ])

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 测试基础聊天模板
    test_data1 = {
        "role": "Python编程",
        "question": "什么是装饰器？",
        "followup_question": "能给我一个例子吗？"
    }

    print("📝 基础聊天模板:")
    print(f"输入数据: {test_data1}")

    try:
        formatted_messages = chat_prompt1.format_messages(**test_data1)
        print("格式化的消息:")
        for i, msg in enumerate(formatted_messages):
            print(f"  {i+1}. {msg.type}: {msg.content}")

        response = llm.invoke(formatted_messages)
        print(f"AI回答: {response.content}\n")
    except Exception as e:
        print(f"格式化错误: {e}\n")

    # 测试带聊天历史的模板
    test_data2 = {
        "role": "AI助手",
        "domain": "机器学习",
        "chat_history": [
            HumanMessage(content="什么是监督学习？"),
            AIMessage(content="监督学习是使用标记数据训练模型的机器学习方法。")
        ],
        "input": "监督学习有哪些优点？",
        "ai_response": "监督学习的主要优点包括：1. 准确性高 2. 可解释性强 3. 成熟的技术",
        "followup": "能推荐一些监督学习的算法吗？"
    }

    print("🔧 带聊天历史的模板:")
    try:
        formatted_messages = chat_prompt3.format_messages(**test_data2)
        print("格式化的消息:")
        for i, msg in enumerate(formatted_messages):
            content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"  {i+1}. {msg.type}: {content_preview}")

        response = llm.invoke(formatted_messages)
        print(f"AI回答: {response.content[:100]}...\n")
    except Exception as e:
        print(f"格式化错误: {e}\n")

def few_shot_prompt_template_example():
    """展示FewShotPromptTemplate的使用"""
    print("🎯 === FewShotPromptTemplate 示例 ===\n")

    from langchain_core.prompts.few_shot import FewShotPromptTemplate
    from langchain_core.prompts.prompt import PromptTemplate

    # 定义示例
    examples = [
        {
            "question": "2 + 2 = ?",
            "answer": "4"
        },
        {
            "question": "5 * 3 = ?",
            "answer": "15"
        },
        {
            "question": "10 - 7 = ?",
            "answer": "3"
        }
    ]

    # 定义示例模板
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="问题: {question}\n回答: {answer}"
    )

    # 定义少样本提示模板
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,                              # 示例列表
        example_prompt=example_prompt,                   # 示例模板
        prefix="以下是一些数学计算的例子：",             # 前缀
        suffix="现在请计算：{question}",                 # 后缀
        input_variables=["question"],                    # 输入变量
        example_separator="\n\n"                         # 示例分隔符
    )

    # 测试少样本提示
    test_question = "8 + 6 = ?"
    print("📝 少样本示例:")
    print(f"问题: {test_question}")

    try:
        formatted_prompt = few_shot_prompt.format(question=test_question)
        print("格式化提示:")
        print(formatted_prompt)

        # 使用LLM回答
        llm = ChatOpenAI(
            model="glm-4",
            temperature=0.1,
            openai_api_key=os.getenv("GLM_API_KEY"),
            openai_api_base=os.getenv("GLM_BASE_URL")
        )

        response = llm.invoke(formatted_prompt)
        print(f"AI回答: {response.content}\n")
    except Exception as e:
        print(f"处理错误: {e}\n")

def pipeline_prompt_template_example():
    """展示PipelinePromptTemplate的使用"""
    print("🔗 === PipelinePromptTemplate 示例 ===\n")

    from langchain_core.prompts.pipeline import PipelinePromptTemplate

    # 定义基础提示
    base_prompt = PromptTemplate(
        input_variables=["topic"],
        template="主题：{topic}"
    )

    # 定义转化提示
    transformation_prompt = PromptTemplate(
        input_variables=["base_prompt"],
        template="基于以下基础信息进行详细分析：\n{base_prompt}"
    )

    # 定义最终提示
    final_prompt = PromptTemplate(
        input_variables=["transformation_result"],
        template="最终分析报告：\n{transformation_result}\n\n请提供专业见解。"
    )

    # 创建管道提示
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=final_prompt,
        pipeline_prompts=[
            ("base", base_prompt),
            ("transformation", transformation_prompt),
        ]
    )

    print("📋 管道提示示例:")

    try:
        # 格式化管道提示
        formatted_prompt = pipeline_prompt.format(topic="机器学习")
        print("管道提示结果:")
        print(formatted_prompt)
    except Exception as e:
        print(f"管道错误: {e}\n")

# ================================
# 3. 模板格式详细说明
# ================================

def template_format_examples():
    """展示不同模板格式的详细用法"""
    print("📚 === 模板格式详细说明 ===\n")

    # === f-string格式（默认）===
    fstring_examples = [
        {
            "template": "请分析{topic}，风格为{style}，长度为{length}字。",
            "variables": ["topic", "style", "length"],
            "description": "基础字符串插值"
        },
        {
            "template": "用户：{name}，年龄：{age}，职业：{job}",
            "variables": ["name", "age", "job"],
            "description": "用户信息展示"
        },
        {
            "template": "公式：{formula}，变量：{variables}，结果：{result}",
            "variables": ["formula", "variables", "result"],
            "description": "数学公式展示"
        }
    ]

    # === Jinja2格式 ===
    jinja2_examples = [
        {
            "template": """
项目名称：{{ project_name }}
开发者：
{% for dev in developers %}
- {{ dev.name }} ({{ dev.role }})
{% endfor %}
功能列表：
{% for feature in features %}
- {{ feature }}
{% endfor %}""",
            "variables": ["project_name", "developers", "features"],
            "description": "项目文档生成"
        },
        {
            "template": """
条件判断示例：
{% if score >= 90 %}
优秀
{% elif score >= 80 %}
良好
{% elif score >= 60 %}
及格
{% else %}
不及格
{% endif %}""",
            "variables": ["score"],
            "description": "条件判断"
        },
        {
            "template": """
数据表格：
| 名称 | 价格 | 数量 |
|------|------|------|
{% for item in items %}
| {{ item.name }} | {{ item.price }} | {{ item.quantity }} |
{% endfor %}""",
            "variables": ["items"],
            "description": "Markdown表格生成"
        }
    ]

    # === Mustache格式 ===
    mustache_examples = [
        {
            "template": "你好{{#name}} {{name}}{{/name}}，欢迎来到{{company}}！",
            "variables": ["name", "company"],
            "description": "有条件的欢迎信息"
        },
        {
            "template": """
购物清单：
{{#items}}
- {{.}}
{{/items}}""",
            "variables": ["items"],
            "description": "简单列表"
        },
        {
            "template": "{{#greeting}}{{greeting}} {{/greeting}}{{name}}！",
            "variables": ["greeting", "name"],
            "description": "可选的问候语"
        }
    ]

    # 测试所有格式
    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 测试f-string
    print("🔤 f-string格式示例:")
    for i, example in enumerate(fstring_examples):
        print(f"\n{i+1}. {example['description']}")
        print(f"模板: {example['template']}")
        print(f"变量: {example['variables']}")

        try:
            # 创建模板
            template = PromptTemplate(
                input_variables=example['variables'],
                template=example['template'],
                template_format="f-string"
            )

            # 测试数据
            if example['variables'] == ["topic", "style", "length"]:
                test_data = {"topic": "AI技术", "style": "科普", "length": "200"}
            elif example['variables'] == ["name", "age", "job"]:
                test_data = {"name": "张三", "age": "30", "job": "工程师"}
            else:
                test_data = {"formula": "E=mc²", "variables": "E=能量, m=质量, c=光速", "result": "质能等价"}

            formatted = template.format(**test_data)
            print(f"格式化结果: {formatted}")
        except Exception as e:
            print(f"错误: {e}")

    # 测试Jinja2
    print("\n\n🔧 Jinja2格式示例:")
    for i, example in enumerate(jinja2_examples):
        print(f"\n{i+1}. {example['description']}")
        print(f"模板: {example['template']}")
        print(f"变量: {example['variables']}")

        try:
            template = PromptTemplate(
                input_variables=example['variables'],
                template=example['template'],
                template_format="jinja2"
            )

            # 测试数据
            if "project_name" in example['variables']:
                test_data = {
                    "project_name": "AI助手",
                    "developers": [
                        {"name": "张三", "role": "前端"},
                        {"name": "李四", "role": "后端"}
                    ],
                    "features": ["聊天功能", "代码生成", "翻译"]
                }
            elif "score" in example['variables']:
                test_data = {"score": 85}
            else:
                test_data = {
                    "items": [
                        {"name": "苹果", "price": "5元", "quantity": "10个"},
                        {"name": "香蕉", "price": "3元", "quantity": "15个"}
                    ]
                }

            formatted = template.format(**test_data)
            print(f"格式化结果:\n{formatted}")
        except Exception as e:
            print(f"错误: {e}")

    # 测试Mustache
    print("\n\n🎯 Mustache格式示例:")
    for i, example in enumerate(mustache_examples):
        print(f"\n{i+1}. {example['description']}")
        print(f"模板: {example['template']}")
        print(f"变量: {example['variables']}")

        try:
            template = PromptTemplate(
                input_variables=example['variables'],
                template=example['template'],
                template_format="mustache"
            )

            # 测试数据
            if "items" in example['variables']:
                test_data = {"items": ["苹果", "香蕉", "橙子"], "company": "超市"}
            elif "name" in example['variables'] and "items" not in example['variables']:
                test_data = {"name": "张三", "company": "科技公司"}
            else:
                test_data = {"greeting": "你好", "name": "李四", "company": "AI公司"}

            formatted = template.format(**test_data)
            print(f"格式化结果: {formatted}")
        except Exception as e:
            print(f"错误: {e}")

# ================================
# 4. 实际应用示例
# ================================

def practical_examples():
    """展示PromptTemplate的实际应用场景"""
    print("🎯 === 实际应用示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # === 1. 代码生成器 ===
    print("💻 1. 代码生成器:")

    code_gen_prompt = PromptTemplate(
        input_variables=["language", "functionality", "requirements"],
        template="""
请用{language}编写一个{functionality}函数。

要求：
{requirements}

请提供：
1. 完整的函数代码
2. 必要的注释
3. 使用示例

代码：
""",
        template_format="f-string",
        metadata={"purpose": "code_generation", "version": "1.0"}
    )

    code_test_data = {
        "language": "Python",
        "functionality": "计算斐波那契数列",
        "requirements": "- 使用递归方法\n- 包含边界检查\n- 时间复杂度优化"
    }

    try:
        formatted = code_gen_prompt.format(**code_test_data)
        response = llm.invoke(formatted)
        print(f"输入数据: {code_test_data}")
        print(f"生成代码:\n{response.content}\n")
    except Exception as e:
        print(f"代码生成错误: {e}\n")

    # === 2. 文档分析器 ===
    print("📄 2. 文档分析器:")

    doc_analysis_prompt = PromptTemplate(
        input_variables=["document_type", "content", "analysis_type"],
        template="""
请对以下{document_type}进行{analysis_type}分析。

文档内容：
{content}

请从以下方面进行分析：
1. 主要内容概述
2. 关键信息提取
3. 结构分析
4. 建议和改进

分析结果：
""",
        validate_template=True
    )

    doc_test_data = {
        "document_type": "技术文档",
        "content": "本文档介绍了微服务架构的设计原则和实施方法，包括服务拆分、API设计、数据一致性等核心概念。",
        "analysis_type": "技术和商业价值"
    }

    try:
        formatted = doc_analysis_prompt.format(**doc_test_data)
        response = llm.invoke(formatted)
        print(f"文档类型: {doc_test_data['document_type']}")
        print(f"分析结果:\n{response.content}\n")
    except Exception as e:
        print(f"文档分析错误: {e}\n")

    # === 3. 多语言翻译器 ===
    print("🌍 3. 多语言翻译器:")

    translation_prompt = PromptTemplate(
        input_variables=["source_text", "source_lang", "target_lang", "style"],
        template="""请将以下{source_lang}文本翻译成{target_lang}。

原文：
{source_text}

翻译要求：
- 风格：{style}
- 保持原文含义
- 符合目标语言习惯

译文：
""",
        input_types={
            "source_text": "str",
            "source_lang": "str",
            "target_lang": "str",
            "style": "str"
        }
    )

    translation_test_data = {
        "source_text": "人工智能正在改变世界。",
        "source_lang": "中文",
        "target_lang": "英语",
        "style": "正式"
    }

    try:
        formatted = translation_prompt.format(**translation_test_data)
        response = llm.invoke(formatted)
        print(f"翻译任务: {translation_test_data['source_lang']} → {translation_test_data['target_lang']}")
        print(f"原文: {translation_test_data['source_text']}")
        print(f"译文: {response.content}\n")
    except Exception as e:
        print(f"翻译错误: {e}\n")

    # === 4. 数据分析报告 ===
    print("📊 4. 数据分析报告:")

    # 使用Jinja2模板生成复杂报告
    data_report_prompt = PromptTemplate(
        input_variables=["dataset_name", "metrics", "insights", "recommendations"],
        template="""数据集：{{ dataset_name }}

关键指标：
{% for metric in metrics %}
- {{ metric.name }}: {{ metric.value }} ({{ metric.unit }})
{% endfor %}

数据洞察：
{% for insight in insights %}
{{ loop.index }}. {{ insight }}
{% endfor %}

建议：
{% for rec in recommendations %}
• {{ rec }}
{% endfor %}

总结：
基于以上分析，该数据集{{ summary }}""",
        template_format="jinja2"
    )

    report_test_data = {
        "dataset_name": "2024年销售数据",
        "metrics": [
            {"name": "总销售额", "value": "1,250,000", "unit": "元"},
            {"name": "增长率", "value": "15.3", "unit": "%"},
            {"name": "客户数量", "value": "3,500", "unit": "个"}
        ],
        "insights": [
            "销售呈上升趋势",
            "客户满意度较高",
            "产品多样性提升"
        ],
        "recommendations": [
            "继续加强营销推广",
            "优化产品结构",
            "提升客户服务质量"
        ],
        "summary": "表现良好，有进一步优化空间"
    }

    try:
        formatted = data_report_prompt.format(**report_test_data)
        print("数据分析报告:")
        print(formatted)

        response = llm.invoke(formatted)
        print(f"\nAI分析:\n{response.content}\n")
    except Exception as e:
        print(f"报告生成错误: {e}\n")

# ================================
# 5. 最佳实践和错误处理
# ================================

def best_practices_and_error_handling():
    """展示最佳实践和错误处理"""
    print("✅ === 最佳实践和错误处理 ===\n")

    # === 1. 模板验证 ===
    print("🔍 1. 模板验证:")

    # 正确的模板
    correct_template = PromptTemplate(
        input_variables=["topic", "style"],
        template="请用{style}风格写一篇关于{topic}的文章。",
        validate_template=True
    )

    # 错误的模板（会触发验证错误）
    try:
        wrong_template = PromptTemplate(
            input_variables=["topic"],
            template="请用{style}风格写一篇关于{topic}的文章。",  # style变量未在input_variables中
            validate_template=True
        )
    except Exception as e:
        print(f"❌ 模板验证捕获错误: {e}")

    # === 2. 类型安全 ===
    print("\n🛡️ 2. 类型安全:")

    typed_template = PromptTemplate(
        input_variables=["name", "age", "email"],
        template="姓名: {name}, 年龄: {age}, 邮箱: {email}",
        input_types={
            "name": "str",
            "age": "int",
            "email": "str"
        }
    )

    try:
        formatted = typed_template.format(
            name="张三",
            age=30,  # 数字类型
            email="zhang@example.com"
        )
        print(f"✅ 类型安全格式化: {formatted}")
    except Exception as e:
        print(f"❌ 类型错误: {e}")

    # === 3. 输入清理和验证 ===
    print("\n🧹 3. 输入清理和验证:")

    def safe_format_template(template, data):
        """安全的模板格式化函数"""
        try:
            # 检查所有必需变量是否提供
            missing_vars = set(template.input_variables) - set(data.keys())
            if missing_vars:
                raise ValueError(f"缺少必需变量: {missing_vars}")

            # 检查是否有额外变量
            extra_vars = set(data.keys()) - set(template.input_variables)
            if extra_vars:
                print(f"⚠️ 警告: 提供了额外变量: {extra_vars}")

            # 格式化模板
            return template.format(**data)
        except Exception as e:
            print(f"❌ 格式化失败: {e}")
            return None

    safe_template = PromptTemplate(
        input_variables=["user_query", "context"],
        template="基于上下文: {context}\n\n回答用户问题: {user_query}"
    )

    test_cases = [
        {"user_query": "什么是AI？", "context": "AI是人工智能的简称。"},  # 正常
        {"user_query": "什么是机器学习？"},  # 缺少context
        {"user_query": "什么是深度学习？", "context": "深度学习是机器学习的子集。", "extra": "不应该存在的变量"}  # 额外变量
    ]

    for i, test_data in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        result = safe_format_template(safe_template, test_data)
        if result:
            print(f"✅ 成功: {result[:50]}...")

    # === 4. 性能优化 ===
    print("\n⚡ 4. 性能优化:")

    import time

    # 预编译模板
    start_time = time.time()
    precompiled_template = PromptTemplate(
        input_variables=["question"],
        template="请详细回答: {question}",
        validate_template=True
    )
    precompile_time = time.time() - start_time

    # 多次使用预编译模板
    questions = [
        "什么是Python？",
        "什么是机器学习？",
        "什么是深度学习？",
        "什么是神经网络？"
    ]

    start_time = time.time()
    for question in questions:
        formatted = precompiled_template.format(question=question)
        # 这里会调用LLM，但我们只测量格式化时间
    usage_time = time.time() - start_time

    print(f"模板预编译时间: {precompile_time:.4f}秒")
    print(f"格式化{len(questions)}个问题时间: {usage_time:.4f}秒")
    print(f"平均每个问题格式化时间: {usage_time/len(questions):.4f}秒")

    # === 5. 内存管理 ===
    print("\n💾 5. 内存管理:")

    # 大模板的内存优化
    large_template_content = """
请对以下内容进行详细分析：

背景信息：{background}

技术细节：{technical_details}

市场分析：{market_analysis}

风险因素：{risk_factors}

建议方案：{recommendations}

实施计划：{implementation_plan}
""" * 5  # 模拟大模板

    large_template = PromptTemplate(
        input_variables=[
            "background", "technical_details", "market_analysis",
            "risk_factors", "recommendations", "implementation_plan"
        ],
        template=large_template_content
    )

    print(f"大模板长度: {len(large_template.template)} 字符")
    print("✅ 大模板创建成功，适合复杂的业务场景")

    # === 6. 国际化支持 ===
    print("\n🌐 6. 国际化支持:")

    i18n_templates = {
        "zh-CN": PromptTemplate(
            input_variables=["name", "topic"],
            template="你好{name}！欢迎学习{topic}。"
        ),
        "en-US": PromptTemplate(
            input_variables=["name", "topic"],
            template="Hello {name}! Welcome to learn {topic}."
        ),
        "ja-JP": PromptTemplate(
            input_variables=["name", "topic"],
            template="こんにちは{name}さん！{topic}の学習へようこそ。"
        )
    }

    for lang, template in i18n_templates.items():
        formatted = template.format(name="张三", topic="AI技术")
        print(f"{lang}: {formatted}")

if __name__ == "__main__":
    import sys
    import io

    # 设置UTF-8编码输出
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("📋 PromptTemplate 完整参数参考指南")
    print("=" * 60)
    print()

    # 检查环境变量
    if not os.getenv("GLM_API_KEY") or not os.getenv("GLM_BASE_URL"):
        print("⚠️ 警告: 未找到GLM_API_KEY或GLM_BASE_URL环境变量")
        print("请确保在.env文件中设置了正确的智谱AI API配置")
        print()

    # 运行所有示例
    examples = [
        ("基础参数参考", basic_prompt_template_parameters),
        ("高级功能", advanced_prompt_template_features),
        ("ChatPromptTemplate", chat_prompt_template_parameters),
        ("FewShotPromptTemplate", few_shot_prompt_template_example),
        ("PipelinePromptTemplate", pipeline_prompt_template_example),
        ("模板格式详细说明", template_format_examples),
        ("实际应用示例", practical_examples),
        ("最佳实践和错误处理", best_practices_and_error_handling)
    ]

    for name, func in examples:
        print(f"\n{'='*60}")
        print(f"📋 运行示例: {name}")
        print('='*60)
        print()

        try:
            func()
        except KeyboardInterrupt:
            print(f"\n⏹️ 用户中断了示例: {name}")
            break
        except Exception as e:
            print(f"❌ 示例 {name} 执行出错: {e}")

        # 询问是否继续
        print("\n" + "="*60)
        try:
            user_input = input("按Enter继续下一个示例，或输入'q'退出: ")
            if user_input.lower() == 'q':
                break
        except (EOFError, KeyboardInterrupt):
            print("\n👋 用户退出程序")
            break

    print("\n" + "="*60)
    print("✨ PromptTemplate 参考指南结束！")
    print("="*60)
    print()
    print("📚 PromptTemplate 核心知识点总结:")
    print()
    print("🔧 核心组件:")
    print("  • PromptTemplate        - 基础提示模板")
    print("  • ChatPromptTemplate   - 聊天提示模板")
    print("  • FewShotPromptTemplate - 少样本提示模板")
    print("  • PipelinePromptTemplate - 管道提示模板")
    print("  • MessagesPlaceholder  - 消息占位符")
    print()
    print("⚙️ 关键参数:")
    print("  • input_variables   - 输入变量列表（必需）")
    print("  • template          - 模板字符串（必需）")
    print("  • template_format   - 模板格式：'f-string'(默认), 'jinja2', 'mustache'")
    print("  • validate_template - 是否验证模板格式（默认True）")
    print("  • metadata          - 额外元数据")
    print("  • input_types       - 输入变量类型")
    print()
    print("🎯 模板格式特性:")
    print("  • f-string   - Python原生，简单快速")
    print("  • Jinja2     - 支持循环、条件、复杂逻辑")
    print("  • Mustache   - 简洁，适合简单模板")
    print()
    print("✅ 最佳实践:")
    print("  1. 总是启用模板验证")
    print("  2. 使用明确的变量命名")
    print("  3. 根据复杂度选择合适的模板格式")
    print("  4. 实现输入验证和错误处理")
    print("  5. 预编译模板以提高性能")
    print("  6. 使用部分模板减少重复")
    print("  7. 合理组织模板结构")
    print()
    print("📖 更多信息:")
    print("  • LangChain提示模板文档: https://python.langchain.com/docs/modules/prompts/")
    print("  • Jinja2模板文档: https://jinja.palletsprojects.com/")
    print("  • Mustache模板文档: https://mustache.github.io/")