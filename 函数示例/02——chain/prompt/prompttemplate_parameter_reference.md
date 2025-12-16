# PromptTemplate 完整参数参考

## 📋 概览

LangChain中的PromptTemplate是用于创建和管理提示文本的核心组件，支持多种模板格式和高级功能。

---

## 🔥 PromptTemplate 核心参数

### 必需参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `input_variables` | `List[str]` | 模板中使用的变量名列表 | `["topic", "style"]` |
| `template` | `str` | 模板字符串，包含变量占位符 | `"请用{style}写{topic}"` |

### 格式参数

| 参数 | 类型 | 默认值 | 说明 | 选项 |
|------|------|--------|------|------|
| `template_format` | `str` | `"f-string"` | 模板格式 | `"f-string"`, `"jinja2"`, `"mustache"` |
| `validate_template` | `bool` | `True` | 是否验证模板格式 | `True`, `False` |

### 元数据参数

| 参数 | 类型 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `metadata` | `dict` | `None` | 额外的元数据 | `{"purpose": "writing"}` |
| `input_types` | `dict` | `None` | 输入变量类型定义 | `{"topic": "str", "count": "int"}` |

---

## 💬 ChatPromptTemplate 参数

### 基础创建方式

```python
# 方式1：从消息列表创建
ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}助手"),
    ("human", "请回答：{question}")
])

# 方式2：直接创建
ChatPromptTemplate(
    input_variables=["role", "question"],
    messages=[
        SystemMessage(content="你是一个{role}助手"),
        HumanMessage(content="请回答：{question}")
    ]
)
```

### 消息格式

| 消息类型 | 说明 | 示例 |
|----------|------|------|
| `("system", "...")` | 系统消息 | `("system", "你是助手")` |
| `("human", "...")` | 人类消息 | `("human", "你好")` |
| `("ai", "...")` | AI消息 | `("ai", "你好！")` |
| `MessagesPlaceholder(...)` | 消息占位符 | `MessagesPlaceholder("chat_history")` |

---

## 🎯 FewShotPromptTemplate 参数

### 核心参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `examples` | `List[dict]` | ✅ | 示例列表 |
| `example_prompt` | `PromptTemplate` | ✅ | 格式化示例的模板 |
| `suffix` | `str` | ✅ | 最终问题模板 |
| `prefix` | `str` | ❌ | 示例前缀文本 |
| `input_variables` | `List[str]` | ✅ | 最终输入变量 |
| `example_separator` | `str` | ❌ | 示例分隔符 |

---

## 🔗 PipelinePromptTemplate 参数

### 核心参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `final_prompt` | `PromptTemplate` | ✅ | 最终模板 |
| `pipeline_prompts` | `List[tuple]` | ✅ | 管道提示列表 |

### 管道提示格式

```python
pipeline_prompts=[
    ("variable_name", prompt_template),  # 变量名和对应的模板
]
```

---

## 🔧 模板格式详细说明

### 1. f-string格式（默认）

**特点：**
- Python原生格式，简单快速
- 支持基本的字符串插值
- 性能最好

**语法：**
```python
template = PromptTemplate(
    input_variables=["name", "age"],
    template="姓名：{name}，年龄：{age}"
)
```

**适用场景：**
- 简单的文本插值
- 性能敏感的应用
- 基础的提示模板

### 2. Jinja2格式

**特点：**
- 支持循环、条件判断等复杂逻辑
- 功能强大，适合复杂模板
- 学习成本稍高

**语法：**
```python
template = PromptTemplate(
    input_variables=["items", "score"],
    template="""
项目列表：
{% for item in items %}
- {{ item.name }} ({{ item.status }})
{% endfor %}

成绩：
{% if score >= 90 %}
优秀
{% elif score >= 60 %}
及格
{% else %}
不及格
{% endif %}""",
    template_format="jinja2"
)
```

**核心功能：**
- `{% for %}` - 循环
- `{% if %}` - 条件判断
- `{{ variable }}` - 变量输出
- `{% filter %}` - 过滤器

**适用场景：**
- 需要循环列表
- 条件判断逻辑
- 复杂的文档生成
- 动态内容模板

### 3. Mustache格式

**特点：**
- 极简语法，易学易用
- 跨语言支持
- 适合简单的模板需求

**语法：**
```python
template = PromptTemplate(
    input_variables=["name", "items"],
    template="""
你好{{#name}} {{name}}{{/name}}！

购物清单：
{{#items}}
- {{.}}
{{/items}}""",
    template_format="mustache"
)
```

**核心功能：**
- `{{variable}}` - 变量输出
- `{{#section}}...{{/section}}` - 条件或循环
- `{{.}}` - 当前项（循环中）

**适用场景：**
- 跨平台模板兼容
- 简单的文本替换
- 前端模板集成

---

## 🛠️ 高级功能

### 1. 模板验证

```python
# 启用验证（推荐）
validated_template = PromptTemplate(
    input_variables=["name", "age"],
    template="姓名：{name}，年龄：{age}",
    validate_template=True  # 默认True
)

# 禁用验证（特殊场景）
no_validate_template = PromptTemplate(
    input_variables=["name"],
    template="姓名：{name}，年龄：{age}",  # age变量未在input_variables中
    validate_template=False  # 允许不验证
)
```

### 2. 类型定义

```python
typed_template = PromptTemplate(
    input_variables=["name", "age", "email"],
    template="姓名：{name}，年龄：{age}，邮箱：{email}",
    input_types={
        "name": "str",
        "age": "int",
        "email": "str"
    }
)
```

### 3. 元数据

```python
metadata_template = PromptTemplate(
    input_variables=["question"],
    template="问题：{question}",
    metadata={
        "version": "1.0",
        "purpose": "question_answering",
        "author": "AI助手",
        "created_at": "2025-11-28",
        "tags": ["qa", "general"]
    }
)
```

### 4. 部分填充

```python
# 部分填充变量
partial_template = PromptTemplate(
    input_variables=["topic", "style", "length"],
    template="请用{style}风格写一篇{length}字关于{topic}的文章。"
).partial(style="专业", length="500")

# 现在只需要提供topic
formatted = partial_template.format(topic="人工智能")
```

---

## 🎯 实际应用场景

### 1. 代码生成

```python
code_template = PromptTemplate(
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
"""
)
```

### 2. 文档分析

```python
doc_analysis_template = PromptTemplate(
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
"""
)
```

### 3. 多语言翻译

```python
translation_template = PromptTemplate(
    input_variables=["source_text", "source_lang", "target_lang", "style"],
    template="""
请将以下{source_lang}文本翻译成{target_lang}。

原文：
{source_text}

翻译要求：
- 风格：{style}
- 保持原文含义
- 符合目标语言习惯

译文：
"""
)
```

### 4. 数据分析报告

```python
# 使用Jinja2生成复杂报告
data_report_template = PromptTemplate(
    input_variables=["dataset_name", "metrics", "insights", "recommendations"],
    template="""
# {{ dataset_name }} 数据分析报告

## 关键指标
{% for metric in metrics %}
- {{ metric.name }}: {{ metric.value }} {{ metric.unit }}
{% endfor %}

## 数据洞察
{% for insight in insights %}
{{ loop.index }}. {{ insight }}
{% endfor %}

## 建议方案
{% for rec in recommendations %}
• {{ rec }}
{% endfor %}

## 总结
基于以上分析，{{ dataset_name }}的表现{{ summary }}。
""",
    template_format="jinja2"
)
```

### 5. 条件逻辑处理

```python
# 根据不同条件生成不同内容
conditional_template = PromptTemplate(
    input_variables=["user_type", "request"],
    template="""
{% if user_type == 'admin' %}
管理员专用响应：{{ request }}
{% elif user_type == 'premium' %}
高级用户响应：{{ request }}
{% else %}
普通用户响应：{{ request }}
{% endif %}

{% if request contains 'urgent' %}
⚠️ 紧急请求，请优先处理！
{% endif %}
""",
    template_format="jinja2"
)
```

---

## ✅ 最佳实践

### 1. 模板设计原则

- **明确变量命名**：使用有意义的变量名
- **模块化设计**：将复杂模板分解为小模板
- **格式一致性**：在同一项目中使用统一的模板格式
- **错误处理**：实现适当的错误处理和验证

### 2. 性能优化

```python
# 预编译常用模板
PRECOMPILED_TEMPLATES = {
    "code_generation": PromptTemplate(
        input_variables=["language", "task"],
        template="用{language}实现：{task}"
    ),
    "translation": PromptTemplate(
        input_variables=["text", "target_lang"],
        template="翻译成{target_lang}：{text}"
    )
}

# 使用预编译模板
def get_prompt(template_name):
    return PRECOMPILED_TEMPLATES[template_name]
```

### 3. 安全考虑

```python
def safe_format_template(template, data):
    """安全的模板格式化"""
    try:
        # 检查必需变量
        missing_vars = set(template.input_variables) - set(data.keys())
        if missing_vars:
            raise ValueError(f"缺少必需变量: {missing_vars}")

        # 检查额外变量
        extra_vars = set(data.keys()) - set(template.input_variables)
        if extra_vars:
            print(f"警告: 提供了额外变量: {extra_vars}")

        return template.format(**data)
    except Exception as e:
        print(f"模板格式化错误: {e}")
        return None
```

### 4. 国际化支持

```python
I18N_TEMPLATES = {
    "zh-CN": {
        "greeting": PromptTemplate(
            input_variables=["name"],
            template="你好{name}！"
        )
    },
    "en-US": {
        "greeting": PromptTemplate(
            input_variables=["name"],
            template="Hello {name}!"
        )
    },
    "ja-JP": {
        "greeting": PromptTemplate(
            input_variables=["name"],
            template="こんにちは{name}さん！"
        )
    }
}

def get_localized_template(locale, template_name):
    return I18N_TEMPLATES[locale][template_name]
```

---

## 📊 格式选择指南

| 场景 | 推荐格式 | 原因 |
|------|----------|------|
| 简单文本插值 | `f-string` | 性能最好，语法简单 |
| 需要循环/条件 | `Jinja2` | 支持复杂逻辑 |
| 前端模板兼容 | `Mustache` | 跨语言标准 |
| 复杂文档生成 | `Jinja2` | 功能最强大 |
| 性能敏感应用 | `f-string` | 速度最快 |
| 对话场景 | `ChatPromptTemplate` | 专为对话设计 |
| 少样本学习 | `FewShotPromptTemplate` | 专门优化 |
| 模板组合 | `PipelinePromptTemplate` | 支持管道处理 |

---

## 🚨 常见错误和解决方案

### 1. 变量未定义

**错误：**
```python
template = PromptTemplate(
    input_variables=["name"],
    template="你好{name}，年龄{age}"  # age变量未定义
)
```

**解决：**
```python
template = PromptTemplate(
    input_variables=["name", "age"],  # 添加缺失变量
    template="你好{name}，年龄{age}"
)
```

### 2. 模板格式错误

**错误：**
```python
# Jinja2语法错误
template = PromptTemplate(
    input_variables=["items"],
    template="{% for item in items %}"  # 缺少endfor
    template_format="jinja2"
)
```

**解决：**
```python
template = PromptTemplate(
    input_variables=["items"],
    template="{% for item in items %}\n- {{ item }}\n{% endfor %}",
    template_format="jinja2"
)
```

### 3. 类型不匹配

**错误：**
```python
# 传入数字但期望字符串
template = PromptTemplate(
    input_variables=["age"],
    template="年龄：{age}岁"
)
formatted = template.format(age=25)  # 数字类型
```

**解决：**
```python
formatted = template.format(age=str(25))  # 转换为字符串
```

---

## 📚 相关资源

- **官方文档：** https://python.langchain.com/docs/modules/prompts/
- **Jinja2文档：** https://jinja.palletsprojects.com/
- **Mustache规范：** https://mustache.github.io/
- **模板模式：** https://python.langchain.com/docs/modules/prompts/prompt_templates/

---

*更新时间：2025-11-28*