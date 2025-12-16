#!/usr/bin/env python3
"""
LangChain 进阶示例 - 智能体和工具 (Agents & Tools)
展示如何创建能够使用外部工具的AI智能体
"""

import os
import json
from dotenv import load_dotenv
from datetime import datetime
from typing import Annotated, Optional, Sequence
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage,SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_openai import ChatOpenAI
# 加载环境变量
load_dotenv()

# ========================
# 定义工具集 (Tool Definitions)
# ========================

@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def calculate(expression: str) -> str:
    """执行数学计算，支持加减乘除等基本运算

    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    """
    try:
        # 安全的数学计算（注意：实际应用中需要更严格的安全检查）
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def search_web(query: str) -> str:
    """模拟网络搜索功能（这里是模拟实现）

    Args:
        query: 搜索关键词
    """
    # 这里是模拟的网络搜索结果
    mock_results = {
        "Python": "Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。",
        "LangChain": "LangChain是一个用于构建LLM应用的框架，提供了模块化的组件。",
        "AI": "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行智能任务的系统。"
    }

    # 简单的关键词匹配
    for keyword in mock_results:
        if keyword.lower() in query.lower():
            return f"搜索结果：{mock_results[keyword]}"

    return f"抱歉，没有找到关于 '{query}' 的相关信息。"

@tool
def save_to_file(content: str, filename: str) -> str:
    """将内容保存到文件中

    Args:
        content: 要保存的内容
        filename: 文件名
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功将内容保存到文件：{filename}"
    except Exception as e:
        return f"保存文件失败：{str(e)}"

@tool
def read_file(filename: str) -> str:
    """读取文件内容

    Args:
        filename: 文件名
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"文件 '{filename}' 的内容：\n{content}"
    except FileNotFoundError:
        return f"文件 '{filename}' 不存在。"
    except Exception as e:
        return f"读取文件失败：{str(e)}"

# ========================
# 创建智能体 (Agent Creation)
# ========================

def create_my_agent():
    """创建一个具有多种工具的智能体"""

    # 定义可用工具列表
    tools = [
        get_current_time,
        calculate,
        search_web,
        save_to_file,
        read_file
    ]

    # 创建LLM实例
    llm = ChatOpenAI(
        model="glm-4.5",
        temperature=0.1,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建提示模板
    system_prompt = SystemMessage(content="""你是一个智能助手，可以使用以下工具来帮助用户：

可用工具：
- get_current_time: 获取当前时间
- calculate: 执行数学计算
- search_web: 搜索网络信息
- save_to_file: 保存内容到文件
- read_file: 读取文件内容

请根据用户的需求，选择合适的工具来完成任务,并说明使用了什么工具。如果需要使用多个工具，可以按步骤执行。
要使用工具时，请使用工具的准确名称和参数。""")
    # 创建智能体 (使用LangGraph的方式)
    try:
        agent_executor = create_agent(llm, tools, system_prompt=system_prompt )
        return agent_executor
    except Exception as e:
        print(f"创建智能体失败: {e}")
        # 如果create_agent失败，尝试直接使用LLM
        return llm

# ========================
# 示例演示 (Example Demonstrations)
# ========================

def basic_agent_example():
    """基础智能体示例"""
    print("=== 基础智能体示例 ===\n")

    agent = create_my_agent()

    # 测试问题
    questions = [
        "现在几点了？",
        "帮我计算 15 * 8 + 32 等于多少",
        "搜索一下 LangChain 的信息",
        "读取SUMMARY.md内容并总结"
    ]

    for question in questions:
        print(f"用户：{question}")
        try:
            # 检查是否是agent还是llm
            if hasattr(agent, 'invoke') and hasattr(agent, 'stream'):
                # 如果是LangGraph agent
                inputs = {"messages": [{"role": "user","content": question}]}
                result = agent.invoke(inputs)
                print(f"助手：{result['messages'][-1].content}\n")
            else:
                # 如果是普通的LLM
                result = agent.invoke(question)
                print(f"助手：{result.content}\n")
        except Exception as e:
            print(f"执行出错：{e}\n")
        print("-" * 50)

def complex_task_example():
    """动态复杂任务示例 - 根据用户输入自动生成TODO列表"""
    print("=== 动态复杂任务示例（智能任务分解） ===\n")

    # 获取用户输入
    print("请输入您想要完成的复杂任务：")
    print("例如：")
    print("- 研究Python编程语言并保存到文件")
    print("- 计算数学表达式并分析结果")
    print("- 搜索AI相关信息并创建学习笔记")
    print("- 获取当前时间并记录到文件")
    print()

    user_task = input("请输入任务描述: ").strip()

    if not user_task:
        user_task = "研究Python编程语言，获取当前时间，并将信息保存到文件中"
        print(f"使用默认任务: {user_task}\n")

    agent = create_my_agent()

    print(f"🎯 用户任务: {user_task}\n")

    # 智能解析任务并生成步骤
    task_steps = parse_user_task_to_steps(user_task)

    # 创建任务列表
    task_list = []
    for i, step in enumerate(task_steps, 1):
        task_list.append({
            "id": i,
            "task": step["task"],
            "tool": step.get("tool", None),
            "status": "待开始"
        })

    def print_task_status():
        """打印当前任务状态"""
        print("📋 任务进度：")
        for task_item in task_list:
            status_symbol = {"待开始": "[待]", "进行中": "[...]", "已完成": "[完成]", "失败": "[失败]"}.get(task_item["status"], "[待]")
            tool_info = f" (工具: {task_item['tool']})" if task_item['tool'] else ""
            print(f"  {status_symbol} [{task_item['id']}] {task_item['task']}{tool_info}")
        print()

    # 初始状态显示
    print("🔍 任务分析结果 - 自动生成的执行步骤：\n")
    print_task_status()

    collected_info = {}  # 收集的信息

    try:
        for i, task_item in enumerate(task_list):
            # 更新任务状态为进行中
            task_item["status"] = "进行中"
            print_task_status()

            print(f"🚀 执行任务 {i}: {task_item['task']}")

            # 构建具体的执行指令
            if task_item['tool'] == 'get_current_time':
                instruction = "请帮我获取当前时间"
            elif task_item['tool'] == 'search_web':
                # 根据任务内容构建搜索指令
                if "Python" in task_item['task']:
                    instruction = "请搜索Python编程语言的基本信息，包括特点、应用领域等"
                elif "AI" in task_item['task'] or "人工智能" in task_item['task']:
                    instruction = "请搜索人工智能相关信息，包括发展历史、应用领域等"
                else:
                    instruction = f"请搜索{task_item['task']}相关信息"
            elif task_item['tool'] == 'calculate':
                instruction = f"请计算{task_item['task'].replace('计算: ', '')}"
            elif task_item['tool'] == 'save_to_file':
                filename = "output.txt"
                if "python" in task_item['task'].lower():
                    filename = "python_study_notes.txt"
                elif "report" in task_item['task'].lower():
                    filename = "report.txt"

                # 之前收集的信息
                content_parts = []
                if "time" in collected_info:
                    content_parts.append(f"时间信息: {collected_info['time']}")
                if any(key in collected_info for key in ["python_info", "search_result"]):
                    for key in ["python_info", "search_result"]:
                        if key in collected_info:
                            content_parts.append(f"搜索信息: {collected_info[key]}")

                content = "\n\n".join(content_parts) if content_parts else "任务执行结果"
                instruction = f"请将以下内容保存到 '{filename}' 文件中: {content}"

            elif task_item['tool'] == 'read_file':
                filename = "python_study_notes.txt" if "python" in task_item['task'].lower() else "output.txt"
                instruction = f"请读取 '{filename}' 文件的内容并确认保存成功"
            else:
                instruction = f"请执行以下任务: {task_item['task']}"

            # 执行任务
            if hasattr(agent, 'stream'):
                # 如果是LangGraph agent
                result = agent.invoke({"messages": [{"role": "user", "content": instruction}]})
                response = result['messages'][-1].content
            else:
                # 如果是普通的LLM
                result = agent.invoke(instruction)
                response = result.content

            print(f"📤 响应: {response}")

            # 智能收集信息
            if task_item['tool'] == 'get_current_time':
                collected_info["time"] = response
            elif task_item['tool'] == 'search_web':
                if "Python" in task_item['task']:
                    collected_info["python_info"] = response
                else:
                    collected_info["search_result"] = response
            elif task_item['tool'] == 'calculate':
                collected_info["calculation_result"] = response

            # 标记任务完成
            task_item["status"] = "已完成"
            print_task_status()
            print("-" * 60)

        # 最终总结
        print("🎉 所有任务已完成！")
        print("📝 任务执行总结：")
        print(f"✅ 原始任务: {user_task}")
        print(f"✅ 分解步骤数: {len(task_list)}")
        print(f"✅ 使用工具: {list(set([item['tool'] for item in task_list if item['tool']]))}")

        if collected_info:
            print("✅ 收集的信息:")
            for key, value in collected_info.items():
                print(f"  - {key}: {value[:100]}{'...' if len(value) > 100 else ''}")

        print("\n✨ 智能任务分解流程:")
        print("  1️⃣ 智能解析用户输入的任务描述")
        print("  2️⃣ 识别任务类型和所需工具")
        print("  3️⃣ 自动生成执行步骤和TODO列表")
        print("  4️⃣ 分步骤执行并实时更新进度")
        print("  5️⃣ 收集、整合并提供完成报告")

    except Exception as e:
        print(f"❌ 执行出错：{e}")
        # 标记失败的任务
        for task_item in task_list:
            if task_item["status"] == "进行中":
                task_item["status"] = "失败"
                break
        print_task_status()

def ai_complex_task_example():
    """AI智能分析复杂任务示例 - AI分析→用户确认→执行"""
    print("=== AI智能分析复杂任务示例 ===\n")

    # 获取用户输入
    print("请输入您想要完成的复杂任务：")
    print("例如：")
    print("- 研究Python编程语言并保存到文件")
    print("- 计算数学表达式并分析结果")
    print("- 搜索AI相关信息并创建学习笔记")
    print("- 获取当前时间并记录到文件")
    print()

    user_task = input("请输入任务描述: ").strip()

    if not user_task:
        user_task = "研究Python编程语言，获取当前时间，并将信息保存到文件中"
        print(f"使用默认任务: {user_task}\n")

    agent = create_my_agent()

    print(f"🎯 用户任务: {user_task}")
    print("\n🤖 正在让AI分析您的任务，生成执行计划...")
    print("-" * 50)

    # AI分析任务并生成TODO列表
    task_steps = ai_analyze_task_to_todo(user_task, agent)

    # 创建任务列表
    task_list = []
    for i, step in enumerate(task_steps, 1):
        task_list.append({
            "id": i,
            "task": step.get("task", f"步骤 {i}"),
            "tool": step.get("tool", None),
            "priority": step.get("priority", i),
            "description": step.get("description", ""),
            "status": "待开始"
        })

    def print_todo_list():
        """打印TODO列表供用户确认"""
        print("📋 AI生成的执行计划：")
        for i, task_item in enumerate(task_list, 1):
            tool_info = f" [工具: {task_item['tool']}]" if task_item['tool'] else ""
            print(f"  {i}. {task_item['task']}{tool_info}")
            if task_item['description']:
                print(f"     说明: {task_item['description']}")
        print()

    # 显示AI分析结果
    print_todo_list()

    # 用户确认环节
    print("请确认以上执行计划：")
    print("1. 继续执行 - 开始按计划执行任务")
    print("2. 修改计划 - 重新生成或修改执行步骤")
    print("3. 取消任务 - 退出当前任务")

    while True:
        try:
            choice = input("请选择 (1/2/3): ").strip()
            if choice == "1":
                print("\n✅ 用户确认，开始执行任务...")
                break
            elif choice == "2":
                print("\n📝 修改计划功能开发中，使用当前计划继续执行...")
                break
            elif choice == "3":
                print("\n❌ 用户取消任务")
                return
            else:
                print("无效选择，请输入 1、2 或 3")
        except (EOFError, KeyboardInterrupt):
            print("\n\n用户取消操作")
            return

    # 更新状态显示函数
    def print_execution_status():
        """打印执行状态"""
        print("📊 任务执行进度：")
        for task_item in task_list:
            status_symbol = {"待开始": "[待]", "进行中": "[...]", "已完成": "[完成]", "失败": "[失败]"}.get(task_item["status"], "[待]")
            tool_info = f" (工具: {task_item['tool']})" if task_item['tool'] else ""
            print(f"  {status_symbol} [{task_item['id']}] {task_item['task']}{tool_info}")
        print()

    # 开始执行任务
    print("🚀 开始执行任务...\n")
    collected_info = {}

    try:
        for task_item in task_list:
            # 更新任务状态为进行中
            task_item["status"] = "进行中"
            print_execution_status()

            print(f"🔄 正在执行: {task_item['task']}")

            # 构建具体的执行指令
            if task_item['tool'] == 'get_current_time':
                instruction = "请帮我获取当前时间"
            elif task_item['tool'] == 'search_web':
                instruction = f"请搜索{task_item['task']}相关信息"
            elif task_item['tool'] == 'calculate':
                instruction = f"请计算{task_item['task'].replace('计算: ', '')}"
            elif task_item['tool'] == 'save_to_file':
                # 确定文件名
                filename = "output.txt"
                if "python" in task_item['task'].lower():
                    filename = "python_study_notes.txt"
                elif "report" in task_item['task'].lower():
                    filename = "report.txt"

                # 准备要保存的内容
                content_parts = []
                if "time" in collected_info:
                    content_parts.append(f"时间信息: {collected_info['time']}")
                if any(key in collected_info for key in ["python_info", "search_result", "calculation_result"]):
                    for key in ["python_info", "search_result", "calculation_result"]:
                        if key in collected_info:
                            content_parts.append(f"{key}: {collected_info[key]}")

                if not content_parts:
                    content = "任务执行结果"
                else:
                    content = "\n\n".join(content_parts)

                instruction = f"请将以下内容保存到 '{filename}' 文件中: {content}"
            elif task_item['tool'] == 'read_file':
                filename = "python_study_notes.txt" if "python" in task_item['task'].lower() else "output.txt"
                instruction = f"请读取 '{filename}' 文件的内容并确认保存成功"
            else:
                instruction = f"请执行以下任务: {task_item['task']}"

            # 执行任务
            if hasattr(agent, 'stream'):
                result = agent.invoke({"messages": [{"role": "user", "content": instruction}]})
                response = result['messages'][-1].content
            else:
                result = agent.invoke(instruction)
                response = result.content

            print(f"📤 执行结果: {response}")

            # 智能收集信息
            if task_item['tool'] == 'get_current_time':
                collected_info["time"] = response
            elif task_item['tool'] == 'search_web':
                if "Python" in task_item['task']:
                    collected_info["python_info"] = response
                else:
                    collected_info["search_result"] = response
            elif task_item['tool'] == 'calculate':
                collected_info["calculation_result"] = response

            # 标记任务完成
            task_item["status"] = "已完成"
            print_execution_status()
            print("-" * 50)

        # 最终总结
        print("🎉 所有任务执行完成！")
        print("\n📝 任务执行总结：")
        print(f"✅ 原始任务: {user_task}")
        print(f"✅ 执行步骤数: {len(task_list)}")
        print(f"✅ 使用工具: {list(set([item['tool'] for item in task_list if item['tool']]))}")

        if collected_info:
            print("✅ 收集的信息:")
            for key, value in collected_info.items():
                preview = value[:100] + ('...' if len(value) > 100 else '')
                print(f"  - {key}: {preview}")

        print("\n🔄 AI分析→用户确认→执行 流程:")
        print("  1️⃣ 用户输入复杂任务描述")
        print("  2️⃣ AI智能分析并生成详细执行计划")
        print("  3️⃣ 用户确认或修改执行计划")
        print("  4️⃣ 按计划分步骤执行任务")
        print("  5️⃣ 提供完整的执行报告和结果")

    except Exception as e:
        print(f"❌ 执行出错：{e}")
        # 标记失败的任务
        for task_item in task_list:
            if task_item["status"] == "进行中":
                task_item["status"] = "失败"
                break
        print_execution_status()

def custom_tool_example():
    """自定义工具示例"""
    print("=== 自定义工具示例 ===\n")

    @tool
    def analyze_code_quality(code: str) -> str:
        """分析代码质量（简单分析）

        Args:
            code: 要分析的代码
        """
        lines = code.split('\n')
        total_lines = len(lines)

        # 简单的代码质量检查
        issues = []
        if total_lines > 50:
            issues.append("函数可能过长，建议拆分")

        if 'print(' in code:
            issues.append("代码中包含print语句，建议在正式代码中使用日志")

        if code.count('for ') + code.count('while ') > 3:
            issues.append("嵌套循环较多，建议优化逻辑")

        if not issues:
            return f"代码质量良好！共 {total_lines} 行代码。"
        else:
            return f"代码分析结果：共 {total_lines} 行代码。发现 {len(issues)} 个潜在问题：\n" + "\n".join(f"- {issue}" for issue in issues)

    # 创建包含自定义工具的智能体
    tools = [
        calculate,
        analyze_code_quality
    ]

    llm = ChatOpenAI(
        model="glm-4.5",
        temperature=0.1,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    system_prompt = """你是一个代码分析助手，可以帮助用户进行数学计算和代码质量分析。

可用工具：
- calculate: 执行数学计算
- analyze_code_quality: 分析代码质量并提供改进建议

请根据用户需求选择合适的工具。"""

    # 创建智能体
    try:
        agent_executor = create_agent(llm, tools,system_prompt)
    except Exception as e:
        print(f"创建智能体失败: {e}")
        # 如果create_agent失败，尝试直接使用LLM
        agent_executor = llm

    # 测试代码分析
    sample_code = """
def calculate_factorial(n):
    result = 1
    for i in range(1, n + 1):
        for j in range(1, 5):  # 内层循环
            print(f"计算 {i} * {j}")
            result *= i
    return result
"""

    questions = [
        "计算 100 的阶除以 5 等于多少",
        f"请分析以下代码的质量：{sample_code}"
    ]

    for question in questions:
        print(f"用户：{question}")
        try:
            # 检查是否是agent还是llm
            if hasattr(agent_executor, 'stream'):
                # 如果是LangGraph agent
                result = agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
                print(f"助手：{result['messages'][-1].content}\n")
            else:
                # 如果是普通的LLM
                result = agent_executor.invoke(question)
                print(f"助手：{result.content}\n")
        except Exception as e:
            print(f"执行出错：{e}\n")
        print("-" * 50)

# ========================
# 动态任务解析函数 (Dynamic Task Parsing)
# ========================

def ai_analyze_task_to_todo(user_task: str, agent) -> list:
    """让AI分析用户任务，智能生成TODO列表"""

    # 构建分析提示
    analysis_prompt = f"""请分析以下用户任务，将其分解为具体的执行步骤，并生成TODO列表：

用户任务：{user_task}

请按照以下格式返回JSON格式的结果：

{{
    "analysis": "对用户任务的理解和分析",
    "steps": [
        {{
            "id": 1,
            "task": "具体的执行步骤描述",
            "tool": "需要的工具名称（get_current_time, search_web, calculate, save_to_file, read_file等）",
            "priority": 1,
            "description": "该步骤的详细说明"
        }},
        ...
    ]
}}

可用工具说明：
- get_current_time: 获取当前时间
- search_web: 搜索网络信息
- calculate: 执行数学计算
- save_to_file: 保存内容到文件
- read_file: 读取文件内容

请只返回JSON格式，不要包含其他解释文字。"""

    try:
        # 使用AI分析任务
        if hasattr(agent, 'stream'):
            # 如果是LangGraph agent
            result = agent.invoke({"messages": [{"role": "user", "content": analysis_prompt}]})
            ai_response = result['messages'][-1].content
        else:
            # 如果是普通的LLM
            result = agent.invoke(analysis_prompt)
            ai_response = result.content

        # 解析AI返回的JSON
        import json
        import re

        # 尝试提取JSON部分
        json_match = re.search(r'\\{.*\\}', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            todo_data = json.loads(json_str)
            return todo_data.get("steps", [])
        else:
            # 如果无法提取JSON，使用备用方法
            return parse_user_task_to_steps(user_task)

    except Exception as e:
        print(f"AI分析失败，使用备用解析方法: {e}")
        return parse_user_task_to_steps(user_task)

def parse_user_task_to_steps(user_task: str) -> list:
    """智能解析用户输入的任务，分解为具体的执行步骤"""

    # 尝试使用AI进行智能分析
    try:
        agent = create_my_agent()
        # 使用更简单的提示词进行快速分析
        analysis_prompt = f"""分析任务并生成执行步骤，返回JSON格式：

用户任务：{user_task}

请分析这个任务需要哪些具体步骤，并返回：
{{
    "steps": [
        {{
            "task": "步骤1描述",
            "tool": "工具名或null",
            "priority": 1
        }},
        ...
    ]
}}

可用工具：get_current_time, search_web, calculate, save_to_file, read_file

只返回JSON，不要其他文字。"""

        if hasattr(agent, 'stream'):
            result = agent.invoke({"messages": [{"role": "user", "content": analysis_prompt}]})
            ai_response = result['messages'][-1].content
        else:
            result = agent.invoke(analysis_prompt)
            ai_response = result.content

        # 尝试解析JSON
        import json, re
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            todo_data = json.loads(json_match.group())
            if todo_data.get("steps"):
                return todo_data["steps"]

    except Exception as e:
        print(f"AI分析失败，使用备用方法: {e}")

    # 备用的简单解析方法
    return _simple_task_parsing(user_task)

def _simple_task_parsing(user_task: str) -> list:
    """简单的任务解析备用方法"""
    user_task_lower = user_task.lower()
    detected_tasks = []

    # 时间相关
    if any(keyword in user_task_lower for keyword in ["时间", "当前", "今天", "现在", "日期"]):
        detected_tasks.append({
            "task": "获取当前时间",
            "tool": "get_current_time",
            "priority": 1
        })

    # 计算
    import re
    calc_patterns = [
        r'(\d+\s*[\+\-\*\/]\s*\d+)',  # 简单运算
        r'计算.*?(\d+.+?\d+)',       # "计算"开头的表达式
        r'等于多少.*?(\d+.+?\d+)'    # "等于多少"结尾的表达式
    ]

    for pattern in calc_patterns:
        match = re.search(pattern, user_task)
        if match:
            detected_tasks.append({
                "task": f"计算: {match.group(1)}",
                "tool": "calculate",
                "priority": 2
            })
            break

    # 搜索/研究
    search_topics = []
    if any(keyword in user_task_lower for keyword in ["研究", "搜索", "了解", "查找"]):
        if "python" in user_task_lower:
            search_topics.append("Python编程语言")
        elif "langchain" in user_task_lower:
            search_topics.append("LangChain框架")
        elif "ai" in user_task_lower or "人工智能" in user_task_lower:
            search_topics.append("人工智能")
        elif "javascript" in user_task_lower:
            search_topics.append("JavaScript编程")
        else:
            # 提取通用搜索主题
            topic_match = re.search(r'(?:研究|搜索)(.+?)(?:信息|资料|内容)', user_task)
            if topic_match:
                search_topics.append(topic_match.group(1))
            else:
                search_topics.append("相关信息")

    for topic in search_topics:
        detected_tasks.append({
            "task": f"搜索{topic}相关信息",
            "tool": "search_web",
            "priority": 3
        })

    # 文件操作
    if any(keyword in user_task_lower for keyword in ["保存", "文件", "创建", "写入", "记录"]):
        # 智能确定文件名
        filename = _generate_filename(user_task)

        detected_tasks.append({
            "task": f"保存结果到 '{filename}' 文件",
            "tool": "save_to_file",
            "priority": 4
        })

    # 如果没有识别到具体任务，生成通用步骤
    if not detected_tasks:
        detected_tasks = [
            {
                "task": "分析任务需求",
                "tool": None,
                "priority": 1
            },
            {
                "task": "收集相关信息",
                "tool": "search_web",
                "priority": 2
            }
        ]

    # 按优先级排序
    detected_tasks.sort(key=lambda x: x["priority"])
    return detected_tasks

def _generate_filename(user_task: str) -> str:
    """根据任务内容智能生成文件名"""
    user_task_lower = user_task.lower()

    # 基于任务内容确定文件名
    if "python" in user_task_lower:
        return "python_study_notes.txt"
    elif "report" in user_task_lower or "报告" in user_task_lower:
        return "task_report.txt"
    elif "note" in user_task_lower or "笔记" in user_task_lower or "记录" in user_task_lower:
        return "study_notes.txt"
    elif "result" in user_task_lower or "结果" in user_task_lower:
        return "calculation_results.txt"
    else:
        # 基于当前时间生成默认文件名
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"task_result_{timestamp}.txt"

# ========================
# 主函数 (Main Function)
# ========================

def test_ai_analysis_noninteractive():
    """非交互式测试AI分析功能"""
    print("=== 非交互式AI分析测试 ===\n")

    # 创建智能体
    agent = create_my_agent()

    # 测试任务
    test_tasks = [
        "研究Python编程语言并保存学习笔记到文件",
        "计算数学表达式 15 * 8 + 32 并验证结果"
    ]

    for i, task in enumerate(test_tasks, 1):
        print(f"测试任务 {i}: {task}")
        print("-" * 50)

        try:
            # 使用AI分析任务
            task_steps = ai_analyze_task_to_todo(task, agent)

            print(f"AI分析的执行步骤 ({len(task_steps)} 个):")
            for j, step in enumerate(task_steps, 1):
                tool_info = f" [工具: {step.get('tool', '无')}]" if step.get('tool') else ""
                desc_info = f" - {step.get('description', '')}" if step.get('description') else ""
                print(f"  {j}. {step.get('task', f'步骤 {j}')}{tool_info}{desc_info}")

            print("-" * 30)

        except Exception as e:
            print(f"分析失败: {e}")
            print("-" * 30)

def test_dynamic_parsing():
    """测试改进后的智能任务解析功能"""
    print("=== 测试改进后的智能任务解析功能 ===\n")

    test_cases = [
        "研究Python编程语言并保存到文件",
        "计算25 + 17等于多少",
        "获取当前时间并记录到文件",
        "搜索AI相关信息并创建学习笔记",
        "分析JavaScript代码质量",
        "研究LangChain框架的详细信息",
        "帮我计算 (15 * 8) + 32 并保存结果",
        "现在几点了？请记录下来",
        "查找React框架信息并生成学习报告",
        "计算圆的面积并保存到数学笔记文件"
    ]

    print("[测试] 多种类型的任务解析：\n")

    for i, test_task in enumerate(test_cases, 1):
        print(f"测试用例 {i}: {test_task}")
        print("-" * 40)

        try:
            steps = parse_user_task_to_steps(test_task)

            print(f"[成功] 解析成功 - 生成 {len(steps)} 个步骤:")
            for j, step in enumerate(steps, 1):
                tool_info = f" [工具: {step.get('tool', '无')}]" if step.get('tool') else ""
                priority_info = f" (优先级: {step.get('priority', 'N/A')})"
                print(f"  {j}. {step.get('task', f'步骤 {j}')}{tool_info}{priority_info}")

        except Exception as e:
            print(f"[失败] 解析失败: {e}")

        print("\n" + "="*50 + "\n")

def compare_parsing_methods():
    """对比新旧解析方法的效果"""
    print("=== 对比新旧任务解析方法 ===\n")

    test_tasks = [
        "研究Python并保存学习笔记",
        "计算100除以5的结果",
        "获取当前时间"
    ]

    for task in test_tasks:
        print(f"[任务]: {task}")
        print("-" * 30)

        # 使用旧方法模拟
        print("[旧方法 - 固定步骤]:")
        old_steps = [
            {"task": "获取相关信息", "tool": "search_web"},
            {"task": "整理关键要点", "tool": None},
            {"task": "总结核心内容", "tool": None},
            {"task": "保存或写入文件", "tool": "save_to_file"},
            {"task": "验证文件内容", "tool": "read_file"}
        ]
        for i, step in enumerate(old_steps, 1):
            tool_info = f" [工具: {step['tool']}]" if step['tool'] else ""
            print(f"  {i}. {step['task']}{tool_info}")

        print("\n[新方法 - 智能解析]:")
        try:
            new_steps = parse_user_task_to_steps(task)
            for i, step in enumerate(new_steps, 1):
                tool_info = f" [工具: {step.get('tool', '无')}]" if step.get('tool') else ""
                print(f"  {i}. {step.get('task', f'步骤 {i}')}{tool_info}")

            print(f"\n[改进效果]: 旧方法总是5个步骤，新方法生成 {len(new_steps)} 个步骤")
        except Exception as e:
            print(f"  解析失败: {e}")

        print("\n" + "="*50 + "\n")

def main():
    """主函数"""
    print("LangChain 智能体和工具进阶示例\n")
    print("本示例展示如何创建能够使用外部工具的AI智能体\n")
    print("新增功能：智能任务拆解 - 根据任务复杂度动态生成步骤\n")

    try:
        # 1. 对比新旧解析方法
        print("\n" + "="*60)
        #print("1. 首先对比新旧任务解析方法的效果")
        print("="*60 + "\n")
        #compare_parsing_methods()

        # 2. 测试改进后的智能解析功能
        print("\n" + "="*60)
        #print("2. 测试改进后的智能任务解析功能")
        print("="*60 + "\n")
        #test_dynamic_parsing()

        # 3. 运行完整的AI智能分析复杂任务示例
        print("\n" + "="*60)
        print("3. 体验完整的AI分析->用户确认->执行流程")
        print("="*60 + "\n")
        basic_agent_example()
        #ai_complex_task_example()

    except Exception as e:
        print(f"运行示例时出错：{e}")
        print("请确保您的环境配置正确。")

if __name__ == "__main__":
    main()