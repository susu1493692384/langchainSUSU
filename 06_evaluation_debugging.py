#!/usr/bin/env python3
"""
LangChain 进阶示例 - 评估和调试 (Evaluation & Debugging)
展示如何评估AI应用性能、调试问题和优化系统
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain.evaluation.criteria import LabeledCriteria
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.callbacks import get_openai_callback, tracing_enabled
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser

# 加载环境变量
load_dotenv()

# ========================
# 自定义输出解析器 (Custom Output Parser)
# ========================

class StructuredOutputParser(BaseOutputParser):
    """结构化输出解析器，用于解析JSON格式的响应"""

    def parse(self, text: str) -> Dict[str, Any]:
        """解析文本为结构化数据"""
        try:
            # 尝试解析JSON
            if text.strip().startswith('{'):
                return json.loads(text)
            else:
                # 如果不是JSON，提取关键信息
                return {
                    "response": text.strip(),
                    "confidence": "medium",
                    "structured": False
                }
        except json.JSONDecodeError:
            return {
                "response": text.strip(),
                "error": "Failed to parse as JSON",
                "structured": False
            }

    def get_format_instructions(self) -> str:
        """获取格式说明"""
        return """请以JSON格式返回你的回答，格式如下：
{
    "response": "你的回答",
    "confidence": "high/medium/low",
    "sources": ["引用来源1", "引用来源2"],
    "additional_info": "额外信息"
}"""

# ========================
# 性能评估指标 (Performance Metrics)
# ========================

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_time": 0.0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "errors": 0
        }

    def start_request(self):
        """开始请求计时"""
        self.start_time = time.time()

    def end_request(self, tokens_used: int = 0, cost: float = 0.0, error: bool = False):
        """结束请求并记录指标"""
        if hasattr(self, 'start_time'):
            request_time = time.time() - self.start_time
            self.metrics["total_requests"] += 1
            self.metrics["total_time"] += request_time
            self.metrics["total_tokens"] += tokens_used
            self.metrics["total_cost"] += cost
            if error:
                self.metrics["errors"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if self.metrics["total_requests"] == 0:
            return self.metrics

        return {
            **self.metrics,
            "avg_time_per_request": self.metrics["total_time"] / self.metrics["total_requests"],
            "avg_tokens_per_request": self.metrics["total_tokens"] / self.metrics["total_requests"],
            "avg_cost_per_request": self.metrics["total_cost"] / self.metrics["total_requests"],
            "error_rate": self.metrics["errors"] / self.metrics["total_requests"] * 100,
            "requests_per_second": self.metrics["total_requests"] / self.metrics["total_time"] if self.metrics["total_time"] > 0 else 0
        }

# ========================
# 质量评估器 (Quality Evaluators)
# ========================

class ResponseQualityEvaluator:
    """回答质量评估器"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.evaluator = load_evaluator(EvaluatorType.CRITERIA, llm=llm)

    def evaluate_response(self, question: str, response: str, reference: Optional[str] = None) -> Dict[str, Any]:
        """评估回答质量"""
        evaluation_criteria = {
            "relevance": "回答是否与问题相关",
            "accuracy": "回答的准确性",
            "clarity": "表达的清晰度",
            "completeness": "回答的完整性",
            "helpfulness": "回答的帮助程度"
        }

        if reference:
            # 有参考答案的评估
            eval_result = self.evaluator.evaluate_strings(
                prediction=response,
                reference=reference,
                input=question,
                criteria=evaluation_criteria
            )
        else:
            # 无参考答案的评估
            eval_result = self.evaluator.evaluate_strings(
                prediction=response,
                input=question,
                criteria=evaluation_criteria
            )

        return eval_result

class CustomEvaluator:
    """自定义评估器"""

    @staticmethod
    def evaluate_length(response: str, min_length: int = 50, max_length: int = 500) -> Dict[str, Any]:
        """评估回答长度是否合适"""
        length = len(response)
        score = 1.0
        feedback = []

        if length < min_length:
            score *= 0.5
            feedback.append(f"回答过短，建议至少 {min_length} 字符")
        elif length > max_length:
            score *= 0.8
            feedback.append(f"回答过长，建议不超过 {max_length} 字符")
        else:
            feedback.append("回答长度适中")

        return {
            "score": score,
            "length": length,
            "feedback": feedback
        }

    @staticmethod
    def evaluate_structure(response: str) -> Dict[str, Any]:
        """评估回答结构"""
        has_numbering = bool(any(char.isdigit() for char in response.split()[:3]))
        has_bullets = '•' in response or '-' in response or '*' in response
        has_paragraphs = response.count('\n') >= 2

        score = 0.0
        feedback = []

        if has_numbering:
            score += 0.3
            feedback.append("使用了数字编号")
        if has_bullets:
            score += 0.3
            feedback.append("使用了项目符号")
        if has_paragraphs:
            score += 0.4
            feedback.append("分段清晰")

        if score == 0:
            feedback.append("建议使用编号或项目符号来组织内容")

        return {
            "score": score,
            "structure_elements": {
                "numbering": has_numbering,
                "bullets": has_bullets,
                "paragraphs": has_paragraphs
            },
            "feedback": feedback
        }

# ========================
# 调试和监控工具 (Debugging & Monitoring Tools)
# ========================

class DebugChain:
    """带调试功能的链"""

    def __init__(self, llm: ChatOpenAI, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose
        self.memory = ConversationBufferMemory(return_messages=True)
        self.debug_info = []

    def log_debug(self, message: str, data: Any = None):
        """记录调试信息"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "data": data
        }
        self.debug_info.append(log_entry)

        if self.verbose:
            print(f"[{timestamp}] DEBUG: {message}")
            if data:
                print(f"  Data: {data}")

    def invoke_with_debug(self, prompt: str, **kwargs) -> str:
        """带调试的调用"""
        self.log_debug("开始处理请求", {"prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt})

        try:
            # 记录输入
            self.memory.chat_memory.add_user_message(prompt)

            # 调用模型
            response = self.llm.invoke(prompt + "\n\n请提供详细的回答。")

            # 记录输出
            self.memory.chat_memory.add_ai_message(response.content)
            self.log_debug("成功生成回答", {"response_length": len(response.content)})

            return response.content

        except Exception as e:
            self.log_debug("处理请求时出错", {"error": str(e)})
            raise

    def get_debug_summary(self) -> Dict[str, Any]:
        """获取调试摘要"""
        return {
            "total_logs": len(self.debug_info),
            "debug_entries": self.debug_info,
            "conversation_length": len(self.memory.chat_memory.messages)
        }

# ========================
# 示例演示 (Example Demonstrations)
# ========================

def performance_monitoring_example():
    """性能监控示例"""
    print("=== 性能监控示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    monitor = PerformanceMonitor()
    test_questions = [
        "什么是机器学习？",
        "解释一下Python的特点",
        "LangChain有什么优势？",
        "如何学习人工智能？"
    ]

    print("开始性能测试...\n")

    for i, question in enumerate(test_questions, 1):
        print(f"问题 {i}: {question}")

        monitor.start_request()

        try:
            with get_openai_callback() as cb:
                response = llm.invoke(question)
                tokens_used = cb.total_tokens
                estimated_cost = cb.total_cost

            monitor.end_request(tokens_used=tokens_used, cost=estimated_cost)
            print(f"回答: {response.content[:100]}...\n")

        except Exception as e:
            monitor.end_request(error=True)
            print(f"错误: {e}\n")

    # 显示统计信息
    stats = monitor.get_stats()
    print("=== 性能统计 ===")
    print(f"总请求数: {stats['total_requests']}")
    print(f"平均响应时间: {stats['avg_time_per_request']:.2f}秒")
    print(f"平均Token数: {stats['avg_tokens_per_request']:.0f}")
    print(f"平均成本: ${stats['avg_cost_per_request']:.4f}")
    print(f"错误率: {stats['error_rate']:.1f}%")
    print(f"每秒请求数: {stats['requests_per_second']:.2f}\n")

def quality_evaluation_example():
    """质量评估示例"""
    print("=== 质量评估示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.1,
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    evaluator = ResponseQualityEvaluator(llm)
    custom_evaluator = CustomEvaluator()

    test_cases = [
        {
            "question": "什么是人工智能？",
            "response": "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "reference": "人工智能是指由计算机系统表现出的智能，能够学习、推理、感知和理解自然语言。"
        },
        {
            "question": "Python适合做什么？",
            "response": "Python是一种高级编程语言，适合web开发、数据分析、机器学习、自动化脚本等多种应用场景。",
            "reference": None
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"测试案例 {i}:")
        print(f"问题: {case['question']}")
        print(f"回答: {case['response']}")

        # 使用标准评估器
        eval_result = evaluator.evaluate_response(
            case["question"],
            case["response"],
            case.get("reference")
        )
        print(f"\n标准评估结果:")
        print(f"得分: {eval_result.get('score', 'N/A')}")
        print(f"评价: {eval_result.get('reasoning', 'N/A')}")

        # 使用自定义评估器
        length_eval = custom_evaluator.evaluate_length(case["response"])
        structure_eval = custom_evaluator.evaluate_structure(case["response"])

        print(f"\n自定义评估结果:")
        print(f"长度评估: {length_eval['score']:.2f} - {'; '.join(length_eval['feedback'])}")
        print(f"结构评估: {structure_eval['score']:.2f} - {'; '.join(structure_eval['feedback'])}")
        print("-" * 50 + "\n")

def debugging_example():
    """调试示例"""
    print("=== 调试示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.3,
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    debug_chain = DebugChain(llm, verbose=True)

    print("开始调试对话...\n")

    questions = [
        "解释一下什么是机器学习",
        "机器学习有哪些主要类型？",
        "如何开始学习机器学习？"
    ]

    for question in questions:
        print(f"用户: {question}")
        try:
            response = debug_chain.invoke_with_debug(question)
            print(f"AI: {response[:200]}...\n")
        except Exception as e:
            print(f"错误: {e}\n")

    # 显示调试摘要
    summary = debug_chain.get_debug_summary()
    print("=== 调试摘要 ===")
    print(f"总调试日志数: {summary['total_logs']}")
    print(f"对话轮次: {summary['conversation_length']}")

    print("\n调试日志摘要:")
    for log in summary['debug_info'][-5:]:  # 显示最后5条日志
        print(f"[{log['timestamp']}] {log['message']}")

def structured_output_example():
    """结构化输出示例"""
    print("=== 结构化输出示例 ===\n")

    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.1,
        openai_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_base=os.getenv("ANTHROPIC_BASE_URL")
    )

    parser = StructuredOutputParser()

    # 创建带格式要求的提示模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""你是一个专业的AI助手，请以结构化格式回答问题。

{parser.get_format_instructions()}

请确保你的回答准确、详细且有条理。"""),
        ("user", "{input}")
    ])

    chain = prompt_template | llm | parser

    test_questions = [
        "什么是深度学习？请从定义、应用、优缺点三个方面回答。",
        "介绍Python编程语言的主要特点"
    ]

    for question in test_questions:
        print(f"问题: {question}")
        try:
            result = chain.invoke({"input": question})
            print(f"结构化回答:\n{json.dumps(result, ensure_ascii=False, indent=2)}\n")
        except Exception as e:
            print(f"处理出错: {e}\n")

# ========================
# 主函数 (Main Function)
# ========================

def main():
    """主函数"""
    print("LangChain 评估和调试进阶示例\n")
    print("本示例展示如何评估AI应用性能、调试问题和优化系统\n")

    try:
        # 运行各种示例
        performance_monitoring_example()
        print("\n" + "="*60 + "\n")

        quality_evaluation_example()
        print("\n" + "="*60 + "\n")

        debugging_example()
        print("\n" + "="*60 + "\n")

        structured_output_example()

        print("\n=== 学习要点总结 ===")
        print("1. 性能监控：跟踪响应时间、Token使用量、成本等指标")
        print("2. 质量评估：使用标准和自定义评估器检查回答质量")
        print("3. 调试工具：记录详细日志，监控处理流程")
        print("4. 结构化输出：确保AI回答的格式一致性")
        print("5. 错误处理：优雅地处理和记录错误情况")

    except Exception as e:
        print(f"运行示例时出错：{e}")
        print("请确保您的环境配置正确。")

if __name__ == "__main__":
    main()