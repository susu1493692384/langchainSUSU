#!/usr/bin/env python3
"""
ChatOpenAI 完整参数参考指南
这个文件包含了 LangChain ChatOpenAI 类的所有可用参数的详细说明和示例
适用于智谱AI (GLM) API 配置，但同样适用于标准 OpenAI API

作者: Claude
版本: 1.0
更新时间: 2025-11-28
"""

import os
import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# 加载环境变量
load_dotenv()

# ================================
# 1. 基础模型参数示例
# ================================

def basic_model_parameters():
    """
    基础模型参数示例
    包括: model, temperature, top_p, max_tokens, n
    """
    print("🔧 === 基础模型参数示例 ===\n")

    # 完整的基础参数配置
    llm = ChatOpenAI(
        # 核心模型参数
        model="glm-4",  # 模型名称，对于智谱AI通常使用 "glm-4"
        temperature=0.7,  # 控制输出随机性: 0.0=确定性, 1.0=标准随机性, 2.0=最大随机性
        top_p=0.9,  # 核采样: 0.1=只考虑前10%概率的tokens, 1.0=考虑所有tokens
        max_tokens=150,  # 生成响应的最大token数量
        n=1,  # 为每个输入生成多少个响应(某些模型支持)

        # API配置
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),
        verbose=True  # 显示详细日志
    )

    message = HumanMessage(content="请简单介绍一下LangChain是什么？")

    try:
        response = llm.invoke([message])
        print(f"问题: {message.content}")
        print(f"回答: {response.content}\n")
        print(f"使用的参数: temperature={llm.temperature}, max_tokens={llm.max_tokens}")
    except Exception as e:
        print(f"调用模型时出错: {e}\n")

# ================================
# 2. 惩罚参数示例
# ================================

def penalty_parameters():
    """
    惩罚参数示例
    包括: frequency_penalty, presence_penalty, logit_bias
    """
    print("⚖️ === 惩罚参数示例 ===\n")

    # 配置惩罚参数
    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.8,
        max_tokens=200,

        # 惩罚参数
        frequency_penalty=0.5,  # 频率惩罚: 正值减少重复，负值增加重复，范围-2到2
        presence_penalty=0.3,   # 存在惩罚: 正值鼓励谈论新话题，范围-2到2

        # API配置
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    message = HumanMessage(content="请重复三次：编程很有趣，编程很有趣，编程很有趣。")

    try:
        response = llm.invoke([message])
        print(f"问题: {message.content}")
        print(f"回答 (应用了惩罚参数): {response.content}\n")
        print(f"使用的参数: frequency_penalty={llm.frequency_penalty}, presence_penalty={llm.presence_penalty}")
    except Exception as e:
        print(f"调用模型时出错: {e}\n")

# ================================
# 3. 输出控制参数示例
# ================================

def output_control_parameters():
    """
    输出控制参数示例
    包括: stop, streaming, logprobs, top_logprobs
    """
    print("🎛️ === 输出控制参数示例 ===\n")

    # 配置输出控制参数
    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=100,

        # 输出控制
        stop=["\n", "回答完毕", "END"],  # 遇到这些序列时停止生成
        streaming=False,  # 是否流式返回响应

        # API配置
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    message = HumanMessage(content="请列举三个编程语言的特点，每行一个。")

    try:
        response = llm.invoke([message])
        print(f"问题: {message.content}")
        print(f"回答: {response.content}\n")
        print(f"使用的停止序列: {llm.stop}")
    except Exception as e:
        print(f"调用模型时出错: {e}\n")

# ================================
# 4. API配置参数示例
# ================================

def api_configuration_parameters():
    """
    API配置参数示例
    包括: timeout, max_retries, organization, custom headers
    """
    print("🌐 === API配置参数示例 ===\n")

    # 配置API参数
    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.5,
        max_tokens=150,

        # API配置
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),
        timeout=30.0,  # 请求超时时间(秒)
        max_retries=3,  # 最大重试次数

        # 自定义配置
        default_headers={  # 默认HTTP头
            "User-Agent": "MyLangChainApp/1.0",
            "X-Custom-Header": "custom-value"
        },

        verbose=True
    )

    message = HumanMessage(content="请解释一下什么是API超时和重试机制？")

    try:
        response = llm.invoke([message])
        print(f"问题: {message.content}")
        print(f"回答: {response.content}\n")
        print(f"API配置: timeout={llm.timeout}s, max_retries={llm.max_retries}")
    except Exception as e:
        print(f"调用模型时出错: {e}\n")

# ================================
# 5. 客户端配置参数示例
# ================================

def client_configuration_parameters():
    """
    客户端配置参数示例
    包括: http_client, http_async_client, custom query params
    """
    print("🔌 === 客户端配置参数示例 ===\n")

    # 自定义HTTP客户端
    custom_client = httpx.Client(
        limits=httpx.Limits(
            max_keepalive_connections=5,
            max_connections=10
        ),
        timeout=httpx.Timeout(10.0, connect=5.0)
    )

    # 配置客户端参数
    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.6,
        max_tokens=120,

        # API配置
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),

        # 客户端配置
        http_client=custom_client,  # 自定义同步HTTP客户端
        default_query={"model_version": "latest"},  # 默认查询参数

        verbose=True
    )

    message = HumanMessage(content="请解释一下HTTP客户端连接池的作用。")

    try:
        response = llm.invoke([message])
        print(f"问题: {message.content}")
        print(f"回答: {response.content}\n")
        print("✅ 使用了自定义HTTP客户端配置")

        # 清理客户端资源
        custom_client.close()
    except Exception as e:
        print(f"调用模型时出错: {e}")
        try:
            custom_client.close()
        except:
            pass

# ================================
# 6. 高级参数示例
# ================================

def advanced_parameters():
    """
    高级参数示例
    包括: seed, model_kwargs, extra_body, disabled_params
    """
    print("🚀 === 高级参数示例 ===\n")

    # 配置高级参数
    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.7,
        max_tokens=150,

        # API配置
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),

        # 高级参数
        seed=42,  # 随机种子，相同种子会得到相同结果
        model_kwargs={  # 传递任何有效的OpenAI API参数
            "response_format": {"type": "text"},  # 响应格式
            # "tools": [...],  # 工具调用
            # "tool_choice": "auto"
        },
        extra_body={  # 向OpenAI兼容API请求中添加额外JSON属性
            "custom_parameter": "custom_value",
            "provider_specific": {"option": true}
        },
        disabled_params={  # 禁用特定模型不支持的参数
            # "parallel_tool_calls": None  # 禁用并行工具调用
        },

        verbose=True
    )

    message = HumanMessage(content="请生成一个关于编程的随机笑话。")

    try:
        response = llm.invoke([message])
        print(f"问题: {message.content}")
        print(f"回答: {response.content}\n")
        print(f"高级参数: seed={llm.seed}")
        print(f"Model kwargs: {llm.model_kwargs}")
    except Exception as e:
        print(f"调用模型时出错: {e}\n")

# ================================
# 7. Responses API参数示例 (适用于新版langchain-openai)
# ================================

def responses_api_parameters():
    """
    Responses API参数示例
    包括: use_responses_api, reasoning, verbosity, include等
    注意: 需要langchain-openai 0.3.24+版本
    """
    print("🆕 === Responses API参数示例 ===\n")

    try:
        # 配置Responses API参数
        llm = ChatOpenAI(
            model="glm-4",
            temperature=0.7,
            max_tokens=150,

            # API配置
            openai_api_key=os.getenv("GLM_API_KEY"),
            openai_api_base=os.getenv("GLM_BASE_URL"),

            # Responses API参数 (如果支持)
            # use_responses_api=True,  # 使用Responses API而非Chat Completions API
            # reasoning={  # 推理模型参数
            #     "effort": "medium",  # "low", "medium", "high"
            #     "summary": "detailed"  # "auto", "concise", "detailed"
            # },
            # verbosity="medium",  # "low", "medium", "high"
            # service_tier="auto",  # "auto", "default", "flex"
            # store=True,  # 是否存储响应数据

            verbose=True
        )

        message = HumanMessage(content="请分析一下人工智能的发展趋势。")

        response = llm.invoke([message])
        print(f"问题: {message.content}")
        print(f"回答: {response.content}\n")
        print("✅ Responses API配置 (如果支持)")

    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            print("⚠️ 当前版本的langchain-openai不支持Responses API参数")
            print("请升级到langchain-openai 0.3.24+版本\n")
        else:
            print(f"调用模型时出错: {e}\n")
    except Exception as e:
        print(f"调用模型时出错: {e}\n")

# ================================
# 8. 完整配置示例
# ================================

def complete_configuration_example():
    """
    完整配置示例
    展示如何组合使用多个参数
    """
    print("🎯 === 完整配置示例 ===\n")

    # 自定义HTTP客户端
    custom_client = httpx.Client(
        limits=httpx.Limits(max_connections=20),
        timeout=httpx.Timeout(60.0)
    )

    # 最完整的配置
    llm = ChatOpenAI(
        # === 核心模型参数 ===
        model="glm-4",
        temperature=0.7,  # 控制随机性
        top_p=0.9,  # 核采样
        max_tokens=300,  # 最大tokens
        n=1,  # 生成响应数量

        # === 惩罚参数 ===
        frequency_penalty=0.1,  # 频率惩罚
        presence_penalty=0.1,   # 存在惩罚

        # === 输出控制 ===
        stop=["\n\n", "===END==="],  # 停止序列
        streaming=False,  # 非流式

        # === API配置 ===
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL"),
        timeout=60.0,  # 超时时间
        max_retries=5,  # 最大重试

        # === 客户端配置 ===
        http_client=custom_client,  # 自定义客户端
        default_headers={
            "User-Agent": "LangChainCompleteExample/1.0",
            "X-Request-ID": "complete-example"
        },
        default_query={"version": "v1"},

        # === 高级参数 ===
        seed=12345,  # 随机种子
        model_kwargs={
            "response_format": {"type": "text"}
        },
        extra_body={
            "custom_provider_config": {
                "optimization": true
            }
        },

        # === 其他参数 ===
        tiktoken_model_name="glm-4",  # 用于token计算的模型名称
        include_response_headers=True,  # 包含响应头

        verbose=True
    )

    messages = [
        SystemMessage(content="你是一个专业的AI助手,擅长技术解释。"),
        HumanMessage(content="请详细解释LangChain的核心概念,包括Chains、Agents和Memory。")
    ]

    try:
        response = llm.invoke(messages)
        print("系统消息:", messages[0].content)
        print("用户消息:", messages[1].content)
        print("\nAI回答:")
        print("=" * 50)
        print(response.content)
        print("=" * 50)

        # 显示响应元数据
        if hasattr(response, 'response_metadata'):
            print(f"\n响应元数据: {response.response_metadata}")

        print(f"\n✅ 完整配置示例成功执行")
        print(f"使用参数总结:")
        print(f"  - 模型: {llm.model}")
        print(f"  - 温度: {llm.temperature}")
        print(f"  - 最大tokens: {llm.max_tokens}")
        print(f"  - 种子: {llm.seed}")
        print(f"  - 超时: {llm.timeout}s")
        print(f"  - 最大重试: {llm.max_retries}")

    except Exception as e:
        print(f"❌ 调用模型时出错: {e}")
    finally:
        # 清理资源
        custom_client.close()

# ================================
# 9. 参数对比示例
# ================================

def parameter_comparison_example():
    """
    参数对比示例
    展示不同参数值对输出结果的影响
    """
    print("⚖️ === 参数对比示例 ===\n")

    question = "请用创意的方式描述一下编程的乐趣。"
    message = HumanMessage(content=question)

    # 低温度配置 - 更确定性
    llm_deterministic = ChatOpenAI(
        model="glm-4",
        temperature=0.1,  # 低温度
        max_tokens=150,
        frequency_penalty=0.0,  # 无惩罚
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 高温度配置 - 更创意性
    llm_creative = ChatOpenAI(
        model="glm-4",
        temperature=1.2,  # 高温度
        max_tokens=150,
        frequency_penalty=0.3,  # 有频率惩罚
        presence_penalty=0.2,    # 有存在惩罚
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    try:
        print(f"问题: {question}\n")

        # 确定性回答
        print("📊 低温度配置 (temperature=0.1, 无惩罚):")
        print("-" * 40)
        response1 = llm_deterministic.invoke([message])
        print(response1.content)
        print()

        # 创意性回答
        print("🎨 高温度配置 (temperature=1.2, 有惩罚):")
        print("-" * 40)
        response2 = llm_creative.invoke([message])
        print(response2.content)
        print()

        print("💡 对比分析:")
        print("  - 低温度: 输出更稳定、可预测，适合事实性回答")
        print("  - 高温度: 输出更有创意、多样性，适合创意性任务")
        print("  - 惩罚参数: 可以减少重复内容，鼓励更多样化表达")

    except Exception as e:
        print(f"❌ 参数对比示例出错: {e}")

# ================================
# 主函数 - 运行所有示例
# ================================

def main():
    """
    主函数 - 运行所有参数示例
    """
    import sys
    import io

    # 设置UTF-8编码输出
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("🤖 ChatOpenAI 完整参数参考指南")
    print("=" * 60)
    print()

    # 检查环境变量
    if not os.getenv("GLM_API_KEY") or not os.getenv("GLM_BASE_URL"):
        print("⚠️ 警告: 未找到GLM_API_KEY或GLM_BASE_URL环境变量")
        print("请确保在.env文件中设置了正确的智谱AI API配置")
        print("示例.env文件内容:")
        print("GLM_API_KEY=your_api_key_here")
        print("GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/")
        print()
        return

    # 运行各个示例
    examples = [
        ("基础模型参数", basic_model_parameters),
        ("惩罚参数", penalty_parameters),
        ("输出控制参数", output_control_parameters),
        ("API配置参数", api_configuration_parameters),
        ("客户端配置参数", client_configuration_parameters),
        ("高级参数", advanced_parameters),
        ("Responses API参数", responses_api_parameters),
        ("完整配置示例", complete_configuration_example),
        ("参数对比示例", parameter_comparison_example)
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
    print("✨ ChatOpenAI 参数参考指南结束！")
    print("="*60)
    print()
    print("📚 更多信息:")
    print("  - LangChain文档: https://python.langchain.com/")
    print("  - ChatOpenAI API参考: https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html")
    print("  - OpenAI API文档: https://platform.openai.com/docs/api-reference/chat")
    print()
    print("💡 提示:")
    print("  - 根据你的具体需求选择合适的参数组合")
    print("  - 生产环境中建议设置适当的超时和重试机制")
    print("  - 使用seed参数可以确保结果的可重现性")
    print("  - 通过model_kwargs可以传递任何新的API参数")

if __name__ == "__main__":
    main()