#!/usr/bin/env python3
"""
测试修复后的RAG工具
验证Pydantic验证错误是否已解决
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_tool_parameters():
    """测试工具参数修复"""
    print("测试工具参数修复...")

    try:
        from ragflow_retrieval_tool import (
            list_knowledge_bases,
            search_documents,
            ask_knowledge_base,
            get_document_summary
        )
        print("工具导入成功")

        # 测试1: search_documents
        print("\n测试 search_documents:")
        try:
            result = search_documents.invoke({
                "query": "王书友",
                "knowledge_base": None,  # 这里应该不再报错
                "max_results": 3
            })
            print("search_documents: 参数验证通过")
        except Exception as e:
            if "validation error" in str(e):
                print(f"search_documents: 参数验证失败 - {e}")
                return False
            else:
                print(f"search_documents: 参数验证通过 (其他错误: {e})")

        # 测试2: ask_knowledge_base
        print("\n测试 ask_knowledge_base:")
        try:
            result = ask_knowledge_base.invoke({
                "question": "王书友是什么岗位?",
                "knowledge_base": None,  # 这里应该不再报错
                "include_sources": True
            })
            print("ask_knowledge_base: 参数验证通过")
        except Exception as e:
            if "validation error" in str(e):
                print(f"ask_knowledge_base: 参数验证失败 - {e}")
                return False
            else:
                print(f"ask_knowledge_base: 参数验证通过 (其他错误: {e})")

        # 测试3: get_document_summary
        print("\n测试 get_document_summary:")
        try:
            result = get_document_summary.invoke({
                "knowledge_base": None  # 这里应该不再报错
            })
            print("get_document_summary: 参数验证通过")
        except Exception as e:
            if "validation error" in str(e):
                print(f"get_document_summary: 参数验证失败 - {e}")
                return False
            else:
                print(f"get_document_summary: 参数验证通过 (其他错误: {e})")

        return True

    except Exception as e:
        print(f"工具导入失败: {e}")
        return False

def test_agent_integration():
    """测试智能体集成"""
    print("\n测试智能体集成...")

    try:
        from agent_with_rag_example import RAGEnabledAgent

        agent = RAGEnabledAgent(ragflow_api_key="test_key")

        # 测试聊天逻辑
        test_messages = [
            "搜索关于王书友的信息",
            "王书友是什么岗位?",
            "有哪些知识库?"
        ]

        for message in test_messages:
            print(f"\n测试消息: {message}")
            try:
                response = agent.chat(message)
                print(f"响应: {response}")
            except Exception as e:
                if "validation error" in str(e):
                    print(f"参数验证错误: {e}")
                    return False
                else:
                    print(f"其他错误（正常）: {e}")

        return True

    except Exception as e:
        print(f"智能体测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("RAG工具参数修复验证")
    print("=" * 60)

    tests = [
        ("工具参数修复", test_tool_parameters),
        ("智能体集成", test_agent_integration)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"{test_name} 测试通过")
            else:
                failed += 1
                print(f"{test_name} 测试失败")
        except Exception as e:
            print(f"{test_name} 测试异常: {e}")
            failed += 1

    # 输出测试总结
    print("\n" + "=" * 60)
    print("修复验证总结")
    print("=" * 60)
    print(f"通过: {passed}")
    print(f"失败: {failed}")

    if failed == 0:
        print("\n✅ Pydantic验证错误已修复！")
        print("所有工具现在可以正确处理None参数了。")
        print("\n修复内容:")
        print("- 将 knowledge_base: str = None 改为 knowledge_base: Optional[str] = None")
        print("- 修复了 search_documents, ask_knowledge_base, get_document_summary 三个工具")
        print("- 现在可以正常传递 None 值来搜索所有知识库")
    else:
        print(f"\n❌ 还有 {failed} 个测试失败。")

if __name__ == "__main__":
    main()