#!/usr/bin/env python3
"""
Test the fixed retriever
"""

from ragflow_langchain_integration import RAGFlowAPIConnector, RAGFlowRetriever

def test_fixed_retriever():
    """Test the fixed retriever"""
    print("=== Testing Fixed RAGFlow Retriever ===")

    # Create connector
    connector = RAGFlowAPIConnector(base_url="http://localhost:9000")

    # Get knowledge bases
    kbs = connector.get_knowledge_bases()
    kb = kbs[0]
    kb_id = kb.get('id')
    print(f"Using knowledge base ID: {kb_id}")

    # Create retriever
    retriever = RAGFlowRetriever(
        connector=connector,
        kb_name=kb_id,
        top_k=3,
        similarity_threshold=0.1
    )

    # Test search
    test_query = "王书友"
    print(f"Query: {test_query}")

    try:
        docs = retriever.get_relevant_documents(test_query)
        print(f"Found {len(docs)} documents:")

        for i, doc in enumerate(docs, 1):
            score = doc.metadata.get("score", 0.0)
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "")
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content

            print(f"\nDocument {i}:")
            print(f"  Score: {score:.3f}")
            print(f"  Source: {source}")
            print(f"  Title: {title}")
            print(f"  Content: {content_preview}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_retriever()