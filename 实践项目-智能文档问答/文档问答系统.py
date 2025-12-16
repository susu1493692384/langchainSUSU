#- 智能记忆管理对话上下文
#- 结合文档检索提供准确答案
#- 实现多轮对话理解和动态相关性评分
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
from collections import Counter, defaultdict
from dotenv import load_dotenv
import math
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import time
from dataclasses import dataclass
# 加载环境变量
load_dotenv()

def window_memory_example():
    """窗口记忆示例 - 简化版本"""
    print("=== 窗口记忆示例 ===")

    llm = ChatOpenAI(
        model="glm-4.6",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建消息历史（手动管理窗口）
    message_history = InMemoryChatMessageHistory()
    window_size = 3  # 保留最近3轮对话

    def get_recent_messages():
        """获取最近的消息"""
        messages = message_history.messages
        # 保留最近6条消息（用户+AI各3条）
        return messages[-window_size*2:] if len(messages) > window_size*2 else messages

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是一个友好的AI助手。你只能记住最近{window_size}轮对话。请根据最近的对话历史回答用户的问题。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 创建链
    chain = prompt | llm | StrOutputParser()
    
    print("请输入对话内容（输入 'exit' 结束对话）：")
    for i in range(1, 6):  # 进行5轮对话
        user_input = input(f"轮次 {i} - 用户: ")
        try:
            # 添加用户消息
            message_history.add_user_message(user_input)

            # 获取最近消息
            recent_messages = get_recent_messages()

            # 调用链
            response = chain.invoke({
                "input": user_input,
                "history": recent_messages
            })

            # 添加AI回复
            message_history.add_ai_message(response)

            print(f"轮次 {i}:")
            print(f"用户: {user_input}")
            print(f"AI助手: {response}\n")

        except Exception as e:
            print(f"处理对话轮次 {i} 时出错: {e}")

        if user_input.lower() == 'exit':
            break

def create_sample_documents():
    """创建示例文档"""
    sample_texts = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子集，它使计算机能够在没有被明确编程的情况下学习和改进。",
        "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。",
        "自然语言处理是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
        "计算机视觉是人工智能的一个分支，使计算机能够从数字图像或视频中获取信息。"
    ]

    documents = []
    for i, text in enumerate(sample_texts):
        doc = Document(
            page_content=text,
            metadata={"source": f"文档_{i+1}"}
        )
        documents.append(doc)

    return documents

def text_splitting_example(documents):
    """文本分割示例"""
    print("=== 文本分割示例 ===\n")

    # 字符分割器
    char_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )

    char_chunks = char_splitter.split_documents(documents)
    print(f"字符分割器生成了 {len(char_chunks)} 个文档块")

    for i, chunk in enumerate(char_chunks, 1):
        print(f"文档块 {i}: {chunk.page_content[:100]}...")

    return char_chunks

class TFIDFRetriever:
    """基于TF-IDF的文档检索器"""

    def __init__(self):
        self.documents = []
        self.word_to_doc_count = defaultdict(int)  # 词出现在多少文档中
        self.doc_word_counts = []  # 每个文档的词频统计
        self.doc_word_tfidf = []  # 每个文档的TF-IDF向量
        self.total_docs = 0

    def add_documents(self, documents):
        """添加文档并构建TF-IDF索引"""
        self.documents = documents
        self.total_docs = len(documents)

        # 统计每个词在多少文档中出现过
        all_words = set()
        for doc in documents:
            words = self._extract_words(doc.page_content)
            unique_words = set(words)
            all_words.update(unique_words)
            for word in unique_words:
                self.word_to_doc_count[word] += 1

        # 计算每个文档的词频和TF-IDF
        self.doc_word_counts = []
        self.doc_word_tfidf = []

        for doc in documents:
            words = self._extract_words(doc.page_content)
            word_count = Counter(words)
            self.doc_word_counts.append(word_count)

            # 计算TF-IDF
            tfidf = {}
            doc_length = len(words)

            for word, count in word_count.items():
                # TF (term frequency)
                tf = count / doc_length

                # IDF (inverse document frequency)
                idf = math.log(self.total_docs / self.word_to_doc_count[word])

                # TF-IDF
                tfidf[word] = tf * idf

            self.doc_word_tfidf.append(tfidf)

    def _extract_words(self, text):
        """提取文本中的词汇"""
        # 使用正则表达式提取中文词汇和英文词汇
        chinese_words = re.findall(r'[\u4e00-\u9fff]+', text.lower())
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        return chinese_words + english_words

    def _get_query_tfidf(self, query):
        """计算查询的TF-IDF向量"""
        query_words = self._extract_words(query)
        word_count = Counter(query_words)
        query_length = len(query_words)

        tfidf = {}
        for word, count in word_count.items():
            # TF
            tf = count / query_length if query_length > 0 else 0

            # IDF (如果词在文档中出现过)
            if word in self.word_to_doc_count:
                idf = math.log(self.total_docs / self.word_to_doc_count[word])
            else:
                idf = 0

            tfidf[word] = tf * idf

        return tfidf

    def _cosine_similarity(self, tfidf1, tfidf2):
        """计算两个TF-IDF向量的余弦相似度"""
        if not tfidf1 or not tfidf2:
            return 0

        # 找到共同的词汇
        common_words = set(tfidf1.keys()) & set(tfidf2.keys())

        if not common_words:
            return 0

        # 计算点积
        dot_product = sum(tfidf1[word] * tfidf2[word] for word in common_words)

        # 计算向量的模长
        norm1 = math.sqrt(sum(tfidf1[word] ** 2 for word in tfidf1))
        norm2 = math.sqrt(sum(tfidf2[word] ** 2 for word in tfidf2))

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    def search(self, query, top_k=3):
        """搜索与查询最相似的文档"""
        if not self.documents:
            return []

        query_tfidf = self._get_query_tfidf(query)

        # 计算相似度
        similarities = []
        for i, doc_tfidf in enumerate(self.doc_word_tfidf):
            similarity = self._cosine_similarity(query_tfidf, doc_tfidf)
            similarities.append((self.documents[i], similarity))

        # 按相似度排序并返回top_k个文档
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in similarities[:top_k]]

@dataclass
class MemoryNode:
    """记忆节点数据结构"""
    content: str
    importance: float
    timestamp: float
    topic: str
    referenced_docs: List[str]
    entity_mentions: Set[str]

class IntelligentMemoryManager:
    """智能记忆管理器"""

    def __init__(self, max_memory_nodes: int = 50, importance_threshold: float = 0.3):
        self.memory_nodes: List[MemoryNode] = []
        self.topic_tracker: Dict[str, float] = {}
        self.entity_graph: Dict[str, Set[str]] = defaultdict(set)
        self.conversation_topics: List[Tuple[str, float]] = []
        self.max_memory_nodes = max_memory_nodes
        self.importance_threshold = importance_threshold
        self.topic_decay_rate = 0.9

    def extract_entities_and_topics(self, text: str) -> Tuple[Set[str], str]:
        """从文本中提取实体和主题"""
        # 提取中文实体（名词性词汇）
        chinese_entities = set(re.findall(r'[\u4e00-\u9fff]{2,}(?:系统|技术|方法|模型|算法)', text))

        # 提取英文实体
        english_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

        # 提取主题词（更通用的词汇）
        topic_words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text.lower())

        # 过滤停用词
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '这个', '那个', 'can', 'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at'}
        topic_words = [word for word in topic_words if word not in stop_words and len(word) > 1]

        # 确定主要主题
        if topic_words:
            topic_counter = Counter(topic_words)
            main_topic = topic_counter.most_common(1)[0][0]
        else:
            main_topic = "general"

        all_entities = chinese_entities.union(english_entities)
        return all_entities, main_topic

    def calculate_importance(self, content: str, entities: Set[str], current_topics: Dict[str, float]) -> float:
        """计算记忆节点的重要性"""
        importance = 0.5  # 基础重要性

        # 实体权重
        for entity in entities:
            if entity in self.entity_graph:
                importance += 0.2

        # 话题相关性权重
        for topic, weight in current_topics.items():
            if topic in content.lower():
                importance += weight * 0.3

        # 内容长度权重（假设更详细的内容更重要）
        importance += min(len(content) / 500, 0.2)

        return min(importance, 1.0)

    def add_memory_node(self, content: str, referenced_docs: List[str] = None):
        """添加记忆节点"""
        entities, main_topic = self.extract_entities_and_topics(content)

        # 更新话题追踪
        self.update_topic_tracking(main_topic)

        # 更新实体图谱
        for entity in entities:
            self.entity_graph[entity].add(main_topic)

        # 计算重要性
        importance = self.calculate_importance(content, entities, self.topic_tracker)

        if importance < self.importance_threshold:
            return  # 重要性太低，不存储

        memory_node = MemoryNode(
            content=content,
            importance=importance,
            timestamp=time.time(),
            topic=main_topic,
            referenced_docs=referenced_docs or [],
            entity_mentions=entities
        )

        self.memory_nodes.append(memory_node)

        # 维护记忆节点数量限制
        if len(self.memory_nodes) > self.max_memory_nodes:
            self._cleanup_old_memories()

    def update_topic_tracking(self, topic: str):
        """更新话题追踪"""
        # 衰减现有话题权重
        for t in self.topic_tracker:
            self.topic_tracker[t] *= self.topic_decay_rate

        # 增加当前话题权重
        self.topic_tracker[topic] = min(self.topic_tracker.get(topic, 0) + 0.3, 1.0)

        # 记录话题变化
        self.conversation_topics.append((topic, time.time()))

        # 保持话题历史在合理范围内
        if len(self.conversation_topics) > 20:
            self.conversation_topics.pop(0)

    def _cleanup_old_memories(self):
        """清理不重要的记忆节点"""
        # 按重要性和时间排序
        self.memory_nodes.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)

        # 保留最重要的节点
        self.memory_nodes = self.memory_nodes[:self.max_memory_nodes]

    def get_relevant_context(self, query: str, current_topic: str = None, max_context: int = 3) -> List[str]:
        """获取与查询相关的上下文"""
        query_entities, query_topic = self.extract_entities_and_topics(query)

        relevant_memories = []

        for memory in self.memory_nodes:
            relevance_score = 0

            # 实体匹配
            entity_overlap = len(memory.entity_mentions & query_entities)
            relevance_score += entity_overlap * 0.4

            # 主题匹配
            if memory.topic == query_topic or memory.topic == current_topic:
                relevance_score += 0.3

            # 时间衰减（最近的记忆更相关）
            time_factor = math.exp(-(time.time() - memory.timestamp) / 3600)  # 1小时衰减
            relevance_score += time_factor * 0.2

            # 重要性权重
            relevance_score += memory.importance * 0.1

            relevant_memories.append((memory.content, relevance_score))

        # 排序并返回最相关的上下文
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [content for content, _ in relevant_memories[:max_context]]

    def get_topic_weights(self) -> Dict[str, float]:
        """获取当前话题权重"""
        return self.topic_tracker.copy()

class EnhancedDocumentRetriever:
    """增强的文档检索器 - 结合TF-IDF和动态相关性评分"""

    def __init__(self, base_retriever: TFIDFRetriever, memory_manager: IntelligentMemoryManager):
        self.base_retriever = base_retriever
        self.memory_manager = memory_manager
        self.doc_topic_weights: Dict[str, Dict[str, float]] = {}
        self.conversation_history_docs: List[str] = []

    def calculate_dynamic_relevance(self, query: str, document: Document,
                                   context_topics: Dict[str, float], referenced_docs: List[str]) -> float:
        """计算动态相关性评分"""
        # 基础TF-IDF相似度
        base_score = 0
        try:
            relevant_chunks = self.base_retriever.search(query, top_k=1)
            if document in relevant_chunks:
                base_score = 0.8  # 基础检索命中
        except:
            base_score = 0

        # 上下文话题相关性
        topic_score = 0
        doc_content = document.page_content.lower()
        for topic, weight in context_topics.items():
            if topic in doc_content:
                topic_score += weight * 0.3

        # 文档历史引用权重
        history_score = 0
        doc_id = f"{document.metadata.get('source', '')}_{document.metadata.get('paragraph', 0)}"
        if doc_id in referenced_docs:
            history_score += 0.2  # 之前引用过的文档
        if doc_id in self.conversation_history_docs[-3:]:  # 最近引用的文档
            history_score += 0.3

        # 实体匹配权重
        entities, _ = self.memory_manager.extract_entities_and_topics(query)
        entity_score = 0
        for entity in entities:
            if entity in doc_content:
                entity_score += 0.1

        # 综合评分
        final_score = (base_score * 0.4 +
                      topic_score * 0.3 +
                      history_score * 0.2 +
                      entity_score * 0.1)

        return min(final_score, 1.0)

    def enhanced_search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """增强的文档搜索"""
        # 获取基础检索结果
        base_results = self.base_retriever.search(query, top_k=top_k * 2)

        # 获取当前上下文
        context_topics = self.memory_manager.get_topic_weights()
        referenced_docs = [doc_id for memory in self.memory_manager.memory_nodes for doc_id in memory.referenced_docs]

        # 计算动态相关性评分
        scored_results = []
        for doc in base_results:
            score = self.calculate_dynamic_relevance(query, doc, context_topics, referenced_docs)
            scored_results.append((doc, score))

        # 按动态评分排序
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # 记录引用的文档
        for doc, _ in scored_results[:top_k]:
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('paragraph', 0)}"
            self.conversation_history_docs.append(doc_id)

        # 保持历史记录在合理范围内
        if len(self.conversation_history_docs) > 10:
            self.conversation_history_docs = self.conversation_history_docs[-10:]

        return scored_results[:top_k]

def simple_text_similarity(text1, text2):
    """简单的文本相似度计算（保留作为备用）"""
    # 使用Jaccard相似度
    words1 = set(re.findall(r'[\w\u4e00-\u9fff]+', text1.lower()))
    words2 = set(re.findall(r'[\w\u4e00-\u9fff]+', text2.lower()))

    if not words1 or not words2:
        return 0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)

def find_similar_chunks(query, chunks, top_k=2):
    """根据查询找到最相似的文本块"""
    similarities = []

    for chunk in chunks:
        similarity = simple_text_similarity(query, chunk.page_content)
        similarities.append((chunk, similarity))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 返回最相似的top_k个文本块
    return [chunk for chunk, _ in similarities[:top_k]]

class IntelligentDocumentQA:
    """智能文档问答系统 - 结合记忆管理和动态检索"""

    def __init__(self, model_name: str = "glm-4.6", temperature: float = 0.7):
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("GLM_API_KEY"),
            openai_api_base=os.getenv("GLM_BASE_URL")
        )

        # 初始化记忆管理器
        self.memory_manager = IntelligentMemoryManager(
            max_memory_nodes=50,
            importance_threshold=0.3
        )

        # 初始化文档系统
        self.base_retriever = TFIDFRetriever()
        self.enhanced_retriever = None  # 将在加载文档后初始化
        self.documents = []
        self.document_chunks = []

        # 对话历史
        self.message_history = InMemoryChatMessageHistory()

        # 创建增强的提示模板
        self.create_enhanced_prompts()

    def create_enhanced_prompts(self):
        """创建增强的提示模板系统"""
        # 基础文档问答提示
        self.base_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能文档助手，具有强大的上下文理解能力。

指导原则：
1. 仔细分析用户的当前问题，考虑对话历史上下文
2. 基于提供的文档内容给出准确、详细的回答
3. 当文档中信息不足时，明确说明并提供合理的推断
4. 保持回答的连贯性和一致性，与之前的对话保持逻辑一致
5. 引用具体的文档内容来支持你的观点
6. 适当提及之前讨论过的相关概念

检索到的相关文档：
{retrieved_docs}

对话上下文记忆：
{context_memory}

当前主要话题：
{current_topics}"""),
            ("human", "{question}")
        ])

        # 引用消解提示
        self.reference_resolution_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的对话上下文分析器。请分析用户的当前问题，识别其中的代词、引用和省略，并转换为完整的问题。

分析要点：
- 识别"它"、"这个"、"那个"、"上述"等代词的具体指代
- 理解"刚才提到的"、"前面说的"等时间引用
- 补充用户省略的上下文信息
- 考虑当前对话主题和之前的文档引用

对话历史：
{conversation_history}

请将用户的当前问题转换为完整、明确的问题："""),
            ("human", "{current_question}")
        ])

        # 记忆更新提示
        self.memory_update_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是对话记忆分析器。请分析这段对话，提取重要信息用于记忆更新。

提取要点：
1. 用户询问的主要概念和实体
2. 涉及的重要文档内容
3. 用户的潜在兴趣点和偏好
4. 对话的主题演变

请以JSON格式返回关键信息：
{
    "main_concepts": ["概念1", "概念2"],
    "referenced_documents": ["文档1", "文档2"],
    "user_interests": ["兴趣点1", "兴趣点2"],
    "topic_evolution": "话题变化描述"
}"""),
            ("human", "用户问题：{user_question}\nAI回答：{ai_answer}")
        ])

    def load_documents(self, file_path: str = None):
        """加载文档并初始化检索系统"""
        if file_path and os.path.exists(file_path):
            self.documents = load_local_documents(file_path)
        else:
            self.documents = create_sample_documents()

        # 文本分割
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )
        self.document_chunks = splitter.split_documents(self.documents)

        # 初始化TF-IDF检索器
        self.base_retriever.add_documents(self.document_chunks)

        # 初始化增强检索器
        self.enhanced_retriever = EnhancedDocumentRetriever(
            self.base_retriever,
            self.memory_manager
        )

        print(f"已加载 {len(self.documents)} 个文档，分割为 {len(self.document_chunks)} 个片段")

    def resolve_references(self, question: str) -> str:
        """解析用户问题中的引用"""
        if not self.message_history.messages:
            return question

        # 获取最近的对话历史
        recent_messages = self.message_history.messages[-6:]  # 最近3轮对话

        conversation_history = ""
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                conversation_history += f"用户：{msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_history += f"助手：{msg.content}\n"

        # 使用LLM解析引用
        try:
            chain = self.reference_resolution_prompt | self.llm | StrOutputParser()
            resolved_question = chain.invoke({
                "current_question": question,
                "conversation_history": conversation_history
            })
            return resolved_question.strip()
        except Exception as e:
            print(f"引用解析失败，使用原问题: {e}")
            return question

    def find_relevant_documents(self, question: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """查找相关文档"""
        if not self.enhanced_retriever:
            return []

        # 使用增强检索器
        scored_docs = self.enhanced_retriever.enhanced_search(question, top_k=top_k)
        return scored_docs

    def get_context_memory(self, question: str) -> str:
        """获取相关记忆上下文"""
        # 获取当前话题
        _, current_topic = self.memory_manager.extract_entities_and_topics(question)

        # 获取相关记忆
        relevant_contexts = self.memory_manager.get_relevant_context(
            question,
            current_topic,
            max_context=3
        )

        if not relevant_contexts:
            return "无相关记忆上下文"

        context_text = ""
        for i, context in enumerate(relevant_contexts, 1):
            context_text += f"记忆{i}: {context[:200]}...\n"

        return context_text

    def generate_answer(self, question: str, relevant_docs: List[Tuple[Document, float]]) -> str:
        """生成回答"""
        # 格式化检索到的文档
        retrieved_docs = ""
        for i, (doc, score) in enumerate(relevant_docs, 1):
            source_info = f"[来源: {doc.metadata.get('source', '未知')}, 相关性: {score:.2f}]"
            retrieved_docs += f"{i}. {source_info}\n{doc.page_content}\n\n"

        # 获取上下文记忆
        context_memory = self.get_context_memory(question)

        # 获取当前话题权重
        topic_weights = self.memory_manager.get_topic_weights()
        current_topics = ", ".join([f"{k}:{v:.2f}" for k, v in topic_weights.items()])

        # 生成回答
        try:
            chain = self.base_qa_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "question": question,
                "retrieved_docs": retrieved_docs,
                "context_memory": context_memory,
                "current_topics": current_topics
            })
            return response.strip()
        except Exception as e:
            return f"生成回答时出错: {e}"

    def update_memory(self, question: str, answer: str, referenced_docs: List[str]):
        """更新记忆系统"""
        # 提取引用的文档ID
        referenced_doc_ids = []
        for doc in referenced_docs:
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('paragraph', 0)}"
            referenced_doc_ids.append(doc_id)

        # 添加记忆节点
        conversation_content = f"用户问题：{question}\nAI回答：{answer[:200]}..."
        self.memory_manager.add_memory_node(
            conversation_content,
            referenced_doc_ids
        )

    def ask_question(self, question: str) -> str:
        """处理用户问题"""
        print(f"\n用户问题：{question}")

        # 1. 解析引用
        resolved_question = self.resolve_references(question)
        if resolved_question != question:
            print(f"解析后的问题：{resolved_question}")

        # 2. 查找相关文档
        relevant_docs = self.find_relevant_documents(resolved_question)

        if not relevant_docs:
            response = "抱歉，我没有找到相关的文档内容来回答您的问题。"
            print(f"AI回答：{response}")
            return response

        print(f"找到 {len(relevant_docs)} 个相关文档片段")

        # 3. 生成回答
        answer = self.generate_answer(resolved_question, relevant_docs)

        # 4. 更新记忆
        self.update_memory(question, answer, [doc for doc, _ in relevant_docs])

        # 5. 添加到对话历史
        self.message_history.add_user_message(question)
        self.message_history.add_ai_message(answer)

        print(f"AI回答：{answer}")
        return answer

    def start_conversation(self):
        """开始对话"""
        print("=== 智能文档问答系统 ===")
        print("我具备了智能记忆管理和动态文档检索能力")
        print("我可以记住我们的对话内容，并根据上下文智能检索相关文档")
        print("请输入您的问题（输入 'exit' 结束对话）：")

        while True:
            print(f"\n=== 记忆系统统计 ===")
            print(f"记忆节点数量: {len(self.memory_manager.memory_nodes)}")
            print(f"当前活跃话题: {list(self.memory_manager.topic_tracker.keys())}")
            print(f"实体图谱大小: {len(self.memory_manager.entity_graph)}")
            print(f"对话话题历史: {len(self.memory_manager.conversation_topics)} 个话题")
            user_input = input("\n用户: ").strip()


            if user_input.lower() == 'exit':
                print("感谢使用智能文档问答系统！")
                break

            if not user_input:
                continue

            try:
                self.ask_question(user_input)
            except Exception as e:
                print(f"处理问题时出错: {e}")

    def get_memory_stats(self):
        """获取记忆统计信息"""
        print(f"\n=== 记忆系统统计 ===")
        print(f"记忆节点数量: {len(self.memory_manager.memory_nodes)}")
        print(f"当前活跃话题: {list(self.memory_manager.topic_tracker.keys())}")
        print(f"实体图谱大小: {len(self.memory_manager.entity_graph)}")
        print(f"对话话题历史: {len(self.memory_manager.conversation_topics)} 个话题")

def load_local_documents(file_path):
    """加载本地文档"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 简单按段落分割文档
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        documents = []
        for i, paragraph in enumerate(paragraphs):
            doc = Document(
                page_content=paragraph,
                metadata={"source": file_path, "paragraph": i+1}
            )
            documents.append(doc)

        return documents
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在，使用示例文档")
        return create_sample_documents()
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return create_sample_documents()

def document_retrieval_qa():
    """文档检索问答系统"""
    print("=== 文档检索问答系统 ===\n")

    # 尝试加载本地文档，如果不存在则使用示例文档
    file_path = "local_documents.txt"  # 你可以修改为你的文档路径
    documents = load_local_documents(file_path)
    print(f"加载了 {len(documents)} 个文档段落")

    # 文本分割
    chunks = text_splitting_example(documents)

    # 创建LLM
    llm = ChatOpenAI(
        model="glm-4.6",
        temperature=0.7,
        openai_api_key=os.getenv("GLM_API_KEY"),
        openai_api_base=os.getenv("GLM_BASE_URL")
    )

    # 创建包含文档检索的提示词模板
    retrieval_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能文档助手。请根据以下检索到的相关文档内容来回答用户问题。

回答要求：
1. 基于提供的文档内容回答
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、有条理
4. 可以引用具体的文档内容来支持你的回答

检索到的相关文档：
{retrieved_docs}"""),
        ("human", "{question}")
    ])

    # 创建链
    chain = retrieval_prompt | llm | StrOutputParser()

    print("\n=== 开始文档问答 ===")
    print("你可以询问关于文档内容的任何问题（输入 'exit' 结束）")

    while True:
        user_input = input("\n请输入你的问题: ").strip()

        if user_input.lower() == 'exit':
            break

        if not user_input:
            continue

        try:
            # 查找最相关的文档块
            relevant_chunks = find_similar_chunks(user_input, chunks, top_k=3)

            if not relevant_chunks:
                print("没有找到相关的文档内容。")
                continue

            # 格式化检索到的文档
            retrieved_docs = ""
            for i, chunk in enumerate(relevant_chunks, 1):
                source_info = f"[来源: {chunk.metadata['source']}, 段落: {chunk.metadata.get('paragraph', i)}]"
                retrieved_docs += f"{i}. {source_info}\n{chunk.page_content}\n\n"

            print(f"\n找到 {len(relevant_chunks)} 个相关文档片段:")
            for i, chunk in enumerate(relevant_chunks, 1):
                print(f"片段 {i}: {chunk.page_content[:100]}...")

            # 调用链生成回答
            response = chain.invoke({
                "question": user_input,
                "retrieved_docs": retrieved_docs
            })

            print(f"\nAI回答: {response}")

        except Exception as e:
            print(f"处理问题时出错: {e}")

def create_sample_document_file():
    """创建示例文档文件"""
    sample_content = """人工智能技术概述

人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。AI系统可以学习、推理、感知、理解语言，并做出决策。

机器学习
机器学习是人工智能的一个子集，它使计算机能够在没有被明确编程的情况下学习和改进。机器学习算法通过分析数据来识别模式，并使用这些模式来对新数据做出预测或决策。

深度学习
深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。深度学习在图像识别、自然语言处理和语音识别等领域取得了重大突破。

自然语言处理
自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。NLP使计算机能够理解、解释和生成人类语言。

计算机视觉
计算机视觉是人工智能的一个分支，使计算机能够从数字图像或视频中获取有意义的信息。计算机视觉系统可以识别物体、检测面孔、分析场景等。

应用领域
人工智能技术在各个领域都有广泛应用，包括医疗诊断、自动驾驶、金融分析、推荐系统、智能助手等。随着技术的发展，AI将继续改变我们的生活方式和工作方式。"""

    try:
        with open("local_documents.txt", "w", encoding="utf-8") as file:
            file.write(sample_content)
        print("已创建示例文档文件 'local_documents.txt'")
    except Exception as e:
        print(f"创建示例文档文件时出错: {e}")

def intelligent_document_qa_demo():
    """智能文档问答系统演示"""
    print("=== 智能文档问答系统演示 ===")

    # 创建智能文档问答系统
    qa_system = IntelligentDocumentQA()

    # 加载文档
    print("正在加载文档...")
    qa_system.load_documents("local_documents.txt")  # 会自动创建示例文档如果不存在

    # 显示记忆统计
    qa_system.get_memory_stats()

    # 开始对话
    qa_system.start_conversation()

if __name__ == "__main__":
    print("选择要运行的示例:")
    print("1. 传统窗口记忆对话示例")
    print("2. 传统文档检索问答示例")
    print("3. 创建示例文档文件")
    print("4. 智能文档问答系统（推荐）- 整合记忆管理和动态检索")

    choice = input("请输入选择 (1/2/3/4): ").strip()

    if choice == "1":
        window_memory_example()
    elif choice == "2":
        document_retrieval_qa()
    elif choice == "3":
        create_sample_document_file()
        print("文档文件创建完成，现在可以运行选项2或选项4来测试文档检索功能")
    elif choice == "4":
        intelligent_document_qa_demo()
    else:
        print("无效选择，运行智能文档问答系统")
        intelligent_document_qa_demo()