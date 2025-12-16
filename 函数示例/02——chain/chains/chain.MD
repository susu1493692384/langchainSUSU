angChain 中的 Chain（链）有很多种类型，让我为您详细介绍：      

  1. 基础链类型

  LLMChain

  from langchain.chains import LLMChain

  # 最基础的链
  chain = LLMChain(
      llm=llm,
      prompt=prompt
  )
  result = chain.run({"input": "数据"})

  SimpleSequentialChain

  from langchain.chains import SimpleSequentialChain

  # 顺序执行多个链
  chain = SimpleSequentialChain(
      chains=[chain1, chain2, chain3]
  )
  result = chain.run("输入数据")

  SequentialChain

  from langchain.chains import SequentialChain

  # 复杂的顺序链，支持多个输入输出
  complex_chain = SequentialChain(
      chains=[prompt_chain, summary_chain, translation_chain],    
      input_variables=["text"],
      output_variables=["summary", "translation"]
  )

  2. 现代链式语法 (LCEL)

  基本管道链

  from langchain_core.prompts import PromptTemplate
  from langchain_core.output_parsers import StrOutputParser       

  # 使用管道操作符
  chain = prompt | llm | StrOutputParser()
  result = chain.invoke({"input": "数据"})

  复杂组合链

  # 您代码中的例子
  story_chain = (
      {"character": RunnablePassthrough()}
      | story_prompt
      | llm
      | StrOutputParser()
      | (lambda story: {"story_summary": story})
  )

  3. 特殊功能链

  ConversationChain

  from langchain.chains import ConversationChain

  # 对话链
  conversation = ConversationChain(
      llm=llm,
      memory=memory,
      prompt=chat_prompt
  )

  RetrievalQA

  from langchain.chains import RetrievalQA

  # 检索问答链
  qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      retriever=retriever,
      chain_type="stuff"  # 或 "map_reduce", "refine"
  )

  SummarizationChain

  from langchain.chains.summarize import load_summarize_chain     

  # 文档摘要链
  summary_chain = load_summarize_chain(
      llm=llm,
      chain_type="map_reduce"
  )

  4. 文档处理链

  load_qa_chain

  from langchain.chains.question_answering import
  load_qa_chain

  # 问答链
  qa_chain = load_qa_chain(
      llm=llm,
      chain_type="stuff"
  )

  result = qa_chain({
      "input_documents": documents,
      "question": question
  })

  TransformChain

  from langchain.chains import TransformChain

  # 数据转换链
  def text_transform(inputs):
      text = inputs["text"]
      return {"transformed_text": text.upper()}

  transform_chain = TransformChain(
      input_variables=["text"],
      output_variables=["transformed_text"],
      transform=text_transform
  )

  5. 路由链

  LLMRouterChain

  from langchain.chains import LLMRouterChain

  # 路由链 - 根据输入选择不同的链
  router_prompt = PromptTemplate.from_template("""
  根据用户的问题，选择最合适的处理方式：
  {input}
  目标：{destinations}
  """)
  router_chain = LLMRouterChain.from_llm(llm, router_prompt)      

  MultiRouteChain

  from langchain.chains import MultiRouteChain

  # 多路由链
  multi_chain = MultiRouteChain(
      router_chain=router_chain,
      destination_chains={
          "math": math_chain,
          "history": history_chain,
          "general": general_chain
      },
      default_chain=default_chain
  )

  6. 时间序列链

  ConversationalRetrievalChain

  from langchain.chains import ConversationalRetrievalChain       

  # 对话检索链
  conv_retrieval_chain = ConversationalRetrievalChain(
      retriever=retriever,
      llm=llm,
      memory=conversation_memory
  )

  7. 自定义链

  继承基类

  from langchain.chains.base import Chain
  from pydantic import BaseModel

  class CustomChain(Chain):
      input_variables: List[str] = ["input"]
      output_variables: List[str] = ["output"]

      def _call(self, inputs):
          # 自定义逻辑
          result = self.process_data(inputs["input"])
          return {"output": result}

      def process_data(self, data):
          # 处理逻辑
          return data.upper()

  8. 工具链

  Tool 链

  from langchain.tools import Tool
  from langchain.agents import initialize_agent, AgentType        

  # 工具链
  search_tool = Tool(
      name="Search",
      func=search_function,
      description="搜索信息"
  )

  agent = initialize_agent(
      tools=[search_tool, calculator_tool],
      llm=llm,
      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
  )

  9. 现代组合模式

  并行链

  from langchain_core.runnables import RunnableParallel

  # 并行执行多个链
  parallel_chain = RunnableParallel({
      "summary": summarize_chain,
      "keywords": keyword_chain,
      "translation": translate_chain
  })

  result = parallel_chain.invoke({"text": "输入"})

  分支链

  from langchain_core.runnables import RunnableBranch

  # 条件分支
  branch_chain = RunnableBranch(
      (lambda x: x.get("type") == "question", qa_chain),
      (lambda x: x.get("type") == "summary", summary_chain),      
      (lambda x: x.get("type") == "translation",
  translate_chain),
      default_chain
  )

  传递链

  # 传递并添加字段
  enhanced_chain = (
      base_chain
      | RunnablePassthrough.assign(
          metadata=lambda x: {"length": len(x["result"])}
      )
      | RunnablePassthrough.assign(
          confidence=lambda x: calculate_confidence(x)
      )
  )

  10. 预构建链

  create_sql_query_chain

  from langchain.chains import create_sql_query_chain

  # SQL查询链
  sql_chain = create_sql_query_chain(llm, db)

  create_history_aware_retriever

  from langchain.chains import create_history_aware_retriever     

  # 历史感知检索器
  retriever_chain = create_history_aware_retriever(
      llm, retriever, prompt
  )

  在您的代码中使用的链

  # 您代码中的链类型：

  # 1. 基本管道链
  story_chain = (
      {"character": RunnablePassthrough()}
      | story_prompt
      | llm
      | StrOutputParser()
      | (lambda story: {"story_summary": story})
  )

  # 2. 复合链
  full_chain = (
      story_chain
      | RunnablePassthrough.assign(title=title_generator)
      | RunnablePassthrough.assign(ending=ending_generator)       
  )

  # 3. 内联链
  title_lambda = lambda x: (title_prompt | llm |
  StrOutputParser()).invoke(...)

  链的创建方式

  1. 传统方式（较少使用）

  chain = LLMChain(llm=llm, prompt=prompt)

  2. 现代方式（推荐）

  chain = prompt | llm | output_parser

  3. 函数式组合

  chain = RunnableParallel({
      "step1": chain1,
      "step2": chain2
  }) | final_processor

  这些链可以像积木一样组合，构建复杂的 AI
  应用！选择合适的链类型取决于您的具体需求。