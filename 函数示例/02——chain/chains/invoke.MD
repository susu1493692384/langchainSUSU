invoke() 是执行链或组件的方法，相当于"运行"或"调用"。

  语法结构

  result = chain.invoke(input_data)
  #              ↑       ↑
  #           链对象    输入数据

  在您的代码中的使用

  result = full_chain.invoke(character)
  #                   ↑       ↑
  #                完整链    "一个会说话的猫咪"

  不同场景下的 invoke

  1. 简单链的 invoke

  # 创建简单链
  chain = prompt | llm | output_parser

  # 调用链
  result = chain.invoke({"text": "你好"})

  2. 在您的代码中的复杂链

  # 第1步：故事概要链
  story_result = story_chain.invoke("一个会说话的猫咪")

  # 第2步：标题生成链（在 lambda 中）
  title = (title_prompt | llm | StrOutputParser()).invoke({       
      "story_summary": story_result["story_summary"]
  })

  # 第3步：结尾生成链（在 lambda 中）
  ending = (ending_prompt | llm | StrOutputParser()).invoke({     
      "title": title,
      "story_summary": story_result["story_summary"]
  })

  invoke 的输入格式

  1. 字典格式（最常见）

  chain.invoke({"text": "输入内容", "style": "正式"})

  2. 字符串格式

  chain.invoke("简单的文本输入")

  3. 复杂对象

  chain.invoke({
      "user_input": "用户输入",
      "context": ["历史1", "历史2"],
      "options": {"temperature": 0.7}
  })

  完整的数据流示例

  from langchain_core.prompts import PromptTemplate
  from langchain_core.runnables import RunnablePassthrough        

  # 1. 定义提示模板
  prompt = PromptTemplate(
      input_variables=["name"],
      template="你好，{name}！"
  )

  # 2. 创建链
  chain = prompt | llm | StrOutputParser()

  # 3. invoke 执行
  result = chain.invoke({"name": "张三"})
  print(result)  # 输出模型的回答

  invoke vs 其他执行方法

  # 1. invoke - 同步执行（最常用）
  result = chain.invoke({"input": "数据"})

  # 2. ainvoke - 异步执行
  result = await chain.ainvoke({"input": "数据"})

  # 3. batch - 批量执行
  results = chain.batch([{"input": "数据1"}, {"input":
  "数据2"}])

  # 4. stream - 流式执行
  for chunk in chain.stream({"input": "数据"}):
      print(chunk, end="")

  在您的复杂链中的 invoke 过程

  full_chain = (
      story_chain                                    # 第1步      
      | RunnablePassthrough.assign(
          title=lambda x: (title_prompt | llm |
  StrOutputParser()).invoke(
              {"story_summary": x["story_summary"]}  # 第2步的    
   invoke
          )
      )
      | RunnablePassthrough.assign(
          ending=lambda x: (ending_prompt | llm |
  StrOutputParser()).invoke({
              "title": x["title"],                   # 第3步的    
   invoke
              "story_summary": x["story_summary"]
          })
      )
  )

  # 最终执行
  result = full_chain.invoke("一个会说话的猫咪")

  错误处理

  try:
      result = chain.invoke({"input": "数据"})
  except Exception as e:
      print(f"调用失败: {e}")

  总结

  invoke 在 LangChain 中：

  - 作用：执行链/组件并获得结果
  - 类比：就像调用函数 function(arguments)
  - 位置：在链的末尾执行整个流程
  - 输入：通常是字典格式的数据
  - 输出：处理后的结果

  简单说，invoke 就是**"运行这个链"**的意思！