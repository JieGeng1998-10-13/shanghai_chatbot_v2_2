from langgraph.graph import StateGraph
import gradio as gr
from utils.tools import GraphState  # 图结构数据类型声明
from langgraph.graph import END
from utils.function_tools import route_question, call_model, call_model_SQL_prompt
from utils.function_tools import call_model_SQL, law_question_prompt
from utils.rag_text import call_model_raglaw
from langgraph.checkpoint.memory import MemorySaver
from utils.enhencement_functions import route_question_enhencement
from utils.noun_retriever import noun_retriever_prompt
from utils.law_selection import route_question_law


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("answer_directly", call_model)  # web search
workflow.add_node("retrieve_SQL_prompt", call_model_SQL_prompt)
# workflow.add_node("retrieve_SQL", call_model_SQL_vanna)  # retrieve， 第一种vanna方案
workflow.add_node("retrieve_SQL", call_model_SQL)  # 第二种SQL查询方案，用langchain自带方法,方案切换只需要注释掉其中一个节点即可

# workflow.add_node("SQL_enhencement", label_predict_SQL)
workflow.add_node("SQL_enhencement", noun_retriever_prompt)
#workflow.add_node("enhencement_processing", call_model_SQL)
workflow.add_node("enhencement_processing", call_model)



#Define four law nodes

# Define three four nodes
workflow.add_node("law_question", law_question_prompt) # 替换为正确的函数

workflow.add_node("retrieve_law", call_model_raglaw)
workflow.add_node("meaning_large_retrieve", call_model_raglaw) #替换为正确的函数
workflow.add_node("similarity_retrieve", call_model_raglaw)

# Define the edges
workflow.add_edge("answer_directly", END)
# workflow.add_edge("retrieve_SQL_prompt", "retrieve_SQL")
workflow.add_edge("retrieve_SQL", END)
workflow.add_edge("retrieve_law", END)
workflow.add_edge("SQL_enhencement", "enhencement_processing")
workflow.add_edge("enhencement_processing", END)


# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "answer_directly": "answer_directly",
        "vectorstore": "retrieve_SQL_prompt",
        "law_query": "law_question",
    },
)


workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "retrieve_SQL_prompt",
    # Next, we pass in the function that will determine which node is called next.
    route_question_enhencement,
    {
        "no_enhencement": "retrieve_SQL",
        "enhencement": "SQL_enhencement"
    },
)


workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "law_question",
    # Next, we pass in the function that will determine which node is called next.
    route_question_law,
    {
        "clear_question": "retrieve_law",
        "law_details": "meaning_large_retrieve",
        "law_examples": "similarity_retrieve"

    },
)


memory = MemorySaver()
graph = workflow.compile()



#
# inputs = {"question": "你好"}
# inputs = {"question": "公路项目一共有几条招标公告"}
#
# if __name__ == "__main__":
#     #检查工作流内容
#     for event in graph.stream(inputs, stream_mode="values"):
#         print(event)
#定义处理函数


def answer_question(question):
    inputs = {"question": question}
    outputs = None
    # 运行图并获取最终输出
    for event in graph.stream(inputs, stream_mode="values"):
        outputs = event  # 假设最后一个事件是最终输出
    # 提取 'generation' 字段
    generation = outputs.get("generation", "未生成回答")
    return generation


# 构建 Gradio 界面
iface = gr.Interface(fn=answer_question,
                     inputs="text",
                     outputs="text",
                     title="上海市交易系统问答平台（Chatglm3）",
                     description="我会根据问题查询已有交通交易及法律知识。",
                     examples=[
                         ["你好"],
                         ["招标人不按规定退保证金是否要罚款？"],
                         ["奉贤区10米级氢能源城市客车采购项目的中标结果是什么"],
                         ["哪家公司中标项目最多"],
                         ["现在正在招标的绿化项目有哪些"],
                         ["中国电力工程顾问集团华东电力设计院中过哪些标"],
                         ["现在正在招标的绿化项目有哪些"],
                         ["招标公告中有多少条航道项目"],
                         ["上海城建审图咨询有限公司能否提供相关网址"],
                         ["阜阳站信什么锁室内设备招标公告这个项目完整名称是什么，我不太记得"]
                     ])

# 启动应用程序
if __name__ == "__main__":
    iface.launch(share=True)
