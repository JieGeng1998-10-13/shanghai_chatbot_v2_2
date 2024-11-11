from typing import Literal

from utils.router_selection import Router  # 导入自定义的选择路由类
from utils.retriever_SQL import SQLModelHandler  # 导入自定义的SQL类
from langchain_core.messages import HumanMessage, SystemMessage
import json
from langgraph.graph import MessagesState, END



router_instance = Router()

llm = router_instance.llm
retriever1 = SQLModelHandler()



# 选择路由函数，决定问答方向是普通回答还是需要查询的专业回答
def route_question(state: MessagesState):
    """
    根据用户的问题，将其路由到适当的datasource。

    Args:
        state (dict): 当前的graph状态

    Returns:
        str: 下一个要调用的节点
    """

    ROUTE_STATUS = "---正在引导问题---"
    print(ROUTE_STATUS)
    route_question = router_instance.llm_json_mode.invoke(
        [SystemMessage(content=router_instance.router_instructions)] +
        [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "answer_directly":
        ROUTE_STATUS = "---正把问题引导至普通问答---"
        print("---ROUTE QUESTION TO ASK LLM---")
        print(ROUTE_STATUS)
        return "answer_directly"
    elif source == "vectorstore":
        ROUTE_STATUS = "---正在把问题引导至SQL查询---"
        print("---ROUTE QUESTION TO RAG---")
        print(ROUTE_STATUS)
        return "vectorstore"

    elif source == "law_query":
        ROUTE_STATUS = "---正在把问题引导至法律咨询---"
        print("---ROUTE QUESTION TO LAW QUERY")
        print(ROUTE_STATUS)
        return "law_query"
    else:
        ROUTE_STATUS = f"正在把问题引导至{source.upper()}---"
        print(f"---ROUTE QUESTION TO {source.upper()}---")
        print(ROUTE_STATUS)
        return source


# 调用llm生成回答，并标记步骤加一
def call_model(state):
    # print(state)
    loop_step = state.get("loop_step", 0)
    question = state['question']
    messages = [HumanMessage(content=question)]
    response = llm.invoke(messages)
    return {"generation": response.content, "loop_step": loop_step + 1}


# 方案一，用langchain自带的TXT2SQL
def call_model_SQL(state):
    question = state["question"]
    loop_step = state.get("loop_step", 0)
    result = retriever1.chain.invoke({"question": question})
    return {"generation": result, "loop_step": loop_step + 1}


def call_model_SQL_enhence(state):
    question = state["question"]
    loop_step = state.get("loop_step", 0)
    result = retriever1.chain.invoke({"question": question})
    return {"generation": question + result, "loop_step": loop_step + 1}
    
def call_model_filter(state):
    question = state["generation"]
    prompt_SQL = """请根据问题和检索信息，把回答整理成一个过滤后的答案，
    即回答出问题需要的信息，
    假如无法做到，就回答请参考上海市交通交易系统的官方网站，
    回答中不要提到问题的信息，只要回答的部分即可
    """
    new_question = question + '，' + prompt_SQL
    messages = [HumanMessage(content=new_question)]
    response = llm.invoke(messages)
    return {"generation": response.content}

   
# 设置节点函数，去给SQL查询生成提示词
def call_model_SQL_prompt(state):
    question = state["question"]
    prompt_SQL = '必须用中文回答'
    new_question = question + '，' + prompt_SQL

    return {"question": new_question}


def law_question_prompt(state):
    question = state["question"] 
    prompt_SQL = '必须用中文回答'
    new_question = question + '，' + prompt_SQL

    return {"question": new_question}


def noun_retriever(state):
    question = state["question"]
    # predicted_labels = str(predict_all_labels(question))
    # new_question = question + ', ' + predicted_labels
    new_question = question
    return {"question": new_question}


def call_model(state):
    # print(state)
    loop_step = state.get("loop_step", 0)
    question = state['question']
    messages = [HumanMessage(content=question)]
    response = llm.invoke(messages)
    return {"generation": response.content, "loop_step": loop_step + 1}
