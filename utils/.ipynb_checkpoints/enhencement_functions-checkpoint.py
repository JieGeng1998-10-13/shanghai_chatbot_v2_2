import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from langgraph.graph import MessagesState

# Initialize the model
local_llm = "EntropyYue/chatglm3:6b"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

# Enhanced router instructions
router_instructions = """
你是一位专家，负责判断用户的问题是否包含具体的公司名称或项目名称。

**规则**：

1. 如果问题中包含具体的公司名称（例如：“上海市XXXX公司”、“北京ABC有限公司”）或具体的项目名称（例如：“XXXXXX招标公告”、“项目Y进展报告”），返回“enhencement”。

2. 如果问题不包含具体的公司名称或项目名称，返回“no_enhencement”。

**注意事项**：

- 一般性的词语如“公司”、“项目”、“公告”或“新闻”不被视为具体的名称。
- 只有当问题中提及了具体的、可识别的公司或项目名称时，才应返回“enhencement”。

**输出格式**：仅返回一个包含单个键 'datasource' 的 JSON 对象，值为 'enhencement' 或 'no_enhencement'。

**示例输出**：

- 问题：“新建潍坊至宿迁高速铁路江苏段监理项目目前什么状态？”  
  输出：{"datasource": "enhencement"}

- 问题：“招标公告中有多少条交通类项目”  
  输出：{"datasource": "no_enhencement"}

- 问题：“有多少项目目前处于异常状态？”  
  输出：{"datasource": "no_enhencement"}

- 问题：“沪昆铁路义乌高架站房建设工程-YWZFTJ-1标哪家公司中标了？”  
  输出：{"datasource": "enhencement"}
  
  
- 问题：“哪家公司中标项目最多？”  
  输出：{"datasource": "no_enhencement"}

**禁止事项**：

- 不要添加任何额外的文本、解释或标点符号，只需返回指定格式的 JSON 对象。

请根据上述规则判断用户的问题，并返回正确的 JSON 格式的输出。
"""

def route_question_enhencement(state):
    """
    使用 LLM 判断问题中是否包含公司名称或项目名称，将其路由到适当的 datasource。

    Args:
        state (dict): 当前的 graph 状态

    Returns:
        str: 下一个要调用的节点
    """
    ROUTE_STATUS = "---正在引导问题---"
    print(ROUTE_STATUS)

    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)] +
        [HumanMessage(content=state["question"])]
    )

    # 解析返回的 JSON 数据
    try:
        source = json.loads(route_question.content.strip())["datasource"]
    except json.JSONDecodeError as e:
        print("JSON 解码错误:", e)
        print("模型输出为:", route_question.content)
        # 当解析出错时，默认返回 'no_enhencement'
        source = "no_enhencement"

    if source == "enhencement":
        ROUTE_STATUS = "---问题包含公司或项目名称，路由至增强---"
        print("---ROUTE QUESTION TO ENHENCEMENT---")
        print(ROUTE_STATUS)
        return "enhencement"
    else:
        ROUTE_STATUS = "---问题未包含公司或项目名称，路由至无增强---"
        print("---ROUTE QUESTION TO NO ENHANCEMENT---")
        print(ROUTE_STATUS)
        return "no_enhencement"

def test_router():
    """
    测试路由功能。
    """
    test_cases = [
        "奉贤区10米级氢能源城市客车采购项目的中标结果是什么",
        "做防水材料的公司，哪几家比较好",
        "临港新片区云鹃路及水芸路（黄日港-绿丽港）贯通工程管线搬迁项目具体的招标信息",
        "现在正在招标的绿化项目有哪些？",
        "芦茂路（老芦公路～潮和路）道路专项整治工程设计服务项目情况",
        "一共有多少条状态处于异常的公告",
        "北京建国伟业防水材料集团有限公司在哪个项目中中标?",
        "招标公告中有多少条铁路建设项目？？",
        "中标人名称为中铁武汉大桥工程咨询监理有限公司的项目编号是什么？"
    ]
    for question in test_cases:
        state = {"question": question}
        result = route_question_enhencement(state)
        print(f"问题: '{question}' -> 返回: {result}")

# 使用示例
if __name__ == "__main__":
    test_router()
