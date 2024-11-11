import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from langgraph.graph import MessagesState


class Router:
    def __init__(self):
        # 初始化模型
        #self.local_llm = "qwen2.5:7b"
        self.local_llm = "EntropyYue/chatglm3:6b"
        self.llm = ChatOllama(model=self.local_llm, temperature=0)
        self.llm_json_mode = ChatOllama(model=self.local_llm, temperature=0, format="json")

        # 路由指令

        self.router_instructions = """
        你是一个专家，负责根据用户的问题将其引导到以下三个数据源之一：'vectorstore'、'answer_directly' 或 'law_query'。

        数据源描述：

        1. **vectorstore**：
           - 包含与招标项目具体相关的文档。
           - 主题包括：中标结果、变更和异常、招标公告、项目分类等。
           - 示例问题：
             - "铁路中标结果一共有多少条？"
             - "请提供最新的招标公告。"
             - "有哪些项目发生了变更？"
           - 对于涉及招标项目细节的问题，选择 'vectorstore'。

        2. **law_query**：
           - 包含与法律法规法条相关的内容。
           - 适用于涉及行为合规性判断、具体法律规定的提问。
           - 示例问题：
             - "中华人民共和国招标投标法实施条例具体涉及哪些信息？"
             - "投标过程中有哪些法律要求？"
             - "这种行为是否符合相关法规？"
           - 对于涉及法律法规的问题，选择 'law_query'。

        3. **answer_directly**：
           - 用于所有其他问题，尤其是问候、闲聊等。
           - 示例问题：
             - "你是谁？"
             - "今天天气怎么样？"
             - "能告诉我一个笑话吗？"
           - 对于无法归类到以上两个数据源的问题，选择 'answer_directly'。

        **注意事项**：

        - **输出格式**：仅返回一个包含单个键 'datasource' 的 JSON 对象，值为 'answer_directly'、'vectorstore' 或 'law_query' 之一。
        - **示例输出**：
          - {"datasource": "answer_directly"}
          - {"datasource": "vectorstore"}
          - {"datasource": "law_query"}
        - **禁止**：不要添加任何额外的文本、解释或标点符号，只需返回指定格式的 JSON 对象。
     
        请根据上述要求，对用户的问题进行分类，并返回正确的 JSON 格式的输出。
        """
    def route_question(self, state):
        """
        根据用户的问题，将其路由到适当的datasource。

        Args:
            state (dict): 当前的graph状态

        Returns:
            str: 下一个要调用的节点
        """

        ROUTE_STATUS = "---正在引导问题---"
        print(ROUTE_STATUS)
        route_question = self.llm_json_mode.invoke(
            [SystemMessage(content=self.router_instructions)] +
            [HumanMessage(content=state["question"])]
        )
        source = json.loads(route_question.content)["datasource"]
        if source == "answer_directly":
            ROUTE_STATUS ="---正把问题引导至普通问答---"
            print("---ROUTE QUESTION TO ASK LLM---")

            print(ROUTE_STATUS)
            return "answer_directly"
        elif source == "vectorstore":
            ROUTE_STATUS="---正在把问题引导至SQL查询---"
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
            state["ROUTE_STATUS"] = ROUTE_STATUS
            print(ROUTE_STATUS)
            return source


    def call_model(self, state: MessagesState):
        question = state['question']
        messages = [HumanMessage(content=question)]
        response = self.llm.invoke(messages)
        return {"generation": response.content}

    def test_router(self):
        """
        测试路由功能。
        """
        test_ask_search = self.llm_json_mode.invoke(
            [SystemMessage(content=self.router_instructions)] +
            [HumanMessage(content="你是谁?")]
        )
        test_vector_store = self.llm_json_mode.invoke(
            [SystemMessage(content=self.router_instructions)] +
            [HumanMessage(content="铁路中标结果一共有多少条?")]
        )

        test_law_query = self.llm_json_mode.invoke(
            [SystemMessage(content=self.router_instructions)] +
            [HumanMessage(content="中华人民共和国招标投标法实施条例具体涉及哪些信息?")]
        )
        print(
            json.loads(test_ask_search.content),
            json.loads(test_vector_store.content),
            json.loads(test_law_query.content),
        )


# 使用示例
if __name__ == "__main__":
    router = Router()
    router.test_router()
