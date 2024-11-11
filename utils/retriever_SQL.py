from langchain_community.utilities import SQLDatabase
from operator import itemgetter
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama
from langchain_core.prompts import  PromptTemplate

class SQLModelHandler:
    def __init__(self,  model_name="jiesql2"):
        # 初始化数据库连接
        self.db = SQLDatabase.from_uri("sqlite:///上海市交通系统交易情况.db")
        self.context = self.db.get_context()
        self.table_info = self.context["table_info"]
        # 初始化LLM模型
        self.llm = ChatOllama(model=model_name)
        # 创建查询执行工具
        self.execute_query = QuerySQLDataBaseTool(db=self.db)
        # 创建SQL查询链
        self.write_query = create_sql_query_chain(self.llm, self.db)
        # 定义答案提示模板
        self.answer_prompt = PromptTemplate.from_template(
            """给出下列用户问题, 根据SQL query, 和 SQL result, 回答用户问题.
               根据输入的问题，创建一个语法正确的 SQLite 查询来运行，然后查看查询结果并返回答案。除非用户指定他们想要获取的具体样例数量，                否则查询结果最多限制为 5 条记录。可以根据相关列对结果进行排序，以返回数据库中最有意义的样本。切勿查询某个表中的所有列，只查                询与问题相关的列。

               你可以使用的工具如下。构建最终答案时只能使用这些工具返回的信息。你必须在执行查询前仔细检查查询的正确性。如果在执行查询时遇                到错误，请重新编写查询并重试。

               不要对数据库执行任何 DML 语句（如 INSERT、UPDATE、DELETE、DROP 等）。
               用中文回答
               Question: {question}
               SQL Query: {query}
               SQL Result: {result}
                回答: """
        )
        # 创建总的处理链
        self.chain = (
            RunnablePassthrough.assign(query=self.write_query).assign(
                result=itemgetter("query") | self.execute_query
            )
            | self.answer_prompt
            | self.llm
            | StrOutputParser()
        )


    def call_model_SQL(self, state: MessagesState):
        question = state["question"]
        result = self.chain.invoke({"question": question})
        return {"answer": result}

    def call_model_SQL_inside(self, question):
        """
        调用模型生成响应。

        Args:
            question (str): 用户提出的问题

        Returns:
            str: 模型的回答
        """
        result = self.chain.invoke({"question": question})
        return result
# 使用示例
if __name__ == "__main__":
    handler = SQLModelHandler()
    question = "奉贤区10米级氢能源城市客车采购项目的中标结果是什么？"
    answer = handler.call_model_SQL_inside(question)
    print(answer)
