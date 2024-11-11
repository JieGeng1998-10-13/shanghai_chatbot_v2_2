import os
from typing import List

from langchain import hub

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.document_loaders import TextLoader

import logging
from typing import Sequence


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RagLaw:
    def __init__(self):
        # 设置环境变量和日志
        #os.environ['USER_AGENT'] = 'MyLangChainApp/1.0'
        logging.basicConfig(level=logging.INFO)

        # 初始化语言模型
        self.llm = ChatOllama(
            #model="EntropyYue/chatglm3:6b ",
            model="EntropyYue/chatglm3:6b",
        )

        # 加载和处理文档
        loader = TextLoader('law_text_splitters_test.txt', encoding='utf-8')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        vectorstore = InMemoryVectorStore.from_documents(
            documents=splits,
            embedding=OllamaEmbeddings(model="milkey/m3e")
            #embedding=OllamaEmbeddings(model="shaw/dmeta-embedding-zh")
        )
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        self.rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )


assistant = RagLaw()


def call_model_raglaw(state):
    # print(state)
    loop_step = state.get("loop_step", 0)
    question = state['question']

    response = assistant.rag_chain.invoke(question)
    return {"generation": response, "loop_step": loop_step + 1}


if __name__ == "__main__":
    result = assistant.rag_chain.invoke("不发中标通知书要承担法律责任吗?")
    print(result)
