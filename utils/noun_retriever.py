from langchain_ollama import  OllamaEmbeddings
from langchain_chroma import Chroma

def noun_retriever_prompt(state):
    question = state["question"]
    # Initialize embeddings and vector store
    embeddings = OllamaEmbeddings(model="milkey/m3e")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_non2.db")

    # Create a retriever and perform a query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    #docs = retriever.invoke("新建杭州乘务员工程施工总价承包招标公告")
    docs = retriever.invoke(question)
    # Print the retrieved documents
    infer_info =[]
    prompt = """
             结合原本问题和检索结果回答问题，
             你要根据问题选出合适的结果回答
             """
    prompt_enhence = """
             结合原本问题和检索结果回答问题，
             每次查询都有四个结果放在列表中,
             按照和问题的相似度从高到低排序，
             每个检索结果被逗号隔开，
             你要根据问题选出合适的结果回答
             
             
             
             """
    for doc in docs:
        #print(doc.page_content)
        infer_info.append(doc.page_content)
    new_question = "原本问题是：" + question + "现在经过检索器后得到下列结果" + str(infer_info) + prompt
    return {"question": new_question}