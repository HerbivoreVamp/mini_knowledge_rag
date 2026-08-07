# LangChain Tool 封装

from langchain.tools import tool

from utils.logger import logger
def create_retrieve_tool(vector_store):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """检索知识库回答问题"""
        logger.info("模型调用retrieve_context工具")
        docs = vector_store.similarity_search(query, k=5)
        logger.info("工具返回搜索结果")
        serialized = "\n\n".join(
            f"Content: {d.page_content}"
            for d in docs
        )

        return serialized, docs

    return retrieve_context


if __name__ == '__main__':
    import sys
    from pathlib import Path
    # backend目录加入搜索路径
    sys.path.append(
        str(Path(__file__).resolve().parent.parent)
    )

    from ingestion.embedding import embeddings
    from vectorstore_manage import load_vectorstore
    from langchain.chat_models import init_chat_model
    from langchain.messages import HumanMessage
    from langchain.agents import create_agent

    emb = embeddings(r"D:\home\models\BAAI", r"bge-small-zh-v1.5")
    vectorstore = load_vectorstore(load_dir=r"D:\dump\my_project\mini_knowledge_rag\backend\database", # 导入测试
                                   embedding=emb,
                                   index_name="langchain_doc")

    retrieve_tool = create_retrieve_tool(vectorstore)
    model = init_chat_model(
        model="qwen/qwen3-8b",
        model_provider="openai",
        base_url="http://localhost:1234/v1",
        api_key="1234",
    )
    agent = create_agent(
        model,
        tools=[retrieve_tool],
        system_prompt="你有检索工具，它有绝大多数可查询的文档，需要时调用它获取知识",
    )
    for event in agent.stream(
            {"messages": [HumanMessage("LangChain 的Agent怎么用？")]},
            stream_mode="values",
    ):
        event["messages"][-1].pretty_print()