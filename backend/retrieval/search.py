# LangChain Tool 封装

from langchain.tools import tool

from utils.logger import logger
from utils.exceptions import RetrievalError


def create_retrieve_tool(vector_store):
    if vector_store is None:
        raise RetrievalError(
            "vector_store不能为空"
        )

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """检索知识库回答问题"""
        try:
            logger.info(
                "模型调用retrieve_context工具 query=%s",
                query
            )
            if not query.strip():
                logger.debug("retrieve_context收到空query")
                return "检索工具调用失败：query为空。请重新生成有效检索关键词。", []
            docs = vector_store.similarity_search(
                query,
                k=5
            )
            logger.info(f"retrieve_context返回结果数量={len(docs)}")
            serialized = "\n\n".join(
                f"""
            来源:
            {d.metadata.get("source")}

            内容:
            {d.page_content}
            """
                for d in docs
            )
            return serialized, docs

        except Exception as e:

            logger.exception("retrieve_context工具执行失败")
            return f"检索工具执行失败：{str(e)}。""请尝试其他方式回答用户。", []

    return retrieve_context


if __name__ == '__main__':
    import sys
    from pathlib import Path

    # backend目录加入搜索路径
    sys.path.append(
        str(Path(__file__).resolve().parent.parent)
    )

    from ingestion.embedding import embeddings
    from backend.knowledge_manage.vectorstore import load_vectorstore
    from langchain.chat_models import init_chat_model
    from langchain.messages import HumanMessage
    from langchain.agents import create_agent

    emb = embeddings(r"D:\home\models\BAAI", r"bge-small-zh-v1.5")
    vectorstore = load_vectorstore(load_dir=r"D:\dump\my_project\mini_knowledge_rag\backend\database",  # 导入测试
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
