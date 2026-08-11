# LangChain Tool 封装

from langchain.tools import tool

from core.logger import logger
from core.exceptions import RetrievalError


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


