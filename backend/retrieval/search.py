# LangChain Tool 封装

from langchain.tools import tool
from langchain_core.documents import Document

from backend.core.logger import logger
from backend.core.exceptions import RetrievalError
from backend.retrieval.hierarchical_retriever import HierarchicalRetriever


def search(query: str, retriever: HierarchicalRetriever, k=2) -> list[Document]:
    try:
        logger.info(
            "search检索知识库 query=%s",
            query
        )
        if not query.strip():
            logger.debug("search收到空query")
            raise RetrievalError("search检索知识库失败：query为空。")
        retriever.rerank_k = k
        docs = retriever.invoke(query)
        logger.info(f"查询数量rerank_k={k} search返回结果数量={len(docs)}")
        for index, doc in enumerate(docs, start=1):
            logger.info(
                "search 检索得到的知识库文档 "
                "index=%d source=%s file_type=%s start_index=%s parent_id=%s",
                index,
                doc.metadata.get("source"),
                doc.metadata.get("file_type"),
                doc.metadata.get("start_index"),
                doc.metadata.get("parent_id"),
            )
    except Exception as e:
        logger.exception("search检索知识库失败")
        raise RetrievalError(f"search检索知识库失败：error={str(e)}") from e
    return docs


def create_retrieve_tool(retriever):
    if retriever is None:
        raise RetrievalError(
            "retriever不能为空"
        )

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """
        知识库检索工具 输入关键词搜索内容效果更好
        """
        try:
            logger.info(
                "模型调用retrieve_context工具 query=%s",
                query
            )
            docs = search(query, retriever, k=2)

        except Exception as e:
            logger.exception("retrieve_context工具执行失败")
            return f"检索工具执行失败：{str(e)}。""请再次尝试 或使用其他方式回答用户。", []

        serialized = "\n\n以下是调用工具retrieve_context返回的结果:\n\n".join(
            f"""
                    来源:
                    {d.metadata.get("source")}

                    内容:
                    {d.page_content}
                    """
            for d in docs
        )

        return serialized, docs

    return retrieve_context
