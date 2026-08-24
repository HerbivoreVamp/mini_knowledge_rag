from collections import defaultdict

from langchain_core.documents import Document

from backend.core.logger import logger
from backend.core.exceptions import RetrievalError


def reciprocal_rank_fusion(
        retriever_results: list[list[Document]],
        k: int = 60,
        top_k: int | None = None,
) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF)

    Args:
        retriever_results:
            多个retriever返回的Document列表
            例如:
            [
                dense_docs,
                sparse_docs,
            ]

        k:
            RRF平滑参数，默认60是论文常用值

        top_k:
            最终返回数量

    Returns:
        根据RRF分数排序后的Document列表
    """
    try:
        scores = defaultdict(float)
        documents = {}

        for docs in retriever_results:
            for rank, doc in enumerate(docs, start=1):

                # Document去重key
                doc_id = _get_doc_id(doc)

                scores[doc_id] += 1 / (k + rank)

                if doc_id not in documents:
                    documents[doc_id] = doc

        ranked_ids = sorted(
            scores,
            key=lambda x: scores[x],
            reverse=True,
        )

        results = [
            documents[doc_id]
            for doc_id in ranked_ids
        ]

        if top_k:
            results = results[:top_k]
    except Exception as e:
        logger.exception("Reciprocal Rank Fusion重排出错")
        raise RetrievalError("Reciprocal Rank Fusion重排出错") from e
    return results


def _get_doc_id(doc: Document) -> str:
    """
    获取Document唯一标识
    """

    metadata = doc.metadata
    if "chunk_id" in metadata:
        return metadata["chunk_id"]

    # fallback
    return doc.page_content
