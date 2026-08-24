from typing import TYPE_CHECKING

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from backend.core.exceptions import RetrievalError
from backend.core.logger import logger
from backend.config.reranker import BGEReranker


class RerankRetriever(BaseRetriever):
    retriever: BaseRetriever
    reranker: BGEReranker
    top_k: int = 5
    if TYPE_CHECKING:
        def __init__(
                self,
                *,
                retriever: BaseRetriever,
                reranker: BGEReranker,
                top_k: int = 5,
        ):
            ...

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        try:
            docs = self.retriever.invoke(query)
            reranked_docs = self.reranker.rerank(
                query,
                docs,
                top_k=self.top_k
            )
            logger.info("RerankRetriever重排成功")
        except Exception as e:
            logger.exception(f"rerank出错 error={e}")
            raise RetrievalError("rerank出错") from e

        return reranked_docs
