from typing import TYPE_CHECKING

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from .fusion.rrf import reciprocal_rank_fusion
from backend.core.exceptions import RetrievalError
from backend.core.logger import logger


class HybridRetriever(BaseRetriever):
    dense_retriever: BaseRetriever
    sparse_retriever: BaseRetriever
    k: int = 50
    if TYPE_CHECKING:  # 方便显示参数 没有实际运行
        def __init__(
                self,
                *,
                dense_retriever: BaseRetriever,
                sparse_retriever: BaseRetriever,
                k: int = 50,
        ):
            ...

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        try:
            dense_docs = self.dense_retriever.invoke(query)
            sparse_docs = self.sparse_retriever.invoke(query)
            logger.info(f"SparseRetriever召回文档成功 total_docs={len(sparse_docs)}")
            docs = reciprocal_rank_fusion(
                [
                    dense_docs,
                    sparse_docs,
                ],
                top_k=self.k,
            )
            logger.info(f"Reciprocal Rank Fusion重排成功 top_k={self.k}")
        except Exception as e:
            logger.exception(f"HybridRetriever出错 error={e}")
            raise RetrievalError("HybridRetriever出错") from e

        return docs
