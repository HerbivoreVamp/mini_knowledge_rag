from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever

from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever
from .rerank_retriever import RerankRetriever
from .hierarchical_retriever import HierarchicalRetriever
from backend.storage.docstore import JsonDocStore
from backend.core.exceptions import RetrievalError
from backend.core.logger import logger


def create_retriever(
        vectorstore: VectorStore,
        parent_store: JsonDocStore,
        reranker=None,
        hierarchical=True,
        hybrid=True,
        k=30,
        sparse_k=30,
        rerank_topk=20,
        parent_k=3,
) -> BaseRetriever:
    try:
        retriever = DenseRetriever(
            vectorstore=vectorstore,
            k=k,
        )
        if hybrid:
            docs = list(vectorstore.docstore._dict.values())  # 临时使用 后续更换为sqilte数据库维护
            sparse = BM25Retriever.from_documents(
                docs,
                k=sparse_k,
            )
            logger.info(
                f"BM25Retriever创建成功 k={sparse_k}"
            )
            retriever = HybridRetriever(dense_retriever=retriever, sparse_retriever=sparse)

        if reranker:
            retriever = RerankRetriever(
                retriever=retriever,
                reranker=reranker,
                top_k=rerank_topk,
            )

        if hierarchical:
            retriever = HierarchicalRetriever(
                child_retriever=retriever,
                parent_store=parent_store,
                parent_k=parent_k,
            )
    except RetrievalError:
        raise

    return retriever
