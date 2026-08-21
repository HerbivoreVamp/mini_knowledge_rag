from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from .normal_retriever import NormalRetriever
from .rerank_retriever import RerankRetriever
from .hierarchical_retriever import HierarchicalRetriever
from backend.storage.docstore import JsonDocStore
from backend.core.exceptions import RetrievalError


def create_retriever(
        vectorstore: VectorStore,
        parent_store: JsonDocStore,
        reranker=None,
        hierarchical=True,
        norm_k=30,
        rerank_topk=5
) -> BaseRetriever:
    try:
        retriever = NormalRetriever(
            vectorstore=vectorstore,
            k=norm_k,
        )

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
            )
    except RetrievalError:
        raise

    return retriever
