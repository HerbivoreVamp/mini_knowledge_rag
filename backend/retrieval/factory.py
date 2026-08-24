from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever

from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever
from .rerank_retriever import RerankRetriever
from .hierarchical_retriever import HierarchicalRetriever
from backend.storage.sqlite_docstore import SqliteDocStore
from backend.core.exceptions import RetrievalError
from backend.core.logger import logger


def create_retriever(
        vectorstore: VectorStore,
        parent_store,
        child_store=None,
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
            # 优先从 sqlite child_store 读取 child 语料 兼容旧的 FAISS docstore
            if isinstance(child_store, SqliteDocStore) and child_store.count() > 0:
                docs = child_store.get_all_documents()
                logger.info("BM25 语料来源 SqliteDocStore(child)")
            else:
                docs = list(vectorstore.docstore._dict.values())
                logger.info("BM25 语料来源 FAISS docstore")
            # 无论哪个分支都关闭 child_store 释放文件锁
            if isinstance(child_store, SqliteDocStore):
                child_store.close()
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
