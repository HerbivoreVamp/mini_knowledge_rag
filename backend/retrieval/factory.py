from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever

from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever
from .rerank_retriever import RerankRetriever
from .hierarchical_retriever import HierarchicalRetriever
from .tokenizer import CjkMode, TokenizerMode, create_tokenizer, describe_tokenizer
from backend.storage.sqlite_docstore import SqliteDocStore
from backend.core.exceptions import RetrievalError
from backend.core.logger import logger


def create_retriever(
        vectorstore: VectorStore,
        parent_store=None,
        child_store=None,
        reranker=None,
        hierarchical=True,
        hybrid=True,
        k=30,
        sparse_k=30,
        hybrid_topk=50,
        rerank_topk=20,
        parent_k=3,
        tokenizer: TokenizerMode | str = TokenizerMode.PROPER,
        tokenizer_lowercase: bool = False,
        tokenizer_cjk: CjkMode | str = CjkMode.RUN,
) -> BaseRetriever:
    """组装检索链 Dense -> Hybrid(BM25 + RRF) -> Rerank -> Hierarchical

    :param tokenizer: BM25 分词方案, 见 backend/retrieval/tokenizer.py
        默认 PROPER(去标点, 且保留 create_agent 这类标识符);
        要旧行为就显式传 TokenizerMode.DEFAULT
    :param tokenizer_lowercase: 是否小写化(实测收益为负, 默认 False)
    :param tokenizer_cjk: 中文切法 run(整段)/char(逐字)
    """
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
            preprocess_func = create_tokenizer(
                mode=tokenizer,
                lowercase=tokenizer_lowercase,
                cjk=tokenizer_cjk,
            )
            sparse = BM25Retriever.from_documents(
                docs,
                k=sparse_k,
                preprocess_func=preprocess_func,
            )
            logger.info(
                "BM25Retriever创建成功 k=%s tokenizer=%s(%s) lowercase=%s cjk=%s",
                sparse_k,
                preprocess_func.mode,
                describe_tokenizer(preprocess_func.mode),
                preprocess_func.lowercase,
                preprocess_func.cjk,
            )
            retriever = HybridRetriever(dense_retriever=retriever, sparse_retriever=sparse,k=hybrid_topk)

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
