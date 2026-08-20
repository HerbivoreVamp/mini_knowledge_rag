from langchain_classic.retrievers import ParentDocumentRetriever

from .splitter import create_text_splitter
from backend.core.logger import logger
from backend.core.exceptions import RetrievalError
from backend.storage.docstore import JsonDocStore


def create_parent_retriever(vectorstore, parent_store: JsonDocStore) -> ParentDocumentRetriever:
    parent_chunk_size = 2000
    parent_chunk_overlap = 200
    child_chunk_size = 400
    child_chunk_overlap = 50
    k = 2

    parent_splitter = create_text_splitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap,
                                           add_start_index=True)
    child_splitter = create_text_splitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap,
                                          add_start_index=True)

    try:
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=parent_store,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
            search_kwargs={
                "k": 2
            }
        )
    except Exception as e:
        logger.error(
            "Hierarchical Retriever 创建失败 error=%s",
            e
        )
        raise RetrievalError(
            "Hierarchical Retriever 创建失败"
        ) from e
    logger.info(
        "Hierarchical Retriever 创建成功 "
        "parent_chunk_size=%s parent_chunk_overlap=%s "
        "child_chunk_size=%s child_chunk_overlap=%s k=%s",
        parent_chunk_size,
        parent_chunk_overlap,
        child_chunk_size,
        child_chunk_overlap,
        k,
    )
    return retriever


def add_documents_to_retriever(retriever, docs) -> ParentDocumentRetriever:
    retriever.add_documents(docs)
    logger.info(f"新文档已添加至retriever docs={len(docs)}")
    return retriever
