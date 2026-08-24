from pathlib import Path

from .loader import load_md
from .hierarchical import ingest_documents
from .splitter import create_hierarchy_splitter
from backend.config.settings import Settings
from backend.core.exceptions import RAGError
from backend.config.reranker import BGEReranker
from backend.storage.sqlite_docstore import create_sqlite_docstore
from backend.storage.vectorstore import save_vectorstore, create_empty_vectorstore
from backend.retrieval.factory import create_retriever


def ingestion_service(settings: Settings, folder, emb, reranker: BGEReranker, vectorstore=None, retriever=None, hybrid=True):
    try:
        document_dir = settings.document_dir
        vectorstore_dir = settings.vectorstore_dir
        index_name = settings.index_name
        parent_store_dir = settings.parent_store_dir
        child_store_dir = settings.child_store_dir

        parent_docstore = create_sqlite_docstore(Path(parent_store_dir))
        child_docstore = create_sqlite_docstore(Path(child_store_dir))
        if vectorstore is None:
            vectorstore = create_empty_vectorstore(emb)
            save_vectorstore(
                vectorstore,
                Path(vectorstore_dir),
                index_name,
            )

        docs = load_md(Path(document_dir), folder)
        parent_splitter, child_splitter = create_hierarchy_splitter()

        ingest_documents(
            docs=docs,
            vectorstore=vectorstore,
            parent_store=parent_docstore,
            child_store=child_docstore,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
        )
        child_docstore.close()
        if retriever is None:
            retriever = create_retriever(
                vectorstore=vectorstore,
                parent_store=parent_docstore,
                child_store=create_sqlite_docstore(settings.child_store_dir),
                reranker=reranker,
            )
        else:
            parent_docstore.close()

        save_vectorstore(
            vectorstore,
            Path(vectorstore_dir),
            index_name,
        )

    except RAGError as e:
        raise e

    return retriever, vectorstore
