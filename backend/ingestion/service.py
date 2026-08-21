from pathlib import Path

from .loader import load_md
from .hierarchical import ingest_documents
from .splitter import create_hierarchy_splitter
from backend.core.exceptions import RAGError
from backend.config.reranker import BGEReranker
from backend.storage.docstore import create_docstore
from backend.storage.vectorstore import save_vectorstore, create_empty_vectorstore
from backend.retrieval.hierarchical import HierarchicalRetriever


def ingestion_service(document_dir: str, folder, emb, vectorstore_dir: str, index_name, parent_store_dir: str,
                      reranker: BGEReranker,
                      vectorstore=None,
                      retriever=None):
    try:
        parent_docstore = create_docstore(Path(parent_store_dir))
        if vectorstore is None:
            vectorstore = create_empty_vectorstore(emb)
            save_vectorstore(
                vectorstore,
                Path(vectorstore_dir),
                index_name,
            )

        docs = load_md(Path(document_dir), folder)
        parent_splitter, child_splitter = create_hierarchy_splitter()

        ingest_documents(docs=docs,
                         vectorstore=vectorstore,
                         parent_store=parent_docstore,
                         parent_splitter=parent_splitter,
                         child_splitter=child_splitter,
                         )
        if retriever is None:
            retriever = HierarchicalRetriever(vectorstore=vectorstore,
                                              parent_store=parent_docstore,
                                              reranker=reranker,
                                              k=30
                                              )

        save_vectorstore(
            vectorstore,
            Path(vectorstore_dir),
            index_name,
        )

    except RAGError as e:
        raise e

    return retriever
