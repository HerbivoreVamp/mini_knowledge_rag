from pathlib import Path

from .loader import load_md
from .hierarchical import ingest_documents
from .splitter import create_hierarchy_splitter
from core.exceptions import RAGError
from config.reranker import BGEReranker
from storage.docstore import create_docstore
from storage.vectorstore import save_vectorstore, create_empty_vectorstore, add_documents
from retrieval.hierarchical import HierarchicalRetriever


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
                                              reranker=reranker
                                              )

        save_vectorstore(
            vectorstore,
            vectorstore_dir,
            index_name,
        )

    except RAGError as e:
        raise e

    return retriever
