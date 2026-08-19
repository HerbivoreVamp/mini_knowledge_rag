from pathlib import Path

from .loader import load_md
from .hierarchical import create_docstore, create_parent_retriever, add_documents_to_retriever
from core.exceptions import RAGError
from storage.vectorstore import save_vectorstore, create_empty_vectorstore, add_documents


def ingestion_service(document_dir: str, folder, emb, vectorstore_dir: str, index_name, parent_store_dir: str,
                      vectorstore=None,
                      retriever=None):
    try:
        docs = load_md(Path(document_dir), folder)

        if vectorstore is None:
            vectorstore = create_empty_vectorstore(emb)
            save_vectorstore(
                vectorstore,
                Path(vectorstore_dir),
                index_name,
            )
        if retriever is None:
            retriever = create_parent_retriever(
                vectorstore=vectorstore,
                parent_store=create_docstore(Path(parent_store_dir))  # 这个json会自动保存
            )
        retriever = add_documents_to_retriever(retriever=retriever, docs=docs)
        save_vectorstore(
            vectorstore,
            vectorstore_dir,
            index_name,
        )

    except RAGError as e:
        raise e

    return retriever
