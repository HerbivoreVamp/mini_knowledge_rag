from .loader import load_md
from .splitter import split_text
from core.exceptions import RAGError
from storage.vectorstore import save_vectorstore, build_vectorstore


def ingestion_service(document_dir, folder, emb, database_dir, database_index_name):
    try:
        docs = load_md(str(document_dir), str(folder))
        splits = split_text(docs)
        vectorstore = build_vectorstore(splits, emb)
        save_vectorstore(vectorstore, str(database_dir), database_index_name)
    except RAGError:
        raise
    return vectorstore
