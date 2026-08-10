from .loader import load_all_md
from .splitter import split_text
from utils.exceptions import RAGError
from knowledge_manage.vectorstore import save_vectorstore, build_vectorstore


def ingestion_service(document_dir, folder, emb, database_dir, database_index_name):
    doc_dir = document_dir / folder
    try:
        docs = load_all_md(str(doc_dir))
        splits = split_text(docs)
        vectorstore = build_vectorstore(splits, emb)
        save_vectorstore(vectorstore, str(database_dir), database_index_name)
    except RAGError:
        raise
    return vectorstore
