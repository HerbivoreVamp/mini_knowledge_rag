from .loader import load_md
from .splitter import split_text
from core.exceptions import RAGError
from storage.vectorstore import save_vectorstore, create_empty_vectorstore,add_documents


def ingestion_service(document_dir, folder, emb, database_dir, database_index_name, vectorstore=None):
    try:
        docs = load_md(str(document_dir), str(folder))
        splits = split_text(docs)

        if vectorstore is None:
            vectorstore = create_empty_vectorstore(emb)

        vectorstore = add_documents(vectorstore, splits)

        save_vectorstore(
            vectorstore,
            str(database_dir),
            database_index_name,
        )

    except RAGError:
        raise

    return vectorstore
