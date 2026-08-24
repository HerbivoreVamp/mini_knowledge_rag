import hashlib


def create_chunk_id(doc):
    content = (
            doc.metadata["source"]
            + str(doc.metadata.get("start_index", ""))
            + doc.page_content
    )

    return hashlib.md5(
        content.encode("utf-8")
    ).hexdigest()


def create_doc_id(doc):
    content = (
            doc.metadata["source"]
            + doc.page_content
    )

    return hashlib.md5(
        content.encode("utf-8")
    ).hexdigest()
