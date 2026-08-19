import uuid

from storage.docstore import JsonDocStore
from core.logger import logger


def ingest_documents(docs, vectorstore, parent_store: JsonDocStore, parent_splitter, child_splitter):
    parents = parent_splitter.split_documents(
        docs
    )
    parent_items = []
    child_docs = []

    for parent in parents:

        parent_id = str(uuid.uuid4())

        parent.metadata["parent_id"] = parent_id

        children = child_splitter.split_documents(
            [parent]
        )

        for child in children:
            child.metadata["parent_id"] = parent_id

        child_docs.extend(children)

        parent_items.append(
            (parent_id, parent)
        )

    # 一次保存所有parent
    parent_store.mset(
        parent_items
    )

    # 一次添加所有child
    vectorstore.add_documents(
        child_docs
    )
    logger.info(
        "导入完成 parents=%s children=%s",
        len(parent_items),
        len(child_docs)
    )
    return {
        "parents": len(parent_items),
        "children": len(child_docs)
    }
