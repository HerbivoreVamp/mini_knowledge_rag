import uuid

from .utils import create_chunk_id

from backend.storage.sqlite_docstore import SqliteDocStore
from backend.storage.vectorstore import add_documents
from backend.core.logger import logger
from backend.core.exceptions import RAGError


def ingest_documents(docs, vectorstore, parent_store: SqliteDocStore, child_store, parent_splitter, child_splitter):
    parents = parent_splitter.split_documents(
        docs
    )
    parent_items = []
    child_docs = []

    for parent in parents:

        parent_id = str(uuid.uuid4())
        parent.metadata["parent_id"] = parent_id
        # 统一语义: file_start_index 永远是"在源文档里的字符偏移"
        # (parent 的 start_index 本来就是文件内偏移, 这里给它一个统一的字段名)
        parent.metadata["file_start_index"] = parent.metadata.get("start_index") or 0

        children = child_splitter.split_documents(
            [parent]
        )

        for child in children:
            # 先补 parent_id: create_chunk_id 依赖它来避免 id 碰撞
            child.metadata["parent_id"] = parent_id
            child.metadata["chunk_id"] = create_chunk_id(child)
            # child 的 start_index 是相对 parent 的, 加上 parent 的文件内偏移
            # 才是它在源文档里的真实位置(评测锚点、日志排查都用这个)
            child.metadata["file_start_index"] = (
                    parent.metadata["file_start_index"]
                    + (child.metadata.get("start_index") or 0)
            )

        child_docs.extend(children)

        parent_items.append(
            (parent_id, parent)
        )

    # 一次保存所有parent
    parent_store.mset(
        parent_items
    )

    # child 文档存入 child_store 用于 BM25 语料
    if child_store is not None and child_docs:
        child_store.mset(
            [(child.metadata["chunk_id"], child) for child in child_docs]
        )

    try:
        # 一次添加所有child
        add_documents(vectorstore=vectorstore, docs=child_docs)
        logger.info(
            "导入成功 parents=%s children=%s",
            len(parent_items),
            len(child_docs)
        )
    except RAGError as e:
        logger.info(
            "导入失败 parents=%s children=%s",
            len(parent_items),
            len(child_docs)
        )
        raise e
    return {
        "parents": len(parent_items),
        "children": len(child_docs)
    }
