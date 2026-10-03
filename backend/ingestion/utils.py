import hashlib

from backend.core.exceptions import RAGError


def create_chunk_id(doc):
    """child 切片的 id

    必须把 parent_id 掺进哈希: child 的 start_index 是"相对 parent"的偏移
    (children 是从 [parent] 切出来的), 所以同一段文本出现在不同 parent 的
    相同相对位置时(样板代码很常见), 只用 source + start_index + content
    会算出同一个 id。

    实测当前语料(9 篇文档)不加 parent_id 时有 12 个重复 id / 32 篇 child 被
    child_store 覆盖丢失, RRF 还会把不同 parent 的切片当成同一个 child 合并。
    """
    parent_id = doc.metadata.get("parent_id")
    if not parent_id:
        raise RAGError(
            "create_chunk_id 需要 metadata['parent_id']: "
            "child 的 start_index 是相对 parent 的偏移, 少了 parent_id 会造成 id 碰撞"
        )

    content = (
            parent_id
            + doc.metadata["source"]
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
