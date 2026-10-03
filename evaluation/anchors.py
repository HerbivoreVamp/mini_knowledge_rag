"""评测用的证据锚点(anchor)

为什么需要:
    QA 数据集原来用 chunk_id / parent_id 当 ground truth, 但两者都不稳定:
      - chunk_id = md5(parent_id + source + start_index + content)
        改 chunk_size / chunk_overlap / id 算法, 或者文档被重新切分 -> 全变
      - parent_id = uuid4() -> 每次重新导入都变
    结果是"每改一次切分就要重做一遍 QA"。

锚点改用「源文档 + 字符区间」定位证据:

    {"source": "langchain_doc\\\\overview.md", "span": [0, 372], "quote": "原文切片..."}

    只要源文档内容没变, 重新导入 / 改切分参数 / 改 id 算法都不影响 ——
    评测时用当前索引现算 ground truth, QA 零维护。
    万一文档内容也改了, reanchor() 还能拿 quote 在新文档里重新定位。

区间从哪来:
    child 的 metadata["file_start_index"] 就是它在源文档里的字符偏移
    (parent 的文件内偏移 + child 相对 parent 的偏移, 见 ingestion/hierarchical.py),
    区间长度就是切片文本的长度 —— 实测当前语料 1423 篇 child 全部能精确还原原文。
"""
from langchain_core.documents import Document

# 一个切片要覆盖锚点区间多大的比例才算命中
# 取 50% 是因为换了切分参数后, 一段证据可能被切成两半, 两半各覆盖一部分;
# 想要更严就调高, 想要更松就调低(评测脚本用 --min-coverage 暴露)
DEFAULT_MIN_COVERAGE = 0.5


def doc_span(doc: Document):
    """取切片在源文档里的 (source, start, end)

    需要 metadata["file_start_index"]; 老索引没有这个字段时返回 None
    (这样调用方会明确地"解析不出来", 而不是拿 parent 内偏移当文件内偏移用错)
    """
    metadata = doc.metadata
    source = metadata.get("source")
    start = metadata.get("file_start_index")
    if source is None or start is None:
        return None
    return source, int(start), int(start) + len(doc.page_content)


def anchor_of(doc: Document):
    """从一个切片造锚点(把旧数据集迁移成锚点口径时用)"""
    span = doc_span(doc)
    if span is None:
        return None
    source, start, end = span
    return {"source": source, "span": [start, end], "quote": doc.page_content}


def coverage(anchor: dict, span) -> float:
    """span 覆盖 anchor 区间的比例(0~1)"""
    if span is None or span[0] != anchor["source"]:
        return 0.0
    anchor_start, anchor_end = anchor["span"]
    length = anchor_end - anchor_start
    if length <= 0:
        return 0.0
    overlap = min(anchor_end, span[2]) - max(anchor_start, span[1])
    return max(0, overlap) / length


def resolve(anchor: dict, docs, min_coverage: float = DEFAULT_MIN_COVERAGE):
    """返回覆盖了锚点区间(>= min_coverage)的切片"""
    return [doc for doc in docs if coverage(anchor, doc_span(doc)) >= min_coverage]


def resolve_ids(anchor: dict, docs, key: str = "chunk_id", min_coverage: float = DEFAULT_MIN_COVERAGE):
    """返回命中的切片 id 集合"""
    return {
        doc.metadata.get(key)
        for doc in resolve(anchor, docs, min_coverage)
    } - {None}


def resolve_gold(items, docs, level: str = "child", min_coverage: float = DEFAULT_MIN_COVERAGE) -> dict:
    """把整个数据集解析成 {item_id: {id...}}

    :param level: child  -> 覆盖锚点的 child 切片 id(默认, 与旧口径一致)
                  parent -> 这些 child 所属的 parent id(parent 口径)
    """
    key = "parent_id" if level == "parent" else "chunk_id"
    gold = {}
    for item in items:
        ids = set()
        for anchor in item.get("gold", []):
            ids |= resolve_ids(anchor, docs, key=key, min_coverage=min_coverage)
        gold[item["id"]] = ids
    return gold


def reanchor(anchor: dict, doc_text: str):
    """文档内容变了以后, 用 quote 在新文档里重新定位区间; 找不到返回 None"""
    quote = anchor.get("quote")
    if not quote:
        return None
    start = doc_text.find(quote)
    if start < 0:
        return None
    return {"source": anchor["source"], "span": [start, start + len(quote)], "quote": quote}
