"""backend/ingestion/utils.py 的单元测试

重点是把"chunk_id 必须唯一"这个不变量的回归测试钉住:
child 的 start_index 是相对 parent 的偏移, 所以 id 必须掺 parent_id,
否则"文件不同章节里相对位置相同的同一段文本"会算出同一个 id。
"""
import pytest

from langchain_core.documents import Document

from backend.core.exceptions import RAGError
from backend.ingestion.utils import create_chunk_id, create_doc_id


def make_doc(content="same text", parent_id="parent-1", source="a.md", start_index=0):
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "parent_id": parent_id,
            "start_index": start_index,
        },
    )


def test_chunk_id_is_deterministic():
    assert create_chunk_id(make_doc()) == create_chunk_id(make_doc())


def test_chunk_id_differs_when_parent_differs():
    """回归测试: 同一段文本落在不同 parent 的相同相对位置 -> 必须是不同 id"""
    assert create_chunk_id(make_doc(parent_id="p1")) != create_chunk_id(make_doc(parent_id="p2"))


def test_chunk_id_differs_when_content_differs():
    assert create_chunk_id(make_doc(content="a")) != create_chunk_id(make_doc(content="b"))


def test_chunk_id_differs_when_start_index_differs():
    assert create_chunk_id(make_doc(start_index=0)) != create_chunk_id(make_doc(start_index=10))


def test_chunk_id_differs_when_source_differs():
    assert create_chunk_id(make_doc(source="a.md")) != create_chunk_id(make_doc(source="b.md"))


def test_chunk_id_requires_parent_id():
    """缺 parent_id 时要显式报错, 不能静默退化成会碰撞的旧算法"""
    doc = Document(page_content="x", metadata={"source": "a.md", "start_index": 0})
    with pytest.raises(RAGError):
        create_chunk_id(doc)


def test_doc_id_depends_on_source_and_content():
    a = Document(page_content="x", metadata={"source": "a.md"})
    b = Document(page_content="x", metadata={"source": "a.md"})
    c = Document(page_content="x", metadata={"source": "b.md"})
    assert create_doc_id(a) == create_doc_id(b)
    assert create_doc_id(a) != create_doc_id(c)
