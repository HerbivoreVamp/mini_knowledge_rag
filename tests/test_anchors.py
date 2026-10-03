"""evaluation/anchors.py 的单元测试(纯函数, 不加载模型、不碰索引)"""
import pytest

from langchain_core.documents import Document

from evaluation.anchors import (
    DEFAULT_MIN_COVERAGE,
    anchor_of,
    coverage,
    doc_span,
    reanchor,
    resolve,
    resolve_gold,
    resolve_ids,
)


def make_doc(content, start, source="a.md", chunk_id="c1", parent_id="p1"):
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "file_start_index": start,
            "chunk_id": chunk_id,
            "parent_id": parent_id,
        },
    )


def test_doc_span_uses_file_start_index():
    assert doc_span(make_doc("abcdef", 10)) == ("a.md", 10, 16)


def test_doc_span_returns_none_without_file_start_index():
    """老索引只有 parent 内偏移的 start_index, 不能当成文件内偏移用"""
    doc = Document(page_content="x", metadata={"source": "a.md", "start_index": 5})
    assert doc_span(doc) is None


def test_anchor_of_builds_anchor_from_doc():
    assert anchor_of(make_doc("abcdef", 10)) == {
        "source": "a.md",
        "span": [10, 16],
        "quote": "abcdef",
    }


def test_coverage_full_partial_and_other_source():
    anchor = {"source": "a.md", "span": [100, 200]}
    assert coverage(anchor, ("a.md", 100, 200)) == 1.0
    assert coverage(anchor, ("a.md", 150, 250)) == 0.5
    assert coverage(anchor, ("a.md", 50, 100)) == 0.0
    assert coverage(anchor, ("b.md", 100, 200)) == 0.0
    assert coverage(anchor, None) == 0.0


def test_resolve_and_min_coverage():
    anchor = {"source": "a.md", "span": [100, 200], "quote": "q"}
    docs = [
        make_doc("x" * 100, 100, chunk_id="full"),
        make_doc("x" * 50, 150, chunk_id="half"),
        make_doc("x" * 10, 100, chunk_id="tiny"),
        make_doc("x" * 100, 100, source="b.md", chunk_id="other_file"),
    ]

    assert {d.metadata["chunk_id"] for d in resolve(anchor, docs)} == {"full", "half"}
    assert resolve_ids(anchor, docs, min_coverage=0.9) == {"full"}
    assert resolve_ids(anchor, docs, min_coverage=0.5) == {"full", "half"}


def test_resolve_gold_child_and_parent_levels():
    anchor = {"source": "a.md", "span": [0, 10], "quote": "0123456789"}
    docs = [
        make_doc("0123456789", 0, chunk_id="c1", parent_id="p1"),
        make_doc("xyz", 500, chunk_id="c2", parent_id="p2"),
    ]
    items = [{"id": "q1", "gold": [anchor]}]

    assert resolve_gold(items, docs, level="child") == {"q1": {"c1"}}
    assert resolve_gold(items, docs, level="parent") == {"q1": {"p1"}}


def test_resolve_gold_unresolvable_anchor_yields_empty_set():
    items = [
        {"id": "q1", "gold": [{"source": "a.md", "span": [0, 5], "quote": "aaaaa"}]},
        {"id": "q2", "gold": [{"source": "missing.md", "span": [0, 5], "quote": "bbbbb"}]},
    ]
    docs = [make_doc("aaaaa", 0, chunk_id="c1", parent_id="p1")]

    gold = resolve_gold(items, docs, level="child")
    assert gold["q1"] == {"c1"}
    assert gold["q2"] == set()


def test_reanchor_after_document_changed():
    anchor = {"source": "a.md", "span": [0, 5], "quote": "hello"}
    assert reanchor(anchor, "xxx hello yyy") == {
        "source": "a.md",
        "span": [4, 9],
        "quote": "hello",
    }
    assert reanchor(anchor, "no such text") is None
    assert reanchor({"source": "a.md", "span": [0, 5]}, "hello") is None


def test_default_min_coverage_is_half():
    assert DEFAULT_MIN_COVERAGE == 0.5
