import pytest

from backend.retrieval.fusion.rrf import reciprocal_rank_fusion, _get_doc_id
from backend.core.exceptions import RetrievalError
from langchain_core.documents import Document


def test_rrf_basic_ranking():
    dense = [
        Document(page_content="doc1", metadata={"chunk_id": "c1"}),
        Document(page_content="doc2", metadata={"chunk_id": "c2"}),
    ]
    sparse = [
        Document(page_content="doc2", metadata={"chunk_id": "c2"}),
        Document(page_content="doc1", metadata={"chunk_id": "c1"}),
    ]

    result = reciprocal_rank_fusion([dense, sparse])

    # doc1: 1/61 + 1/62, doc2: 1/62 + 1/61 分数相同 排序稳定
    assert len(result) == 2
    assert {d.metadata["chunk_id"] for d in result} == {"c1", "c2"}


def test_rrf_doc_ranked_high_in_both_wins():
    dense = [
        Document(page_content="docA", metadata={"chunk_id": "A"}),
        Document(page_content="docB", metadata={"chunk_id": "B"}),
    ]
    sparse = [
        Document(page_content="docA", metadata={"chunk_id": "A"}),
        Document(page_content="docC", metadata={"chunk_id": "C"}),
    ]

    result = reciprocal_rank_fusion([dense, sparse])

    # docA在两个列表都排第一 分数最高
    assert result[0].metadata["chunk_id"] == "A"
    assert len(result) == 3


def test_rrf_dedup_by_chunk_id():
    dense = [Document(page_content="doc1", metadata={"chunk_id": "c1"})]
    sparse = [Document(page_content="doc1", metadata={"chunk_id": "c1"})]

    result = reciprocal_rank_fusion([dense, sparse])

    assert len(result) == 1
    assert result[0].metadata["chunk_id"] == "c1"


def test_rrf_fallback_to_page_content_when_no_chunk_id():
    dense = [Document(page_content="same content")]
    sparse = [Document(page_content="same content")]

    result = reciprocal_rank_fusion([dense, sparse])

    # 无chunk_id时用page_content去重
    assert len(result) == 1


def test_rrf_top_k_truncation():
    docs = [
        Document(page_content=f"doc{i}", metadata={"chunk_id": f"c{i}"})
        for i in range(5)
    ]

    result = reciprocal_rank_fusion([docs], top_k=3)

    assert len(result) == 3


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_score_accumulates_across_lists():
    # doc1在dense排第2(1/62) 在sparse排第1(1/61) 总分应高于只在dense排第1的doc2(1/61)
    dense = [
        Document(page_content="doc2", metadata={"chunk_id": "c2"}),
        Document(page_content="doc1", metadata={"chunk_id": "c1"}),
    ]
    sparse = [
        Document(page_content="doc1", metadata={"chunk_id": "c1"}),
    ]

    result = reciprocal_rank_fusion([dense, sparse])

    assert result[0].metadata["chunk_id"] == "c1"


def test_get_doc_id_prefers_chunk_id():
    doc = Document(page_content="content", metadata={"chunk_id": "abc"})
    assert _get_doc_id(doc) == "abc"


def test_get_doc_id_fallback():
    doc = Document(page_content="content")
    assert _get_doc_id(doc) == "content"


def test_rrf_invalid_input_raises():
    with pytest.raises(RetrievalError):
        reciprocal_rank_fusion(None)
