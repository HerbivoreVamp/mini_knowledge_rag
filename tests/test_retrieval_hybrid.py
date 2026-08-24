import pytest

from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.core.exceptions import RetrievalError
from langchain_core.documents import Document


def test_hybrid_retriever_invoke_success(mocker):
    dense_docs = [
        Document(page_content="doc1", metadata={"chunk_id": "c1"}),
        Document(page_content="doc2", metadata={"chunk_id": "c2"}),
    ]
    sparse_docs = [
        Document(page_content="doc2", metadata={"chunk_id": "c2"}),
        Document(page_content="doc3", metadata={"chunk_id": "c3"}),
    ]
    fused_docs = [
        Document(page_content="doc2", metadata={"chunk_id": "c2"}),
        Document(page_content="doc1", metadata={"chunk_id": "c1"}),
        Document(page_content="doc3", metadata={"chunk_id": "c3"}),
    ]

    dense_retriever = mocker.Mock()
    dense_retriever.invoke.return_value = dense_docs

    sparse_retriever = mocker.Mock()
    sparse_retriever.invoke.return_value = sparse_docs

    mock_rrf = mocker.patch(
        "backend.retrieval.hybrid_retriever.reciprocal_rank_fusion",
        return_value=fused_docs,
    )

    retriever = HybridRetriever.model_construct(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        k=50,
    )
    result = retriever.invoke("test query")

    assert result == fused_docs
    dense_retriever.invoke.assert_called_once_with("test query")
    sparse_retriever.invoke.assert_called_once_with("test query")
    mock_rrf.assert_called_once_with([dense_docs, sparse_docs], top_k=50)


def test_hybrid_retriever_invoke_dense_failure(mocker):
    dense_retriever = mocker.Mock()
    dense_retriever.invoke.side_effect = Exception("dense检索失败")

    retriever = HybridRetriever.model_construct(
        dense_retriever=dense_retriever,
        sparse_retriever=mocker.Mock(),
    )

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_hybrid_retriever_invoke_sparse_failure(mocker):
    sparse_retriever = mocker.Mock()
    sparse_retriever.invoke.side_effect = Exception("sparse检索失败")

    retriever = HybridRetriever.model_construct(
        dense_retriever=mocker.Mock(),
        sparse_retriever=sparse_retriever,
    )

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_hybrid_retriever_invoke_fusion_failure(mocker):
    dense_retriever = mocker.Mock()
    dense_retriever.invoke.return_value = [
        Document(page_content="doc1", metadata={"chunk_id": "c1"}),
    ]

    sparse_retriever = mocker.Mock()
    sparse_retriever.invoke.return_value = [
        Document(page_content="doc2", metadata={"chunk_id": "c2"}),
    ]

    mock_rrf = mocker.patch(
        "backend.retrieval.hybrid_retriever.reciprocal_rank_fusion",
        side_effect=RetrievalError("融合失败"),
    )

    retriever = HybridRetriever.model_construct(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")
