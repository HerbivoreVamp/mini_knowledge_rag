import pytest

from backend.retrieval.rerank_retriever import RerankRetriever
from backend.core.exceptions import RetrievalError
from langchain_core.documents import Document


def test_rerank_retriever_invoke_success(mocker):
    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
    ]
    reranked = [
        Document(page_content="doc2"),
        Document(page_content="doc1"),
    ]

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = docs

    mock_reranker = mocker.Mock()
    mock_reranker.rerank.return_value = reranked

    retriever = RerankRetriever.model_construct(retriever=mock_retriever, reranker=mock_reranker, top_k=2)
    result = retriever.invoke("test query")

    assert result == reranked
    mock_retriever.invoke.assert_called_once_with("test query")
    mock_reranker.rerank.assert_called_once_with("test query", docs, top_k=2)


def test_rerank_retriever_invoke_default_top_k(mocker):
    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
    ]

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = docs

    mock_reranker = mocker.Mock()
    mock_reranker.rerank.return_value = docs

    retriever = RerankRetriever.model_construct(retriever=mock_retriever, reranker=mock_reranker)
    result = retriever.invoke("test query")

    mock_reranker.rerank.assert_called_once_with("test query", docs, top_k=5)


def test_rerank_retriever_invoke_retrieval_failure(mocker):
    mock_retriever = mocker.Mock()
    mock_retriever.invoke.side_effect = Exception("检索失败")

    retriever = RerankRetriever.model_construct(retriever=mock_retriever, reranker=mocker.Mock())

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_rerank_retriever_invoke_rerank_failure(mocker):
    docs = [Document(page_content="doc1")]

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = docs

    mock_reranker = mocker.Mock()
    mock_reranker.rerank.side_effect = Exception("rerank失败")

    retriever = RerankRetriever.model_construct(retriever=mock_retriever, reranker=mock_reranker)

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")