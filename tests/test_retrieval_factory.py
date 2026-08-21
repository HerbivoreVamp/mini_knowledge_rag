import pytest

from backend.retrieval.factory import create_retriever
from backend.core.exceptions import RetrievalError


def test_create_retriever_full_chain(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = mocker.Mock()

    mock_normal = mocker.Mock()
    mock_rerank = mocker.Mock()
    mock_hierarchical = mocker.Mock()

    mock_NormalRetriever = mocker.patch(
        "backend.retrieval.factory.NormalRetriever",
        return_value=mock_normal,
    )
    mock_RerankRetriever = mocker.patch(
        "backend.retrieval.factory.RerankRetriever",
        return_value=mock_rerank,
    )
    mock_HierarchicalRetriever = mocker.patch(
        "backend.retrieval.factory.HierarchicalRetriever",
        return_value=mock_hierarchical,
    )

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=reranker,
        hierarchical=True,
        norm_k=30,
        rerank_topk=5,
    )

    mock_NormalRetriever.assert_called_once_with(vectorstore=vectorstore, k=30)
    mock_RerankRetriever.assert_called_once_with(retriever=mock_normal, reranker=reranker, top_k=5)
    mock_HierarchicalRetriever.assert_called_once_with(child_retriever=mock_rerank, parent_store=parent_store)
    assert result == mock_hierarchical


def test_create_retriever_no_reranker(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()

    mock_normal = mocker.Mock()
    mock_hierarchical = mocker.Mock()

    mocker.patch("backend.retrieval.factory.NormalRetriever", return_value=mock_normal)
    mock_RerankRetriever = mocker.patch("backend.retrieval.factory.RerankRetriever")
    mocker.patch("backend.retrieval.factory.HierarchicalRetriever", return_value=mock_hierarchical)

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=None,
        hierarchical=True,
    )

    mock_RerankRetriever.assert_not_called()
    assert result == mock_hierarchical


def test_create_retriever_no_hierarchical(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = mocker.Mock()

    mock_normal = mocker.Mock()
    mock_rerank = mocker.Mock()

    mocker.patch("backend.retrieval.factory.NormalRetriever", return_value=mock_normal)
    mocker.patch("backend.retrieval.factory.RerankRetriever", return_value=mock_rerank)
    mock_HierarchicalRetriever = mocker.patch("backend.retrieval.factory.HierarchicalRetriever")

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=reranker,
        hierarchical=False,
    )

    mock_HierarchicalRetriever.assert_not_called()
    assert result == mock_rerank


def test_create_retriever_plain(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()

    mock_normal = mocker.Mock()

    mocker.patch("backend.retrieval.factory.NormalRetriever", return_value=mock_normal)
    mock_RerankRetriever = mocker.patch("backend.retrieval.factory.RerankRetriever")
    mock_HierarchicalRetriever = mocker.patch("backend.retrieval.factory.HierarchicalRetriever")

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=None,
        hierarchical=False,
    )

    mock_RerankRetriever.assert_not_called()
    mock_HierarchicalRetriever.assert_not_called()
    assert result == mock_normal


def test_create_retriever_normal_retriever_fails(mocker):
    mocker.patch(
        "backend.retrieval.factory.NormalRetriever",
        side_effect=RetrievalError("创建失败"),
    )

    with pytest.raises(RetrievalError):
        create_retriever(
            vectorstore=mocker.Mock(),
            parent_store=mocker.Mock(),
        )