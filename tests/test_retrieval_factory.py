import pytest

from backend.retrieval.factory import create_retriever
from backend.core.exceptions import RetrievalError


def _make_vectorstore(mocker, docs=None):
    vectorstore = mocker.MagicMock()
    vectorstore.docstore._dict.values.return_value = docs or []
    return vectorstore


def test_create_retriever_full_chain(mocker):
    from langchain_core.documents import Document

    docs = [Document(page_content="doc1")]
    vectorstore = _make_vectorstore(mocker, docs=docs)
    parent_store = mocker.Mock()
    reranker = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_sparse = mocker.Mock()
    mock_hybrid = mocker.Mock()
    mock_rerank = mocker.Mock()
    mock_hierarchical = mocker.Mock()

    mock_DenseRetriever = mocker.patch(
        "backend.retrieval.factory.DenseRetriever",
        return_value=mock_dense,
    )
    mock_BM25Retriever = mocker.patch(
        "backend.retrieval.factory.BM25Retriever",
    )
    mock_BM25Retriever.from_documents.return_value = mock_sparse
    mock_HybridRetriever = mocker.patch(
        "backend.retrieval.factory.HybridRetriever",
        return_value=mock_hybrid,
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
        hybrid=True,
        k=30,
        sparse_k=20,
        rerank_topk=5,
        parent_k=3,
    )

    mock_DenseRetriever.assert_called_once_with(vectorstore=vectorstore, k=30)
    mock_BM25Retriever.from_documents.assert_called_once_with(docs, k=20)
    mock_HybridRetriever.assert_called_once_with(dense_retriever=mock_dense, sparse_retriever=mock_sparse)
    mock_RerankRetriever.assert_called_once_with(retriever=mock_hybrid, reranker=reranker, top_k=5)
    mock_HierarchicalRetriever.assert_called_once_with(
        child_retriever=mock_rerank,
        parent_store=parent_store,
        parent_k=3,
    )
    assert result == mock_hierarchical


def test_create_retriever_no_hybrid(mocker):
    vectorstore = _make_vectorstore(mocker)
    parent_store = mocker.Mock()
    reranker = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_rerank = mocker.Mock()
    mock_hierarchical = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_HybridRetriever = mocker.patch("backend.retrieval.factory.HybridRetriever")
    mocker.patch("backend.retrieval.factory.RerankRetriever", return_value=mock_rerank)
    mocker.patch("backend.retrieval.factory.HierarchicalRetriever", return_value=mock_hierarchical)

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=reranker,
        hierarchical=True,
        hybrid=False,
    )

    mock_BM25Retriever.from_documents.assert_not_called()
    mock_HybridRetriever.assert_not_called()
    assert result == mock_hierarchical


def test_create_retriever_no_reranker(mocker):
    vectorstore = _make_vectorstore(mocker)
    parent_store = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_sparse = mocker.Mock()
    mock_hybrid = mocker.Mock()
    mock_hierarchical = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_BM25Retriever.from_documents.return_value = mock_sparse
    mock_HybridRetriever = mocker.patch(
        "backend.retrieval.factory.HybridRetriever",
        return_value=mock_hybrid,
    )
    mock_RerankRetriever = mocker.patch("backend.retrieval.factory.RerankRetriever")
    mocker.patch("backend.retrieval.factory.HierarchicalRetriever", return_value=mock_hierarchical)

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=None,
        hierarchical=True,
        hybrid=True,
    )

    mock_RerankRetriever.assert_not_called()
    assert result == mock_hierarchical


def test_create_retriever_no_hierarchical(mocker):
    vectorstore = _make_vectorstore(mocker)
    parent_store = mocker.Mock()
    reranker = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_rerank = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_HybridRetriever = mocker.patch("backend.retrieval.factory.HybridRetriever")
    mocker.patch("backend.retrieval.factory.RerankRetriever", return_value=mock_rerank)
    mock_HierarchicalRetriever = mocker.patch("backend.retrieval.factory.HierarchicalRetriever")

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=reranker,
        hierarchical=False,
        hybrid=False,
    )

    mock_HierarchicalRetriever.assert_not_called()
    assert result == mock_rerank


def test_create_retriever_hybrid_only(mocker):
    vectorstore = _make_vectorstore(mocker)
    parent_store = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_sparse = mocker.Mock()
    mock_hybrid = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_BM25Retriever.from_documents.return_value = mock_sparse
    mocker.patch("backend.retrieval.factory.HybridRetriever", return_value=mock_hybrid)
    mock_RerankRetriever = mocker.patch("backend.retrieval.factory.RerankRetriever")
    mock_HierarchicalRetriever = mocker.patch("backend.retrieval.factory.HierarchicalRetriever")

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=None,
        hierarchical=False,
        hybrid=True,
    )

    mock_RerankRetriever.assert_not_called()
    mock_HierarchicalRetriever.assert_not_called()
    assert result == mock_hybrid


def test_create_retriever_plain(mocker):
    vectorstore = _make_vectorstore(mocker)
    parent_store = mocker.Mock()

    mock_dense = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_HybridRetriever = mocker.patch("backend.retrieval.factory.HybridRetriever")
    mock_RerankRetriever = mocker.patch("backend.retrieval.factory.RerankRetriever")
    mock_HierarchicalRetriever = mocker.patch("backend.retrieval.factory.HierarchicalRetriever")

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=None,
        hierarchical=False,
        hybrid=False,
    )

    mock_BM25Retriever.from_documents.assert_not_called()
    mock_HybridRetriever.assert_not_called()
    mock_RerankRetriever.assert_not_called()
    mock_HierarchicalRetriever.assert_not_called()
    assert result == mock_dense


def test_create_retriever_dense_retriever_fails(mocker):
    mocker.patch(
        "backend.retrieval.factory.DenseRetriever",
        side_effect=RetrievalError("创建失败"),
    )

    with pytest.raises(RetrievalError):
        create_retriever(
            vectorstore=_make_vectorstore(mocker),
            parent_store=mocker.Mock(),
        )
