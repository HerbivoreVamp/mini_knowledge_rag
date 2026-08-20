import pytest

from backend.retrieval.hierarchical import HierarchicalRetriever
from backend.core.exceptions import RetrievalError
from langchain_core.documents import Document


class FakeReranker:
    def rerank(self, query, docs, top_k=5):
        return docs[:top_k]


def test_hierarchical_retriever_init_success(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = FakeReranker()

    retriever = HierarchicalRetriever(vectorstore, parent_store, reranker, k=10)

    assert retriever is not None
    vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 10})


def test_hierarchical_retriever_init_default_k(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = FakeReranker()

    retriever = HierarchicalRetriever(vectorstore, parent_store, reranker)

    vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 30})


def test_hierarchical_retriever_init_failure(mocker):
    vectorstore = mocker.Mock()
    vectorstore.as_retriever.side_effect = Exception("创建失败")
    parent_store = mocker.Mock()
    reranker = FakeReranker()

    with pytest.raises(RetrievalError):
        HierarchicalRetriever(vectorstore, parent_store, reranker)


def test_hierarchical_retriever_invoke_success(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = FakeReranker()

    child_docs = [
        Document(page_content="child1", metadata={"parent_id": "p1"}),
        Document(page_content="child2", metadata={"parent_id": "p2"}),
        Document(page_content="child3", metadata={"parent_id": "p3"}),
    ]

    parent_docs = [
        Document(page_content="parent1"),
        Document(page_content="parent2"),
        Document(page_content="parent3"),
    ]

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = child_docs
    vectorstore.as_retriever.return_value = mock_retriever
    parent_store.mget.return_value = parent_docs

    retriever = HierarchicalRetriever(vectorstore, parent_store, reranker)
    result = retriever.invoke("test query", k=3)

    assert len(result) == 3
    assert result == parent_docs
    parent_store.mget.assert_called_once_with(["p1", "p2", "p3"])


def test_hierarchical_retriever_invoke_custom_k(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = FakeReranker()

    child_docs = [
        Document(page_content="child1", metadata={"parent_id": "p1"}),
        Document(page_content="child2", metadata={"parent_id": "p2"}),
    ]

    parent_docs = [
        Document(page_content="parent1"),
        Document(page_content="parent2"),
    ]

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = child_docs
    vectorstore.as_retriever.return_value = mock_retriever
    parent_store.mget.return_value = parent_docs

    retriever = HierarchicalRetriever(vectorstore, parent_store, reranker)
    result = retriever.invoke("test query", k=2)

    assert len(result) == 2


def test_hierarchical_retriever_invoke_child_retrieval_failure(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = FakeReranker()

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.side_effect = Exception("检索失败")
    vectorstore.as_retriever.return_value = mock_retriever

    retriever = HierarchicalRetriever(vectorstore, parent_store, reranker)

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_hierarchical_retriever_invoke_rerank_failure(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = mocker.Mock()
    reranker.rerank.side_effect = Exception("rerank失败")

    child_docs = [Document(page_content="child1", metadata={"parent_id": "p1"})]

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = child_docs
    vectorstore.as_retriever.return_value = mock_retriever

    retriever = HierarchicalRetriever(vectorstore, parent_store, reranker)

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_hierarchical_retriever_invoke_parent_lookup_failure(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    parent_store.mget.side_effect = Exception("parent lookup失败")
    reranker = FakeReranker()

    child_docs = [Document(page_content="child1", metadata={"parent_id": "p1"})]

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = child_docs
    vectorstore.as_retriever.return_value = mock_retriever

    retriever = HierarchicalRetriever(vectorstore, parent_store, reranker)

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_hierarchical_retriever_invoke_empty_child_results(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = FakeReranker()

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = []
    vectorstore.as_retriever.return_value = mock_retriever

    parent_store.mget.return_value = []

    retriever = HierarchicalRetriever(
        vectorstore,
        parent_store,
        reranker
    )

    result = retriever.invoke("test query")

    assert result == []


def test_hierarchical_retriever_invoke_parent_not_found(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    parent_store.mget.return_value = [None]
    reranker = FakeReranker()

    child_docs = [
        Document(
            page_content="child1",
            metadata={"parent_id": "p1"}
        )
    ]

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = child_docs
    vectorstore.as_retriever.return_value = mock_retriever

    retriever = HierarchicalRetriever(
        vectorstore,
        parent_store,
        reranker
    )

    result = retriever.invoke("test query")

    assert result == []
