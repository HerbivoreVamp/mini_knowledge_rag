import pytest

from backend.retrieval.hierarchical_retriever import HierarchicalRetriever
from backend.core.exceptions import RetrievalError
from langchain_core.documents import Document


def _make_retriever(mocker, vectorstore=None, parent_store=None, reranker=None, k=30, rerank_k=5, *, run_validator=False):
    """绕过 pydantic 类型校验构造 HierarchicalRetriever"""
    vs = vectorstore or mocker.Mock()
    ps = parent_store or mocker.Mock()
    rr = reranker or mocker.Mock()
    retriever = HierarchicalRetriever.model_construct(
        vectorstore=vs, parent_store=ps, reranker=rr, k=k, rerank_k=rerank_k
    )
    if run_validator:
        retriever = retriever.create_child_retriever()
    return retriever


def test_hierarchical_retriever_init_success(mocker):
    vectorstore = mocker.Mock()

    retriever = _make_retriever(mocker, vectorstore=vectorstore, k=10, run_validator=True)

    assert retriever is not None
    vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 10})


def test_hierarchical_retriever_init_default_k(mocker):
    vectorstore = mocker.Mock()

    _make_retriever(mocker, vectorstore=vectorstore, run_validator=True)

    vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 30})


def test_hierarchical_retriever_init_failure(mocker):
    vectorstore = mocker.Mock()
    vectorstore.as_retriever.side_effect = Exception("创建失败")

    retriever = _make_retriever(mocker, vectorstore=vectorstore)

    with pytest.raises(RetrievalError):
        retriever.create_child_retriever()


def test_hierarchical_retriever_invoke_success(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = mocker.Mock()
    reranker.rerank.return_value = [
        Document(page_content="child1", metadata={"parent_id": "p1"}),
        Document(page_content="child2", metadata={"parent_id": "p2"}),
        Document(page_content="child3", metadata={"parent_id": "p3"}),
    ]

    parent_docs = [
        Document(page_content="parent1"),
        Document(page_content="parent2"),
        Document(page_content="parent3"),
    ]

    mock_child_retriever = mocker.Mock()
    mock_child_retriever.invoke.return_value = [
        Document(
            page_content="child",
            metadata={"parent_id": "p1"}
        )
    ]
    parent_store.mget.return_value = parent_docs

    retriever = _make_retriever(mocker, vectorstore=vectorstore, parent_store=parent_store, reranker=reranker)
    retriever._child_retriever = mock_child_retriever

    result = retriever.invoke("test query")

    assert len(result) == 3
    assert result == parent_docs
    parent_store.mget.assert_called_once_with(["p1", "p2", "p3"])


def test_hierarchical_retriever_invoke_rerank_k(mocker):
    vectorstore = mocker.Mock()
    parent_store = mocker.Mock()
    reranker = mocker.Mock()
    reranker.rerank.return_value = [
        Document(page_content="child1", metadata={"parent_id": "p1"}),
        Document(page_content="child2", metadata={"parent_id": "p2"}),
    ]

    parent_docs = [
        Document(page_content="parent1"),
        Document(page_content="parent2"),
    ]

    mock_child_retriever = mocker.Mock()
    mock_child_retriever.invoke.return_value = []
    parent_store.mget.return_value = parent_docs

    retriever = _make_retriever(mocker, vectorstore=vectorstore, parent_store=parent_store, reranker=reranker, rerank_k=2)
    retriever._child_retriever = mock_child_retriever

    result = retriever.invoke("test query")
    reranker.rerank.assert_called_once_with(
        "test query",
        mock_child_retriever.invoke.return_value,
        top_k=2
    )
    assert len(result) == 2


def test_hierarchical_retriever_invoke_child_retrieval_failure(mocker):
    mock_child_retriever = mocker.Mock()
    mock_child_retriever.invoke.side_effect = Exception("检索失败")

    retriever = _make_retriever(mocker)
    retriever._child_retriever = mock_child_retriever

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_hierarchical_retriever_invoke_rerank_failure(mocker):
    reranker = mocker.Mock()
    reranker.rerank.side_effect = Exception("rerank失败")

    mock_child_retriever = mocker.Mock()
    mock_child_retriever.invoke.return_value = [Document(page_content="child1", metadata={"parent_id": "p1"})]

    retriever = _make_retriever(mocker, reranker=reranker)
    retriever._child_retriever = mock_child_retriever

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_hierarchical_retriever_invoke_parent_lookup_failure(mocker):
    parent_store = mocker.Mock()
    parent_store.mget.side_effect = Exception("parent lookup失败")
    reranker = mocker.Mock()
    reranker.rerank.return_value = [Document(page_content="child1", metadata={"parent_id": "p1"})]

    mock_child_retriever = mocker.Mock()
    mock_child_retriever.invoke.return_value = []

    retriever = _make_retriever(mocker, parent_store=parent_store, reranker=reranker)
    retriever._child_retriever = mock_child_retriever

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_hierarchical_retriever_invoke_empty_child_results(mocker):
    reranker = mocker.Mock()
    reranker.rerank.return_value = []

    mock_child_retriever = mocker.Mock()
    mock_child_retriever.invoke.return_value = []

    parent_store = mocker.Mock()
    parent_store.mget.return_value = []

    retriever = _make_retriever(mocker, parent_store=parent_store, reranker=reranker)
    retriever._child_retriever = mock_child_retriever

    result = retriever.invoke("test query")

    assert result == []


def test_hierarchical_retriever_invoke_parent_not_found(mocker):
    reranker = mocker.Mock()
    reranker.rerank.return_value = [
        Document(page_content="child1", metadata={"parent_id": "p1"})
    ]

    mock_child_retriever = mocker.Mock()
    mock_child_retriever.invoke.return_value = [
        Document(
            page_content="child1",
            metadata={"parent_id": "p1"}
        )
    ]

    parent_store = mocker.Mock()
    parent_store.mget.return_value = [None]

    retriever = _make_retriever(mocker, parent_store=parent_store, reranker=reranker)
    retriever._child_retriever = mock_child_retriever

    result = retriever.invoke("test query")

    assert result == []