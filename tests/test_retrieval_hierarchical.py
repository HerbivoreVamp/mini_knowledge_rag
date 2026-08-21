import pytest

from backend.retrieval.hierarchical_retriever import HierarchicalRetriever
from backend.core.exceptions import RetrievalError
from langchain_core.documents import Document


def test_hierarchical_retriever_invoke_success(mocker):
    child_retriever = mocker.Mock()
    child_retriever.invoke.return_value = [
        Document(page_content="child1", metadata={"parent_id": "p1"}),
        Document(page_content="child2", metadata={"parent_id": "p2"}),
        Document(page_content="child3", metadata={"parent_id": "p3"}),
    ]

    parent_docs = [
        Document(page_content="parent1"),
        Document(page_content="parent2"),
        Document(page_content="parent3"),
    ]

    parent_store = mocker.Mock()
    parent_store.mget.return_value = parent_docs

    retriever = HierarchicalRetriever.model_construct(child_retriever=child_retriever, parent_store=parent_store)
    result = retriever.invoke("test query")

    assert len(result) == 3
    assert result == parent_docs
    child_retriever.invoke.assert_called_once_with("test query")
    parent_store.mget.assert_called_once_with(["p1", "p2", "p3"])


def test_hierarchical_retriever_invoke_child_retrieval_failure(mocker):
    child_retriever = mocker.Mock()
    child_retriever.invoke.side_effect = Exception("检索失败")

    retriever = HierarchicalRetriever.model_construct(child_retriever=child_retriever, parent_store=mocker.Mock())

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_hierarchical_retriever_invoke_parent_lookup_failure(mocker):
    child_retriever = mocker.Mock()
    child_retriever.invoke.return_value = [
        Document(page_content="child1", metadata={"parent_id": "p1"})
    ]

    parent_store = mocker.Mock()
    parent_store.mget.side_effect = Exception("parent lookup失败")

    retriever = HierarchicalRetriever.model_construct(child_retriever=child_retriever, parent_store=parent_store)

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_hierarchical_retriever_invoke_empty_child_results(mocker):
    child_retriever = mocker.Mock()
    child_retriever.invoke.return_value = []

    parent_store = mocker.Mock()
    parent_store.mget.return_value = []

    retriever = HierarchicalRetriever.model_construct(child_retriever=child_retriever, parent_store=parent_store)
    result = retriever.invoke("test query")

    assert result == []


def test_hierarchical_retriever_invoke_parent_not_found(mocker):
    child_retriever = mocker.Mock()
    child_retriever.invoke.return_value = [
        Document(page_content="child1", metadata={"parent_id": "p1"})
    ]

    parent_store = mocker.Mock()
    parent_store.mget.return_value = [None]

    retriever = HierarchicalRetriever.model_construct(child_retriever=child_retriever, parent_store=parent_store)
    result = retriever.invoke("test query")

    assert result == []


def test_hierarchical_retriever_invoke_duplicate_parent_ids(mocker):
    child_retriever = mocker.Mock()
    child_retriever.invoke.return_value = [
        Document(page_content="child1", metadata={"parent_id": "p1"}),
        Document(page_content="child2", metadata={"parent_id": "p1"}),
        Document(page_content="child3", metadata={"parent_id": "p2"}),
    ]

    parent_docs = [
        Document(page_content="parent1"),
        Document(page_content="parent2"),
    ]

    parent_store = mocker.Mock()
    parent_store.mget.return_value = parent_docs

    retriever = HierarchicalRetriever.model_construct(child_retriever=child_retriever, parent_store=parent_store)
    result = retriever.invoke("test query")

    assert len(result) == 2
    parent_store.mget.assert_called_once_with(["p1", "p2"])


def test_hierarchical_retriever_invoke_no_parent_id_in_metadata(mocker):
    child_retriever = mocker.Mock()
    child_retriever.invoke.return_value = [
        Document(page_content="child1", metadata={"other": "value"}),
    ]

    parent_store = mocker.Mock()
    parent_store.mget.return_value = []

    retriever = HierarchicalRetriever.model_construct(child_retriever=child_retriever, parent_store=parent_store)
    result = retriever.invoke("test query")

    assert result == []
    parent_store.mget.assert_called_once_with([])