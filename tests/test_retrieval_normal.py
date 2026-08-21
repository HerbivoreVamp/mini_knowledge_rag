import pytest

from backend.retrieval.normal_retriever import NormalRetriever
from backend.core.exceptions import RetrievalError
from langchain_core.documents import Document


def test_normal_retriever_init_success(mocker):
    vectorstore = mocker.Mock()

    retriever = NormalRetriever.model_construct(vectorstore=vectorstore, k=10)
    retriever.create_retriever()

    assert retriever is not None
    vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 10})


def test_normal_retriever_init_default_k(mocker):
    vectorstore = mocker.Mock()

    retriever = NormalRetriever.model_construct(vectorstore=vectorstore)
    retriever.create_retriever()

    vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 30})


def test_normal_retriever_init_failure(mocker):
    vectorstore = mocker.Mock()
    vectorstore.as_retriever.side_effect = Exception("创建失败")

    retriever = NormalRetriever.model_construct(vectorstore=vectorstore)

    with pytest.raises(RetrievalError):
        retriever.create_retriever()


def test_normal_retriever_invoke_success(mocker):
    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
    ]

    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = docs

    retriever = NormalRetriever.model_construct(vectorstore=mocker.Mock(), k=30)
    retriever._retriever = mock_retriever

    result = retriever.invoke("test query")

    assert result == docs
    mock_retriever.invoke.assert_called_once_with("test query")


def test_normal_retriever_invoke_not_initialized(mocker):
    retriever = NormalRetriever.model_construct(vectorstore=mocker.Mock(), k=30)
    retriever._retriever = None

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")


def test_normal_retriever_invoke_failure(mocker):
    mock_retriever = mocker.Mock()
    mock_retriever.invoke.side_effect = Exception("检索失败")

    retriever = NormalRetriever.model_construct(vectorstore=mocker.Mock(), k=30)
    retriever._retriever = mock_retriever

    with pytest.raises(RetrievalError):
        retriever.invoke("test query")