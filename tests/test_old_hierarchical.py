import pytest

from backend.ingestion.hierarchical_old import (
    create_parent_retriever,
    add_documents_to_retriever,
)
from backend.storage.docstore import JsonDocStore
from backend.core.exceptions import RetrievalError
from langchain_core.documents import Document


class TestCreateParentRetriever:
    def test_create_parent_retriever(self, mocker):
        mock_vectorstore = mocker.MagicMock()
        mock_parent_store = mocker.MagicMock(spec=JsonDocStore)

        mock_splitter = mocker.MagicMock()
        mock_create_splitter = mocker.patch(
            "backend.ingestion.hierarchical_old.create_text_splitter",
            return_value=mock_splitter,
        )

        mock_parent_retriever_cls = mocker.patch(
            "backend.ingestion.hierarchical_old.ParentDocumentRetriever",
        )

        retriever = create_parent_retriever(mock_vectorstore, mock_parent_store)

        assert mock_create_splitter.call_count == 2
        mock_parent_retriever_cls.assert_called_once_with(
            vectorstore=mock_vectorstore,
            docstore=mock_parent_store,
            parent_splitter=mock_splitter,
            child_splitter=mock_splitter,
            search_kwargs={"k": 2},
        )

    def test_create_parent_retriever_error(self, mocker):
        mock_vectorstore = mocker.MagicMock()
        mock_parent_store = mocker.MagicMock(spec=JsonDocStore)

        mocker.patch(
            "backend.ingestion.hierarchical_old.create_text_splitter",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "backend.ingestion.hierarchical_old.ParentDocumentRetriever",
            side_effect=ValueError("init error"),
        )

        with pytest.raises(RetrievalError):
            create_parent_retriever(mock_vectorstore, mock_parent_store)


class TestAddDocumentsToRetriever:
    def test_add_documents_to_retriever(self, mocker):
        mock_retriever = mocker.MagicMock()
        docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
        ]

        result = add_documents_to_retriever(mock_retriever, docs)

        mock_retriever.add_documents.assert_called_once_with(docs)
        assert result is mock_retriever
