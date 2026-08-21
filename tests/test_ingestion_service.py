import pytest

from backend.ingestion.service import ingestion_service


def test_ingestion_service(mocker, tmp_path):
    mock_docstore = mocker.MagicMock()
    mock_create_docstore = mocker.patch(
        "backend.ingestion.service.create_docstore",
        return_value=mock_docstore
    )

    mock_vectorstore = mocker.MagicMock()
    mock_create_empty_vectorstore = mocker.patch(
        "backend.ingestion.service.create_empty_vectorstore",
        return_value=mock_vectorstore
    )

    mock_save = mocker.patch(
        "backend.ingestion.service.save_vectorstore"
    )

    mock_load = mocker.patch(
        "backend.ingestion.service.load_md",
        return_value=["doc"]
    )

    mock_parent_splitter = mocker.MagicMock()
    mock_child_splitter = mocker.MagicMock()
    mock_create_hierarchy_splitter = mocker.patch(
        "backend.ingestion.service.create_hierarchy_splitter",
        return_value=(mock_parent_splitter, mock_child_splitter)
    )

    mock_ingest = mocker.patch(
        "backend.ingestion.service.ingest_documents"
    )

    mock_retriever = mocker.MagicMock()
    mock_HierarchicalRetriever = mocker.patch(
        "backend.ingestion.service.HierarchicalRetriever",
        return_value=mock_retriever
    )

    mock_reranker = mocker.MagicMock()

    result = ingestion_service(
        str(tmp_path),
        "document",
        "fake_embedding",
        str(tmp_path),
        "test_index",
        str(tmp_path),
        mock_reranker,
    )

    mock_create_docstore.assert_called_once()
    mock_create_empty_vectorstore.assert_called_once_with("fake_embedding")
    mock_load.assert_called_once()
    mock_create_hierarchy_splitter.assert_called_once()
    mock_ingest.assert_called_once_with(
        docs=["doc"],
        vectorstore=mock_vectorstore,
        parent_store=mock_docstore,
        parent_splitter=mock_parent_splitter,
        child_splitter=mock_child_splitter,
    )
    mock_HierarchicalRetriever.assert_called_once_with(
        vectorstore=mock_vectorstore,
        parent_store=mock_docstore,
        reranker=mock_reranker,
        k=30,
        rerank_k=5,
    )
    assert mock_save.call_count == 2
    assert result == (mock_retriever, mock_vectorstore)
