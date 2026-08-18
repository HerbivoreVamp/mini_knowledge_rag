import pytest

from ingestion.loader import load_md
from ingestion.hierarchical import create_docstore, create_parent_retriever, add_documents_to_retriever
from storage.vectorstore import create_empty_vectorstore, save_vectorstore,add_documents

# tests/test_ingestion_service.py

from ingestion.service import ingestion_service


def test_ingestion_service(mocker, tmp_path):
    mock_load = mocker.patch(
        "ingestion.service.load_md",
        return_value=["doc"]
    )

    mock_create_empty_vectorstore = mocker.patch(
        "ingestion.service.create_empty_vectorstore",
        return_value="fake_vectorstore"
    )

    mock_create_parent_retriever = mocker.patch(
        "ingestion.service.create_parent_retriever",
        return_value="fake_retriever"
    )

    mock_add = mocker.patch(
        "ingestion.service.add_documents",
        return_value="add_vectorstore"
    )

    mock_add_documents_to_retriever = mocker.patch(
        "ingestion.service.add_documents_to_retriever",
        return_value="add_documents_to_retriever"
    )

    mock_save = mocker.patch(
        "ingestion.service.save_vectorstore"
    )

    result = ingestion_service(
        tmp_path,
        "document",
        "fake_embedding",
        tmp_path,
        "test_index",
        tmp_path,
    )

    assert result == "add_documents_to_retriever"

