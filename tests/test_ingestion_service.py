import pytest

from ingestion.loader import load_md
from ingestion.splitter import split_text
from storage.vectorstore import build_vectorstore, save_vectorstore

# tests/test_ingestion_service.py

from ingestion.service import ingestion_service


def test_ingestion_service(mocker, tmp_path):
    mock_load = mocker.patch(
        "ingestion.service.load_md",
        return_value=["doc"]
    )

    mock_split = mocker.patch(
        "ingestion.service.split_text",
        return_value=["chunk"]
    )

    mock_build = mocker.patch(
        "ingestion.service.build_vectorstore",
        return_value="fake_vectorstore"
    )

    mock_save = mocker.patch(
        "ingestion.service.save_vectorstore"
    )

    result = ingestion_service(
        tmp_path,
        "document",
        "fake_embedding",
        tmp_path,
        "test_index"
    )

    assert result == "fake_vectorstore"

