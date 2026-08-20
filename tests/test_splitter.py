import pytest
from backend.ingestion.splitter import split_text
from backend.core.exceptions import SplitError
from random import randrange
from langchain_core.documents import Document


def test_not_docs():
    docs = "12345"
    with pytest.raises(SplitError):
        splits = split_text([docs])


def test_chunk_size_below_0():
    docs = "12345"
    document = Document(page_content=docs)
    with pytest.raises(SplitError):
        splits = split_text([document], chunk_size=randrange(-100, 0))


def test_chunk_overlap_below_0():
    docs = "12345"
    document = Document(page_content=docs)
    with pytest.raises(SplitError):
        splits = split_text([document], chunk_overlap=randrange(-100, 0))


def test_chunk_overlap_above_chunk_size():
    docs = "12345"
    document = Document(page_content=docs)
    with pytest.raises(SplitError):
        splits = split_text([document], chunk_size=50, chunk_overlap=100)


def test_all_documents_is_empty():
    docs = ""
    document = Document(page_content=docs)
    with pytest.raises(SplitError):
        splits = split_text([document])


def test_split_text():
    docs = "hello world " * 100
    document = Document(page_content=docs)
    splits = split_text([document])
    assert len(splits) > 1
