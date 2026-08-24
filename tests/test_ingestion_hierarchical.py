import pytest

from langchain_core.documents import Document

from backend.ingestion.hierarchical import ingest_documents
from backend.core.exceptions import RAGError, VectorStoreError
from backend.storage.docstore import JsonDocStore


class FakeSplitter:
    def split_documents(self, docs):
        return docs


def test_ingest_documents_success(mocker, tmp_path):
    from backend.storage.docstore import JsonDocStore

    parent_store = JsonDocStore(tmp_path)
    vectorstore = mocker.Mock()
    parent_splitter = FakeSplitter()
    child_splitter = FakeSplitter()

    docs = [Document(page_content="test content", metadata={"source": "test.md"})]

    result = ingest_documents(docs, vectorstore, parent_store, parent_splitter, child_splitter)

    assert result["parents"] == 1
    assert result["children"] == 1
    vectorstore.add_documents.assert_called_once()


def test_ingest_documents_multiple_docs(mocker, tmp_path):
    from backend.storage.docstore import JsonDocStore

    parent_store = JsonDocStore(tmp_path)
    vectorstore = mocker.Mock()
    parent_splitter = FakeSplitter()
    child_splitter = FakeSplitter()

    docs = [
        Document(page_content="doc1", metadata={"source": "test.md"}),
        Document(page_content="doc2", metadata={"source": "test.md"}),
        Document(page_content="doc3", metadata={"source": "test.md"}),
    ]

    result = ingest_documents(docs, vectorstore, parent_store, parent_splitter, child_splitter)

    assert result["parents"] == 3
    assert result["children"] == 3


def test_ingest_documents_parent_splitter_produces_multiple(mocker, tmp_path):
    from backend.storage.docstore import JsonDocStore

    parent_store = JsonDocStore(tmp_path)
    vectorstore = mocker.Mock()

    class MultiParentSplitter:
        def split_documents(self, docs):
            return [
                Document(page_content="parent1", metadata={"source": "test.md"}),
                Document(page_content="parent2", metadata={"source": "test.md"}),
            ]

    child_splitter = FakeSplitter()

    docs = [Document(page_content="original", metadata={"source": "test.md"})]

    result = ingest_documents(docs, vectorstore, parent_store, MultiParentSplitter(), child_splitter)

    assert result["parents"] == 2
    assert result["children"] == 2


def test_ingest_documents_child_splitter_produces_multiple(mocker, tmp_path):
    from backend.storage.docstore import JsonDocStore

    parent_store = JsonDocStore(tmp_path)
    vectorstore = mocker.Mock()
    parent_splitter = FakeSplitter()

    class MultiChildSplitter:
        def split_documents(self, docs):
            return [
                Document(page_content="child1", metadata={"source": "test.md"}),
                Document(page_content="child2", metadata={"source": "test.md"}),
                Document(page_content="child3", metadata={"source": "test.md"}),
            ]

    docs = [Document(page_content="original", metadata={"source": "test.md"})]

    result = ingest_documents(docs, vectorstore, parent_store, parent_splitter, MultiChildSplitter())

    assert result["parents"] == 1
    assert result["children"] == 3


def test_ingest_documents_empty_docs(mocker, tmp_path):
    with pytest.raises(VectorStoreError):
        parent_store = JsonDocStore(tmp_path)
        vectorstore = mocker.Mock()
        parent_splitter = FakeSplitter()
        child_splitter = FakeSplitter()

        result = ingest_documents([], vectorstore, parent_store, parent_splitter, child_splitter)


def test_ingest_documents_parent_id_assigned(mocker, tmp_path):
    from backend.storage.docstore import JsonDocStore

    parent_store = JsonDocStore(tmp_path)
    vectorstore = mocker.Mock()
    parent_splitter = FakeSplitter()
    child_splitter = FakeSplitter()

    docs = [Document(page_content="test", metadata={"source": "test.md"})]

    result = ingest_documents(docs, vectorstore, parent_store, parent_splitter, child_splitter)

    # 验证 parent_store 中保存的 parent 有 parent_id
    keys = list(parent_store.yield_keys())
    assert len(keys) == 1
    parent = parent_store.mget(keys)[0]
    assert "parent_id" in parent.metadata
    assert parent.metadata["parent_id"] == keys[0]


def test_ingest_documents_child_has_parent_id(mocker, tmp_path):
    from backend.storage.docstore import JsonDocStore

    parent_store = JsonDocStore(tmp_path)
    vectorstore = mocker.Mock()
    parent_splitter = FakeSplitter()
    child_splitter = FakeSplitter()

    docs = [Document(page_content="test", metadata={"source": "test.md"})]

    ingest_documents(docs, vectorstore, parent_store, parent_splitter, child_splitter)

    # 验证 add_documents 被调用时传入的 child 有 parent_id
    called_docs = vectorstore.add_documents.call_args[0][0]
    assert len(called_docs) == 1
    assert "parent_id" in called_docs[0].metadata


def test_ingest_documents_add_documents_failure(mocker, tmp_path):
    from backend.storage.docstore import JsonDocStore

    parent_store = JsonDocStore(tmp_path)
    vectorstore = mocker.Mock()
    vectorstore.add_documents.side_effect = RAGError("模拟失败")
    parent_splitter = FakeSplitter()
    child_splitter = FakeSplitter()

    docs = [Document(page_content="test", metadata={"source": "test.md"})]

    with pytest.raises(RAGError):
        ingest_documents(docs, vectorstore, parent_store, parent_splitter, child_splitter)
