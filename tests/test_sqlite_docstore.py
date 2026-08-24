import sqlite3
import pytest

from backend.storage.sqlite_docstore import (
    SqliteDocStore,
    create_sqlite_docstore,
)
from backend.core.exceptions import DocStoreError
from langchain_core.documents import Document


class TestSqliteDocStoreInit:
    def test_init_empty_dir(self, tmp_path):
        store = SqliteDocStore(tmp_path)

        assert store.store_dir.exists()
        assert store.path.exists()
        assert store.count() == 0

    def test_init_creates_dir(self, tmp_path):
        store_dir = tmp_path / "nested" / "parent_store"
        store = SqliteDocStore(store_dir)

        assert store_dir.exists()
        assert store.path.exists()

    def test_init_existing_db(self, tmp_path):
        store1 = SqliteDocStore(tmp_path)
        store1.mset([("key1", Document(page_content="hello"))])
        store1.close()

        store2 = SqliteDocStore(tmp_path)
        assert store2.count() == 1

        docs = store2.mget(["key1"])
        assert docs[0].page_content == "hello"

    def test_init_corrupted_db(self, tmp_path):
        db_file = tmp_path / "docstore.db"
        db_file.write_bytes(b"not a sqlite file")

        with pytest.raises(DocStoreError):
            SqliteDocStore(tmp_path)


class TestSqliteDocStoreCRUD:
    def test_mset(self, tmp_path):
        store = SqliteDocStore(tmp_path)
        docs = [
            ("key1", Document(page_content="hello", metadata={"a": 1})),
            ("key2", Document(page_content="world", metadata={"b": 2})),
        ]
        store.mset(docs)

        assert store.count() == 2
        result = store.mget(["key1", "key2"])
        assert result[0].page_content == "hello"
        assert result[0].metadata == {"a": 1}
        assert result[1].page_content == "world"
        assert result[1].metadata == {"b": 2}

    def test_mset_upsert_overwrites(self, tmp_path):
        store = SqliteDocStore(tmp_path)
        store.mset([("key1", Document(page_content="old"))])
        store.mset([("key1", Document(page_content="new"))])

        assert store.count() == 1
        assert store.mget(["key1"])[0].page_content == "new"

    def test_mset_empty_items(self, tmp_path):
        store = SqliteDocStore(tmp_path)
        store.mset([])

        assert store.count() == 0

    def test_mget_missing_key(self, tmp_path):
        store = SqliteDocStore(tmp_path)
        store.mset([("key1", Document(page_content="hello"))])

        result = store.mget(["key1", "missing_key"])
        assert len(result) == 2
        assert result[0] is not None
        assert result[1] is None

    def test_mget_empty_keys(self, tmp_path):
        store = SqliteDocStore(tmp_path)

        assert store.mget([]) == []

    def test_mdelete(self, tmp_path):
        store = SqliteDocStore(tmp_path)
        store.mset([
            ("key1", Document(page_content="hello")),
            ("key2", Document(page_content="world")),
        ])

        store.mdelete(["key1", "missing_key"])
        assert store.count() == 1
        assert store.mget(["key1"])[0] is None
        assert store.mget(["key2"])[0] is not None

    def test_metadata_chinese(self, tmp_path):
        store = SqliteDocStore(tmp_path)
        store.mset([
            ("key1", Document(page_content="中文内容", metadata={"来源": "测试文档"})),
        ])

        doc = store.mget(["key1"])[0]
        assert doc.page_content == "中文内容"
        assert doc.metadata == {"来源": "测试文档"}


class TestSqliteDocStorePersistence:
    def test_data_survives_reopen(self, tmp_path):
        store1 = SqliteDocStore(tmp_path)
        store1.mset([
            ("key1", Document(page_content="hello", metadata={"a": 1})),
            ("key2", Document(page_content="world", metadata={"b": 2})),
        ])
        store1.close()

        store2 = SqliteDocStore(tmp_path)
        assert store2.count() == 2

        result = store2.mget(["key1", "key2"])
        assert result[0].page_content == "hello"
        assert result[1].page_content == "world"

    def test_delete_survives_reopen(self, tmp_path):
        store1 = SqliteDocStore(tmp_path)
        store1.mset([
            ("key1", Document(page_content="hello")),
            ("key2", Document(page_content="world")),
        ])
        store1.mdelete(["key1"])
        store1.close()

        store2 = SqliteDocStore(tmp_path)
        assert store2.count() == 1
        assert store2.mget(["key1"])[0] is None


class TestSqliteDocStoreYieldKeys:
    def test_yield_keys_all(self, tmp_path):
        store = SqliteDocStore(tmp_path)
        store.mset([
            ("a_key1", Document(page_content="hello")),
            ("a_key2", Document(page_content="world")),
            ("b_key1", Document(page_content="foo")),
        ])

        keys = list(store.yield_keys())
        assert sorted(keys) == ["a_key1", "a_key2", "b_key1"]

    def test_yield_keys_with_prefix(self, tmp_path):
        store = SqliteDocStore(tmp_path)
        store.mset([
            ("a_key1", Document(page_content="hello")),
            ("a_key2", Document(page_content="world")),
            ("b_key1", Document(page_content="foo")),
        ])

        keys = list(store.yield_keys(prefix="a_"))
        assert sorted(keys) == ["a_key1", "a_key2"]

    def test_yield_keys_empty_store(self, tmp_path):
        store = SqliteDocStore(tmp_path)

        assert list(store.yield_keys()) == []


class TestSqliteDocStoreGetAll:
    def test_get_all_documents(self, tmp_path):
        store = SqliteDocStore(tmp_path)
        store.mset([
            ("key1", Document(page_content="hello", metadata={"a": 1})),
            ("key2", Document(page_content="world", metadata={"b": 2})),
        ])

        docs = store.get_all_documents()
        assert len(docs) == 2
        contents = {doc.page_content for doc in docs}
        assert contents == {"hello", "world"}

    def test_get_all_documents_empty(self, tmp_path):
        store = SqliteDocStore(tmp_path)

        assert store.get_all_documents() == []


class TestCreateSqliteDocstore:
    def test_create_sqlite_docstore(self, tmp_path):
        store = create_sqlite_docstore(tmp_path)
        assert isinstance(store, SqliteDocStore)
        assert store.store_dir == tmp_path
        assert store.path == tmp_path / "docstore.db"

    def test_create_sqlite_docstore_str_path(self, tmp_path):
        store = create_sqlite_docstore(str(tmp_path))
        assert isinstance(store, SqliteDocStore)
        assert store.store_dir == tmp_path
