import json
import pytest

from storage.docstore import (
    JsonDocStore,
    create_docstore,
)
from core.exceptions import DocStoreError
from langchain_core.documents import Document


class TestJsonDocStore:
    def test_init_empty_dir(self, tmp_path):
        store = JsonDocStore(tmp_path)

        assert store.data == {}
        assert store.parent_store_dir.exists()
        assert not store.path.exists()

    def test_init_existing_data(self, tmp_path):
        data = {
            "key1": {"page_content": "hello", "metadata": {"a": 1}},
            "key2": {"page_content": "world", "metadata": {"b": 2}},
        }
        parents_file = tmp_path / "parents.json"
        with open(parents_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        store = JsonDocStore(tmp_path)
        assert store.data == data

    def test_init_corrupted_json(self, tmp_path):
        parents_file = tmp_path / "parents.json"
        parents_file.write_text("not valid json", encoding="utf-8")

        with pytest.raises(DocStoreError):
            JsonDocStore(tmp_path)

    def test_mset(self, tmp_path):
        store = JsonDocStore(tmp_path)
        docs = [
            ("key1", Document(page_content="hello", metadata={"a": 1})),
            ("key2", Document(page_content="world", metadata={"b": 2})),
        ]
        store.mset(docs)
        assert "key1" in store.data
        assert store.data["key1"]["page_content"] == "hello"
        assert store.data["key1"]["metadata"] == {"a": 1}
        assert "key2" in store.data
        assert store.data["key2"]["page_content"] == "world"
        assert store.data["key2"]["metadata"] == {"b": 2}

        # 验证持久化
        with open(store.path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == store.data

    def test_mget(self, tmp_path):
        store = JsonDocStore(tmp_path)
        docs = [
            ("key1", Document(page_content="hello", metadata={"a": 1})),
            ("key2", Document(page_content="world", metadata={"b": 2})),
        ]
        store.mset(docs)

        result = store.mget(["key1", "key2", "missing_key"])
        assert len(result) == 3
        assert result[0].page_content == "hello"
        assert result[0].metadata == {"a": 1}
        assert result[1].page_content == "world"
        assert result[1].metadata == {"b": 2}
        assert result[2] is None

    def test_mdelete(self, tmp_path):
        store = JsonDocStore(tmp_path)
        docs = [
            ("key1", Document(page_content="hello")),
            ("key2", Document(page_content="world")),
        ]
        store.mset(docs)

        store.mdelete(["key1", "missing_key"])
        assert "key1" not in store.data
        assert "key2" in store.data

        # 验证持久化
        with open(store.path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert "key1" not in saved
        assert "key2" in saved

    def test_yield_keys_all(self, tmp_path):
        store = JsonDocStore(tmp_path)
        docs = [
            ("a_key1", Document(page_content="hello")),
            ("a_key2", Document(page_content="world")),
            ("b_key1", Document(page_content="foo")),
        ]
        store.mset(docs)

        keys = list(store.yield_keys())
        assert len(keys) == 3
        assert "a_key1" in keys
        assert "a_key2" in keys
        assert "b_key1" in keys

    def test_yield_keys_with_prefix(self, tmp_path):
        store = JsonDocStore(tmp_path)
        docs = [
            ("a_key1", Document(page_content="hello")),
            ("a_key2", Document(page_content="world")),
            ("b_key1", Document(page_content="foo")),
        ]
        store.mset(docs)

        keys = list(store.yield_keys(prefix="a_"))
        assert keys == ["a_key1", "a_key2"]


class TestCreateDocstore:
    def test_create_docstore(self, tmp_path):
        store = create_docstore(str(tmp_path))
        assert isinstance(store, JsonDocStore)
        assert store.parent_store_dir == tmp_path