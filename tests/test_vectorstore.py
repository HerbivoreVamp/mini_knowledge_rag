import pytest
from storage.vectorstore import build_vectorstore, save_vectorstore, load_vectorstore
from core.exceptions import VectorStoreError, VectorStoreNotFoundError
from langchain_core.documents import Document


class FakeEmbedding:

    def embed_documents(self, texts):
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def test_build_vectorstore_docs_is_none():
    fake_emb = FakeEmbedding()
    with pytest.raises(VectorStoreError):
        splits = None
        vectorstore = build_vectorstore(
            splits,
            fake_emb
        )


def test_build_vectorstore_docs_isnt_doc():
    fake_emb = FakeEmbedding()
    with pytest.raises(VectorStoreError):
        splits = [
            "测试文本"
        ]
        vectorstore = build_vectorstore(
            splits,
            fake_emb
        )


def test_build_vectorstore_emb_is_none():
    fake_emb = None
    with pytest.raises(VectorStoreError):
        splits = [
            Document(page_content="测试文本")
        ]
        vectorstore = build_vectorstore(
            splits,
            fake_emb
        )


def test_build_vectorstore_success():
    fake_emb = FakeEmbedding()
    splits = [
        Document(page_content="测试文本")
    ]
    vectorstore = build_vectorstore(
        splits,
        fake_emb
    )
    assert vectorstore is not None


def test_save_vectorstore_is_none(mocker):
    fake_dir = mocker.Mock()
    fake_index = mocker.Mock()
    fake_vec = None
    with pytest.raises(VectorStoreError):
        save_vectorstore(fake_vec, fake_dir, fake_index)


def test_save_dir_isnt_str(mocker):
    fake_vec = mocker.Mock()
    fake_dir = 123
    fake_index = mocker.Mock()
    with pytest.raises(VectorStoreError):
        save_vectorstore(fake_vec, fake_dir, fake_index)


def test_save_index_isnt_str(mocker):
    fake_dir = mocker.Mock()
    fake_index = 123
    fake_vec = mocker.Mock()
    with pytest.raises(VectorStoreError):
        save_vectorstore(fake_vec, fake_dir, fake_index)


def test_save_vectorstore_success(mocker, tmp_path):
    fake_vec = mocker.Mock()

    path = save_vectorstore(
        fake_vec,
        str(tmp_path),
        "test_index"
    )

    assert path == tmp_path / "test_index"

    fake_vec.save_local.assert_called_once_with(
        folder_path=str(tmp_path / "test_index"),
        index_name="test_index"
    )


def test_load_vectorstore_isnt_found():
    fake_dir = "不存在的文件夹"
    fake_emb = FakeEmbedding()
    fake_index = "不存在的文件"
    with pytest.raises(VectorStoreNotFoundError):
        load_vectorstore(fake_dir, fake_emb, fake_index)


def test_load_vectorstore_success(tmp_path):
    embedding = FakeEmbedding()

    docs = [
        Document(page_content="测试文本")
    ]

    vectorstore = build_vectorstore(
        docs,
        embedding
    )

    save_vectorstore(
        vectorstore,
        str(tmp_path),
        "test_index"
    )

    loaded = load_vectorstore(
        str(tmp_path),
        embedding,
        "test_index"
    )

    assert loaded is not None
