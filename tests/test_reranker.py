import pytest
from pathlib import Path

from backend.config.reranker import BGEReranker, init_reranker_model
from backend.core.exceptions import RerankerError
from langchain_core.documents import Document


class FakeCrossEncoder:
    def __init__(self, model_name_or_path, max_length, device):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.device = device

    def predict(self, pairs):
        return [0.5 + i * 0.1 for i in range(len(pairs))]


def test_bge_reranker_init_success(mocker):
    mocker.patch(
        "backend.config.reranker.CrossEncoder",
        new=FakeCrossEncoder
    )

    reranker = BGEReranker(Path("/fake/path"), device="cpu")

    assert reranker is not None
    assert reranker.model is not None


def test_bge_reranker_init_failure(mocker):
    mocker.patch(
        "backend.config.reranker.CrossEncoder",
        side_effect=Exception("模型加载失败")
    )

    with pytest.raises(RerankerError):
        BGEReranker(Path("/fake/path"), device="cpu")


def test_bge_reranker_rerank_success(mocker):
    mocker.patch(
        "backend.config.reranker.CrossEncoder",
        new=FakeCrossEncoder
    )

    reranker = BGEReranker(Path("/fake/path"), device="cpu")

    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
        Document(page_content="doc4"),
        Document(page_content="doc5"),
    ]

    result = reranker.rerank("test query", docs, top_k=3)

    assert len(result) == 3
    # 最高分的是最后一个（0.5 + 4*0.1 = 0.9），所以返回 doc5, doc4, doc3
    assert result[0].page_content == "doc5"
    assert result[1].page_content == "doc4"
    assert result[2].page_content == "doc3"


def test_bge_reranker_rerank_default_top_k(mocker):
    mocker.patch(
        "backend.config.reranker.CrossEncoder",
        new=FakeCrossEncoder
    )

    reranker = BGEReranker(Path("/fake/path"), device="cpu")

    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
    ]

    result = reranker.rerank("test query", docs)

    assert len(result) == 3


def test_bge_reranker_rerank_top_k_larger_than_docs(mocker):
    mocker.patch(
        "backend.config.reranker.CrossEncoder",
        new=FakeCrossEncoder
    )

    reranker = BGEReranker(Path("/fake/path"), device="cpu")

    docs = [Document(page_content="doc1")]

    result = reranker.rerank("test query", docs, top_k=5)

    assert len(result) == 1


def test_bge_reranker_rerank_empty_docs(mocker):
    mocker.patch(
        "backend.config.reranker.CrossEncoder",
        new=FakeCrossEncoder
    )

    reranker = BGEReranker(Path("/fake/path"), device="cpu")

    result = reranker.rerank("test query", [], top_k=3)

    assert result == []


def test_bge_reranker_rerank_predict_failure(mocker):
    fake_model = mocker.Mock()
    fake_model.predict.side_effect = Exception("预测失败")
    mocker.patch(
        "backend.config.reranker.CrossEncoder",
        return_value=fake_model
    )

    reranker = BGEReranker(Path("/fake/path"), device="cpu")

    docs = [Document(page_content="doc1")]

    with pytest.raises(RerankerError):
        reranker.rerank("test query", docs)


def test_init_reranker_model_success(mocker):
    mock_reranker_class = mocker.patch(
        "backend.config.reranker.BGEReranker",
        return_value="fake_reranker"
    )

    result = init_reranker_model(Path("/fake/dir"), "bge-model", device="cpu")

    assert result == "fake_reranker"
    mock_reranker_class.assert_called_once_with(
        Path("/fake/dir") / "bge-model",
        device="cpu"
    )


def test_init_reranker_model_default_device(mocker):
    mock_reranker_class = mocker.patch(
        "backend.config.reranker.BGEReranker",
        return_value="fake_reranker"
    )

    result = init_reranker_model(Path("/fake/dir"), "bge-model")

    mock_reranker_class.assert_called_once_with(
        Path("/fake/dir") / "bge-model",
        device="cpu"
    )


def test_init_reranker_model_failure(mocker):
    mocker.patch(
        "backend.config.reranker.BGEReranker",
        side_effect=Exception("模型加载失败")
    )

    with pytest.raises(RerankerError):
        init_reranker_model(Path("/fake/dir"), "bge-model")