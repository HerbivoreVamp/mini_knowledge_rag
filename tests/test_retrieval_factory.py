import pytest

from backend.retrieval.factory import create_retriever
from backend.core.exceptions import RetrievalError


def _make_vectorstore(mocker, docs=None):
    vectorstore = mocker.MagicMock()
    vectorstore.docstore._dict.values.return_value = docs or []
    return vectorstore


def test_create_retriever_full_chain(mocker):
    from langchain_core.documents import Document

    docs = [Document(page_content="doc1")]
    vectorstore = _make_vectorstore(mocker, docs=docs)
    parent_store = mocker.Mock()
    reranker = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_sparse = mocker.Mock()
    mock_hybrid = mocker.Mock()
    mock_rerank = mocker.Mock()
    mock_hierarchical = mocker.Mock()

    mock_DenseRetriever = mocker.patch(
        "backend.retrieval.factory.DenseRetriever",
        return_value=mock_dense,
    )
    mock_BM25Retriever = mocker.patch(
        "backend.retrieval.factory.BM25Retriever",
    )
    mock_BM25Retriever.from_documents.return_value = mock_sparse
    mock_HybridRetriever = mocker.patch(
        "backend.retrieval.factory.HybridRetriever",
        return_value=mock_hybrid,
    )
    mock_RerankRetriever = mocker.patch(
        "backend.retrieval.factory.RerankRetriever",
        return_value=mock_rerank,
    )
    mock_HierarchicalRetriever = mocker.patch(
        "backend.retrieval.factory.HierarchicalRetriever",
        return_value=mock_hierarchical,
    )

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=reranker,
        hierarchical=True,
        hybrid=True,
        k=30,
        sparse_k=20,
        rerank_topk=5,
        parent_k=3,
    )

    mock_DenseRetriever.assert_called_once_with(vectorstore=vectorstore, k=30)
    # BM25 现在会带一个分词函数进去 所以按调用参数断言而不是整体相等
    bm25_args, bm25_kwargs = mock_BM25Retriever.from_documents.call_args
    assert bm25_args == (docs,)
    assert bm25_kwargs["k"] == 20
    # 默认分词方案: 去标点, 且保留 create_agent 这类标识符
    assert bm25_kwargs["preprocess_func"]("`create_agent`:") == ["create_agent"]
    mock_HybridRetriever.assert_called_once_with(
        dense_retriever=mock_dense,
        sparse_retriever=mock_sparse,
        k=50,
    )
    mock_RerankRetriever.assert_called_once_with(retriever=mock_hybrid, reranker=reranker, top_k=5)
    mock_HierarchicalRetriever.assert_called_once_with(
        child_retriever=mock_rerank,
        parent_store=parent_store,
        parent_k=3,
    )
    assert result == mock_hierarchical


def test_create_retriever_no_hybrid(mocker):
    vectorstore = _make_vectorstore(mocker)
    parent_store = mocker.Mock()
    reranker = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_rerank = mocker.Mock()
    mock_hierarchical = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_HybridRetriever = mocker.patch("backend.retrieval.factory.HybridRetriever")
    mocker.patch("backend.retrieval.factory.RerankRetriever", return_value=mock_rerank)
    mocker.patch("backend.retrieval.factory.HierarchicalRetriever", return_value=mock_hierarchical)

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=reranker,
        hierarchical=True,
        hybrid=False,
    )

    mock_BM25Retriever.from_documents.assert_not_called()
    mock_HybridRetriever.assert_not_called()
    assert result == mock_hierarchical


def test_create_retriever_no_reranker(mocker):
    vectorstore = _make_vectorstore(mocker)
    parent_store = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_sparse = mocker.Mock()
    mock_hybrid = mocker.Mock()
    mock_hierarchical = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_BM25Retriever.from_documents.return_value = mock_sparse
    mock_HybridRetriever = mocker.patch(
        "backend.retrieval.factory.HybridRetriever",
        return_value=mock_hybrid,
    )
    mock_RerankRetriever = mocker.patch("backend.retrieval.factory.RerankRetriever")
    mocker.patch("backend.retrieval.factory.HierarchicalRetriever", return_value=mock_hierarchical)

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=None,
        hierarchical=True,
        hybrid=True,
    )

    mock_RerankRetriever.assert_not_called()
    assert result == mock_hierarchical


def test_create_retriever_no_hierarchical(mocker):
    vectorstore = _make_vectorstore(mocker)
    parent_store = mocker.Mock()
    reranker = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_rerank = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_HybridRetriever = mocker.patch("backend.retrieval.factory.HybridRetriever")
    mocker.patch("backend.retrieval.factory.RerankRetriever", return_value=mock_rerank)
    mock_HierarchicalRetriever = mocker.patch("backend.retrieval.factory.HierarchicalRetriever")

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=reranker,
        hierarchical=False,
        hybrid=False,
    )

    mock_HierarchicalRetriever.assert_not_called()
    assert result == mock_rerank


def test_create_retriever_hybrid_only(mocker):
    vectorstore = _make_vectorstore(mocker)
    parent_store = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_sparse = mocker.Mock()
    mock_hybrid = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_BM25Retriever.from_documents.return_value = mock_sparse
    mocker.patch("backend.retrieval.factory.HybridRetriever", return_value=mock_hybrid)
    mock_RerankRetriever = mocker.patch("backend.retrieval.factory.RerankRetriever")
    mock_HierarchicalRetriever = mocker.patch("backend.retrieval.factory.HierarchicalRetriever")

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=None,
        hierarchical=False,
        hybrid=True,
    )

    mock_RerankRetriever.assert_not_called()
    mock_HierarchicalRetriever.assert_not_called()
    assert result == mock_hybrid


def test_create_retriever_plain(mocker):
    vectorstore = _make_vectorstore(mocker)
    parent_store = mocker.Mock()

    mock_dense = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_HybridRetriever = mocker.patch("backend.retrieval.factory.HybridRetriever")
    mock_RerankRetriever = mocker.patch("backend.retrieval.factory.RerankRetriever")
    mock_HierarchicalRetriever = mocker.patch("backend.retrieval.factory.HierarchicalRetriever")

    result = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        reranker=None,
        hierarchical=False,
        hybrid=False,
    )

    mock_BM25Retriever.from_documents.assert_not_called()
    mock_HybridRetriever.assert_not_called()
    mock_RerankRetriever.assert_not_called()
    mock_HierarchicalRetriever.assert_not_called()
    assert result == mock_dense


def test_create_retriever_dense_retriever_fails(mocker):
    mocker.patch(
        "backend.retrieval.factory.DenseRetriever",
        side_effect=RetrievalError("创建失败"),
    )

    with pytest.raises(RetrievalError):
        create_retriever(
            vectorstore=_make_vectorstore(mocker),
            parent_store=mocker.Mock(),
        )


def test_create_retriever_child_store_source(mocker, tmp_path):
    from langchain_core.documents import Document
    from backend.storage.sqlite_docstore import SqliteDocStore

    child_docs = [
        Document(page_content="child1", metadata={"chunk_id": "c1"}),
        Document(page_content="child2", metadata={"chunk_id": "c2"}),
    ]
    child_store = SqliteDocStore(tmp_path / "child")
    child_store.mset([("c1", child_docs[0]), ("c2", child_docs[1])])

    vectorstore = _make_vectorstore(mocker, docs=[])
    parent_store = mocker.Mock()
    reranker = mocker.Mock()

    mock_dense = mocker.Mock()
    mock_sparse = mocker.Mock()
    mock_hybrid = mocker.Mock()
    mock_rerank = mocker.Mock()
    mock_hierarchical = mocker.Mock()

    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mock_dense)
    mock_BM25Retriever = mocker.patch("backend.retrieval.factory.BM25Retriever")
    mock_BM25Retriever.from_documents.return_value = mock_sparse
    mocker.patch("backend.retrieval.factory.HybridRetriever", return_value=mock_hybrid)
    mocker.patch("backend.retrieval.factory.RerankRetriever", return_value=mock_rerank)
    mocker.patch("backend.retrieval.factory.HierarchicalRetriever", return_value=mock_hierarchical)

    # 用 spy 监控 close 调用
    close_spy = mocker.patch.object(child_store, "close", wraps=child_store.close)

    create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        child_store=child_store,
        reranker=reranker,
    )

    # BM25 语料来自 child_store 的文档
    called_docs = mock_BM25Retriever.from_documents.call_args[0][0]
    assert len(called_docs) == 2
    contents = {d.page_content for d in called_docs}
    assert contents == {"child1", "child2"}
    # child_store 用完关闭
    close_spy.assert_called_once()


def _tokenizer_used(mock_BM25Retriever):
    """取出 factory 传给 BM25 的分词函数"""
    return mock_BM25Retriever.from_documents.call_args.kwargs["preprocess_func"]


def _patch_bm25(mocker):
    mock_BM25Retriever = mocker.patch(
        "backend.retrieval.factory.BM25Retriever",
    )
    mock_BM25Retriever.from_documents.return_value = mocker.Mock()
    mocker.patch("backend.retrieval.factory.DenseRetriever", return_value=mocker.Mock())
    mocker.patch("backend.retrieval.factory.HybridRetriever", return_value=mocker.Mock())
    return mock_BM25Retriever


def test_default_tokenizer_is_proper(mocker):
    """不传 tokenizer 时默认用 proper"""
    mock_BM25Retriever = _patch_bm25(mocker)

    create_retriever(vectorstore=_make_vectorstore(mocker), hybrid=True, hierarchical=False)

    tokenize = _tokenizer_used(mock_BM25Retriever)
    assert tokenize.mode == "proper"
    assert tokenize.lowercase is False
    assert tokenize("`create_agent`:") == ["create_agent"]


def test_tokenizer_can_restore_legacy_split(mocker):
    """显式传 default 时恢复旧行为(text.split)"""
    mock_BM25Retriever = _patch_bm25(mocker)

    create_retriever(
        vectorstore=_make_vectorstore(mocker),
        hybrid=True,
        hierarchical=False,
        tokenizer="default",
    )

    tokenize = _tokenizer_used(mock_BM25Retriever)
    assert tokenize.mode == "default"
    assert tokenize("`create_agent`:") == ["`create_agent`:"]


def test_tokenizer_lowercase_and_cjk_are_forwarded(mocker):
    """lowercase / cjk 两个开关要真的传到分词函数里"""
    mock_BM25Retriever = _patch_bm25(mocker)

    create_retriever(
        vectorstore=_make_vectorstore(mocker),
        hybrid=True,
        hierarchical=False,
        tokenizer="punct",
        tokenizer_lowercase=True,
        tokenizer_cjk="char",
    )

    tokenize = _tokenizer_used(mock_BM25Retriever)
    assert tokenize.mode == "punct"
    assert tokenize.lowercase is True
    assert tokenize.cjk == "char"
    assert tokenize("Create_Agent 中文") == ["create_agent", "中", "文"]


def test_invalid_tokenizer_mode_raises(mocker):
    from backend.retrieval.tokenizer import TokenizerMode

    _patch_bm25(mocker)

    with pytest.raises(RetrievalError):
        create_retriever(
            vectorstore=_make_vectorstore(mocker),
            hybrid=True,
            hierarchical=False,
            tokenizer="no_such_mode",
        )

    # 合法取值仍然正常
    assert TokenizerMode("proper") is TokenizerMode.PROPER
