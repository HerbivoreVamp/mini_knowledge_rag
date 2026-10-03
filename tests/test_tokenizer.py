"""backend/retrieval/tokenizer.py 的单元测试

这里全部是纯函数测试(不写文件、不加载模型), 因为本机 pytest 的 tmp_path
在沙箱下拿不到写权限(会 PermissionError)。
"""
import pytest

from backend.core.exceptions import RetrievalError
from backend.retrieval.tokenizer import (
    CjkMode,
    TokenizerMode,
    create_tokenizer,
    describe_tokenizer,
)

BACKTICK = chr(96)


# ---------- 1. 四种预置方案的核心差异 ----------

def test_default_mode_is_plain_whitespace_split():
    """default 复现 LangChain 现状: 标点和反引号黏在 token 上"""
    tokenize = create_tokenizer("default")
    assert tokenize(BACKTICK + "create_agent" + BACKTICK + ": is a function") == [
        BACKTICK + "create_agent" + BACKTICK + ":",
        "is",
        "a",
        "function",
    ]


def test_proper_strips_punctuation_but_keeps_identifier():
    """proper: 去标点, 但 create_agent 是一个完整 token"""
    tokenize = create_tokenizer("proper")
    tokens = tokenize(BACKTICK + "create_agent" + BACKTICK + ":")
    assert tokens == ["create_agent"]
    assert "create" not in tokens
    assert "agent" not in tokens


def test_proper_keeps_dotted_and_dashed_identifiers():
    """proper: . 和 - 保留在标识符内部"""
    tokenize = create_tokenizer("proper")
    assert tokenize("create_agent.responses") == ["create_agent.responses"]
    assert tokenize("bge-small-zh-v1.5") == ["bge-small-zh-v1.5"]
    assert tokenize("docs.langchain.com") == ["docs.langchain.com"]


def test_proper_does_not_swallow_trailing_punctuation():
    """proper: 句尾的句号/逗号不能被吃进 token"""
    tokenize = create_tokenizer("proper")
    assert tokenize("Use an agent.") == ["Use", "an", "agent"]
    assert tokenize("agent.") == ["agent"]
    assert tokenize("agent,") == ["agent"]
    assert tokenize("(agent)") == ["agent"]


def test_proper_handles_url():
    """proper: URL 被拆成有意义的片段, 且不留下带点的词尾"""
    tokenize = create_tokenizer("proper")
    assert tokenize("https://docs.langchain.com/llms.txt") == [
        "https",
        "docs.langchain.com",
        "llms.txt",
    ]


def test_punct_splits_dotted_identifier():
    """punct 与 proper 的差别: 点号被当标点, 所以点分标识符会被拆开"""
    assert create_tokenizer("punct")("create_agent.responses") == [
        "create_agent",
        "responses",
    ]
    # 下划线仍然保留
    assert create_tokenizer("punct")("create_agent") == ["create_agent"]


def test_dotdash_keeps_trailing_dot():
    """dotdash 的已知缺陷: 句尾的点会被带进 token(所以不推荐)"""
    assert create_tokenizer("dotdash")("agent.") == ["agent."]
    assert create_tokenizer("proper")("agent.") == ["agent"]


# ---------- 2. 两个开关 ----------

def test_lowercase_switch():
    assert create_tokenizer("proper")("Create_Agent") == ["Create_Agent"]
    assert create_tokenizer("proper", lowercase=True)("Create_Agent") == ["create_agent"]
    # default 方案下也能生效
    assert create_tokenizer("default", lowercase=True)("Agent.") == ["agent."]


def test_cjk_run_vs_char():
    text = "中文查询abc"
    assert create_tokenizer("proper", cjk="run")(text) == ["中文查询", "abc"]
    assert create_tokenizer("proper", cjk="char")(text) == ["中", "文", "查", "询", "abc"]


def test_cjk_switch_accepts_enum():
    assert create_tokenizer("proper", cjk=CjkMode.CHAR)("中文") == ["中", "文"]
    assert create_tokenizer("proper", cjk=CjkMode.RUN)("中文") == ["中文"]


# ---------- 3. 边界与错误处理 ----------

def test_empty_and_whitespace_input():
    for mode in TokenizerMode:
        tokenize = create_tokenizer(mode)
        assert tokenize("") == []
        assert tokenize("   \n\t ") == []


def test_non_str_input_raises():
    tokenize = create_tokenizer("proper")
    with pytest.raises(RetrievalError):
        tokenize(None)
    with pytest.raises(RetrievalError):
        tokenize(123)


def test_invalid_mode_raises():
    with pytest.raises(RetrievalError):
        create_tokenizer("no_such_mode")
    with pytest.raises(RetrievalError):
        create_tokenizer(None)


def test_invalid_cjk_raises():
    with pytest.raises(RetrievalError):
        create_tokenizer("proper", cjk="no_such_cjk")


def test_str_and_enum_are_equivalent():
    text = BACKTICK + "create_agent.response" + BACKTICK
    assert create_tokenizer("proper")(text) == create_tokenizer(TokenizerMode.PROPER)(text)
    assert create_tokenizer("punct")(text) == create_tokenizer(TokenizerMode.PUNCT)(text)


def test_describe_tokenizer_covers_all_modes():
    for mode in TokenizerMode:
        assert describe_tokenizer(mode)
    with pytest.raises(RetrievalError):
        describe_tokenizer("no_such_mode")


def test_default_tokenizer_mode_is_proper():
    """create_tokenizer() 不传参时就是推荐方案"""
    assert create_tokenizer().mode == TokenizerMode.PROPER.value


# ---------- 4. 与 BM25Retriever 的串联 ----------

def test_bm25_receives_tokenizer_for_both_index_and_query():
    """BM25Retriever 会把 preprocess_func 同时用于建索引和查询"""
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    docs = [Document(page_content="use " + BACKTICK + "create_agent" + BACKTICK)]
    tokenize = create_tokenizer("proper")
    retriever = BM25Retriever.from_documents(docs, k=1, preprocess_func=tokenize)

    assert retriever.preprocess_func is tokenize
    assert tokenize(retriever.docs[0].page_content) == ["use", "create_agent"]


def test_proper_tokenizer_finds_document_by_identifier():
    """回归测试: 语料里只有带反引号的 `create_agent` 时, 查询 create_agent 能排到第一

    注意 rank_bm25 的 IDF = log(N - df + 0.5) - log(df + 0.5),
    语料太小(N=2, df=1)时 IDF 恰好等于 0, 排序会退化成按索引顺序,
    所以这里用 6 篇文档让 IDF 为正。
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="use " + BACKTICK + "create_agent" + BACKTICK + " to build it",
            metadata={"chunk_id": "hit"},
        )
    ] + [
        Document(
            page_content=f"filler document number {i} about weather and cooking",
            metadata={"chunk_id": f"filler{i}"},
        )
        for i in range(5)
    ]
    retriever = BM25Retriever.from_documents(
        docs, k=3, preprocess_func=create_tokenizer("proper")
    )

    assert [d.metadata["chunk_id"] for d in retriever.invoke("create_agent")][0] == "hit"


def test_default_tokenizer_cannot_match_bare_identifier():
    """对照测试: 旧分词下语料 token 是 `create_agent`(带反引号), 裸标识符匹配不到"""
    tokenize = create_tokenizer("default")
    corpus_tokens = tokenize(
        "use " + BACKTICK + "create_agent" + BACKTICK + " to build it"
    )
    assert "create_agent" not in corpus_tokens
    assert BACKTICK + "create_agent" + BACKTICK in corpus_tokens
