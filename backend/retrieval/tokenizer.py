"""BM25 分词方案

为什么需要这个模块:
    LangChain BM25Retriever 的默认 preprocess_func 就是 ``text.split()``
    (langchain_community/retrievers/bm25.py:11)。在这个项目的语境下有两个问题:

    1. markdown 语料里 `` `create_agent`: `` 会整体变成一个 token, 标点和反引号
       黏在词上, 查询侧的 create_agent 永远匹配不到它。
       全量 6583 个 token 里有 464 个词尾带标点。
    2. 中文查询没有空格, 整句会变成一个 token; 而本项目语料是全英文,
       词表里中文 token 数为 0, 于是 BM25 只能提供"问题里那几个英文词"这一路信号。

实测(71 条中文 QA, 1423 片 child, bge-small-zh-v1.5 + bge-reranker-v2-m3,
     dense k=30 + sparse k=30 -> RRF top50):
    default   recall@5=0.380  recall@20=0.662  gold未进候选池=26.8%
    proper    recall@5=0.451  recall@20=0.718  gold未进候选池=19.7%
    加 reranker 后: default 0.718/0.732 -> proper 0.775/0.803

注意: 小写化(lowercase=True)实测会让 recall@20 变差(0.718 -> 0.662/0.704),
因为 LangChain/langchain 这类大小写不同的 token 被合并后改变了 IDF。默认关闭。
中文侧的单字/整段开关在当前英文语料上几乎无差别(中文 token 匹配不到任何东西),
它是为"语料以后换成中文"准备的。
"""
import re
from enum import Enum
from typing import Callable

from backend.core.exceptions import RetrievalError

CJK_RUN = r"[\u4e00-\u9fff]+"
CJK_CHAR = r"[\u4e00-\u9fff]"


class TokenizerMode(str, Enum):
    """BM25 分词方案"""

    DEFAULT = "default"  # 现状: 按空白切, 标点/反引号黏在 token 上
    PUNCT = "punct"  # 去标点; 标识符只保留下划线, create_agent.responses 会被拆成两段
    DOTDASH = "dotdash"  # 去标点, 但 . 和 - 也算词内字符 -> 句尾 "agent." 会把点带上
    PROPER = "proper"  # 推荐: 标识符内部保留 . 和 -, 但不在词尾吞标点


class CjkMode(str, Enum):
    """中文切法(只影响 punctuation 系列方案)"""

    RUN = "run"  # 连续中文算一个 token
    CHAR = "char"  # 中文逐字切


# mode -> (正则, 说明) ; DEFAULT 不使用正则
_PATTERNS = {
    TokenizerMode.PUNCT: r"[a-zA-Z0-9_]+",
    TokenizerMode.DOTDASH: r"[a-zA-Z0-9_.\-]+",
    TokenizerMode.PROPER: r"[a-zA-Z0-9_]+(?:[.\-][a-zA-Z0-9_]+)*",
}


def _coerce_mode(mode) -> TokenizerMode:
    try:
        return TokenizerMode(mode)
    except ValueError as e:
        raise RetrievalError(
            f"不支持的分词方案 mode={mode} 可选值={[m.value for m in TokenizerMode]}"
        ) from e


def _coerce_cjk(cjk) -> CjkMode:
    try:
        return CjkMode(cjk)
    except ValueError as e:
        raise RetrievalError(
            f"不支持的中文切法 cjk={cjk} 可选值={[m.value for m in CjkMode]}"
        ) from e


def create_tokenizer(
        mode: TokenizerMode | str = TokenizerMode.PROPER,
        lowercase: bool = False,
        cjk: CjkMode | str = CjkMode.RUN,
) -> Callable[[str], list[str]]:
    """按给定方案创建 BM25 分词函数

    :param mode: 分词方案, 见 TokenizerMode
    :param lowercase: 是否先转小写(实测收益为负, 默认 False)
    :param cjk: 中文切法, 见 CjkMode
    :return: Callable[[str], list[str]] 可直接作为 BM25Retriever 的 preprocess_func
    """
    mode = _coerce_mode(mode)
    cjk = _coerce_cjk(cjk)

    pattern = None
    if mode is not TokenizerMode.DEFAULT:
        cjk_pattern = CJK_CHAR if cjk is CjkMode.CHAR else CJK_RUN
        try:
            pattern = re.compile(_PATTERNS[mode] + "|" + cjk_pattern)
        except re.error as e:
            raise RetrievalError(f"分词正则构造失败 mode={mode.value}") from e

    def tokenize(text: str) -> list[str]:
        if not isinstance(text, str):
            raise RetrievalError(
                f"分词输入必须是str 实际={type(text).__name__}"
            )
        if lowercase:
            text = text.lower()
        if pattern is None:
            return text.split()
        return pattern.findall(text)

    # 便于日志/调试时看出用的是哪个方案
    tokenize.mode = mode.value
    tokenize.lowercase = lowercase
    tokenize.cjk = cjk.value

    return tokenize


def describe_tokenizer(mode: TokenizerMode | str) -> str:
    """给日志/README 用的一句话说明"""
    mode = _coerce_mode(mode)
    return {
        TokenizerMode.DEFAULT: "按空白切分(标点黏在token上)",
        TokenizerMode.PUNCT: "去标点, 标识符只保留下划线",
        TokenizerMode.DOTDASH: "去标点, . 和 - 也算词内字符",
        TokenizerMode.PROPER: "去标点, 标识符内保留 . 和 - 且词尾不吞标点",
    }[mode]
