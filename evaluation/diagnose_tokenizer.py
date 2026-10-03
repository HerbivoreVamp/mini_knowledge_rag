# 诊断脚本: BM25 分词方案对比
# 重点回答: "去标点" 会不会把 `create_agent` 切成 create + agent?
# 对比方案:
#   default  : LangChain 现状 text.split()
#   punct    : 我上次用的 [a-zA-Z0-9_]+|CJK  (下划线保留, 点/连字符丢弃)
#   dotdash  : 我上次用的 CNCHAR [a-zA-Z0-9_.\-]+|CJK (点/连字符保留, 但句尾 "agent." 会把点带上)
#   proper   : [a-zA-Z0-9_]+(?:[.\-][a-zA-Z0-9_]+)*|CJK  (标识符完整保留, 且不在词尾吞标点)
#
# 用法(项目根目录):
#   D:\anaconda3\envs\langchain_learning\python.exe evaluation\diagnose_tokenizer.py
#   D:\anaconda3\envs\langchain_learning\python.exe evaluation\diagnose_tokenizer.py --rerank
import json
import logging
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "backend" / ".env")
logging.disable(logging.CRITICAL)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_community.retrievers import BM25Retriever  # noqa: E402

from backend.config.model import create_embedding, create_reranker  # noqa: E402
from backend.config.settings import get_settings  # noqa: E402
from backend.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from backend.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from backend.retrieval.rerank_retriever import RerankRetriever  # noqa: E402
from backend.storage.sqlite_docstore import create_sqlite_docstore  # noqa: E402
from backend.storage.vectorstore import load_vectorstore  # noqa: E402

BACKTICK = chr(96)
CJK = r"[\u4e00-\u9fff]"

# 三种"去标点"写法 + 是否小写化(上次两次测试唯一的差异就是大小写)
PUNCT_RE = r"[a-zA-Z0-9_]+|" + CJK + r"+"
DOTDASH_RE = r"[a-zA-Z0-9_.\-]+|" + CJK
PROPER_RE = r"[a-zA-Z0-9_]+(?:[.\-][a-zA-Z0-9_]+)*|" + CJK + r"+"


def _make(regex, lower):
    if lower:
        return lambda t: re.findall(regex, t.lower())
    return lambda t: re.findall(regex, t)


# 展示用(4 种)
TOKENIZERS = {
    "default": lambda t: t.split(),
    "punct": _make(PUNCT_RE, True),
    "dotdash": _make(DOTDASH_RE, True),
    "proper": _make(PROPER_RE, True),
}

# 评估用(补上不大写的版本, 隔离"大小写"这个变量)
EVAL = {
    "default": lambda t: t.split(),
    "punct": _make(PUNCT_RE, True),
    "punct_nolower": _make(PUNCT_RE, False),
    "dotdash": _make(DOTDASH_RE, True),
    "dotdash_nolower": _make(DOTDASH_RE, False),
    "proper": _make(PROPER_RE, True),
    "proper_nolower": _make(PROPER_RE, False),
}

K, SPARSE_K, HYBRID_TOPK, RERANK_TOPK = 30, 30, 50, 20
K_VALUES = (5, 20)


def main():
    settings = get_settings()
    dataset = json.loads(
        (settings.database_dir / settings.database_name / "langchain_qa_dataset.json")
        .read_text(encoding="utf-8")
    )
    store = create_sqlite_docstore(settings.child_store_dir)
    corpus = store.get_all_documents()
    store.close()

    # ---------- 1. 标识符会不会被切开 ----------
    print("=" * 78)
    print("[1] 关键验证: `create_agent` / create_agent.responses 这类标识符")
    print("=" * 78)
    sample = next(
        d for d in corpus
        if BACKTICK + "create_agent" + BACKTICK in d.page_content
        and "create_agent.responses" in d.page_content
    ) if any(
        BACKTICK + "create_agent" + BACKTICK in d.page_content
        and "create_agent.responses" in d.page_content for d in corpus
    ) else next(
        d for d in corpus if BACKTICK + "create_agent" + BACKTICK in d.page_content
    )
    text = sample.page_content
    print(f"来源: {sample.metadata.get('source')}")
    for name in ("default", "punct", "dotdash", "proper"):
        toks = TOKENIZERS[name](text)
        related = sorted({t for t in toks if "create" in t or "agent" in t})
        print(f"  {name:<8} 与 create/agent 相关的 token: {related[:10]}")

    # ---------- 2. 语料词表层面的差异 ----------
    print()
    print("=" * 78)
    print("[2] 全量语料词表对比 (1423 片 child)")
    print("=" * 78)
    corpora = {name: [fn(d.page_content) for d in corpus] for name, fn in TOKENIZERS.items()}
    for name, toks in corpora.items():
        vocab = {t for doc in toks for t in doc}
        has_create_agent = "create_agent" in vocab
        has_backticked = BACKTICK + "create_agent" + BACKTICK in vocab
        trailing = sum(1 for t in vocab if t.endswith((".", "-")))
        print(
            f"  {name:<8} 词表={len(vocab):<6} 完整 'create_agent'={has_create_agent!s:<6} "
            f"带反引号的版本={has_backticked!s:<6} 词尾带标点的token={trailing}"
        )

    # ---------- 3. 检索效果 ----------
    print()
    print("=" * 78)
    print("[3] 检索效果对比 (dense k=30 + sparse k=30 -> RRF top50)")
    print("=" * 78)
    import torch

    if not torch.cuda.is_available():
        settings.emb_device = "cpu"
        settings.reranker_device = "cpu"
    emb = create_embedding(settings)
    reranker = create_reranker(settings) if "--rerank" in sys.argv else None
    vs = load_vectorstore(settings.vectorstore_dir, emb, settings.index_name)

    def evaluate(name, use_rerank):
        sparse = BM25Retriever.from_documents(
            corpus, k=SPARSE_K, preprocess_func=EVAL[name]
        )
        child = HybridRetriever(
            dense_retriever=DenseRetriever(vectorstore=vs, k=K),
            sparse_retriever=sparse,
            k=HYBRID_TOPK,
        )
        if use_rerank:
            child = RerankRetriever(retriever=child, reranker=reranker, top_k=RERANK_TOPK)
        hits = {k: 0 for k in K_VALUES}
        miss = 0
        t0 = time.time()
        for item in dataset:
            gold = {c["chunk_id"] for c in item["contexts"]}
            docs = child.invoke(item["question"])
            ids = [d.metadata.get("chunk_id") for d in docs]
            for k in K_VALUES:
                if any(g in ids[:k] for g in gold):
                    hits[k] += 1
            if not any(g in ids for g in gold):
                miss += 1
        n = len(dataset)
        print(
            f"  {name:<8} rerank={use_rerank!s:<6} recall@5={hits[5]/n:.3f} "
            f"recall@20={hits[20]/n:.3f} gold未进top{HYBRID_TOPK}={miss/n:.3f}  ({time.time()-t0:.1f}s)"
        )

    for name in EVAL:
        evaluate(name, use_rerank=False)
    if reranker is not None:
        print("  --- 加 reranker ---")
        for name in ("default", "proper_nolower", "dotdash_nolower"):
            evaluate(name, use_rerank=True)


if __name__ == "__main__":
    main()
