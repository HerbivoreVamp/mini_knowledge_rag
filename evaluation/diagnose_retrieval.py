# 诊断脚本
# 目的1: 统计切分语料的成分(代码块/网址/import 占比) 验证"脏切片"假说
# 目的2: 检查 BM25 的默认分词(whitespace split)在中文 query 上是否失效
# 目的3: 用真实索引对比 dense / hybrid / hybrid+中文分词 各配置的 recall 与脏切片占比
#
# 用法(在项目根目录):
#   D:\anaconda3\envs\langchain_learning\python.exe evaluation\diagnose_retrieval.py
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

logging.disable(logging.CRITICAL)  # 静音项目日志 必须在导入 backend 前调用

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from backend.config.model import create_embedding, create_reranker  # noqa: E402
from backend.config.settings import get_settings  # noqa: E402
from backend.ingestion.loader import load_md  # noqa: E402
from backend.ingestion.splitter import create_hierarchy_splitter  # noqa: E402
from backend.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from backend.retrieval.hierarchical_retriever import HierarchicalRetriever  # noqa: E402
from backend.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from backend.retrieval.rerank_retriever import RerankRetriever  # noqa: E402
from backend.storage.sqlite_docstore import create_sqlite_docstore  # noqa: E402
from backend.storage.vectorstore import load_vectorstore  # noqa: E402

# 与 main.py / factory 默认参数保持一致
K = 30
SPARSE_K = 30
HYBRID_TOPK = 50
RERANK_TOPK = 20
PARENT_K = 3
K_VALUES = (5, 20)

URL_RE = re.compile(r"https?://")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
IMPORT_RE = re.compile(r"^\s*(?:from\s+[\w\.]+\s+import|import\s+[\w\.]+)", re.M)
PAIRED_FENCE_RE = re.compile(r"```.*?```", re.S)
# 中文友好的分词: 英文/数字按词 中文按单字
CN_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\.\-]+|[\u4e00-\u9fff]")


def chunk_stats(text: str) -> dict:
    """切片的成分指标"""
    if not text:
        return {"len": 0, "fence_marks": 0, "code_ratio": 0.0, "url": 0, "import": 0}
    paired = sum(len(m.group(0)) for m in PAIRED_FENCE_RE.finditer(text))
    return {
        "len": len(text),
        "fence_marks": text.count("```"),
        "code_ratio": round(paired / len(text), 3),
        "url": len(URL_RE.findall(text)),
        "import": len(IMPORT_RE.findall(text)),
    }


def is_dirty(text: str) -> bool:
    """疑似"脏切片": 代码占比高 / import 行多 / 网址堆砌 / 被 ``` 从中间切断"""
    s = chunk_stats(text)
    if s["code_ratio"] >= 0.4:
        return True
    if s["import"] >= 3 or s["url"] >= 3:
        return True
    if s["fence_marks"] % 2 == 1 and s["code_ratio"] >= 0.2:
        return True
    return False


def dirty_rate(docs) -> dict:
    if not docs:
        return {"n": 0, "dirty": None, "code": None, "url": None, "import": None}
    texts = [d.page_content for d in docs]
    return {
        "n": len(docs),
        "dirty": round(sum(1 for t in texts if is_dirty(t)) / len(docs), 3),
        "code": round(
            sum(1 for t in texts if chunk_stats(t)["code_ratio"] >= 0.4) / len(docs), 3
        ),
        "url": round(sum(1 for t in texts if chunk_stats(t)["url"] >= 1) / len(docs), 3),
        "import": round(
            sum(1 for t in texts if chunk_stats(t)["import"] >= 1) / len(docs), 3
        ),
    }


def hit_at_k(docs, gold: set, k: int) -> bool:
    return any(d.metadata.get("chunk_id") in gold for d in docs[:k])


def parent_hit_at_k(docs, gold_parents: set, k: int) -> bool:
    return any(d.metadata.get("parent_id") in gold_parents for d in docs[:k])


def preview(doc: Document, width: int = 110) -> str:
    text = re.sub(r"\s+", " ", doc.page_content).strip()
    return text[:width]


def main():
    settings = get_settings()
    report = {"time": datetime.now().isoformat(timespec="seconds"), "config": {}}

    # ================= 1. 语料 / 切片成分统计 =================
    print("=" * 78)
    print("[1] 语料与切片成分统计 (只切分 不加载模型)")
    print("=" * 78)
    docs = load_md(settings.document_dir, "langchain_doc")
    parent_splitter, child_splitter = create_hierarchy_splitter()
    parents = parent_splitter.split_documents(docs)
    children = []
    for p in parents:
        children.extend(child_splitter.split_documents([p]))

    for name, group in (("parent(2000/200)", parents), ("child(400/50)", children)):
        st = dirty_rate(group)
        avg_len = round(sum(len(d.page_content) for d in group) / len(group))
        print(
            f"  {name:<18} n={st['n']:<5} avg_len={avg_len:<5} "
            f"dirty={st['dirty']:<6} code_ratio>=0.4:{st['code']:<6} "
            f"含网址:{st['url']:<6} 含import:{st['import']}"
        )
    report["config"]["corpus"] = {
        "docs": len(docs),
        "parents": len(parents),
        "children": len(children),
        "parent_dirty": dirty_rate(parents),
        "child_dirty": dirty_rate(children),
    }

    # child 脏切片的来源分布
    per_source = Counter()
    per_source_total = Counter()
    for c in children:
        src = c.metadata.get("source")
        per_source_total[src] += 1
        if is_dirty(c.page_content):
            per_source[src] += 1
    print("\n  child 脏切片来源 top:")
    for src, cnt in per_source.most_common(5):
        print(f"    {cnt:>4}/{per_source_total[src]:<4}  {src}")
    report["config"]["corpus"]["dirty_by_source"] = {
        s: [per_source[s], per_source_total[s]] for s, _ in per_source.most_common()
    }

    # ================= 2. BM25 分词诊断 =================
    print()
    print("=" * 78)
    print("[2] BM25 默认分词诊断 (rank_bm25 + LangChain 默认 preprocess_func)")
    print("=" * 78)
    dataset = json.loads(
        (settings.database_dir / settings.database_name / "langchain_qa_dataset.json")
        .read_text(encoding="utf-8")
    )

    child_store = create_sqlite_docstore(settings.child_store_dir)
    corpus_docs = child_store.get_all_documents()
    child_store.close()
    print(f"  BM25 语料(child_store) docs={len(corpus_docs)}")

    # 复现 LangChain 默认分词
    default_tokens = [t.split() for t in (d.page_content for d in corpus_docs)]
    vocab = Counter(tok for toks in default_tokens for tok in toks)
    cn_vocab_tokens = sum(1 for t in vocab if CJK_RE.search(t))
    print(f"  默认分词后词表大小 vocab={len(vocab)}")
    print(f"  语料里出现过的中文 token 数={cn_vocab_tokens}")

    q_cn = dataset[0]["question"]
    q_en = "what is the create_agent function in langchain"
    for label, q in (("中文query", q_cn), ("英文query", q_en)):
        toks = q.split()
        hit = [t for t in toks if t in vocab]
        print(
            f"  {label}: len(query)={len(q):<3} 默认分词得到 {len(toks)} 个token "
            f"命中语料词表的token={len(hit)}  →  token示例={ [t[:40] for t in toks[:2]] }"
        )
    report["config"]["bm25"] = {
        "corpus_docs": len(corpus_docs),
        "vocab_size": len(vocab),
        "cn_vocab_tokens": cn_vocab_tokens,
        "cn_query_tokens": len(q_cn.split()),
        "cn_query_token_hits": len([t for t in q_cn.split() if t in vocab]),
        "en_query_tokens": len(q_en.split()),
        "en_query_token_hits": len([t for t in q_en.split() if t in vocab]),
    }

    # ================= 3. 各检索配置对比 =================
    print()
    print("=" * 78)
    print("[3] 检索配置对比 (真实索引 backend/data/database/vanilla)")
    print("=" * 78)
    import torch

    if not torch.cuda.is_available():
        print("  [warn] CUDA 不可用 reranker/embedding 回退 CPU")
        settings.emb_device = "cpu"
        settings.reranker_device = "cpu"

    emb = create_embedding(settings)
    reranker = create_reranker(settings)
    vectorstore = load_vectorstore(settings.vectorstore_dir, emb, settings.index_name)

    def build_hybrid(tokenizer=None):
        """构造 HybridRetriever(可选自定义分词) 返回 (retriever, 需关闭的store)"""
        store = create_sqlite_docstore(settings.child_store_dir)
        cs_docs = store.get_all_documents()
        store.close()
        if tokenizer is None:
            sparse = BM25Retriever.from_documents(cs_docs, k=SPARSE_K)
        else:
            sparse = BM25Retriever.from_documents(
                cs_docs, k=SPARSE_K, preprocess_func=tokenizer
            )
        dense = DenseRetriever(vectorstore=vectorstore, k=K)
        return HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, k=HYBRID_TOPK)

    def make(name):
        """返回 (child级retriever, parent级retriever, 需要close的store列表)"""
        closes = []
        dense = DenseRetriever(vectorstore=vectorstore, k=K)
        if name == "dense":
            child_r = dense
        elif name == "hybrid":
            child_r = build_hybrid()
        elif name == "hybrid_cn":
            child_r = build_hybrid(CN_TOKEN_RE.findall)
        elif name == "dense_rerank":
            child_r = RerankRetriever(retriever=dense, reranker=reranker, top_k=RERANK_TOPK)
        elif name == "hybrid_rerank":
            child_r = RerankRetriever(
                retriever=build_hybrid(), reranker=reranker, top_k=RERANK_TOPK
            )
        elif name == "hybrid_cn_rerank":
            child_r = RerankRetriever(
                retriever=build_hybrid(CN_TOKEN_RE.findall),
                reranker=reranker,
                top_k=RERANK_TOPK,
            )
        else:
            raise ValueError(name)
        pstore = create_sqlite_docstore(settings.parent_store_dir)
        closes.append(pstore)
        parent_r = HierarchicalRetriever(
            child_retriever=child_r, parent_store=pstore, parent_k=PARENT_K
        )
        return child_r, parent_r, closes

    configs = ["dense", "hybrid", "hybrid_cn", "dense_rerank", "hybrid_rerank", "hybrid_cn_rerank"]
    summary = {}
    snippet_samples = {}

    for name in configs:
        child_r, parent_r, closes = make(name)
        agg = {
            "child": {f"hit@{k}": 0 for k in K_VALUES},
            "child_dirty": {f"dirty@{k}": [] for k in K_VALUES},
            "parent": {"hit@3": 0, "dirty@3": []},
            "rank_of_gold": [],
        }
        t0 = time.time()
        for item in dataset:
            gold = {c["chunk_id"] for c in item["contexts"]}
            # 注意: 数据集把 parent_id 放在 item["metadata"] 里 不在 contexts 内
            gold_parents = {c.get("parent_id") for c in item["contexts"] if c.get("parent_id")}
            if not gold_parents and item.get("metadata", {}).get("parent_id"):
                gold_parents = {item["metadata"]["parent_id"]}
            q = item["question"]

            cdocs = child_r.invoke(q)
            for k in K_VALUES:
                if hit_at_k(cdocs, gold, k):
                    agg["child"][f"hit@{k}"] += 1
                agg["child_dirty"][f"dirty@{k}"].append(dirty_rate(cdocs[:k])["dirty"])

            pdocs = parent_r.invoke(q)
            if parent_hit_at_k(pdocs, gold_parents, PARENT_K):
                agg["parent"]["hit@3"] += 1
            agg["parent"]["dirty@3"].append(dirty_rate(pdocs)["dirty"])

            rank = next(
                (i for i, d in enumerate(cdocs, 1) if d.metadata.get("chunk_id") in gold), None
            )
            agg["rank_of_gold"].append(rank)

            if name in ("dense", "hybrid") and item["id"] in ("lc_0001", "lc_0002"):
                snippet_samples.setdefault(name, []).append(
                    {
                        "id": item["id"],
                        "top5": [
                            {
                                **chunk_stats(d.page_content),
                                "source": d.metadata.get("source"),
                                "preview": preview(d),
                                "is_gold": d.metadata.get("chunk_id") in gold,
                            }
                            for d in cdocs[:5]
                        ],
                    }
                )

        n = len(dataset)
        res = {
            "recall": {k: round(v / n, 3) for k, v in agg["child"].items()},
            "dirty_child": {
                k: round(sum(v) / len(v), 3) for k, v in agg["child_dirty"].items()
            },
            "parent_recall@3": round(agg["parent"]["hit@3"] / n, 3),
            "dirty_parent@3": round(sum(agg["parent"]["dirty@3"]) / n, 3),
            "gold_missing_topk": round(
                sum(1 for r in agg["rank_of_gold"] if r is None) / n, 3
            ),
            "gold_mean_rank": round(
                sum(r for r in agg["rank_of_gold"] if r) / max(1, sum(1 for r in agg["rank_of_gold"] if r)),
                2,
            ),
            "seconds": round(time.time() - t0, 1),
        }
        summary[name] = res
        for c in closes:
            c.close()
        print(
            f"  {name:<17} recall@5={res['recall']['hit@5']:<6} recall@20={res['recall']['hit@20']:<6} "
            f"dirty@5={res['dirty_child']['dirty@5']:<6} dirty@20={res['dirty_child']['dirty@20']:<6} "
            f"parent_hit@3={res['parent_recall@3']:<6} dirty_parent@3={res['dirty_parent@3']:<6} "
            f"gold未进top{K}={res['gold_missing_topk']:<6} gold均rank={res['gold_mean_rank']:<6} "
            f"({res['seconds']}s)"
        )

    report["config"]["retrieval"] = summary
    report["samples"] = snippet_samples

    out = Path(__file__).resolve().parent / f"diagnose_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已保存: {out}")

    # ================= 4. dense vs hybrid top5 切片预览 =================
    print()
    print("=" * 78)
    print("[4] dense / hybrid 的 top5 切片预览 (lc_0001)")
    print("=" * 78)
    for name, items in snippet_samples.items():
        for entry in items:
            if entry["id"] != "lc_0001":
                continue
            print(f"\n  --- {name} ---")
            for i, d in enumerate(entry["top5"], 1):
                print(
                    f"   {i}. code_ratio={d['code_ratio']:<6} url={d['url']:<3} import={d['import']:<3} "
                    f"gold={d['is_gold']} src={(d['source'] or '')[-28:]}"
                )
                print(f"      {d['preview']}")

    return summary


if __name__ == "__main__":
    main()
