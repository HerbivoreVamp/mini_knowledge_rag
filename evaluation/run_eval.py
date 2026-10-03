"""RAG 检索评测

原来有两个脚本, 差别只有 ground truth 的口径:
    rag_test.py              -> child  口径, 命中 gold chunk_id, hierarchical=False
    rag_test_hierarchical.py -> parent 口径, 命中 gold parent_id, hierarchical=True
现在用一个 --level 参数切换, 顺便把 retriever 组合和分词方案都做成可选。

用法(项目根目录):
    python evaluation/run_eval.py --level child
    python evaluation/run_eval.py --level parent
    python evaluation/run_eval.py --level child --retriever dense_rerank,hybrid_rerank --tokenizer proper,default
    python evaluation/run_eval.py --dataset evaluation/data/database/vanilla/langchain_qa_dataset.json   # 旧口径

ground truth 有两种口径:
    anchor(推荐) - 数据集里存"源文档 + 字符区间", 评测时按当前索引现算,
                   重新导入 / 改切分参数 / 改 id 算法都不用改数据集(见 anchors.py)。
                   存在 <data-dir>/<db>/langchain_qa_dataset_anchors.json, 有人就优先用它。
    legacy       - 数据集里存 chunk_id / parent_id, 换一次切分就失效, 只做兼容用。

只读: 只加载 evaluation/data/database 下的索引副本, 不写任何数据库。

注意:
    - 默认参数与 backend/retrieval/factory.py 的默认值一致(k=30, sparse_k=30,
      hybrid_topk=50, rerank_topk=20, parent_k=3), 也就是"评测 = 线上实际配置"。
      想跑更宽的候选池就显式加大 --k/--sparse-k, 但比较 hybrid 与 dense 时
      要让 --hybrid-topk >= --k, 否则 hybrid 会先被 RRF 截断而假输。
    - parent 口径返回的条数受 --parent-k 限制(默认 3), 所以 recall@5 和 recall@20
      会一样; 想看更大的 K 要同时加大 --parent-k。
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
load_dotenv(BASE.parent / "backend" / ".env")
logging.disable(logging.CRITICAL)  # 评测时关掉项目日志

from anchors import DEFAULT_MIN_COVERAGE, resolve_gold  # noqa: E402
from backend.config.model import create_embedding, create_reranker  # noqa: E402
from backend.config.settings import get_settings  # noqa: E402
from backend.retrieval.factory import create_retriever  # noqa: E402
from backend.retrieval.tokenizer import TokenizerMode  # noqa: E402
from backend.storage.sqlite_docstore import create_sqlite_docstore  # noqa: E402
from backend.storage.vectorstore import load_vectorstore  # noqa: E402

# retriever 组合 -> 开关
RETRIEVERS = {
    "dense": {"hybrid": False, "use_reranker": False},
    "hybrid": {"hybrid": True, "use_reranker": False},
    "dense_rerank": {"hybrid": False, "use_reranker": True},
    "hybrid_rerank": {"hybrid": True, "use_reranker": True},
}
# level -> 结果里用来比对的 metadata key
RESULT_KEY = {"child": "chunk_id", "parent": "parent_id"}

ANCHOR_DATASET = "langchain_qa_dataset_anchors.json"
LEGACY_DATASET = "langchain_qa_dataset.json"


def parse_args():
    p = argparse.ArgumentParser(description="RAG 检索评测")
    p.add_argument("--level", choices=sorted(RESULT_KEY), default="child",
                   help="child=命中 gold chunk, parent=命中 gold parent")
    p.add_argument("--retriever", default=",".join(RETRIEVERS),
                   help=f"逗号分隔, 可选 {sorted(RETRIEVERS)}")
    p.add_argument("--tokenizer", default=TokenizerMode.PROPER.value,
                   help=f"逗号分隔的 BM25 分词方案, 可选 {[m.value for m in TokenizerMode]}")
    p.add_argument("--k", type=int, default=30, help="dense 候选数(与 factory 默认一致)")
    p.add_argument("--sparse-k", type=int, default=30, help="BM25 候选数(与 factory 默认一致)")
    p.add_argument("--hybrid-topk", type=int, default=50, help="RRF 融合后保留数(要 >= --k 才公平)")
    p.add_argument("--rerank-topk", type=int, default=20, help="rerank 后保留数")
    p.add_argument("--parent-k", type=int, default=3, help="parent 扩展后保留数")
    p.add_argument("--k-values", default="5,20", help="评测的 K 列表")
    p.add_argument("--limit", type=int, default=0, help="只评测前 N 条(0=全部)")
    p.add_argument("--data-dir", default=str(BASE / "data" / "database"),
                   help="索引副本目录, 默认 evaluation/data/database")
    p.add_argument("--dataset", default="", help=f"数据集路径, 默认优先 {ANCHOR_DATASET}")
    p.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                   help="锚点口径下, 切片覆盖锚点区间多少比例算命中")
    p.add_argument("--out", default="", help="结果输出路径, 默认 evaluation/eval_<level>_<时间>.json")
    return p.parse_args()


def load_dataset(path: Path):
    """返回 (items, is_anchor)"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "items" in raw:
        return raw["items"], True
    return raw, False


def legacy_gold(item, level):
    """旧口径: 直接从 chunk_id / parent_id 取 ground truth"""
    if level == "child":
        return {c["chunk_id"] for c in item["contexts"]}
    # 旧数据集把 parent_id 放在 item["metadata"] 里(不在 contexts 内)
    ids = {c.get("parent_id") for c in item["contexts"] if c.get("parent_id")}
    if not ids and item.get("metadata", {}).get("parent_id"):
        ids = {item["metadata"]["parent_id"]}
    return ids


def build_gold_map(items, level, child_dir, min_coverage):
    """锚点口径: 用当前索引把区间锚点解析成 id"""
    store = create_sqlite_docstore(child_dir)
    child_docs = store.get_all_documents()
    store.close()

    if not any("file_start_index" in d.metadata for d in child_docs):
        raise SystemExit(
            "锚点解析失败: 当前索引的切片没有 file_start_index 字段, 说明这份索引是旧代码建的。\n"
            "  请用新代码重新导入后再跑, 或者 --dataset 指定旧口径的数据集。"
        )

    gold_map = resolve_gold(items, child_docs, level=level, min_coverage=min_coverage)
    unresolved = [item["id"] for item in items if not gold_map[item["id"]]]
    print(
        f"锚点解析(level={level} min_coverage={min_coverage}): "
        f"{len(items) - len(unresolved)}/{len(items)} 条命中当前索引"
    )
    if unresolved:
        print(f"  未命中(这些条目必然算 miss): {unresolved}")
    return gold_map


def build_retriever(name, tokenizer, args, emb, reranker, vectorstore, dirs):
    """按配置组装检索器, 返回 (retriever, 需要关闭的 store)"""
    cfg = RETRIEVERS[name]
    parent_store = create_sqlite_docstore(dirs["parent"])
    child_store = create_sqlite_docstore(dirs["child"])
    retriever = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        child_store=child_store,  # factory 内部读完会自己 close
        reranker=reranker if cfg["use_reranker"] else None,
        hierarchical=args.level == "parent",
        hybrid=cfg["hybrid"],
        k=args.k,
        sparse_k=args.sparse_k,
        hybrid_topk=args.hybrid_topk,
        rerank_topk=args.rerank_topk,
        parent_k=args.parent_k,
        tokenizer=tokenizer,
    )
    return retriever, parent_store


def evaluate(name, tokenizer, args, dataset, gold_map, emb, reranker, vectorstore, dirs, k_values):
    retriever, parent_store = build_retriever(
        name, tokenizer, args, emb, reranker, vectorstore, dirs
    )
    result_key = RESULT_KEY[args.level]
    hits = {k: 0 for k in k_values}
    started = time.time()
    for item in dataset:
        gold = gold_map[item["id"]] if gold_map is not None else legacy_gold(item, args.level)
        docs = retriever.invoke(item["question"])
        ids = [d.metadata.get(result_key) for d in docs]
        for k in k_values:
            if gold & set(ids[:k]):
                hits[k] += 1
    parent_store.close()

    total = len(dataset)
    return {
        "retriever": name,
        "tokenizer": tokenizer,
        **{f"recall@{k}": round(hits[k] / total, 3) for k in k_values},
        **{f"hits@{k}": hits[k] for k in k_values},
        "seconds": round(time.time() - started, 1),
    }


def main():
    args = parse_args()
    k_values = [int(x) for x in args.k_values.split(",")]
    retrievers = [x.strip() for x in args.retriever.split(",") if x.strip()]
    tokenizers = [x.strip() for x in args.tokenizer.split(",") if x.strip()]
    for name in retrievers:
        if name not in RETRIEVERS:
            raise SystemExit(f"未知 retriever: {name}, 可选 {sorted(RETRIEVERS)}")

    settings = get_settings()
    data_dir = Path(args.data_dir)
    db_dir = data_dir / settings.database_name
    dirs = {
        "vectorstore": db_dir / "vectorstore",
        "parent": db_dir / "parent_store",
        "child": db_dir / "child_store",
    }
    # 让 embedding 走 .env 配置, 索引指向评测副本
    settings.vectorstore_dir = dirs["vectorstore"]
    settings.parent_store_dir = dirs["parent"]
    settings.child_store_dir = dirs["child"]

    if args.dataset:
        qa_path = Path(args.dataset)
    else:
        qa_path = db_dir / ANCHOR_DATASET
        if not qa_path.exists():
            qa_path = db_dir / LEGACY_DATASET
    dataset, is_anchor = load_dataset(qa_path)
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"数据集: {qa_path.name} ({'锚点口径' if is_anchor else '旧 chunk_id 口径'})")
    gold_map = (
        build_gold_map(dataset, args.level, dirs["child"], args.min_coverage)
        if is_anchor
        else None
    )

    import torch

    if not torch.cuda.is_available():
        settings.emb_device = "cpu"
        settings.reranker_device = "cpu"

    emb = create_embedding(settings)
    reranker = create_reranker(settings)
    vectorstore = load_vectorstore(dirs["vectorstore"], emb, settings.index_name)

    params = {
        "k": args.k, "sparse_k": args.sparse_k, "hybrid_topk": args.hybrid_topk,
        "rerank_topk": args.rerank_topk, "parent_k": args.parent_k,
        "k_values": k_values, "questions": len(dataset),
        "dataset": qa_path.name, "min_coverage": args.min_coverage if is_anchor else None,
    }
    print(f"level={args.level} questions={len(dataset)} params={params}")
    print(f"{'retriever':<16}{'tokenizer':<10}" + "".join(f"{'recall@'+str(k):<12}" for k in k_values) + "sec")
    print("-" * 78)

    results = []
    for name in retrievers:
        # 分词方案只影响 hybrid 系列, 其余组合跑一次就够
        schemes = tokenizers if RETRIEVERS[name]["hybrid"] else tokenizers[:1]
        for tokenizer in schemes:
            row = evaluate(
                name, tokenizer, args, dataset, gold_map, emb, reranker, vectorstore, dirs, k_values
            )
            results.append(row)
            recalls = "".join(f"{row['recall@'+str(k)]:<12}" for k in k_values)
            print(f"{name:<16}{tokenizer:<10}{recalls}{row['seconds']}")

    out = Path(args.out) if args.out else BASE / f"eval_{args.level}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(
            {
                "model": settings.emb_model_name,
                "model_reranker": settings.reranker_model_name,
                "level": args.level,
                "params": params,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
