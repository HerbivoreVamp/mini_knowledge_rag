# RAG 检索评估脚本 参照 main.py
# 指标 Hit@K: 召回的前K个child chunk里是否包含正确chunk
# 一次性评估三种 retriever 配置并合并保存到一个结果文件
import json
from datetime import datetime
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import logging

from backend.config.settings import get_settings
from backend.config.model import create_embedding, create_reranker
from backend.retrieval.factory import create_retriever
from backend.retrieval.tokenizer import TokenizerMode
from backend.storage.sqlite_docstore import create_sqlite_docstore
from backend.storage.vectorstore import load_vectorstore
from backend.core.logger import setup_logger
from backend.core.exceptions import RAGError

logging.disable(logging.CRITICAL)  # 关闭日志
# ===== 评估配置 =====
RETRIEVERS = ["dense", "hybrid", "dense_rerank"]
K_VALUES = [5, 20]  # Hit@K 的 K 值列表
MAX_K = max(K_VALUES)
# 固定用未做处理的分词(旧行为)，保证和历史结果 20260827_225845.json 同一口径
TOKENIZER = TokenizerMode.DEFAULT

RETRIEVER_CONFIG = {
    "dense": {"hierarchical": False, "hybrid": False, "use_reranker": False},
    "hybrid": {"hierarchical": False, "hybrid": True, "use_reranker": False},
    "dense_rerank": {"hierarchical": False, "hybrid": False, "use_reranker": True},
}

logger = setup_logger()
logger.info("RAG评估启动")

# --- 配置加载 指向 evaluation 数据库 ---
settings = get_settings()
eval_base = Path(__file__).resolve().parent
eval_db_dir = eval_base / "data" / "database"
settings.vectorstore_dir = eval_db_dir / settings.database_name / "vectorstore"
settings.child_store_dir = eval_db_dir / settings.database_name / "child_store"

# 参数
k_0 = 100
sparse_k = 100
hybrid_topk = 100
rerank_topk = 100
# --- 模型初始化（embedding 与 reranker 复用） ---
emb = create_embedding(settings)
reranker = create_reranker(settings)  # 不使用 reranker 的配置传 None

# --- 向量库（复用） ---
try:
    vectorstore = load_vectorstore(settings.vectorstore_dir, emb, settings.index_name)
except RAGError as e:
    print(e)
    raise

# --- 加载 QA 数据集 ---
qa_path = eval_db_dir / settings.database_name / "langchain_qa_dataset.json"
with open(qa_path, encoding="utf-8") as f:
    qa_dataset = json.load(f)
logger.info(f"QA数据集加载成功 total={len(qa_dataset)}")


def build_retriever(name: str):
    """根据 retriever 名称创建检索器，每次重新打开 child_store 避免 hybrid 关闭后复用问题。"""
    cfg = RETRIEVER_CONFIG[name]
    child_store = create_sqlite_docstore(settings.child_store_dir)
    use_reranker = reranker if cfg["use_reranker"] else None
    retriever = create_retriever(
        vectorstore=vectorstore,
        child_store=child_store,
        reranker=use_reranker,
        hierarchical=cfg["hierarchical"],
        hybrid=cfg["hybrid"],
        k=k_0,
        sparse_k=sparse_k,
        hybrid_topk=hybrid_topk,
        rerank_topk=rerank_topk,
        tokenizer=TOKENIZER,
    )
    logger.info(f"检索器创建成功 retriever={name} k={MAX_K}")
    return retriever, child_store


def evaluate(name: str):
    """评估单个 retriever 配置，返回结果 dict 与 misses 列表。"""
    retriever, child_store = build_retriever(name)
    hits = {k: 0 for k in K_VALUES}
    misses = []
    for item in tqdm(qa_dataset, desc=name):
        query = item["question"]
        gold_chunk_ids = {ctx["chunk_id"] for ctx in item["contexts"]}
        docs = retriever.invoke(query)
        for k in K_VALUES:
            retrieved_ids = {doc.metadata.get("chunk_id") for doc in docs[:k]}
            if gold_chunk_ids & retrieved_ids:
                hits[k] += 1
            if not gold_chunk_ids & retrieved_ids and k == 5:
                misses.append({
                    "question": query,
                    "category": item["metadata"]["category"],
                    "source": item["contexts"][0]["source"],
                    "gold": list(gold_chunk_ids),
                    "retrieved": list(retrieved_ids),
                })

    hit_at_k = {k: hits[k] / len(qa_dataset) for k in K_VALUES}
    for k in K_VALUES:
        logger.info(f"[{name}] 评估完成 Hit@{k}={hit_at_k[k]:.4f} hits={hits[k]}/{len(qa_dataset)}")

    result = {
        "retriever": name,
    }
    for k in K_VALUES:
        result[f"recall@{k}"] = round(hit_at_k[k], 2)

    child_store.close()
    return result, misses


# --- 依次评估三个 retriever ---
all_results = []
all_misses = {}
for name in RETRIEVERS:
    result, misses = evaluate(name)
    all_results.append(result)
    all_misses[name] = misses

# --- 保存合并结果（公共信息放顶层，参数只出现一次） ---
output = {
    "model": settings.emb_model_name,
    "model_reranker": settings.reranker_model_name,
    "params": {
        "k": k_0,
        "sparse_k": sparse_k,
        "hybrid_topk": hybrid_topk,
        "rerank_topk": rerank_topk,
    },
    "results": all_results,
}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = eval_base / f"{timestamp}.json"
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
logger.info(f"结果已保存 path={result_path}")
print(json.dumps(output, ensure_ascii=False, indent=2))

# --- Miss 分析（按 retriever 分组） ---
child_store = create_sqlite_docstore(settings.child_store_dir)
all_docs = child_store.get_all_documents()
doc_map = {doc.metadata["chunk_id"]: doc for doc in all_docs}
child_store.close()

for name in RETRIEVERS:
    misses = all_misses[name]
    print(f"\n========== {name} ==========")
    category_counts = Counter(missed["category"] for missed in misses)
    print("Miss by category:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

    for missed in misses:
        print("\nQUESTION:", missed["question"])
        print("CATEGORY:", missed["category"])
        print("SOURCE:", missed["source"])

        print("GOLD:")
        for chunk_id in missed["gold"]:
            doc = doc_map.get(chunk_id)
            print(doc.page_content if doc else "NOT FOUND")

        print("\nRETRIEVED:")
        for chunk_id in missed["retrieved"][:5]:
            doc = doc_map.get(chunk_id)
            print(f"\nchunk_id:[{chunk_id}]")
            print(f"source:[{doc.metadata.get('source')}]")
            print(f"index:[{doc.metadata.get('start_index')}]")
            print(doc.page_content if doc else "NOT FOUND")
