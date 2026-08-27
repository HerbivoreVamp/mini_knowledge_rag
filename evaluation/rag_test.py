# RAG 检索评估脚本 参照 main.py
# 指标 Hit@K: 召回的前K个child chunk里是否包含正确chunk
import json
from datetime import datetime
from pathlib import Path

from backend.config.settings import get_settings
from backend.config.model import create_embedding, create_reranker
from backend.retrieval.factory import create_retriever
from backend.storage.sqlite_docstore import create_sqlite_docstore
from backend.storage.vectorstore import load_vectorstore
from backend.core.logger import setup_logger
from backend.core.exceptions import RAGError

# ===== 评估配置 =====
RETRIEVER = "dense"
K_VALUES = [5, 20]    # Hit@K 的 K 值列表
MAX_K = max(K_VALUES)

RETRIEVER_CONFIG = {
    "dense":        {"hierarchical": False, "hybrid": False, "use_reranker": False},
    "hybrid":       {"hierarchical": False, "hybrid": True,  "use_reranker": False},
    "dense_rerank": {"hierarchical": False, "hybrid": False, "use_reranker": True},
    "hybrid_rerank": {"hierarchical": False, "hybrid": True, "use_reranker": True},
}

logger = setup_logger()
logger.info("RAG评估启动")

# --- 配置加载 指向 evaluation 数据库 ---
settings = get_settings()
eval_base = Path(__file__).resolve().parent
eval_db_dir = eval_base / "data" / "database"
settings.vectorstore_dir = eval_db_dir / settings.database_name / "vectorstore"
settings.parent_store_dir = eval_db_dir / settings.database_name / "parent_store"
settings.child_store_dir = eval_db_dir / settings.database_name / "child_store"

# --- 模型初始化 ---
emb = create_embedding(settings)
cfg = RETRIEVER_CONFIG[RETRIEVER]
reranker = create_reranker(settings) if cfg["use_reranker"] else None

# --- 向量库与检索器 ---
try:
    vectorstore = load_vectorstore(settings.vectorstore_dir, emb, settings.index_name)
    parent_store = create_sqlite_docstore(settings.parent_store_dir) if cfg["hierarchical"] else None
    child_store = create_sqlite_docstore(settings.child_store_dir) if cfg["hybrid"] else None
    retriever = create_retriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
        child_store=child_store,
        reranker=reranker,
        hierarchical=cfg["hierarchical"],
        hybrid=cfg["hybrid"],
        k=MAX_K,
        rerank_topk=MAX_K,
    )
    logger.info(f"检索器创建成功 retriever={RETRIEVER} k={MAX_K}")
except RAGError as e:
    print(e)
    raise

# --- 加载 QA 数据集 ---
qa_path = eval_db_dir / settings.database_name / "langchain_qa_dataset.json"
with open(qa_path, encoding="utf-8") as f:
    qa_dataset = json.load(f)
logger.info(f"QA数据集加载成功 total={len(qa_dataset)}")

# --- 评估 Hit@K ---
hits = {k: 0 for k in K_VALUES}
for item in qa_dataset:
    query = item["question"]
    gold_chunk_ids = {ctx["chunk_id"] for ctx in item["contexts"]}
    docs = retriever.invoke(query)
    for k in K_VALUES:
        retrieved_ids = {doc.metadata.get("chunk_id") for doc in docs[:k]}
        if gold_chunk_ids & retrieved_ids:
            hits[k] += 1

hit_at_k = {k: hits[k] / len(qa_dataset) for k in K_VALUES}
for k in K_VALUES:
    logger.info(f"评估完成 Hit@{k}={hit_at_k[k]:.4f} hits={hits[k]}/{len(qa_dataset)}")

# --- 保存结果 ---
result = {
    "model": settings.emb_model_name,
    "retriever": RETRIEVER,
}
for k in K_VALUES:
    result[f"recall@{k}"] = round(hit_at_k[k], 2)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = eval_base / f"{timestamp}.json"
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
logger.info(f"结果已保存 path={result_path}")
print(json.dumps(result, ensure_ascii=False, indent=2))
