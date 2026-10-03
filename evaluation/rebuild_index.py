"""重建索引副本(改了切分参数 / id 算法之后必须重建才能评测)

用法(项目根目录):
    python evaluation/rebuild_index.py                                   # 重建 evaluation/data/database
    python evaluation/rebuild_index.py --data-dir backend/data/database   # 重建主库(小心, 见下)

行为:
    - 先删掉目标库下的 vectorstore / parent_store / child_store 三个目录, 再重新导入
      (不会碰数据集 json), 避免"追加式导入"造成向量和 parent 翻倍。
    - 只加载 Embedding 模型(建 FAISS 要用), 不需要 Reranker / LLM。
    - 导完做一次自检: child_store 行数必须等于 children 数, 不等就是 chunk_id 又碰撞了。

注意: --data-dir 指向主库时会重建主库, 里面的 parent_id 会全部换新
(数据集已改成锚点口径, 所以不会再失效; 但 main.py 对话记忆不受影响)。
"""
import argparse
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
load_dotenv(BASE.parent / "backend" / ".env")

from backend.config.model import create_embedding  # noqa: E402
from backend.config.settings import get_settings  # noqa: E402
from backend.ingestion.hierarchical import ingest_documents  # noqa: E402
from backend.ingestion.loader import load_md  # noqa: E402
from backend.ingestion.splitter import create_hierarchy_splitter  # noqa: E402
from backend.storage.sqlite_docstore import create_sqlite_docstore  # noqa: E402
from backend.storage.vectorstore import create_empty_vectorstore, save_vectorstore  # noqa: E402

INDEX_DIRS = ("vectorstore", "parent_store", "child_store")


def parse_args():
    p = argparse.ArgumentParser(description="重建索引副本")
    p.add_argument("--data-dir", default=str(BASE / "data" / "database"),
                   help="目标数据库目录, 默认 evaluation/data/database")
    p.add_argument("--folder", default="langchain_doc",
                   help="data/document 下的文档文件夹, 逗号分隔")
    p.add_argument("--keep", action="store_true", help="保留旧索引目录(不推荐, 会变成追加)")
    return p.parse_args()


def main():
    args = parse_args()
    settings = get_settings()
    db_dir = Path(args.data_dir) / settings.database_name

    if not args.keep:
        for name in INDEX_DIRS:
            path = db_dir / name
            if path.exists():
                shutil.rmtree(path)
                print(f"已删除旧索引: {path}")

    emb = create_embedding(settings)
    vectorstore = create_empty_vectorstore(emb)
    parent_store = create_sqlite_docstore(db_dir / "parent_store")
    child_store = create_sqlite_docstore(db_dir / "child_store")
    parent_splitter, child_splitter = create_hierarchy_splitter()

    total_parents = total_children = 0
    for folder in [f.strip() for f in args.folder.split(",") if f.strip()]:
        docs = load_md(settings.document_dir, folder)
        result = ingest_documents(
            docs=docs,
            vectorstore=vectorstore,
            parent_store=parent_store,
            child_store=child_store,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
        )
        total_parents += result["parents"]
        total_children += result["children"]
        print(f"  {folder}: parents={result['parents']} children={result['children']}")

    child_store.close()
    parent_store.close()
    save_vectorstore(vectorstore, db_dir / "vectorstore", settings.index_name)

    check = create_sqlite_docstore(db_dir / "child_store")
    rows = check.count()
    check.close()
    unique = rows == total_children
    print(f"\n重建完成 parents={total_parents} children={total_children} child_store行数={rows}")
    print("chunk_id 唯一性自检:", "通过" if unique else f"失败! 丢了 {total_children - rows} 篇")
    if not unique:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
