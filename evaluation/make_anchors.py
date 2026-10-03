"""把 QA 数据集从 chunk_id 口径迁移成「证据锚点」口径(迁移一次, 以后不用再动)

用法(项目根目录):
    python evaluation/make_anchors.py                          # 默认迁 evaluation/data/database
    python evaluation/make_anchors.py --data-dir backend/data/database

读取: 旧数据集 + child_store + parent_store + 源文档
写出: <data-dir>/<知识库名>/langchain_qa_dataset_anchors.json  (原文件保持不动)

迁移原理:
    旧 child 在源文档里的区间
        = parent_store[parent_id].metadata["start_index"]   (parent 的文件内偏移)
        + child.metadata["start_index"]                     (child 相对 parent 的偏移)
    长度就是切片文本长度。迁移时逐条校验 源文档[区间] == 切片原文,
    实测当前语料 1423 篇 child / 71 条 gold 全部精确命中。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))

from backend.config.settings import get_settings  # noqa: E402
from backend.storage.sqlite_docstore import create_sqlite_docstore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="QA 数据集 -> 证据锚点口径")
    p.add_argument("--data-dir", default=str(BASE / "data" / "database"))
    p.add_argument("--out", default="", help="输出路径, 默认 <data-dir>/<db>/langchain_qa_dataset_anchors.json")
    return p.parse_args()


def main():
    args = parse_args()
    settings = get_settings()
    db_dir = Path(args.data_dir) / settings.database_name

    dataset = json.loads((db_dir / "langchain_qa_dataset.json").read_text(encoding="utf-8"))

    child_store = create_sqlite_docstore(db_dir / "child_store")
    children = {d.metadata["chunk_id"]: d for d in child_store.get_all_documents()}
    child_store.close()
    parent_store = create_sqlite_docstore(db_dir / "parent_store")
    parents = {d.metadata["parent_id"]: d for d in parent_store.get_all_documents()}
    parent_store.close()

    text_cache = {}
    items, ok, bad = [], 0, []
    for item in dataset:
        gold = []
        for ctx in item["contexts"]:
            child = children.get(ctx["chunk_id"])
            if child is None:
                bad.append((ctx["chunk_id"], "child_store 里找不到"))
                continue
            parent = parents.get(child.metadata.get("parent_id"))
            if parent is None:
                bad.append((ctx["chunk_id"], "parent_store 里找不到 parent"))
                continue
            source = child.metadata["source"]
            if source not in text_cache:
                text_cache[source] = (settings.document_dir / source).read_text(encoding="utf-8")
            file_text = text_cache[source]
            start = (parent.metadata.get("start_index") or 0) + (child.metadata.get("start_index") or 0)
            end = start + len(child.page_content)
            if file_text[start:end] != child.page_content:
                bad.append((ctx["chunk_id"], f"区间校验失败 {source}[{start}:{end}]"))
                continue
            ok += 1
            gold.append(
                {
                    "source": source,
                    "span": [start, end],
                    "quote": child.page_content,
                    "excerpt": ctx["content"],
                }
            )
        # parent_id 是 uuid4, 每次导入都变, 不再写进数据集
        metadata = {k: v for k, v in item.get("metadata", {}).items() if k != "parent_id"}
        items.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": item.get("answer", ""),
                "gold": gold,
                "metadata": metadata,
            }
        )

    out_path = Path(args.out) if args.out else db_dir / "langchain_qa_dataset_anchors.json"
    out_path.write_text(
        json.dumps(
            {
                "format": "anchor-v1",
                "generated": datetime.now().isoformat(timespec="seconds"),
                "note": (
                    "ground truth = 源文档字符区间, 评测时按当前索引现算; "
                    "只要源文档内容不变, 重新导入 / 改切分参数 / 改 id 算法都不需要改这个文件。"
                    "解析逻辑见 evaluation/anchors.py"
                ),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total = sum(len(i["gold"]) for i in items)
    print(f"数据集条目={len(items)}  gold 锚点={total}  区间校验通过={ok}  失败={len(bad)}")
    for cid, why in bad[:10]:
        print(f"  [失败] {cid} {why}")
    print(f"已写出: {out_path}")


if __name__ == "__main__":
    main()
