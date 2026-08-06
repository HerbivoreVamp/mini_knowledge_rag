# 删除向量库
import shutil
from pathlib import Path


def delete_vectorstore(delete_dir: str, index_name: str) -> bool:
    """删除指定名称的向量库文件夹"""
    path = Path(delete_dir) / index_name
    if path.exists():
        shutil.rmtree(str(path))
        print(f"向量库 {index_name} 已删除")
        return True
    else:
        print(f"向量库 {index_name} 不存在，无需删除")
        return False


if __name__ == "__main__":
    delete_vectorstore(
        r"D:\dump\my_project\mini_knowledge_rag\backend\database",
        "vanilla"
    )