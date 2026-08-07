# 删除向量库
import shutil
from pathlib import Path
from utils.logger import logger


def delete_vectorstore(delete_dir: str, index_name: str) -> bool:
    """删除指定名称的向量库文件夹"""
    path = Path(delete_dir) / index_name
    if path.exists():
        shutil.rmtree(str(path))
        logger.info(f"向量库删除成功 路径{path}")
        return True
    else:
        logger.info(f"向量库删除失败 向量库 {index_name} 不存在")
        return False


if __name__ == "__main__":
    delete_vectorstore(
        r"D:\dump\my_project\mini_knowledge_rag\backend\database",
        "vanilla"
    )
