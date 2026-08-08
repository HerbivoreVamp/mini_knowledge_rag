# 删除向量库
import shutil
from pathlib import Path
from utils.logger import logger
from utils.exceptions import VectorStoreNotFoundError, DeletionError


def delete_vectorstore(delete_dir: str, index_name: str):
    """删除指定名称的向量库文件夹"""

    path = Path(delete_dir) / index_name

    try:
        if not path.exists():
            raise VectorStoreNotFoundError(
                f"向量库不存在 path={path}"
            )

        shutil.rmtree(path)

        logger.info(
            f"向量库删除成功 path={path}"
        )

    except Exception as e:
        raise DeletionError(f"删除出错 path={path}") from e


if __name__ == "__main__":
    delete_vectorstore(
        r"D:\dump\my_project\mini_knowledge_rag\backend\database",
        "vanilla"
    )
