import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.stores import BaseStore

from backend.core.logger import logger
from backend.core.exceptions import DocStoreError

# 临时使用 未来更换为sqlite
class JsonDocStore(BaseStore):
    def __init__(self, parent_store_dir):
        self.parent_store_dir = Path(parent_store_dir)

        try:
            self.parent_store_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            self.path = self.parent_store_dir / "parents.json"
            self.data = self._load()

        except OSError as e:
            logger.error(
                "Parent Document Store 初始化失败 path=%s error=%s",
                self.parent_store_dir,
                e
            )
            raise DocStoreError(
                f"Parent Document Store 初始化失败: "
                f"{self.parent_store_dir}"
            ) from e

        logger.info(
            "Parent Document Store 加载成功 path=%s docs=%s",
            self.path,
            len(self.data)
        )

    def _load(self):
        if not self.path.exists():
            logger.info(
                "Parent Document Store 文件不存在，将创建空存储 path=%s",
                self.path
            )
            return {}

        try:
            with open(
                    self.path,
                    "r",
                    encoding="utf-8"
            ) as f:
                return json.load(f)

        except json.JSONDecodeError as e:
            logger.error(
                "Parent Document Store JSON 解析失败 path=%s error=%s",
                self.path,
                e
            )
            raise DocStoreError(
                f"Parent Document Store JSON 数据损坏: {self.path}"
            ) from e

        except OSError as e:
            logger.error(
                "Parent Document Store 读取失败 path=%s error=%s",
                self.path,
                e
            )
            raise DocStoreError(
                f"Parent Document Store 读取失败: {self.path}"
            ) from e

    def mset(self, items):
        try:
            count = 0

            for key, doc in items:
                self.data[key] = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                count += 1

            self._save()

            logger.info(
                "Parent Documents 保存成功 path=%s docs=%s",
                self.path,
                count
            )

        except (OSError, TypeError, ValueError) as e:
            logger.error(
                "Parent Documents 保存失败 path=%s error=%s",
                self.path,
                e
            )
            raise DocStoreError(
                f"Parent Documents 保存失败: {self.path}"
            ) from e

    def mget(self, keys):
        result = []

        for key in keys:
            item = self.data.get(key)

            if item is not None:
                result.append(
                    Document(
                        page_content=item["page_content"],
                        metadata=item["metadata"]
                    )
                )
            else:
                result.append(None)

        return result

    def mdelete(self, keys):
        try:
            count = 0

            for key in keys:
                if key in self.data:
                    self.data.pop(key)
                    count += 1

            self._save()

            logger.info(
                "Parent Documents 删除成功 path=%s docs=%s",
                self.path,
                count
            )

        except OSError as e:
            logger.error(
                "Parent Documents 删除失败 path=%s error=%s",
                self.path,
                e
            )
            raise DocStoreError(
                f"Parent Documents 删除失败: {self.path}"
            ) from e

    def yield_keys(self, prefix=None):
        for key in self.data.keys():
            if prefix is None or key.startswith(prefix):
                yield key

    def _save(self):
        try:
            with open(
                    self.path,
                    "w",
                    encoding="utf-8"
            ) as f:
                json.dump(
                    self.data,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        except (OSError, TypeError, ValueError) as e:
            logger.error(
                "Parent Document Store 写入失败 path=%s error=%s",
                self.path,
                e
            )
            raise DocStoreError(
                f"Parent Document Store 写入失败: {self.path}"
            ) from e


def create_docstore(parent_store_dir: Path) -> JsonDocStore:
    store = JsonDocStore(
        parent_store_dir
    )
    return store
