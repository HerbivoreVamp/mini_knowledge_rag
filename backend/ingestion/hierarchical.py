import json
from pathlib import Path

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import BaseStore

from .splitter import create_text_splitter
from core.logger import logger
from core.exceptions import RetrievalError, DocStoreError


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

            if item:
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


def create_docstore(parent_store_dir: str) -> JsonDocStore:
    store = JsonDocStore(
        Path(parent_store_dir)
    )
    return store


def create_parent_retriever(vectorstore, parent_store: JsonDocStore) -> ParentDocumentRetriever:
    parent_chunk_size = 2000
    parent_chunk_overlap = 200
    child_chunk_size = 400
    child_chunk_overlap = 50
    k = 2

    parent_splitter = create_text_splitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap,
                                           add_start_index=True)
    child_splitter = create_text_splitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap,
                                          add_start_index=True)

    try:
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=parent_store,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
            search_kwargs={
                "k": 2
            }
        )
    except Exception as e:
        logger.error(
            "Hierarchical Retriever 创建失败 error=%s",
            e
        )
        raise RetrievalError(
            "Hierarchical Retriever 创建失败"
        ) from e
    logger.info(
        "Hierarchical Retriever 创建成功 "
        "parent_chunk_size=%s parent_chunk_overlap=%s "
        "child_chunk_size=%s child_chunk_overlap=%s k=%s",
        parent_chunk_size,
        parent_chunk_overlap,
        child_chunk_size,
        child_chunk_overlap,
        k,
    )
    return retriever


def add_documents_to_retriever(retriever, docs) -> ParentDocumentRetriever:
    retriever.add_documents(docs)
    logger.info(f"新文档已添加至retriever docs={len(docs)}")
    return retriever
