# SqliteDocStore 替代 JsonDocStore
import sqlite3
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.stores import BaseStore

from backend.core.logger import logger
from backend.core.exceptions import DocStoreError

# sqlite 连接不可跨线程使用 LangGraph 可能在线程中调用检索
# check_same_thread=False + 每次操作后 commit 保证单进程内安全
class SqliteDocStore(BaseStore):
    def __init__(self, store_dir):
        self.store_dir = Path(store_dir)

        try:
            self.store_dir.mkdir(
                parents=True,
                exist_ok=True
            )
            self.path = self.store_dir / "docstore.db"
            self.conn = sqlite3.connect(
                self.path,
                check_same_thread=False
            )
            self._init_table()

        except (OSError, sqlite3.Error) as e:
            logger.error(
                "SqliteDocStore 初始化失败 path=%s error=%s",
                self.store_dir,
                e
            )
            raise DocStoreError(
                f"SqliteDocStore 初始化失败: {self.store_dir}"
            ) from e

        logger.info(
            "SqliteDocStore 初始化成功 path=%s docs=%s",
            self.path,
            self.count()
        )

    def _init_table(self):
        try:
            with self.conn:
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        key TEXT PRIMARY KEY,
                        page_content TEXT NOT NULL,
                        metadata TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as e:
            logger.error(
                "SqliteDocStore 建表失败 path=%s error=%s",
                self.path,
                e
            )
            raise DocStoreError(
                f"SqliteDocStore 建表失败: {self.path}"
            ) from e

    @staticmethod
    def _doc_to_row(doc: Document):
        import json
        return (
            doc.page_content,
            json.dumps(doc.metadata, ensure_ascii=False)
        )

    @staticmethod
    def _row_to_doc(row):
        import json
        if row is None:
            return None
        return Document(
            page_content=row[0],
            metadata=json.loads(row[1])
        )

    def mset(self, items):
        try:
            with self.conn:
                self.conn.executemany(
                    """
                    INSERT INTO documents (key, page_content, metadata)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        page_content=excluded.page_content,
                        metadata=excluded.metadata
                    """,
                    [
                        (key, *self._doc_to_row(doc))
                        for key, doc in items
                    ]
                )

            logger.info(
                "SqliteDocStore 保存成功 path=%s docs=%s",
                self.path,
                len(items)
            )

        except (sqlite3.Error, AttributeError, TypeError) as e:
            logger.error(
                "SqliteDocStore 保存失败 path=%s error=%s",
                self.path,
                e
            )
            raise DocStoreError(
                f"SqliteDocStore 保存失败: {self.path}"
            ) from e

    def mget(self, keys):
        result = []

        try:
            for key in keys:
                row = self.conn.execute(
                    """
                    SELECT page_content, metadata
                    FROM documents
                    WHERE key = ?
                    """,
                    (key,)
                ).fetchone()
                result.append(
                    self._row_to_doc(row)
                )

        except sqlite3.Error as e:
            logger.error(
                "SqliteDocStore 读取失败 path=%s error=%s",
                self.path,
                e
            )
            raise DocStoreError(
                f"SqliteDocStore 读取失败: {self.path}"
            ) from e

        return result

    def mdelete(self, keys):
        try:
            with self.conn:
                cursor = self.conn.executemany(
                    """
                    DELETE FROM documents
                    WHERE key = ?
                    """,
                    [(key,) for key in keys]
                )

            logger.info(
                "SqliteDocStore 删除成功 path=%s docs=%s",
                self.path,
                cursor.rowcount if cursor else 0
            )

        except sqlite3.Error as e:
            logger.error(
                "SqliteDocStore 删除失败 path=%s error=%s",
                self.path,
                e
            )
            raise DocStoreError(
                f"SqliteDocStore 删除失败: {self.path}"
            ) from e

    def yield_keys(self, prefix=None):
        if prefix is None:
            cursor = self.conn.execute(
                "SELECT key FROM documents"
            )
        else:
            cursor = self.conn.execute(
                "SELECT key FROM documents WHERE key LIKE ?",
                (prefix + "%",)
            )

        for row in cursor:
            yield row[0]

    def get_all_documents(self):
        """读取全部文档 用于BM25等需要全量语料的场景"""
        docs = []

        try:
            cursor = self.conn.execute(
                "SELECT page_content, metadata FROM documents"
            )
            for row in cursor:
                docs.append(
                    self._row_to_doc(row)
                )

        except sqlite3.Error as e:
            logger.error(
                "SqliteDocStore 全量读取失败 path=%s error=%s",
                self.path,
                e
            )
            raise DocStoreError(
                f"SqliteDocStore 全量读取失败: {self.path}"
            ) from e

        return docs

    def count(self):
        row = self.conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()
        return row[0]

    def close(self):
        self.conn.close()


def create_sqlite_docstore(store_dir) -> SqliteDocStore:
    store = SqliteDocStore(
        store_dir
    )
    return store
