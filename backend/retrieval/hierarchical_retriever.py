from typing import TYPE_CHECKING

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from backend.storage.docstore import JsonDocStore
from backend.core.exceptions import RetrievalError
from backend.core.logger import logger


class HierarchicalRetriever(BaseRetriever):
    child_retriever: BaseRetriever
    parent_store: JsonDocStore
    parent_k: int
    if TYPE_CHECKING:  # 方便显示参数 没有实际运行
        def __init__(
                self,
                *,
                child_retriever: BaseRetriever,
                parent_store: JsonDocStore,
                parent_k: int,
        ):
            ...

    def _expand(self, child_docs) -> list[Document | None]:
        parent_ids = []
        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)
        logger.info(f"child_docs已传入HierarchicalRetriever total_child={len(child_docs)}")
        return self.parent_store.mget(parent_ids)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        try:
            child_docs = self.child_retriever.invoke(query)
        except Exception as e:
            logger.exception(f"child检索阶段失败 error={e}")
            raise RetrievalError("child检索失败") from e

        try:
            parent_docs = self._expand(child_docs)
        except Exception as e:
            logger.exception(f"parent扩展阶段失败 error={e}")
            raise RetrievalError("parent扩展失败") from e

        total_docs = len(parent_docs)
        parent_docs = [doc for doc in parent_docs if doc is not None]  # 去掉None的部分 也就是没找到的
        missing_docs = total_docs - len(parent_docs)

        if missing_docs > 0:
            logger.warning(f"异常:部分 parent_docs 未找到"
                           f"parent_docs召回成功 保留parent_k={self.parent_k} total_docs={total_docs}, missing_docs={missing_docs}")
        else:
            logger.info(f"parent_docs召回成功 保留parent_k={self.parent_k} total_docs={total_docs}, missing_docs={missing_docs}")
        return parent_docs[:self.parent_k]
