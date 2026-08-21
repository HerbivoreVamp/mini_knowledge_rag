from typing import TYPE_CHECKING
from pydantic import model_validator, PrivateAttr

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from backend.storage.docstore import JsonDocStore
from backend.config.reranker import BGEReranker
from backend.core.exceptions import RetrievalError
from backend.core.logger import logger


class HierarchicalRetriever(BaseRetriever):
    vectorstore: VectorStore
    parent_store: JsonDocStore
    reranker: BGEReranker
    k: int = 30
    rerank_k: int = 5
    _child_retriever: BaseRetriever | None = PrivateAttr(default=None)  # 动态生成的内部对象(由VectorStore生成) 私有属性 / 内部使用
    if TYPE_CHECKING:  # 方便显示参数 没有实际运行
        def __init__(
                self,
                *,
                vectorstore: VectorStore,
                parent_store: JsonDocStore,
                reranker: BGEReranker,
                k: int = 30,
                rerank_k: int = 5,
        ):
            ...

    @model_validator(mode="after")  # 对象创建完成以后，根据已有字段做初始化
    def create_child_retriever(self):
        try:
            self._child_retriever = (
                self.vectorstore.as_retriever(
                    search_kwargs={"k": self.k}
                )
            )

            logger.info(
                f"HierarchicalRetriever创建成功 k={self.k}"
            )

            return self

        except Exception as e:
            logger.exception(
                f"HierarchicalRetriever创建失败 error={e}"
            )
            raise RetrievalError(
                "HierarchicalRetriever创建失败"
            ) from e

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        try:
            # 1. child召回
            child_docs = self._child_retriever.invoke(query)
            logger.info(f"child召回成功 total_child={len(child_docs)}")
        except Exception as e:
            logger.exception(f"child召回出错 error={e}")
            raise RetrievalError("child召回出错") from e
        try:
            # 2. rerank
            child_docs = self.reranker.rerank(
                query,
                child_docs,
                top_k=self.rerank_k
            )
        except Exception as e:
            logger.exception(f"rerank阶段出错 error={e}")
            raise RetrievalError("rerank阶段出错") from e
        try:
            # 3. child -> parent
            parent_ids = []

            for child in child_docs:
                parent_id = child.metadata.get("parent_id")

                if parent_id and parent_id not in parent_ids:
                    parent_ids.append(parent_id)

            parent_docs = self.parent_store.mget(parent_ids)
            total_docs = len(parent_docs)
            parent_docs = [
                doc for doc in parent_docs
                if doc is not None
            ]
            missing_docs = total_docs - len(parent_docs)
        except Exception as e:
            logger.exception(f"parent_docs处理阶段出错 error={e}")
            raise RetrievalError("parent_docs处理阶段出错") from e
        if missing_docs > 0:
            logger.warning(f"异常:部分 parent_docs 未找到"
                           f"parent_docs召回成功 total_docs={total_docs}, missing_docs={missing_docs}")
        else:
            logger.info(f"parent_docs召回成功 total_docs={total_docs}, missing_docs={missing_docs}")
        return parent_docs
