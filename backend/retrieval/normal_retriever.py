from typing import TYPE_CHECKING
from pydantic import model_validator, PrivateAttr

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from backend.core.exceptions import RetrievalError
from backend.core.logger import logger


class NormalRetriever(BaseRetriever):
    vectorstore: VectorStore
    k: int = 30
    _retriever: BaseRetriever | None = PrivateAttr(default=None)  # 动态生成的内部对象(由VectorStore生成) 私有属性 / 内部使用
    if TYPE_CHECKING:  # 方便显示参数 没有实际运行
        def __init__(
                self,
                *,
                vectorstore: VectorStore,
                k: int = 30,
        ):
            ...

    @model_validator(mode="after")  # 对象创建完成以后，根据已有字段做初始化
    def create_retriever(self):
        try:
            self._retriever = (
                self.vectorstore.as_retriever(
                    search_kwargs={"k": self.k}
                )
            )

            logger.info(
                f"NormalRetriever创建成功 k={self.k}"
            )

            return self

        except Exception as e:
            logger.exception(
                f"NormalRetriever创建失败 error={e}"
            )
            raise RetrievalError(
                "NormalRetriever创建失败"
            ) from e

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        if self._retriever is None:
            raise RetrievalError(
                "NormalRetriever未初始化"
            )
        try:
            docs = self._retriever.invoke(query)
            logger.info(f"docs召回成功 total_docs={len(docs)}")
        except Exception as e:
            logger.exception(f"docs召回出错 error={e}")
            raise RetrievalError("docs召回出错") from e

        return docs
