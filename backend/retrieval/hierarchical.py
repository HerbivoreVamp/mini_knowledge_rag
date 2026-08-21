from backend.storage.docstore import JsonDocStore
from backend.config.reranker import BGEReranker
from backend.core.exceptions import RetrievalError
from backend.core.logger import logger


# 带reranker的HierarchicalRetriever

class HierarchicalRetriever:
    def __init__(
            self,
            vectorstore,
            parent_store: JsonDocStore,
            reranker: BGEReranker,
            k=30
    ):
        try:
            self.vectorstore = vectorstore
            self.parent_store = parent_store
            self.reranker = reranker

            self.child_retriever = (
                vectorstore.as_retriever(
                    search_kwargs={
                        "k": k
                    }
                )
            )
            logger.info(f"HierarchicalRetriever创建成功 k={k}")
        except Exception as e:
            logger.exception(f"HierarchicalRetriever创建失败 error={e}")
            raise RetrievalError("HierarchicalRetriever创建失败") from e

    def invoke(self, query, rerank_k=5):
        try:
            # 1. child召回
            child_docs = self.child_retriever.invoke(query)
        except Exception as e:
            logger.exception(f"child召回出错 error={e}")
            raise RetrievalError("child召回出错") from e
        try:
            # 2. rerank
            child_docs = self.reranker.rerank(
                query,
                child_docs,
                top_k=rerank_k
            )
        except Exception as e:
            logger.exception(f"rerank阶段出错 error={e}")
            raise RetrievalError("rerank阶段出错")
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
            raise RetrievalError("parent_docs处理阶段出错")
        if missing_docs > 0:
            logger.warning(f"异常:部分 parent_docs 未找到"
                           f"parent_docs召回成功 total_docs={total_docs}, missing_docs={missing_docs}")
        else:
            logger.info(f"parent_docs召回成功 total_docs={total_docs}, missing_docs={missing_docs}")
        return parent_docs
