from storage.docstore import JsonDocStore
from config.reranker import BGEReranker


# 带reranker的HierarchicalRetriever

class HierarchicalRetriever:
    def __init__(
            self,
            vectorstore,
            parent_store: JsonDocStore,
            reranker: BGEReranker,
    ):
        self.vectorstore = vectorstore
        self.parent_store = parent_store
        self.reranker = reranker

        self.child_retriever = (
            vectorstore.as_retriever(
                search_kwargs={
                    "k": 30
                }
            )
        )

    def invoke(self, query, k=5):
        # 1. child召回
        child_docs = self.child_retriever.invoke(query)

        # 2. rerank
        child_docs = self.reranker.rerank(
            query,
            child_docs,
            top_k=k
        )

        # 3. child -> parent
        parent_docs = []

        for child in child_docs:
            parent_id = child.metadata["parent_id"]

            parent = self.parent_store.mget(
                [parent_id]
            )[0]

            parent_docs.append(parent)

        return parent_docs
