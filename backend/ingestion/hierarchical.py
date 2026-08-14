import json
from pathlib import Path

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import BaseStore

from .splitter import create_text_splitter


class JsonDocStore(BaseStore):
    def __init__(self, parent_store_dir):
        self.parent_store_dir = Path(parent_store_dir)

        self.parent_store_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        self.path = self.parent_store_dir / "parents.json"
        self.data = self._load()

    def _load(self):
        if not self.path.exists():
            return {}

        with open(
                self.path,
                "r",
                encoding="utf-8"
        ) as f:
            return json.load(f)

    def mset(self, items):
        for key, doc in items:
            self.data[key] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }

        self._save()

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

        for key in keys:
            self.data.pop(key, None)

        self._save()

    def yield_keys(self, prefix=None):

        for key in self.data.keys():
            if prefix is None or key.startswith(prefix):
                yield key

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                self.data,
                f,
                ensure_ascii=False,
                indent=2
            )


def create_docstore(parent_store_dir: str) -> JsonDocStore:
    store = JsonDocStore(
        Path(parent_store_dir)
    )
    return store


def create_parent_retriever(vectorstore, parent_store: JsonDocStore) -> ParentDocumentRetriever:
    parent_splitter = create_text_splitter(chunk_size=2000, chunk_overlap=200, add_start_index=True)
    child_splitter = create_text_splitter(chunk_size=400, chunk_overlap=50, add_start_index=True)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,  # Child 存这里，用来做向量搜索
        docstore=parent_store,  # Parent 存这里
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        search_kwargs={
            "k": 2
        }
    )
    return retriever


def add_documents_to_retriever(retriever, docs) -> ParentDocumentRetriever:
    retriever.add_documents(docs)
    return retriever
