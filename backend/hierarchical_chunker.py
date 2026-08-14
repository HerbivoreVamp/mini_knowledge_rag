from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from config.settings import get_settings
from config.model import create_embedding
from ingestion.loader import load_md
settings = get_settings()
# --- 模型初始化 ---
emb = create_embedding(settings)


parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)
vectorstore = FAISS.from_texts(
    ["初始化"],
    emb
)
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,       # Child 存这里，用来做向量搜索
    docstore=store,                # Parent 存这里
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
)
docs = load_md(str(settings.document_dir), "langchain_doc")
retriever.add_documents(docs)

query = "langchain的tool"

results = retriever.invoke(query)


for i, result in enumerate(results, 1):
    print(f"\n========== Result {i} ==========")
    print(result.page_content)
    print(result.metadata)
