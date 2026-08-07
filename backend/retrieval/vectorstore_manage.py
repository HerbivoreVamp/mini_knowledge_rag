# FAISS 构建/保存/加载

from pathlib import Path
from langchain_community.vectorstores import FAISS

from utils.logger import logger
def build_vectorstore(docs, embedding):
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embedding
    )
    logger.info(f'向量库建造成功 len(docs)={len(docs)}')
    return vectorstore


def save_vectorstore(vector_store, save_dir, index_name):
    path = Path(save_dir) / index_name

    vector_store.save_local(
        folder_path=str(path),
        index_name=index_name
    )
    logger.info(f"向量库保存至{str(path)}")
    return True


def load_vectorstore(load_dir, embedding, index_name):
    path = Path(load_dir) / index_name
    vectorstore = FAISS.load_local(
        folder_path=str(path),
        embeddings=embedding,
        index_name=index_name,
        allow_dangerous_deserialization=True
    )
    logger.info(f"向量库载入成功 path={str(path)}")
    return vectorstore


if __name__ == '__main__':  #创建一个索引集
    import sys

    # backend目录加入搜索路径
    sys.path.append(
        str(Path(__file__).resolve().parent.parent)
    )

    from ingestion.embedding import embeddings
    from ingestion.loader import load_all_md
    from ingestion.splitter import split_text

    # docs = load_all_md(
    #     r"D:\dump\my_project\mini_knowledge_rag\backend\document\langchain_doc")
    emb = embeddings(r"D:\home\models\BAAI", r"bge-small-zh-v1.5")
    # splits = split_text(docs)
    # vectorstore = build_vectorstore(docs, emb)
    #
    # save_vectorstore(vectorstore,
    #                  save_dir=r"D:\dump\my_project\mini_knowledge_rag\backend\database",
    #                  index_name="langchain_doc")

    vectorstore = load_vectorstore(load_dir=r"D:\dump\my_project\mini_knowledge_rag\backend\database", # 导入测试
                                   embedding=emb,
                                   index_name="langchain_doc")
