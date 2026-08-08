# FAISS 构建/保存/加载

from pathlib import Path
from langchain_community.vectorstores import FAISS

from utils.logger import logger
from utils.exceptions import VectorStoreLoadError,VectorStoreNotFoundError,VectorStoreError

def build_vectorstore(docs, embedding):
    if not docs:
        raise VectorStoreError(
            "没有文档，无法创建向量库"
        )

    if embedding is None:
        raise VectorStoreError(
            "embedding模型不能为空"
        )

    try:
        vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=embedding
        )

        logger.info(
            f"向量库创建成功 len(docs)={len(docs)}"
        )

        return vectorstore

    except Exception as e:
        logger.exception("向量库创建失败")
        raise VectorStoreError(
            "向量库创建失败"
        ) from e


def save_vectorstore(vector_store, save_dir, index_name):

    if vector_store is None:
        raise VectorStoreError(
            "vectorstore不能为空"
        )

    if not index_name:
        raise VectorStoreError(
            "index_name不能为空"
        )

    path = Path(save_dir) / index_name

    try:
        vector_store.save_local(
            folder_path=str(path),
            index_name=index_name
        )

        logger.info(
            f"向量库保存成功 path={path}"
        )

    except Exception as e:
        logger.exception(
            f"向量库保存失败 path={path}"
        )
        raise VectorStoreError(
            "向量库保存失败"
        ) from e


def load_vectorstore(load_dir, embedding, index_name):
    path = Path(load_dir) / index_name

    if not path.exists():
        raise VectorStoreNotFoundError(
            f"向量库不存在 path={path}"
        )

    try:
        vectorstore = FAISS.load_local(
            folder_path=str(path),
            embeddings=embedding,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )

        logger.info(
            f"向量库加载成功 path={path}"
        )

        return vectorstore

    except Exception as e:
        logger.exception(
            f"向量库加载失败 path={path}"
        )
        raise VectorStoreLoadError(
            f"向量库加载失败 path={path}"
        ) from e


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
