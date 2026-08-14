# FAISS 构建/保存/加载
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from core.logger import logger
from core.exceptions import VectorStoreLoadError, VectorStoreNotFoundError, VectorStoreError, VectorStoreDeleteError

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


def create_empty_vectorstore(embeddings):
    if embeddings is None:
        raise VectorStoreError(
            "embedding模型不能为空"
        )
    dimension = len(embeddings.embed_query(""))

    index = faiss.IndexFlatL2(dimension)
    try:
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
    except Exception as e:
        logger.exception("空向量库创建失败")
        raise VectorStoreError(
            "空向量库创建失败"
        ) from e
    return vectorstore


def add_documents(vectorstore, docs):
    """将已经切分好的 Documents 加入 VectorStore"""
    if vectorstore is None:
        raise VectorStoreError(
            "vectorstore不能为None"
        )
    if not isinstance(docs, list):
        raise VectorStoreError(
            "docs必须是由Document组成的列表[Document,...]"
        )

    if not docs:
        raise VectorStoreError(
            "docs不能为空"
        )

    if not all(isinstance(doc, Document) for doc in docs):
        raise VectorStoreError(
            "docs必须是由Document组成的列表[Document,...]"
        )
    try:
        vectorstore.add_documents(docs)
    except Exception as e:
        raise VectorStoreError("Documents加入出错")from e
    return vectorstore


# 已弃用 被create_empty_vectorstore和add_documents取代
def build_vectorstore(docs, embeddings):
    if not docs:
        raise VectorStoreError(
            "没有文档，无法创建向量库"
        )
    if not isinstance(docs[0], Document):
        raise VectorStoreError(
            "docs必须是由Document组成的的列表[Document,...]"
        )
    if embeddings is None:
        raise VectorStoreError(
            "embedding模型不能为空"
        )

    try:
        vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=embeddings
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


def save_vectorstore(vectorstore, save_dir, index_name):
    if vectorstore is None:
        raise VectorStoreError(
            "vectorstore不能为None"
        )
    if not isinstance(save_dir, str):
        raise VectorStoreError(
            "save_dir必须是字符串str"
        )
    if not isinstance(index_name, str):
        raise VectorStoreError(
            "index_name必须是字符串str"
        )
    path = Path(save_dir) / index_name

    try:
        vectorstore.save_local(
            folder_path=str(path),
            index_name=index_name
        )

        logger.info(
            f"向量库保存成功 path={path}"
        )
        return path
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


def delete_vectorstore(delete_dir: str, index_name: str):
    """删除指定名称的向量库文件夹"""

    path = Path(delete_dir) / Path(index_name)

    try:
        if not path.exists():
            raise VectorStoreNotFoundError(
                f"向量库不存在 path={path}"
            )

        shutil.rmtree(path)

        logger.info(
            f"向量库删除成功 path={path}"
        )

    except Exception as e:
        logger.exception(f"删除失败 path={path}")
        raise VectorStoreDeleteError(f"删除出错 path={path}") from e
