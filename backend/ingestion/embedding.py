# HuggingFace 嵌入封装

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings


def embeddings(dir_path, model_name):
    emb = HuggingFaceEmbeddings(
        model_name=str(Path(dir_path) / model_name)
    )
    return emb


if __name__ == "__main__":
    dir_path = r"D:\home\models\BAAI"
    model_name = r"bge-small-zh-v1.5"
    embeddings(dir_path, model_name)
    print(f"{model_name}加载成功")
