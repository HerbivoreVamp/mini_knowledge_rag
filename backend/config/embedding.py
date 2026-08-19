# HuggingFace 嵌入封装

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

from core.logger import logger
from core.exceptions import EmbeddingError


def embeddings(dir_path: Path, model_name: str, device: str = 'cpu'):
    try:
        emb = HuggingFaceEmbeddings(
            model_name=str(dir_path / model_name),
            model_kwargs={'device': device}
        )
        logger.info(f"嵌入模型加载成功 model_name={model_name}")
    except Exception as e:
        logger.exception(f"嵌入模型加载失败 model_name={model_name}")
        raise EmbeddingError(f"嵌入模型加载失败: {model_name}") from e
    return emb


if __name__ == "__main__":
    dir_path = r"D:\home\models\BAAI"
    model_name = r"bge-small-zh-v1.5"
    embeddings(dir_path, model_name)
    print(f"{model_name}加载成功")
