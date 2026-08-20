# 配置相关
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path

from backend.core.logger import logger


@dataclass
class Settings:
    base_dir: Path
    database_dir: Path
    memory_dir: Path
    document_dir: Path
    database_name: str
    index_name: str
    vectorstore_dir: Path
    parent_store_dir: Path

    model: str
    model_provider: str
    base_url: str
    api_key: str

    emb_dir_path: str
    emb_model_name: str
    emb_device: str
    reranker_dir_path: str
    reranker_model_name: str
    reranker_device: str
    config: dict


def get_settings():
    load_dotenv()
    base = Path(__file__).resolve().parent.parent

    base_dir = base
    database_dir = base / "data" / "database"
    memory_dir = base / "data" / "memory"
    document_dir = base / "data" / "document"

    database_name = "vanilla"
    index_name = "index"
    vectorstore_dir = database_dir / database_name / "vectorstore"
    parent_store_dir = database_dir / database_name / "parent_store"

    model = os.getenv("MODEL")
    api_key = os.getenv("API_KEY")
    model_provider = os.getenv("MODEL_PROVIDER")
    base_url = os.getenv("BASE_URL")

    emb_dir_path = os.getenv("EMB_DIR_PATH")
    emb_model_name = os.getenv("EMB_MODEL_NAME")
    emb_device = os.getenv("EMB_DEVICE")
    reranker_dir_path = os.getenv("RERANKER_DIR_PATH")
    reranker_model_name = os.getenv("RERANKER_MODEL_NAME")
    reranker_device = os.getenv("RERANKER_DEVICE")
    config = {"configurable": {"thread_id": "user0"}}

    logger.info("配置加载成功")
    return Settings(
        base_dir=base_dir,
        database_dir=database_dir,
        memory_dir=memory_dir,
        document_dir=document_dir,

        database_name=database_name,
        index_name=index_name,
        vectorstore_dir=vectorstore_dir,
        parent_store_dir=parent_store_dir,

        model=model,
        api_key=api_key,
        model_provider=model_provider,
        base_url=base_url,

        emb_dir_path=emb_dir_path,
        emb_model_name=emb_model_name,
        emb_device=emb_device,
        reranker_dir_path=reranker_dir_path,
        reranker_model_name=reranker_model_name,
        reranker_device=reranker_device,
        config=config,
    )
