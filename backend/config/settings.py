# 配置相关
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path

from core.logger import logger


@dataclass
class Settings:
    base_dir: Path
    database_dir: Path
    memory_dir: Path
    document_dir: Path
    index_name: str

    model: str
    model_provider: str
    base_url: str
    api_key: str

    emb_dir_path: str
    emb_model_name: str
    config: dict


def get_settings():
    load_dotenv()
    base = Path(__file__).resolve().parent.parent
    logger.info("配置加载成功")
    return Settings(
        base_dir=base,
        database_dir=base / "data" / "database",
        memory_dir=base / "data" / "memory",
        document_dir=base / "data" / "document",
        index_name="vanilla",

        model=os.getenv("MODEL"),
        api_key=os.getenv("API_KEY"),
        model_provider=os.getenv("MODEL_PROVIDER"),
        base_url=os.getenv("BASE_URL"),

        emb_dir_path=os.getenv("EMB_DIR_PATH"),
        emb_model_name=os.getenv("EMB_MODEL_NAME"),
        config={"configurable": {"thread_id": "user0"}}
    )
