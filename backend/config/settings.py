# 配置相关


from dataclasses import dataclass
from pathlib import Path


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


def get_settings():
    base = Path(__file__).resolve().parent.parent

    return Settings(
        base_dir=base,
        database_dir=base / "database",
        memory_dir=base / "memory",
        document_dir=base / "document",
        index_name="vanilla",

        model="qwen/qwen3-8b",  # 这是作者自己部署的模型
        model_provider="openai",
        base_url=r"http://localhost:1234/v1",
        api_key="1234",

        emb_dir_path=r"D:\home\models\BAAI",
        emb_model_name=r"bge-small-zh-v1.5",

    )

