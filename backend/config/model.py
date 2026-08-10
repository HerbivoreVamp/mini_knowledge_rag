from langchain.chat_models import init_chat_model

from utils.logger import logger
from utils.exceptions import LLMError
from config.embedding import embeddings


def create_embedding(settings):
    emb = embeddings(settings.emb_dir_path, settings.emb_model_name)
    return emb


def create_llm(settings):
    try:
        llm = init_chat_model(model=settings.model, model_provider=settings.model_provider, base_url=settings.base_url,
                              api_key=settings.api_key)
        logger.info(f"语言模型配置加载成功 model={settings.model}")
        return llm
    except Exception as e:
        logger.exception(f"语言模型配置加载出错 model={settings.model}")
        raise LLMError(f"语言模型配置加载出错 model={settings.model}") from e
