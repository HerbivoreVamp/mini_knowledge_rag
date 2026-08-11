# Agent 对话封装
from langchain.messages import HumanMessage

from core.logger import logger
from core.exceptions import GenerationError


def chat_generation(agent, input_message, config):
    try:
        output_message = agent.invoke(
            {"messages": [HumanMessage(input_message)]},
            stream_mode="values",
            config=config
        )

        logger.info("模型文本生成完成")
        return output_message

    except Exception as e:
        logger.exception(
            "模型文本生成失败"
        )
        raise GenerationError(
            f"模型文本生成失败{e}"
        ) from e
