# Agent 对话封装
from langchain.messages import HumanMessage

from utils.logger import logger
from utils.exceptions import GenerationError


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
            "模型文本生成失败"
        ) from e
