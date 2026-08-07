# Agent 对话封装
from langchain.messages import HumanMessage

from utils.logger import logger


def chat_generation(agent, input_message, config):
    output_message = agent.invoke({"messages": [HumanMessage(input_message)]},
                                  stream_mode="values",
                                  config=config)
    logger.info("模型文本生成")
    return output_message
