from langchain.agents import create_agent

from retrieval.search import create_retrieve_tool
from core.logger import logger
from core.exceptions import AgentError


def create_rag_agent(llm, system_prompt, vectorstore, checkpointer):
    """创建RAG Agent"""
    if vectorstore is None:
        logger.error("vectorstore不存在")
        raise AgentError(
            "vectorstore不能为空"
        )
    try:
        tool = create_retrieve_tool(vectorstore)
        agent = create_agent(
            llm,
            tools=[tool],
            system_prompt=system_prompt,
            checkpointer=checkpointer
        )


        logger.info("Agent创建成功")

        return agent

    except Exception as e:
        logger.exception("Agent创建失败")
        raise AgentError(
            "Agent创建失败"
        ) from e
