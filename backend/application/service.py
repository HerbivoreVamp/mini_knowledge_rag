from backend.core.exceptions import GenerationError, RAGError
from backend.agent.agent import create_rag_agent
from .chat import chat_generation


def create_generation_service(llm, system_prompt,retriever, checkpointer):
    try:
        agent = create_rag_agent(llm, system_prompt, retriever, checkpointer)
    except RAGError:
        raise

    def generator(input_mes, config):
        try:
            mes = chat_generation(agent=agent, input_message=input_mes, config=config)
            return mes
        except GenerationError:
            raise

    return generator
