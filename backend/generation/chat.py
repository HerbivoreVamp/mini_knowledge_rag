# Agent 对话封装

from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

def chat_generation(agent, input_message, config):
    output_message = agent.invoke({"messages": [HumanMessage(input_message)]},
                                  stream_mode="values",
                                  config=config)
    return output_message


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # backend目录加入搜索路径
    sys.path.append(
        str(Path(__file__).resolve().parent.parent)
    )
    from retrieval.search import create_retrieve_tool
    from ingestion.embedding import embeddings
    from retrieval.vectorstore_manage import load_vectorstore
    from langchain.chat_models import init_chat_model
    from langchain.agents import create_agent

    emb = embeddings(r"D:\home\models\BAAI", r"bge-small-zh-v1.5")
    vectorstore = load_vectorstore(load_dir=r"D:\dump\my_project\mini_knowledge_rag\backend\database", # 导入测试
                                   embedding=emb,
                                   index_name="langchain_doc")

    retrieve_tool = create_retrieve_tool(vectorstore)
    model = init_chat_model(
        model="qwen/qwen3-8b",
        model_provider="openai",
        base_url="http://localhost:1234/v1",
        api_key="1234",
    )
    with SqliteSaver.from_conn_string(r"D:\dump\my_project\mini_knowledge_rag\backend\memory\checkpoints.db") as checkpointer:
        agent = create_agent(
            model,
            tools=[retrieve_tool],
            system_prompt="你有检索工具，它有绝大多数可查询的文档，需要时调用它获取知识",
            checkpointer=checkpointer
        )

        mes = chat_generation(agent=agent,
                              input_message="我之前问了你什么？",
                              config={"configurable": {"thread_id": "user0"}})
        mes["messages"][-1].pretty_print()
