from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import get_settings
from config.prompts import SYSTEM_PROMPT
from ingestion.embedding import embeddings
from ingestion.loader import load_all_md
from ingestion.splitter import split_text
from ingestion.delete import delete_vectorstore
from retrieval.vectorstore_manage import load_vectorstore, save_vectorstore, build_vectorstore
from retrieval.search import create_retrieve_tool
from generation.chat import chat_generation

# --- 配置加载 ---
settings = get_settings()
print("配置加载成功")
# --- 模型初始化 ---
emb = embeddings(settings.emb_dir_path, settings.emb_model_name)
print(f"嵌入模型{settings.emb_model_name} 加载成功")
llm = init_chat_model(model=settings.model, model_provider=settings.model_provider, base_url=settings.base_url,
                      api_key=settings.api_key)


# --- 向量库加载 ---
def _load_vs(vector_embedding):
    index_path = settings.database_dir / settings.index_name

    if index_path.exists():
        return load_vectorstore(
            str(settings.database_dir),
            vector_embedding,
            settings.index_name
        )

    return None


def _make_agent(vectorstore, checkpointer):
    """根据当前 vectorstore 和 checkpointer 创建 agent"""
    tool = create_retrieve_tool(vectorstore)
    return create_agent(llm, tools=[tool], system_prompt=SYSTEM_PROMPT, checkpointer=checkpointer)


vectorstore = _load_vs(emb)
print(f"向量库{settings.index_name}加载成功")
# --- 主循环 ---
print("1. 导入文档到数据库")
print("2. 查询知识库")
print("3. 删除数据库")
print("exit : 退出")

while True:
    with SqliteSaver.from_conn_string(str(settings.memory_dir / "checkpoints.db")) as checkpointer:
        agent = _make_agent(vectorstore, checkpointer)

        option = input("输入对应数字来使用功能:\n")

        if option == "1":
            print("功能1: 导入文档")
            folder = input("请输入文档文件夹名称:\n")
            doc_dir = settings.document_dir / folder
            if not doc_dir.exists():
                print(f"错误输入!! 文件夹 {doc_dir} 不存在!!")
                continue
            docs = load_all_md(str(doc_dir))
            splits = split_text(docs)
            vectorstore = build_vectorstore(splits, emb)
            save_vectorstore(vectorstore, str(settings.database_dir), settings.index_name)
            print("保存成功")

        elif option == "2":
            if vectorstore is None:
                print("错误!! vectorstore 为空，请先导入文档!!")
                continue
            print("功能2: 查询知识库")
            mes = chat_generation(
                agent=agent,
                input_message=input("您想查询什么？\n"),
                config={"configurable": {"thread_id": "user0"}},
            )
            mes["messages"][-1].pretty_print()

        elif option == "3":
            print("功能3: 删除数据库")
            if delete_vectorstore(str(settings.database_dir), settings.index_name):
                vectorstore = None

        elif option in ("exit", "退出"):
            print("退出中...")
            break

        else:
            print(f"错误输入 !! option={option} !!")
