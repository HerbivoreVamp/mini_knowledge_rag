import os
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver

from ingestion.embedding import embeddings
from ingestion.loader import load_all_md
from ingestion.splitter import split_text
from ingestion.delete import delete_vectorstore
from retrieval.vectorstore_manage import load_vectorstore, save_vectorstore, build_vectorstore
from retrieval.search import create_retrieve_tool
from generation.chat import chat_generation

# --- 路径配置 ---
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
MEMORY_DIR = BASE_DIR / "memory"
DOCUMENT_DIR = BASE_DIR / "document"
INDEX_NAME = "vanilla"

# --- 模型初始化 ---
emb_dir_path = r"D:\home\models\BAAI"
emb_model_name = r"bge-small-zh-v1.5"
emb = embeddings(emb_dir_path, emb_model_name)
print(f"{emb_model_name} 加载成功")

model = init_chat_model(
    model="qwen/qwen3-8b",
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="1234",
)


# --- 向量库加载 ---
def _load_vs():
    if (DATABASE_DIR / INDEX_NAME).exists():
        return load_vectorstore(str(DATABASE_DIR), emb, INDEX_NAME)
    return None


def _make_agent(vs, checkpointer):
    """根据当前 vectorstore 和 checkpointer 创建 agent"""
    tool = create_retrieve_tool(vs)
    return create_agent(
        model,
        tools=[tool],
        system_prompt="你是知识库管理者，有检索搜寻工具，它有绝大多数可查询的文档，调用它获取专业知识回答用户",
        checkpointer=checkpointer,
    )


vectorstore = _load_vs()

# --- 主循环 ---
print("1. 导入文档到数据库")
print("2. 查询知识库")
print("3. 删除数据库")
print("exit : 退出")

while True:
    with SqliteSaver.from_conn_string(str(MEMORY_DIR / "checkpoints.db")) as checkpointer:
        agent = _make_agent(vectorstore, checkpointer)

        option = input("输入对应数字来使用功能:\n")

        if option == "1":
            print("功能1: 导入文档")
            folder = input("请输入文档文件夹名称:\n")
            doc_dir = DOCUMENT_DIR / folder
            if not doc_dir.exists():
                print(f"错误输入!! 文件夹 {doc_dir} 不存在!!")
                continue
            docs = load_all_md(str(doc_dir))
            splits = split_text(docs)
            vectorstore = build_vectorstore(splits, emb)
            save_vectorstore(vectorstore, str(DATABASE_DIR), INDEX_NAME)
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
            if delete_vectorstore(str(DATABASE_DIR), INDEX_NAME):
                vectorstore = None

        elif option in ("exit", "退出"):
            print("退出中...")
            break

        else:
            print(f"错误输入 !! option={option} !!")
