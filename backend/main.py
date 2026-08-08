from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import get_settings
from config.prompts import SYSTEM_PROMPT
from ingestion.embedding import embeddings
from ingestion.loader import load_all_md
from ingestion.splitter import split_text
from ingestion.delete import delete_vectorstore
from retrieval.vectorstore_manage import load_vectorstore, save_vectorstore, build_vectorstore
from generation.chat import chat_generation
from agent.agent import create_rag_agent
from utils.logger import setup_logger
from utils.exceptions import *

logger = setup_logger()
logger.info("程序启动")
# --- 配置加载 ---
settings = get_settings()
# --- 模型初始化 ---
emb = embeddings(settings.emb_dir_path, settings.emb_model_name)
llm = init_chat_model(model=settings.model, model_provider=settings.model_provider, base_url=settings.base_url,
                      api_key=settings.api_key)
logger.info(f"语言模型配置加载成功 model={settings.model}")
# --- 向量库加载 ---
try:
    vectorstore = load_vectorstore(settings.database_dir, emb, settings.index_name)
except VectorStoreNotFoundError as e:
    logger.warning(e)
    vectorstore = None

# --- 主循环 ---
print("1. 导入文档到数据库")
print("2. 查询知识库")
print("3. 删除数据库")
print("exit : 退出")
with SqliteSaver.from_conn_string(str(settings.memory_dir / "checkpoints.db")) as checkpointer:
    while True:
        option = input("输入对应数字来使用功能:\n")
        if option == "1":
            print("功能1: 导入文档")
            folder = input("请输入文档文件夹名称:\n")
            doc_dir = settings.document_dir / folder
            try:
                docs = load_all_md(str(doc_dir))
            except RAGError as e:
                logger.error(e)
                print(e)
                continue
            splits = split_text(docs)
            vectorstore = build_vectorstore(splits, emb)
            save_vectorstore(vectorstore, str(settings.database_dir), settings.index_name)
            print("保存成功")


        elif option == "2":
            try:
                agent = create_rag_agent(llm, SYSTEM_PROMPT, vectorstore, checkpointer)
            except VectorStoreNotFoundError as e:
                logger.error(e)
                print("错误，vectorstore 为空，请先导入文档")
                continue
            print("功能2: 查询知识库，输入exit退出问询")
            while True:
                input_mes = input("您想查询什么？\n")
                if input_mes in ("exit", "退出"):
                    logger.info("用户手动结束问询")
                    print("结束问询")
                    break
                try:
                    mes = chat_generation(agent=agent, input_message=input_mes, config=settings.config)
                    mes["messages"][-1].pretty_print()
                except GenerationError as e:
                    logger.error(e)
                    print(e)
                    continue

        elif option == "3":

            print("功能3: 删除数据库")
            try:
                delete_vectorstore(str(settings.database_dir), settings.index_name)
                vectorstore = None
            except RAGError as e:
                logger.error(e)
                print(e)
                continue


        elif option in ("exit", "退出"):
            print("退出中...")
            break

        else:
            print(f"错误输入， option={option} !!")
            logger.info(f"用户错误输入了{option}")
