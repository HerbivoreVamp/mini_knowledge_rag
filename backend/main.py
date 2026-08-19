from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import get_settings
from config.prompts import SYSTEM_PROMPT
from config.model import create_embedding, create_llm
from ingestion.service import ingestion_service
from ingestion.hierarchical import create_docstore, create_parent_retriever
from storage.vectorstore import load_vectorstore, delete_vectorstore
from application.service import create_generation_service
from core.logger import setup_logger
from core.exceptions import RAGError
from core.utils import clear_checkpoints

logger = setup_logger()
logger.info("程序启动")
# --- 配置加载 ---
settings = get_settings()
# --- 模型初始化 ---
emb = create_embedding(settings)
llm = create_llm(settings)
# --- 向量库加载 ---
try:
    vectorstore = load_vectorstore(settings.vectorstore_dir, emb, settings.index_name)
    retriever = create_parent_retriever(vectorstore=vectorstore,
                                        parent_store=create_docstore(settings.parent_store_dir)
                                        )
    logger.info(f"vectorstore导入成功 path={settings.vectorstore_dir}")
    logger.info(f"retriever导入成功 path={settings.parent_store_dir}")
except RAGError as e:
    print(e)
    vectorstore = None
    retriever = None
    logger.info("导入失败，初始化vectorstore和retriever")

# --- 主循环 ---
print("1. 导入文档到数据库")
print("2. 查询知识库")
print("3. 删除数据库并删除记忆")
print("4. 仅删除记忆")
print("exit : 退出")

while True:
    option = input("输入对应数字来使用功能:\n")
    if option == "1":
        print("功能1: 导入文档")
        folder = input("请输入文档文件夹名称:\n")
        try:
            retriever = ingestion_service(document_dir=settings.document_dir,
                                          folder=folder,
                                          emb=emb,
                                          vectorstore_dir=settings.vectorstore_dir,
                                          index_name=settings.index_name,
                                          parent_store_dir=settings.parent_store_dir,
                                          vectorstore=vectorstore,
                                          retriever=retriever,
                                          )
        except RAGError:
            continue
        print("保存成功")
    elif option == "2":
        with SqliteSaver.from_conn_string(str(settings.memory_dir / "checkpoints.db")) as checkpointer:
            try:
                generator = create_generation_service(llm, SYSTEM_PROMPT, retriever, checkpointer)
            except RAGError as e:
                print(e)
                continue
            print("功能2: 查询知识库，输入exit退出问询")
            while True:
                input_mes = input("您想查询什么？\n")
                try:
                    if input_mes in ("exit", "退出"):
                        logger.info("用户手动结束问询")
                        print("结束问询")
                        break
                    mes = generator(input_mes, settings.config)
                    mes["messages"][-1].pretty_print()
                except RAGError as e:
                    print(e)
                    continue
    elif option == "3":
        print("功能3: 删除数据库并删除记忆")
        try:
            delete_vectorstore(str(settings.database_dir), settings.database_name)
            vectorstore = None
            retriever = None
        except RAGError as e:
            print(e)
            continue
        clear_checkpoints(settings.memory_dir / "checkpoints.db")
    elif option == "4":
        clear_checkpoints(settings.memory_dir / "checkpoints.db")  # 删除记忆
        print("记忆已删除")
    elif option in ("exit", "退出"):
        logger.info(f"用户手动退出")
        print("退出中...")
        break
    else:
        print(f"错误输入， option={option} !!")
        logger.info(f"用户错误输入了{option}")
