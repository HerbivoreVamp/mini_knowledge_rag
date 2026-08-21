from langgraph.checkpoint.sqlite import SqliteSaver

from backend.config.settings import get_settings
from backend.config.prompts import SYSTEM_PROMPT
from backend.config.model import create_embedding, create_llm, create_reranker
from backend.ingestion.service import ingestion_service
from backend.retrieval.hierarchical_retriever import HierarchicalRetriever
from backend.retrieval.search import search
from backend.storage.docstore import create_docstore
from backend.storage.vectorstore import load_vectorstore, delete_vectorstore
from backend.application.service import create_generation_service
from backend.core.logger import setup_logger
from backend.core.exceptions import RAGError
from backend.core.utils import clear_checkpoints

logger = setup_logger()
logger.info("程序启动")
# --- 配置加载 ---
settings = get_settings()
# --- 模型初始化 ---
emb = create_embedding(settings)
reranker = create_reranker(settings)
llm = create_llm(settings)
# --- 向量库加载 ---
try:
    vectorstore = load_vectorstore(settings.vectorstore_dir, emb, settings.index_name)
    retriever = HierarchicalRetriever(vectorstore=vectorstore,
                                      parent_store=create_docstore(settings.parent_store_dir),
                                      reranker=reranker
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
print("5. 手动查询知识库")
print("exit : 退出")

while True:
    option = input("输入对应数字来使用功能:\n")
    if option == "1":
        print("功能1: 导入文档")
        folder = input("请输入文档文件夹名称:\n")
        try:
            retriever, vectorstore = ingestion_service(document_dir=settings.document_dir,
                                                       folder=folder,
                                                       emb=emb,
                                                       vectorstore_dir=settings.vectorstore_dir,
                                                       index_name=settings.index_name,
                                                       parent_store_dir=settings.parent_store_dir,
                                                       vectorstore=vectorstore,
                                                       retriever=retriever,
                                                       reranker=reranker
                                                       )
        except RAGError:
            print(f"找不到对应文件夹{folder}")
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
            delete_vectorstore(settings.database_dir, settings.database_name)
            vectorstore = None
            retriever = None
        except RAGError:
            print("删除失败")
            continue
        clear_checkpoints(settings.memory_dir / "checkpoints.db")
    elif option == "4":
        if clear_checkpoints(settings.memory_dir / "checkpoints.db"):  # 删除记忆
            print("记忆已删除")
        else:
            print("记忆不存在 无需删除")
    elif option == "5":
        if (not vectorstore) or (not retriever):
            print("数据库不存在")
            print("请先导入文档")
            continue
        docs = []
        try:
            docs = search(retriever=retriever, query=input("输入检索关键词"), k=int(input("输入检索返回的数量")))
        except RAGError as e:
            print(f"检索出错 error={e}")
        for doc in docs:
            print(f"""来源:{doc.metadata.get("source")}\n内容:{doc.page_content}\n""")
    elif option in ("exit", "退出"):
        logger.info(f"用户手动退出")
        print("退出中...")
        break
    else:
        print(f"错误输入， option={option} !!")
        logger.info(f"用户错误输入了{option}")
