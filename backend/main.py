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
from utils.logger import setup_logger
from utils.exceptions import RAGError,LoaderError,EmbeddingError,RetrievalError
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
def _load_vs(vector_embedding):
    index_path = settings.database_dir / settings.index_name

    if index_path.exists():
        try:
            vectorstore = load_vectorstore(
                str(settings.database_dir),
                vector_embedding,
                settings.index_name
            )
            logger.info(f"向量库{settings.index_name}加载成功")
            return vectorstore
        except Exception:
            logger.exception("向量库出错成功")
            raise
    logger.info(f"向量库{settings.index_name}文件不存在,需先导入文档进行初始化")
    return None


def _make_agent(vectorstore, checkpointer):
    """根据当前 vectorstore 和 checkpointer 创建 agent"""
    try:
        tool = create_retrieve_tool(vectorstore)
        agent = create_agent(llm, tools=[tool], system_prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
        logger.info("agent初始化创建成功")
        return agent
    except Exception:
        logger.exception("agent初始化创建失败")
        raise


vectorstore = _load_vs(emb)
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
            if vectorstore is None:
                logger.info("vectorstore为空，查询功能不可使用")
                print("错误，vectorstore 为空，请先导入文档")
                continue
            agent = _make_agent(vectorstore, checkpointer)
            print("功能2: 查询知识库，输入exit退出问询")
            while True:
                input_mes = input("您想查询什么？\n")
                if input_mes in ("exit", "退出"):
                    logger.info("用户手动结束问询")
                    print("结束问询")
                    break
                mes = chat_generation(agent=agent, input_message=input_mes, config=settings.config)
                mes["messages"][-1].pretty_print()

        elif option == "3":
            print("功能3: 删除数据库")
            if delete_vectorstore(str(settings.database_dir), settings.index_name):
                vectorstore = None
            else:
                print("数据库已经空了")

        elif option in ("exit", "退出"):
            print("退出中...")
            break

        else:
            print(f"错误输入， option={option} !!")
            logger.info(f"用户错误输入了{option}")
