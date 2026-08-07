# 递归文本分块

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import logger


def split_text(docs, chunk_size=800, chunk_overlap=100, add_start_index=True, preview=False):
    """
    :param docs: 由Document组成的的列表
    :param chunk_size: 分块大小
    :param chunk_overlap: 重叠大小
    :param add_start_index: 记录文本开始的序号
    :param preview: 展示分块部分内容
    :return:
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )
    splits = text_splitter.split_documents(docs)
    logger.info(f"文本切分完成 len(splits)={len(splits)}")
    if preview:
        for i, s in enumerate(splits[:]):
            print(f"\n--- 第 {i + 1} 块 ---")
            print("page_content:", s.page_content[:100], "...略...")
            print("metadata:", s.metadata)
    return splits


if __name__ == "__main__":
    # from langchain_core.documents import Document
    #
    # intro_text = [Document(page_content=
    #                        "LangChain 是一个用于开发由大语言模型驱动的应用程序的开源框架。"
    #                        "它提供了一套统一的抽象和组件，帮助开发者将 LLM 与数据源、工具、记忆等结合起来，"
    #                        "构建诸如检索增强生成、智能体、聊天机器人等应用。\n"
    #                        "LangChain 的核心组件包括：Models（模型接口）、Prompts（提示词管理）、"
    #                        "Chains（链式调用）、Agents（智能体）、Memory（记忆）、Retrievers（检索器）、"
    #                        "Tools（工具）等。通过 LCEL 表达式语言，可以用管道符把这些组件组装成可流式、可并行的链。"
    #                        ),
    #               Document(
    #                   page_content="检索增强生成（RAG）通过先从外部知识库检索相关文档，再将其作为上下文喂给 LLM，从而缓解模型幻觉、引入私有知识。向量数据库用于存储文本的 "
    #                                "embedding，并支持基于语义相似度的近邻检索，常见实现有 Chroma、FAISS、Milvus 等。"),
    #               ]
    # splits = split_text(intro_text, chunk_size=80, chunk_overlap=20, preview=True)
    from loader import *

    docs = load_all_md(
        r"D:\dump\my_project\mini_knowledge_rag\backend\document\langchain_doc")
    splits = split_text(docs, preview=True)
    print(type(splits))
