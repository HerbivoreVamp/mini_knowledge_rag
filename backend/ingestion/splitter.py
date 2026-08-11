# 递归文本分块
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import logger
from core.exceptions import SplitError


def split_text(
        docs,
        chunk_size=800,
        chunk_overlap=100,
        add_start_index=True,
        preview=False
):
    """
    文档切分
    :param docs: 由Document组成的的列表
    :param chunk_size: 分块大小
    :param chunk_overlap: 重叠大小
    :param add_start_index: 记录文本开始的序号
    :param preview: 展示分块部分内容
    :return:
    """

    if not docs:
        raise SplitError("没有文档可进行切分")

    if chunk_size <= 0:
        raise SplitError("chunk_size必须大于0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise SplitError(
            "chunk_overlap必须>=0且小于chunk_size"
        )

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
        )

        splits = text_splitter.split_documents(docs)

    except Exception as e:
        logger.exception("文本切分失败")
        raise SplitError("文本切分失败") from e

    if not splits:
        raise SplitError("文本切分后没有产生有效chunk")

    logger.info(
        f"文本切分完成 chunks={len(splits)}"
    )

    if preview:
        for i, s in enumerate(splits):
            print(f"\n--- 第 {i + 1} 块 ---")
            print("page_content:", s.page_content[:100], "...略...")
            print("metadata:", s.metadata)

    return splits


if __name__ == "__main__":
    from loader import *

    docs = load_all_md(
        r"D:\dump\my_project\mini_knowledge_rag\backend\document\langchain_doc")
    splits = split_text(docs, preview=True)
    print(type(splits))
