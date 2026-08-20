# 递归文本分块
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.core.logger import logger
from backend.core.exceptions import SplitError


def create_text_splitter(chunk_size=800, chunk_overlap=100, add_start_index=True):
    """创建文本分块器"""
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
    except Exception as e:
        logger.error("splitter创建失败")
        raise SplitError("splitter创建失败") from e
    return text_splitter


# 弃用
def split_text(
        docs,
        chunk_size=800,
        chunk_overlap=100,
        add_start_index=True,
        preview=False
):
    """
    文档切分
    :param docs: 由Document组成的的列表[Document,...]
    :param chunk_size: 分块大小
    :param chunk_overlap: 重叠大小
    :param add_start_index: 记录文本开始的序号
    :param preview: 展示分块部分内容
    :return:
    """

    if not docs:
        raise SplitError("没有文档可进行切分")

    if not isinstance(docs[0], Document):
        raise SplitError(
            "docs必须是由Document组成的的列表[Document,...]"
        )
    text_splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                         add_start_index=add_start_index)
    try:
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


def create_hierarchy_splitter(parent_chunk_size=2000,
                              parent_chunk_overlap=200,
                              child_chunk_size=400,
                              child_chunk_overlap=50):
    try:
        parent_splitter = create_text_splitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap,
                                               add_start_index=True)
        child_splitter = create_text_splitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap,
                                              add_start_index=True)
    except Exception as e:
        logger.exception(f"hierarchy_splitter创建失败 error={e}")
        raise SplitError("hierarchy_splitter创建失败") from e
    return parent_splitter, child_splitter


