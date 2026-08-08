# 加载 .md 文件

from pathlib import Path
from functools import partial
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from utils.logger import logger
from utils.exceptions import LoaderError


def load_all_md(
        dir_path: str,
        show_progress: bool = False,
        preview: bool = False,
        single_file: bool = False
):
    """加载 markdown 文件，支持目录批量导入和单文件导入"""

    path = Path(dir_path)

    if not path.exists():
        raise LoaderError(
            f"文档路径不存在 path={path}"
        )

    try:
        if single_file:
            loader = TextLoader(
                str(path),
                encoding="utf-8"
            )
        else:
            loader = DirectoryLoader(
                str(path),
                glob="**/*.md",
                loader_cls=partial(
                    TextLoader,
                    encoding="utf-8"
                ),
                show_progress=show_progress
            )

        docs = loader.load()

    except Exception as e:
        raise LoaderError(
            f"文档加载失败 path={path} error={e}"
        ) from e

    if not docs:
        raise LoaderError(
            f"没有找到 markdown 文件 path={path}"
        )

    valid_docs = []  # 筛除空文档
    skip_count = 0
    for doc in docs:
        if not doc.page_content.strip():
            if not single_file:
                logger.warning(
                    f"跳过空文档 path={doc.metadata.get('source')}"
                )
            skip_count += 1
            continue

        file_path = doc.metadata.get("source", "")
        doc.metadata.update({
            "source": file_path,
            "file_type": "markdown"
        })

        valid_docs.append(doc)
    if not valid_docs:
        if single_file:
            raise LoaderError(f"指定markdown文件为空文件 path={path}")
        else:
            raise LoaderError(f"没有有效 markdown 文件 path={path}")
    logger.info(
        f"文件加载完成 path={path};"
        f"success={len(valid_docs)} skipped={skip_count}"
    )
    if preview:
        print(f"加载文件数量: {len(docs)}")
        doc = valid_docs[0]
        print("类型:", type(doc))
        print("长度:", len(doc.page_content))
        print("来源:", doc.metadata.get("source"))
        print(doc.page_content[:800])
    return valid_docs


if __name__ == "__main__":
    docs = load_all_md(
        r"D:\dump\my_project\mini_knowledge_rag\backend\document\langchain_doc",
        show_progress=True,
        preview=True
    )
