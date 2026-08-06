# 加载 .md 文件

from pathlib import Path
from functools import partial
from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_all_md(
        dir_path: str,
        show_progress: bool = False,
        preview: bool = False
):
    """加载目录及子目录下所有 markdown 文件"""

    path = Path(dir_path)

    if not path.exists():
        raise FileNotFoundError(dir_path)

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

    for doc in docs:
        file_path = doc.metadata.get("source", "")
        doc.metadata.update({
            "source": file_path,
            "file_type": "markdown"
        })

    if preview:
        print(f"加载文件数量: {len(docs)}")

        if docs:
            doc = docs[0]
            print("类型:", type(doc))
            print("长度:", len(doc.page_content))
            print("来源:", doc.metadata.get("source"))
            print(doc.page_content[:800])
    return docs


def load_md(file_path: str):
    """加载单个 markdown 文件"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(file_path)

    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source": file_path,
            "file_type": "markdown"
        })

    return docs


if __name__ == "__main__":
    docs = load_all_md(
        r"D:\dump\my_project\mini_knowledge_rag\backend\document\langchain_doc",
        show_progress=True,
        preview=True
    )