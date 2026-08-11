from unittest.mock import MagicMock
import pytest

from retrieval.search import create_retrieve_tool
from core.exceptions import RetrievalError


class FakeDoc:
    def __init__(self, content, source):
        self.page_content = content
        self.metadata = {
            "source": source
        }


def test_create_retrieve_tool_vectorstore_none():
    with pytest.raises(RetrievalError):
        create_retrieve_tool(None)


def test_retrieve_context_success():
    fake_vectorstore = MagicMock()

    fake_vectorstore.similarity_search.return_value = [
        FakeDoc(
            "这是测试内容",
            "test.md"
        )
    ]

    tool = create_retrieve_tool(fake_vectorstore)

    result = tool.invoke(  # invoke直接会返回context
        {
            "query": "测试问题"
        }
    )

    assert "这是测试内容" in result
    assert "test.md" in result

    fake_vectorstore.similarity_search.assert_called_once_with(  # 提醒自己:如果k改成(模型)自定义检索数量需要修改
        "测试问题",
        k=5
    )


def test_retrieve_context_empty_query():
    fake_vectorstore = MagicMock()
    tool = create_retrieve_tool(fake_vectorstore)
    result = tool.invoke(
        {
            "query": " "
        }
    )

    assert "query为空" in result


def test_retrieve_context_exception():
    fake_vectorstore = MagicMock()
    fake_vectorstore.similarity_search.side_effect = Exception(  # 手动报错
        "faiss error"
    )
    tool = create_retrieve_tool(fake_vectorstore)
    result = tool.invoke(
        {
            "query": "测试"
        }
    )
    assert "检索工具执行失败" in result
