import pytest
from pathlib import Path

from backend.ingestion.loader import load_md
from backend.core.exceptions import LoaderError


def test_missing_file(tmp_path):
    with pytest.raises(LoaderError):
        load_md(
            root_path=tmp_path,
            dir_path=Path("不存在的文件"),
            single_file=True
        )


def test_empty_file(tmp_path):
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("")

    with pytest.raises(LoaderError):
        load_md(
            root_path=tmp_path,
            dir_path=Path("empty.md"),
            single_file=True
        )


def test_load_markdown(tmp_path):
    md_file = tmp_path / "测试用md文件.md"
    md_file.write_text("# hello")

    docs = load_md(
        root_path=tmp_path,
        dir_path=Path("测试用md文件.md"),
        single_file=True
    )

    assert len(docs) > 0
