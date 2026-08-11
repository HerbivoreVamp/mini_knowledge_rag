import pytest
from ingestion.loader import load_md
from core.exceptions import LoaderError


def test_missing_file(tmp_path):
    missing_file = tmp_path / "不存在的文件"
    with pytest.raises(LoaderError):
        load_md(
            dir_path=missing_file,
            single_file=True
        )


def test_empty_file(tmp_path):
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("")

    with pytest.raises(LoaderError):
        load_md(
            dir_path=empty_file,
            single_file=True
        )


def test_load_markdown(tmp_path):
    md_file = tmp_path / "测试用md文件.md"
    md_file.write_text("# hello")

    docs = load_md(
        dir_path=md_file,
        single_file=True
    )

    assert len(docs) > 0
