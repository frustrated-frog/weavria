"""语料来源和标注完整性测试;不下载文档,不运行真实编码器。"""

import hashlib
from itertools import pairwise
from pathlib import Path

import pytest

from examples.document_memory_retrieval import (
    Corpus,
    Document,
    Question,
    load_corpus,
    passages,
    snapshot,
)


def test_document_packing_preserves_all_source_lines() -> None:
    text = "Title\n=====\n\n" + "".join(f"Paragraph {i}.\nMore text.\n\n" for i in range(200))
    result = passages("example.rst", text)
    assert "".join(p.text for p in result) == text
    assert result[0].start == 1 and result[-1].end == len(text.splitlines())
    assert all(a.end + 1 == b.start for a, b in pairwise(result))


def test_pinned_corpus_integrity_and_labels_never_enter_snapshot(tmp_path: Path) -> None:
    data = b"Paragraph one.\n\nParagraph two.\n"
    path = tmp_path / "example.rst"
    path.write_bytes(data)
    corpus = Corpus(
        version="test/1",
        documents=(
            Document(
                name=path.name,
                url="https://example.org/document",
                sha256=hashlib.sha256(data).hexdigest(),
            ),
        ),
        questions=(
            Question(
                id="secret-label",
                split="validation",
                category="literal",
                query="Do not leak this question.",
                document=path.name,
                lines=(3, 3),
            ),
        ),
    )
    result = load_corpus(corpus, tmp_path, download=False)
    serialized = snapshot(corpus, result).model_dump_json()
    assert "secret-label" not in serialized and "Do not leak" not in serialized
    assert "Paragraph two." in serialized
    path.write_bytes(data + b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_corpus(corpus, tmp_path, download=False)


def test_labels_outside_source_fail_instead_of_silently_scoring_zero(tmp_path: Path) -> None:
    data = b"Paragraph.\n"
    (tmp_path / "example.rst").write_bytes(data)
    corpus = Corpus(
        version="test/1",
        documents=(
            Document(
                name="example.rst",
                url="https://example.org/document",
                sha256=hashlib.sha256(data).hexdigest(),
            ),
        ),
        questions=(
            Question(
                id="bad",
                split="validation",
                category="literal",
                query="question",
                document="example.rst",
                lines=(2, 3),
            ),
        ),
    )
    with pytest.raises(ValueError, match="complete passage"):
        load_corpus(corpus, tmp_path, download=False)
