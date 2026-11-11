"""检索表示中的语言归一化;不重写原文、事实键或缺口语义。"""

import importlib
import re
from functools import lru_cache
from importlib.metadata import version
from typing import Protocol, cast

from .retrieval import surface_terms


class _Stemmer(Protocol):
    def stemWord(self, word: str) -> str: ...


class _Snowball(Protocol):
    def stemmer(self, language: str) -> _Stemmer: ...


class EnglishSnowballTerms:
    """显式英文模式;标准 Porter2 算法,原词与词干在不同命名空间。

    仅归一化纯 ASCII 单词。下划线/连字符标识、数字、中文仍精确匹配。
    每次检索拥有自己的词干器和有界缓存,不会跨租户保存原文词项,
    也不会并发使用 Snowball 的可变内部状态。
    """

    def __init__(self) -> None:
        try:
            backend = cast(_Snowball, importlib.import_module("snowballstemmer"))
        except ImportError as exc:
            raise ImportError("english_snowball requires weavria[lexical-memory]") from exc
        self.version = f"english-snowball/1:{version('snowballstemmer')}"
        stemmer = backend.stemmer("english")
        self._stem = lru_cache(maxsize=4096)(stemmer.stemWord)

    def __call__(self, text: str) -> tuple[str, ...]:
        output: list[str] = []
        for term in surface_terms(text):
            # 保留精确词面作为一份证据,再附加词干。词干相等只参与软排名,
            # 不能把不同 API、条件或主张合并为同一个事实族。
            output.append("surface:" + term)
            if re.fullmatch(r"[a-z]{3,64}", term):
                output.append("stem:" + self._stem(term))
        return tuple(output)
