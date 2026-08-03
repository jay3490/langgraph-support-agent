"""Okapi BM25 search over the Markdown help-center articles in ``kb/``.

Each article is one document. The first ``# `` heading is the title and is indexed
twice so that title matches outweigh incidental body matches. The snippet returned for
a hit is the article paragraph that shares the most terms with the query.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORD_TEXT = """
a about after all an and any are as at be because been before but by can cannot do does
for from has have how i if in into is it its me my no not of on once only or our so than
that the their them then there these this to up us was we what when where which while who
why will with you your
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def _stem(token: str) -> str:
    """A deliberately small suffix stripper; it only needs to be consistent."""
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Article:
    slug: str
    title: str
    body: str

    @property
    def paragraphs(self) -> list[str]:
        return [" ".join(p.split()) for p in re.split(r"\n\s*\n", self.body) if p.strip()]


@dataclass(frozen=True)
class SearchHit:
    slug: str
    title: str
    score: float
    snippet: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "article": self.slug,
            "title": self.title,
            "score": round(self.score, 3),
            "snippet": self.snippet,
        }


def parse_article(slug: str, text: str) -> Article:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("# "):
        return Article(slug=slug, title=lines[0][2:].strip(), body="\n".join(lines[1:]).strip())
    return Article(slug=slug, title=slug.replace("-", " ").capitalize(), body=text.strip())


class KnowledgeBase:
    def __init__(self, articles: list[Article], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.articles = articles
        self.k1 = k1
        self.b = b
        self._term_freqs = [
            Counter(tokenize(f"{a.title} {a.title} {a.body}")) for a in self.articles
        ]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self.articles else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        n = len(self.articles)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    @classmethod
    def from_directory(cls, directory: Path) -> KnowledgeBase:
        if not directory.is_dir():
            raise FileNotFoundError(f"Knowledge base directory not found: {directory}")
        articles = [
            parse_article(path.stem, path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.md"))
        ]
        return cls(articles)

    def _score(self, index: int, terms: list[str]) -> float:
        tf = self._term_freqs[index]
        norm = self.k1 * (1 - self.b + self.b * self._lengths[index] / self._avg_length)
        score = 0.0
        for term in terms:
            freq = tf.get(term, 0)
            if freq:
                score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
        return score

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not self.articles:
            return []
        scored = [(self._score(i, terms), i) for i in range(len(self.articles))]
        ranked = sorted((pair for pair in scored if pair[0] > 0), key=lambda p: (-p[0], p[1]))
        return [
            SearchHit(
                slug=self.articles[i].slug,
                title=self.articles[i].title,
                score=score,
                snippet=self._snippet(self.articles[i], set(terms)),
            )
            for score, i in ranked[:k]
        ]

    @staticmethod
    def _snippet(article: Article, terms: set[str]) -> str:
        paragraphs = article.paragraphs
        if not paragraphs:
            return ""
        return max(paragraphs, key=lambda p: len(terms.intersection(tokenize(p))))
