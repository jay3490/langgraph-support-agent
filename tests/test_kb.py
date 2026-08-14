from __future__ import annotations

import pytest

from support_agent.kb import Article, KnowledgeBase, parse_article, tokenize
from tests.conftest import KB_DIR


def test_tokenize_drops_stopwords_and_stems() -> None:
    assert tokenize("How do I reset my passwords?") == ["reset", "password"]
    assert tokenize("Shipped shipping") == ["shipp", "shipp"]
    assert tokenize("address") == ["address"]


def test_parse_article_uses_heading_as_title() -> None:
    article = parse_article("x", "# Title here\n\nBody one.\n\nBody two.")
    assert article.title == "Title here"
    assert article.paragraphs == ["Body one.", "Body two."]
    assert parse_article("no-heading", "Just text").title == "No heading"


def test_loads_every_article(kb: KnowledgeBase) -> None:
    assert len(kb.articles) == len(list(KB_DIR.glob("*.md")))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("I forgot my password and the reset link expired", "password-reset"),
        ("lost my phone with the authenticator app", "two-factor-authentication"),
        ("how long until my refund appears on my statement", "refund-policy"),
        ("my tracking number has not updated", "shipping-and-delivery"),
        ("the app won't sync my orders", "app-sync-troubleshooting"),
        ("I see a duplicate charge on my card", "payment-methods"),
        ("can I cancel an order", "order-cancellation"),
    ],
)
def test_top_hit(kb: KnowledgeBase, query: str, expected: str) -> None:
    hits = kb.search(query)
    assert hits, query
    assert hits[0].slug == expected


def test_results_are_ranked_and_limited(kb: KnowledgeBase) -> None:
    hits = kb.search("refund order delivery", k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_snippet_is_the_best_matching_paragraph(kb: KnowledgeBase) -> None:
    hit = kb.search("backup codes")[0]
    assert "backup codes" in hit.snippet
    assert "\n" not in hit.snippet


def test_no_match_returns_nothing(kb: KnowledgeBase) -> None:
    assert kb.search("zeppelin") == []
    assert kb.search("the and of") == []


def test_title_terms_are_boosted() -> None:
    kb = KnowledgeBase(
        [
            Article("a", "Gift cards", "Some text about vouchers and balances."),
            Article("b", "Vouchers", "You can also buy a gift card here among other things."),
        ]
    )
    assert kb.search("gift card")[0].slug == "a"


def test_missing_directory() -> None:
    with pytest.raises(FileNotFoundError):
        KnowledgeBase.from_directory(KB_DIR / "missing")
