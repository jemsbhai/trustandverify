"""Shared pytest fixtures for trustandverify tests."""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from trustandverify.core.models import Evidence, Source


@pytest.fixture
def source_high_trust() -> Source:
    return Source(
        url="https://www.cdc.gov/example",
        title="CDC Study",
        content_snippet="Government source content.",
        trust_score=0.9,
    )


@pytest.fixture
def source_low_trust() -> Source:
    return Source(
        url="https://www.reddit.com/r/example",
        title="Reddit Post",
        content_snippet="Reddit content.",
        trust_score=0.35,
    )


@pytest.fixture
def opinion_strong_support() -> Opinion:
    return Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)


@pytest.fixture
def opinion_strong_refute() -> Opinion:
    return Opinion(belief=0.1, disbelief=0.7, uncertainty=0.2, base_rate=0.5)


@pytest.fixture
def opinion_vacuous() -> Opinion:
    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)


@pytest.fixture
def evidence_supporting(source_high_trust: Source) -> Evidence:
    return Evidence(
        text="Studies show significant productivity gains.",
        supports_claim=True,
        relevance=0.9,
        confidence_raw=0.8,
        source=source_high_trust,
    )


@pytest.fixture
def evidence_contradicting(source_low_trust: Source) -> Evidence:
    return Evidence(
        text="No measurable productivity difference found.",
        supports_claim=False,
        relevance=0.7,
        confidence_raw=0.7,
        source=source_low_trust,
    )
